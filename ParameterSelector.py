import math
import pandas as pd
from sklearn.metrics import confusion_matrix
from Classifiers import *
from FilesIO import *
from sklearn import metrics
from DataUtilities import *
from sklearn.model_selection import StratifiedKFold, GridSearchCV


class ParameterSelector:
    def __init__(self, data, x_test_ids, x_valid_ids, classifiers, vectorization_types, param_selection_folder,
                 verbose=True,
                 load_from_file=True):
        print("ParameterSelector: starts")
        self.data = data
        self.x_test_ids = x_test_ids
        self.x_valid_ids = x_valid_ids
        self.classifiers = classifiers
        self.vectorization_types = vectorization_types
        self.selected_parameters = {}
        self.param_selection_folder = param_selection_folder
        self.verbose = verbose
        if load_from_file:
            self.load_params_from_json()
        else:
            self.run_parameter_selection()
        print("ParameterSelector: finished")

    def run_parameter_selection(self):
        for clf in self.classifiers:
            for vectorization_type in self.vectorization_types:
                print("ParameterSelector: running parameter selection for classifier: " + clf.name + " with vectorization type: "
                      + str(vectorization_type))
                self.__run_parameter_selection(clf, vectorization_type)

    def __run_parameter_selection(self, clf, vectorization_type):
        try:
            for level in range(len(self.data.categories_ids_by_levels)):
                for category_id in self.data.categories_ids_by_levels[level]:
                    if len(self.data.categories[category_id].sub_categories_ids) == 0:
                        continue
                    if self.verbose:
                        print("ParameterSelector: level: " + str(level) + ", " + str(vectorization_type) + ", clf: " + clf.name)
                    key = str((clf.name, vectorization_type, category_id))
                    if key not in self.selected_parameters:
                        x_train, y_train, x_valid, y_valid, _, _ = \
                            extract_article_samples_of_sub_categories_of(category_id, self.data, self.x_test_ids,
                                                                         self.x_valid_ids, vectorization_type)

                        categories_num = len(set(y_train))
                        if categories_num < 2:
                            continue
                        if clf.name == "Neural Network":
                            # we reinitialise the neural network classifier before each use:
                            clf = KerasNeuralNetwork()
                            # the neural network classifier needs to know how many categories there are in order to build its model
                            clf.categories_num = categories_num
                        elif clf.name == "KNN":
                            max_k = math.floor(len(x_train) / 2)
                            clf.param_grid["n_neighbors"] = range(3, min(max_k, 21))

                        category_name = self.data.categories[category_id].title.replace(':', '.')
                        path_name = self.param_selection_folder + clf.name + "_" + str(
                            vectorization_type) + "_" + category_name + ".csv"
                        save_data = True if level < 2 else False

                        best_params = self.run_params_grid_search(clf, x_train, y_train, save_data, path_name)

                        clf.classifier.set_params(**best_params)
                        clf.classifier.fit(x_train, y_train)
                        y_pred = clf.classifier.predict(x_valid)
                        score_result = metrics.accuracy_score(y_valid, y_pred)

                        self.add_best_params(best_params, score_result, clf.name, vectorization_type, category_id)
                        self.save_params_to_json()
                        if self.verbose:
                            print("ParameterSelector:" + str(best_params))
            if self.verbose:
                print("ParameterSelector: parameter selection is done, saving data to json...")
            self.save_params_to_json()

        except:
            print("ParameterSelector: ERROR during parameter selection, saving data to json...")
            self.save_params_to_json()
            raise

    def run_params_grid_search(self, clf: GenericClassifier, x, y, save_data, csv_path):
        k = 5 if len(x) >= 100 else max(2, math.ceil(len(x) / 30))
        k_fold_generator = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
        grid_search = GridSearchCV(estimator=clf.classifier, param_grid=clf.param_grid, cv=k_fold_generator,
                                   scoring=self.scorer,
                                   n_jobs=-1, pre_dispatch='2*n_jobs', verbose=1)
        # grid_search = GridSearchCV(estimator=clf.classifier, param_grid=clf.param_grid, cv=KFoldGenerator,
        #                            scoring=self.scorer,
        #                            n_jobs=-1, verbose=1)
        grid_search.fit(x, y)
        if save_data:
            results = {'score': grid_search.cv_results_['mean_test_score']}
            for param_name in clf.param_grid.keys():
                for _ in grid_search.cv_results_['params']:
                    results[param_name] = []
            for param_name in clf.param_grid.keys():
                for params_set in grid_search.cv_results_['params']:
                    results[param_name].append(params_set[param_name])

            df = pd.DataFrame(results)
            df.to_csv(path_or_buf=csv_path)

        return grid_search.best_params_

    def scorer(self, estimator, x, y):
        y_pred = estimator.predict(x)
        cm = confusion_matrix(y, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        precisions = []
        for i in range(cm.shape[0]):
            correct_i_predictions = cm[i, i]
            total_i_samples = sum(cm[:, i])
            if total_i_samples > 0:
                i_precision = correct_i_predictions / total_i_samples
                precisions.append(i_precision)
        if len(precisions) == 0:
            return -1
        sorted_precisions = np.sort(precisions)
        return np.mean(sorted_precisions[:min(3, len(sorted_precisions))])

    def add_best_params(self, best_params, score_result, clf_name, vectorization_type, category_id):
        self.selected_parameters[str((clf_name, vectorization_type, category_id))] = [best_params, score_result]

    def get_best_params(self, clf_name, vectorization_type, category_id):
        key = str((clf_name, vectorization_type, category_id))
        if key in self.selected_parameters:
            return self.selected_parameters[key]
        else:
            return None

    def save_params_to_json(self):
        save_JSON_data(self.selected_parameters, self.param_selection_folder + "best_parameters.json", beautify=True)

    def load_params_from_json(self):
        try:
            self.selected_parameters = load_JSON_data(self.param_selection_folder + "best_parameters.json")
        except:
            pass
