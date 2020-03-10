from DataClasses import *
from ParameterSelector import *
from DataUtilities import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.multiclass import unique_labels
import itertools


class HierarchicalArticleClassifier:
    def __init__(self, selected_parameters, data, x_test_ids, x_valid_ids, classifiers,
                 vectorization_types, n_estimators, trained_classifiers_file_name, load_from_file=False, verbose = True):
        if verbose:
            print("HierarchicalArticleClassifier: starts")
        self.selected_parameters = selected_parameters
        self.data = data
        self.x_test_ids = x_test_ids
        self.x_valid_ids = x_valid_ids
        self.classifier_names = [classifier.name for classifier in classifiers]
        self.vectorization_types = vectorization_types
        self.n_estimators = n_estimators
        self.trained_classifiers_file_name = trained_classifiers_file_name
        self.classifiers_by_category_id = {}
        if load_from_file:
            if verbose:
                print("HierarchicalArticleClassifier: load classifiers from file")
            self.load_classifiers_from_file()
        if verbose:
            print("HierarchicalArticleClassifier: finished")

    def save_classifiers_to_file(self):
        save_JSON_data(self.classifiers_by_category_id, self.trained_classifiers_file_name + ".json", beautify=True)

    def load_classifiers_from_file(self):
        try:
            self.classifiers_by_category_id = load_JSON_data(self.trained_classifiers_file_name + ".json")
        except:
            pass

    def get_best_combinations_for_category(self, category_id):
        combinations = []
        for clf_name in self.classifier_names:
            for vec_type in self.vectorization_types:
                best_params, accuracy = self.selected_parameters[str((clf_name, vec_type, category_id))]
                combination = (clf_name, vec_type, best_params, accuracy)
                combinations.append(combination)
        combinations = sorted(combinations, key=lambda c: c[3], reverse=True)
        return combinations[:self.n_estimators]

    def get_classifiers_for_combinations(self, combinations, category_id):
        classifiers_list = []
        for combination in combinations:
            classifier_name, vec_type, best_params, _ = combination
            if classifier_name == 'Neural Network':
                keras_classifier = KerasNeuralNetwork()
                keras_classifier.categories_num = len(self.data.categories[category_id].sub_categories_ids)
                classifier = keras_classifier.classifier
            if classifier_name == 'SVM':
                classifier = svm.SVC()
            if classifier_name == 'KNN':
                classifier = KNeighborsClassifier()
            if classifier_name == 'Random_Forest':
                classifier = RandomForestClassifier()
            if classifier_name == 'AdaBoost':
                classifier = AdaBoostClassifier()
            if classifier_name == 'Id3':
                classifier = DecisionTreeClassifier()
            classifier.set_params(**best_params)
            classifiers_list.append((classifier, vec_type))
        return classifiers_list

    def select_and_fit_best_models_for_each_category(self):
        for level in range(3):
            for category_id in self.data.categories_ids_by_levels[level]:
                if len(self.data.categories[category_id].sub_categories_ids) == 0 or \
                        category_id in self.classifiers_by_category_id.keys():
                    continue
                self.classifiers_by_category_id[category_id] = []
                combinations = self.get_best_combinations_for_category(category_id)
                classifiers_list = self.get_classifiers_for_combinations(combinations, category_id)
                for vec_type in self.vectorization_types:
                    x_train, y_train, x_valid, y_valid, x_test, y_test = \
                        extract_article_samples_of_sub_categories_of(category_id, self.data, self.x_test_ids,
                                                                     self.x_valid_ids, vec_type)
                    x_train = x_train + x_valid
                    y_train = y_train + y_valid
                    for classifier, clf_vec_type in classifiers_list:
                        print(category_id + " " + str(vec_type) + " " + str(classifier))
                        if vec_type == clf_vec_type:
                            classifier.fit(x_train, y_train)
                            self.classifiers_by_category_id[category_id].append((classifier, clf_vec_type))
        self.save_classifiers_to_file()

    def most_frequent(self, in_list):
        counter = 0
        num = in_list[0]
        for i in in_list:
            current_frequent = in_list.count(i)
            if current_frequent > counter:
                counter = current_frequent
                num = i
        return num

    def predict_by_votes(self, article_id, classifiers_list):
        results_list = []
        for classifier, clf_vec_type in classifiers_list:
            vector = self.data.articles[str(article_id)].vectors[str(clf_vec_type)]
            prediction = classifier.predict(vector)
            results_list.append(prediction[0])
        return self.most_frequent(results_list)

    def is_category_in_article_classification_path(self, article_id, category_id):
        current_category_id = self.data.articles[str(article_id)].categories_ids.pop()
        self.data.articles[str(article_id)].categories_ids.add(current_category_id)
        while current_category_id is not None:
            if current_category_id == category_id:
                return True
            current_category_id = self.data.categories[current_category_id].parent_id
        return False

    def get_article_id_classification_path_length(self, article_id):
        current_category_id = self.data.articles[str(article_id)].categories_ids.pop()
        self.data.articles[str(article_id)].categories_ids.add(current_category_id)
        path_length = 0
        while current_category_id is not '0':
            path_length += 1
            current_category_id = self.data.categories[current_category_id].parent_id
        return path_length

    def evaluate_accuracies_by_level_and_path_length(self):
        path_accuracies = {}
        for i in range(1, 4):
            path_accuracies[str(i)] = []
            path_accuracies["level " + str(i - 1)] = []
        path_accuracies['all lengths'] = []
        x_test_ids = list(self.x_test_ids)
        for article_id in x_test_ids:
            original_path_length = self.get_article_id_classification_path_length(article_id)
            correct_path_length = 0
            root_category_id = '0'
            while True:
                classifiers_list = self.classifiers_by_category_id.get(root_category_id)
                if classifiers_list is None:
                    # we reached the last level
                    break
                predicted_category_id = str(self.predict_by_votes(article_id, classifiers_list))
                if not self.is_category_in_article_classification_path(article_id, predicted_category_id):
                    break
                else:
                    correct_path_length += 1
                    root_category_id = predicted_category_id
            for i in range(0, correct_path_length):
                path_accuracies["level " + str(i)].append(1)
            if correct_path_length != original_path_length:
                path_accuracies["level " + str(correct_path_length)].append(0)
            path_accuracy = correct_path_length / original_path_length
            path_accuracies[str(original_path_length)].append(path_accuracy)
            path_accuracies['all lengths'].append(path_accuracy)
        for key in path_accuracies.keys():
            path_accuracies[key] = np.mean(path_accuracies[key])
        return path_accuracies


    def predict_by_votes_for_online_article(self, article, classifiers_list):
        results_list = []
        for classifier, clf_vec_type in classifiers_list:
            vector = article.vectors[str(clf_vec_type)]
            prediction = classifier.predict(vector)
            results_list.append(prediction[0])
        return self.most_frequent(results_list)


    def predict_path_for_online_article(self, article_title, desired_path_length):
        print("Downloading article from Wikipedia...")
        try:
            article = get_article_with_title(article_title)
        except:
            print("An error occured while trying to download the article.")
            print("Please check you internet connection or the article's title and try again.")
            return
        print("Done downloading.")
        print("vectorizing article using BERT...")
        vectorizer = TextVectorizer(data=None,vectorized_data_file_name=None,vectorization_types=None,
                                    should_vectorize=False,should_normalize=True)
        for vec_type in [VectorizationType.first_sentence,VectorizationType.summary,VectorizationType.full_text]:
            vectorizer.vectorize = vectorizer.enum_to_vectorization_function(vec_type)
            article.vectors[str(vec_type)] = vectorizer.vectorize(article)
        print("Done vectorizing.")
        print("Predicting a path for the article...")
        path_list = ['Wikipedia-Root']
        current_path_length = 0
        root_category_id = '0'
        while current_path_length < desired_path_length:
            classifiers_list = self.classifiers_by_category_id.get(root_category_id)
            predicted_category_id = str(self.predict_by_votes_for_online_article(article, classifiers_list))
            predict_category_title = self.data.categories[predicted_category_id].title
            path_list.append(predict_category_title)
            current_path_length += 1
            root_category_id = predicted_category_id
        print("Prediction done:")
        print()
        print("#################################################################################################")
        print("The predicted path in the Wikipedia categories hierarchy for the article \"" + article.title + "\" is:")
        for category_title in path_list[:-1]:
            print(category_title, end=' ---> ')
        print(path_list[-1])
        print("#################################################################################################")


    def generate_confusion_matrix(self, output_path):
        root_category_id = '0'
        dummy_vec_type = VectorizationType.full_text
        x_all, y_all, _, x_all_ids = extract_all_article_samples_of_sub_categories_of(root_category_id, self.data,
                                                                                      dummy_vec_type)
        y_pred = []
        y_test = []
        classifiers_list = self.classifiers_by_category_id.get(root_category_id)
        for y_sample, x_id in zip(y_all, x_all_ids):
            if x_id in self.x_test_ids:
                predicted_category_id = str(self.predict_by_votes(x_id, classifiers_list))
                y_pred.append(self.data.categories[predicted_category_id].title.replace('category:', '').title())
                y_test.append(self.data.categories[str(y_sample)].title.replace('category:', '').title())

        categories_names = list(unique_labels(y_test,y_pred))
        cm = confusion_matrix(y_test, y_pred, labels=categories_names)
        category_name = self.data.categories[root_category_id].title.replace('category:','').title()

        plot_confusion_matrix(cm, normalize=False,title=category_name.replace('_',' '),classes=categories_names,
                              output_path=output_path)
        plot_confusion_matrix(cm, normalize=True,title=category_name.replace('_',' '),classes=categories_names,
                              output_path=output_path+"_normalized")


def plot_confusion_matrix(cm, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, classes=None,
                          output_path = None):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        np.set_printoptions(precision=2)
        fig, ax = plt.subplots()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, pad=10)
        plt.colorbar()
        tick_marks = np.arange(5)
        plt.xticks(tick_marks, rotation=45)
        ax = plt.gca()
        labels = (ax.get_xticks() + 1).astype(str)
        ax.set_xticklabels(classes)
        ax.set_yticklabels([''] + classes)

        thresh = cm.max() / 2.
        fmt = '.2f' if normalize else 'd'
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j],fmt),
                     horizontalalignment="center",
                     verticalalignment='center',
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True')
        plt.xlabel('Predicted', labelpad=-10)
        plt.savefig(output_path + '.png', dpi=300)
        plt.clf()

########################################################################################################################
########################################## Global Functions ############################################################
########################################################################################################################

def get_article_with_title(article_title):
    import wikipediaapi
    article_page = wikipediaapi.Wikipedia('en').page(article_title)  # TODO: if this stay hear we should import wikipedia api
    return Article(article_page)

