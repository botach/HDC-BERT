import matplotlib.pyplot as plt
import csv
import re
import numpy as np


class GraphCreation:
    def __init__(self, max_level, data, classifiers, vectorization_types, selected_parameters, param_selection_folder, graphs_folder):
        print("GraphCreation: starts")
        self.param_selection_folder = param_selection_folder
        self.graphs_folder = graphs_folder
        self.classifier_names = [classifier.name for classifier in classifiers]
        self.category_names = []
        self.category_ids = []
        self.selected_parameters = selected_parameters
        for level in range(max_level+1):
            for category_id in data.categories_ids_by_levels[level]:
                self.category_ids.append(category_id)
                self.category_names.append(data.categories[category_id].title.replace('category:', ''))
        self.vectorization_types_names = [str(vec_type).replace('VectorizationType.', '') for vec_type in vectorization_types]
        self.vectorization_types = vectorization_types
        self.parameters = {}
        for clf_name in self.classifier_names:
            self.parameters[clf_name] = {}
            for category_name in self.category_names:
                self.parameters[clf_name][category_name] = {}
        self.load_files()
        self.create_basic_graphs()
        self.create_accuracy_histograms()
        print("GraphCreation: finished")

    def get_path(self, clf_name, vectorization_type, category_name):
        return self.param_selection_folder + clf_name + "_VectorizationType." + vectorization_type + "_category." + category_name \
               + ".csv"

    def load_files(self):
        for clf_name in self.classifier_names:
            for category_name in self.category_names:
                for vectorization_type in self.vectorization_types_names:
                    file_path = self.get_path(clf_name, vectorization_type, category_name)
                    with open(file_path, mode='r') as file:
                        reader = csv.reader(file)
                        headers = next(reader,None)[1:]
                        columns = {}
                        for h in headers:
                            columns[h] = []
                        for row in reader:
                            row = row[1:]
                            for h,v in zip(headers,row):
                                columns[h].append(v)
                        self.parameters[clf_name][category_name][vectorization_type] = columns

    def get_best_parameters_for(self, clf_name, category_name):
        max_score = -1
        best_params = {}
        current_params = self.parameters[clf_name][category_name]
        for vec_type in self.vectorization_types_names:
            for s in current_params[vec_type]['score']:
                if float(s) > max_score:
                    max_score = float(s)
                    s_index = current_params[vec_type]['score'].index(s)
                    for key in current_params[vec_type].keys():
                        if key != 'score':
                            best_params[key] = current_params[vec_type][key][s_index]
        return best_params

    def get_column_for_free_param(self, fixed_params, free_param, clf_name, category_name, vec_type):
        columns = {'score': [], free_param: []}
        current_params = self.parameters[clf_name][category_name][vec_type]
        for i in range(len(current_params['score'])):
            should_take_row_i = True
            for key in fixed_params:
                if current_params[key][i] != fixed_params[key]:
                    should_take_row_i = False
                    break
            if should_take_row_i:
                columns['score'].append(float(current_params['score'][i]))
                if re.search('[a-zA-Z]', current_params[free_param][i]):
                    columns[free_param].append(current_params[free_param][i])
                else:
                    columns[free_param].append(float(current_params[free_param][i]))
        return columns

    def create_graphs_for(self, clf_name, category_name):
        best_params = self.get_best_parameters_for(clf_name,category_name)
        for free_param in best_params.keys():
            fixed_params = dict(best_params)
            fixed_params.pop(free_param, None)
            plt.xlabel(beautify_string(free_param))
            plt.ylabel(beautify_string('score'))
            for vec_type in self.vectorization_types_names:
                graph_series = self.get_column_for_free_param(fixed_params, free_param, clf_name, category_name, vec_type)
                plt.plot(graph_series[free_param], graph_series['score'])
            plt.legend([beautify_string(vec_type) for vec_type in self.vectorization_types_names])
            plt.title(beautify_string(category_name), pad=30, fontsize=18, x=0.44)
            fixed_params_string = ": "
            for best_param_name in best_params.keys():
                if best_param_name != free_param:
                    fixed_params_string += beautify_string(best_param_name) + " = " + best_params[best_param_name] + " "
            if fixed_params_string == ": ":
                fixed_params_string = ""
            plt.suptitle(beautify_string(clf_name) + fixed_params_string,fontsize=11,y=0.89)
            plt.grid(linewidth=0.2)
            plt.tight_layout()
            plt.savefig(self.graphs_folder + clf_name + "_" + free_param + "_" + category_name + '.png', dpi=300)
            plt.clf()

    def create_basic_graphs(self):
        for clf_name in self.classifier_names:
            for category_name in self.category_names:
                self.create_graphs_for(clf_name, category_name)

    def create_accuracy_histograms(self):
        for category_id in self.category_ids:
            self.create_accuracy_histograms_for(category_id)

    def create_accuracy_histograms_for(self, category_id):
        best_parameters = self.selected_parameters
        x = np.arange(len(self.classifier_names))
        width = 0.25
        x_coordinates = [x-width, x, x+width]
        fig, ax = plt.subplots()
        for vec_type in self.vectorization_types:
            y_values = []
            for clf_name in self.classifier_names:
                y_values.append(best_parameters[str((clf_name, vec_type, category_id))][1])
            ax.bar(x_coordinates.pop(), y_values, width, label=beautify_string(str(vec_type)))
        plt.yticks(np.arange(0.4, 1.05, 0.05))
        plt.ylim(0.4, 1)
        ax.set_xticks(x)
        classifier_names = [beautify_string(classifier_name, is_for_histogram_axis=True) for classifier_name in
                            self.classifier_names]
        ax.set_xticklabels(classifier_names)
        ax.set_ylabel("Accuracy")
        category_name = beautify_string(self.category_names[int(category_id)])
        plt.title(category_name, fontsize=18, pad=10, x=0.47)
        plt.grid(linewidth=0.2)
        ax.legend()
        plt.savefig(self.graphs_folder + category_name + '_accuracy_histogram.png', dpi=300)
        plt.clf()


def beautify_string(string, is_for_histogram_axis=False):
    if is_for_histogram_axis:
        if string == 'KNN':
            return 'k-NN'
        if string == 'Id3':
            return 'ID3'
        string = string.replace('_', '\n')
        string = string.replace(' ', '\n')
        return string
    string = string.replace('nn__', '')
    string = string.replace('_', ' ')
    string = string.replace('VectorizationType.', '')
    string = string.title()
    if string == "Svm":
        return "SVM"
    if string == "Knn":
        return "k-Nearest Neighbors"
    if string == "N Estimators":
        return "N-Estimators"
    if string == "N Neighbors":
        return "N-Neighbors"
    return string
