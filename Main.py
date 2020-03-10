import nltk
nltk.download('punkt')
from Miner import *
from HierarchicalArticleClassifier import *
from GraphsCreation import *
from StatisticsGenerator import *
from DataUtilities import *
import warnings

warnings.filterwarnings('ignore')

# Categories to be mined by levels:
level_1_categories = ["Category:Politics",
                      "Category:Health",
                      "Category:Crimes",
                      "Category:Technology",
                      "Category:Entertainment"]
level_2_categories = ["Category:Jewish political status", "Category:Voting", "Category:Government", "Category:Clothes in politics", "Category:Political corruption",
                      "Category:Nutrition", "Category:Mental health", "Category:Sexual health", "Category:Physical exercise",
                      "Category:Theft", "Category:Torture", "Category:Espionage", "Category:Cruelty to animals", "Category:Driving under the influence",
                      "Category:Mobile technology", "Category:Engineering", "Category:Computers", "Category:Science and technology ministries",
                      "Category:Comedy", "Category:Drama", "Category:Sports entertainment", "Category:Magic (illusion)", "Category:Gaming"]
level_3_categories = ["Category:Court Jews", "Category:Jewish political organizations",
                      "Category:Elections", "Category:Political whips",
                      "Category:National security", "Category:State ritual and ceremonies", "Category:Judiciaries",
                      "Category:Bribery", "Category:Electoral fraud", "Category:Lobbying",
                      "Category:Hunger", "Category:Infant feeding", "Category:Human weight",
                      "Category:Books about mental health", "Category:Mental and behavioural disorders",
                      "Category:Birth control", "Category:Andrology",
                      "Category:Sexually transmitted diseases and infections",
                      "Category:Running", "Category:Gymnastics", "Category:Tai chi",
                      "Category:Plagiarism", "Category:Car theft", "Category:Identity theft",
                      "Category:Physical torture techniques", "Category:Mind control",
                      "Category:Espionage devices", "Category:Espionage techniques",
                      "Category:Baiting (blood sport)", "Category:Animal sacrifice", "Category:Zoophilia",
                      "Category:Smartwatches", "Category:Mobile security", "Category:Mobile computers",
                      "Category:Construction", "Category:Industrial equipment", "Category:Standards",
                      "Category:Computer programming", "Category:Computer architecture", "Category:Data centers",
                      "Category:Clowning", "Category:Comedy genres",
                      "Category:Monologues", "Category:Tragedies (dramas)",
                      "Category:Cheerleading", "Category:Red Bull sports events",
                      "Category:Competitive eating",
                      "Category:Magic organizations", "Category:Magic tricks",
                      "Category:Game awards", "Category:Gamebooks", "Category:Game equipment"]


def main_regular():
    print("Main: starts")

    # data_file_name - the file where the articles & categories data are saved
    data_file_name = "./Data/3-Level-DataSet-Final"
    # root_category - the wikipedia root category from which the mining procedure begins
    root_category = "Category:Main_topic_classifications"

    ####################################################################################################################
    # Mining
    # - Data Mining
    # - Statistics
    ####################################################################################################################
    # - Data Mining
    # load_data_from_file - should the data be mined from wikipedia first or loaded directly from the JSON file
    load_data_from_file = True
    # max_level - the max level (inclusive) in the categories hierarchy which will be mined (level 0 is the root level)
    max_level = 3
    miner = Miner(root_category,
                  max_level=max_level,
                  load_from_file=load_data_from_file,
                  file_name=data_file_name,
                  level_1_categories=level_1_categories,
                  level_2_categories=level_2_categories,
                  level_3_categories=level_3_categories)
    data = miner.data

    # - Statistics
    statistics_by_category_file_name = './Data/statistics_by_category.csv'
    statistics_by_level_file_name = './Data/statistics_by_level.csv'
    StatisticsGenerator(data, statistics_by_level_file_name, statistics_by_category_file_name)

    ####################################################################################################################
    # Vectorization:
    ####################################################################################################################
    vectorization_types = [VectorizationType.full_text, VectorizationType.summary, VectorizationType.first_sentence]
    # should_normalize - should the resulting BERT vectors be normalized - this is a MUST if comparing using euclidean distance!
    should_normalize = True
    # should_vectorize - should the articles loaded from the file be vectorize
    should_vectorize = False
    vectorizer = TextVectorizer(data,
                                data_file_name,
                                vectorization_types,
                                should_vectorize,
                                should_normalize)
    data = vectorizer.data

    ####################################################################################################################
    # Data Splitting:
    ####################################################################################################################
    test_and_validation_sizes = 0.15
    data_splitter = DataSplitter(data, test_and_validation_sizes)
    x_test_ids, x_valid_ids = data_splitter.x_test_and_validation_ids

    ####################################################################################################################
    # Vectorization Evaluation:
    # - Parameter Selection
    # - Results Evaluation
    ####################################################################################################################
    # - Parameter Selection
    classifiers = [KNNClassifier(), Id3(), AdaBoost(), RandomForest(), SVM(), KerasNeuralNetwork()]
    # parameter_selection_folder - the folder where the results of the parameter selection process are saved
    parameter_selection_folder = "./Parameter Selection/Selected Parameters/"
    parameter_selector = ParameterSelector(data,
                                           x_test_ids,
                                           x_valid_ids,
                                           classifiers,
                                           vectorization_types,
                                           parameter_selection_folder,
                                           verbose=True,
                                           load_from_file=True)
    selected_parameters = parameter_selector.selected_parameters

    # - Results Evaluation
    # the max level for graph generation (inclusive)
    max_level_for_graphing = 1
    # the folder where the statistical graphs are saved
    graphs_folder = "./Parameter Selection/Graphs/"
    GraphCreation(max_level_for_graphing,
                  data,
                  classifiers,
                  vectorization_types,
                  selected_parameters,
                  parameter_selection_folder,
                  graphs_folder)

    ####################################################################################################################
    # Hierarchical Article Classifier
    ####################################################################################################################
    # the number of best estimator instances that will be used as part of the voting classification procedure
    voting_number_of_estimators = 3
    trained_classifiers_file_name = './trained_classifiers'
    confusion_matrix_file_name = './root_category_confusion_matrix'
    article_classifier = HierarchicalArticleClassifier(selected_parameters,
                                                       data,
                                                       x_test_ids,
                                                       x_valid_ids,
                                                       classifiers,
                                                       vectorization_types,
                                                       voting_number_of_estimators,
                                                       trained_classifiers_file_name,
                                                       load_from_file=True)
    article_classifier.select_and_fit_best_models_for_each_category()
    # The next line generates the statistical results shown in the report for the hierarchical classifier
    article_classifier.evaluate_accuracies_by_level_and_path_length()
    article_classifier.generate_confusion_matrix(confusion_matrix_file_name)

    print("Main: finished")


def main_online_classification(article_title, desired_path_length):
    if desired_path_length < 0 or desired_path_length > 3:
        print("An illegal value for the desired path length parameter was entered.")
        print("Please try again using a value between 0 and 3.")
        return
    print()
    print("Loading categories hierarchy data from file...")
    data_file_name = "./Data/3-Level-DataSet-Final"
    data = load_JSON_data(data_file_name + ".json")
    print("Loading done.")
    classifiers = [KNNClassifier(), Id3(), AdaBoost(), RandomForest(), SVM(), KerasNeuralNetwork()]
    parameter_selection_folder = "./Parameter Selection/Selected Parameters/"
    selected_parameters = load_JSON_data(parameter_selection_folder + "best_parameters.json")
    voting_number_of_estimators = 3
    trained_classifiers_file_name = './trained_classifiers'
    print("Initiating the hierarchical classifier...")
    online_article_classifier = HierarchicalArticleClassifier(selected_parameters,data,None,None,[],None,
                voting_number_of_estimators,trained_classifiers_file_name,load_from_file=True, verbose=False)
    print("Classifier initiation done.")
    print("Fitting the hierarchical classifier...")
    online_article_classifier.select_and_fit_best_models_for_each_category()
    print("Fitting process done.")
    print("Initiating the classification process...")
    online_article_classifier.predict_path_for_online_article(article_title, desired_path_length)
    print()
    print("Classification process done.")



#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
if __name__ == "__main__":
    # There are 2 running modes to choose from:
    ####################################################################################################################
    # Regular Mode - Includes all the project's steps one after the other. This is the default mode.
    ####################################################################################################################
    main_regular()

    ####################################################################################################################
    # Online mode - allows the user to input an article title and a max path length, and predicts a path
    #               of that length in the Wikipedia categories hierarchy.
    ####################################################################################################################
    # the article title to predict (as it appears on Wikipedia)
    article_title = "Parallel voting"
    # the desired path length to predict. the entered value must be between 0 and 3.
    desired_path_length = 3
    # main_online_classification(article_title, desired_path_length)
