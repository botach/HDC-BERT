from DataClasses import RANDOM_SEED
from sklearn.model_selection import train_test_split
from TextVectorizer import *
import numpy as np


class DataSplitter:
    def __init__(self, data, test_and_validation_sizes=0.15):
        print("DataSplitter: starts")
        self.raw_data = data
        self.x_test_ids = set()
        self.x_valid_ids = set()
        self.x_test_and_validation_ids = None
        self.__generate_test_and_validation_sets(test_and_validation_sizes)
        print("DataSplitter: finished")

    def __generate_test_and_validation_sets(self, test_and_validation_sizes=0.15):
        root_category_id = '0'
        # the vec_type is irrelevant here
        dummy_vectoriation_type = VectorizationType.summary
        x, y, _, article_ids = extract_all_article_samples_of_sub_categories_of(root_category_id, self.raw_data, dummy_vectoriation_type)
        # note that the random state is fixed. This is important, as we want the same split for each vectorization type
        zipped = zip(x, article_ids)
        zipped_list = list(zipped)
        x_learn_zipped, x_test_zipped, y_learn, y_test = train_test_split(zipped_list, y, test_size=test_and_validation_sizes, stratify=y,
                                                                          random_state=RANDOM_SEED)
        # we want the size of the validation test group to be equal to the size of the test group
        validation_test_size = len(y_test)/len(y_learn)
        x_train_zipped, x_valid_zipped, y_train, y_valid = train_test_split(x_learn_zipped, y_learn, test_size=validation_test_size, stratify=y_learn,
                                                                            random_state=RANDOM_SEED)
        x_test, x_test_ids = zip(*x_test_zipped)
        x_valid, x_valid_ids = zip(*x_valid_zipped)
        self.x_test_ids = x_test_ids
        self.x_valid_ids = x_valid_ids
        self.x_test_and_validation_ids = (self.x_test_ids, self.x_valid_ids)


########################################################################################################################
########################################## Category-Based Set Extractor ################################################
########################################################################################################################
# this function is used for train, test & validation set generation given a category id
# based on the absolute test & validation sets
def extract_article_samples_of_sub_categories_of(category_id, data, x_test_ids_all, x_valid_ids_all, vectorization_type):
    x_all, y_all, _, article_ids = extract_all_article_samples_of_sub_categories_of(category_id, data, vectorization_type)
    x_train_of_sub_categories = []
    y_train_of_sub_categories = []
    x_valid_of_sub_categories = []
    y_valid_of_sub_categories = []
    x_test_of_sub_categories = []
    y_test_of_sub_categories = []

    for x_sample, y_sample, article_id in zip(x_all, y_all, article_ids):
        if article_id not in (set(x_test_ids_all) | set(x_valid_ids_all)):
            x_train_of_sub_categories.append(x_sample)
            y_train_of_sub_categories.append(y_sample)
        elif article_id in x_valid_ids_all:
            x_valid_of_sub_categories.append(x_sample)
            y_valid_of_sub_categories.append(y_sample)
        elif article_id in x_test_ids_all:
            x_test_of_sub_categories.append(x_sample)
            y_test_of_sub_categories.append(y_sample)

    return x_train_of_sub_categories, y_train_of_sub_categories, x_valid_of_sub_categories, y_valid_of_sub_categories, x_test_of_sub_categories, y_test_of_sub_categories


########################################################################################################################
########################################## Internal Helper Functions ###################################################
########################################################################################################################
def convert_to_numpy_arrays(x, y, articles_names):
    if not x:
        return [], [], []
    # convert the data to Numpy arrays:
    x_np_3_dim = np.array([np.array(x_i) for x_i in x])
    x_np = x_np_3_dim.reshape(x_np_3_dim.shape[0], x_np_3_dim.shape[2])
    y_np = np.array([int(y_i) for y_i in y])
    articles_names = np.array(articles_names)
    return x_np, y_np, articles_names


def article_in_more_than_one_category(article, data):
    intersection_size = len(article.categories_ids & data.categories_ids)
    return True if intersection_size > 1 else False


def article_has_all_vector_types(article):
    return article.vectors[str(VectorizationType.first_sentence)] is not None and\
                article.vectors[str(VectorizationType.summary)] is not None and \
                article.vectors[str(VectorizationType.full_text)] is not None


def extract_article_samples_of_category(category_id, data, vectorization_type):
    x = []
    articles_names = []
    article_ids = []
    category = data.categories[category_id]
    for article_id in category.descendant_articles:
        article = data.articles[article_id]
        if article_in_more_than_one_category(article, data) or not article_has_all_vector_types(article)\
                or vectorization_type is None:
            continue
        else:
            x.append(article.vectors[str(vectorization_type)])
            articles_names.append(article.title)
            article_ids.append(int(article.id))
    y = [category.id]*len(x)
    x, y, articles_names = convert_to_numpy_arrays(x, y, articles_names)
    return x, y, articles_names, article_ids


# the name implies 'all' because this includes the test and validation sets
def extract_all_article_samples_of_sub_categories_of(category_id, data, vectorization_type):
    x = []
    y = []
    articles_names = []
    article_ids = []
    root_category = data.categories[category_id]
    for sub_category_id in root_category.sub_categories_ids:
        sub_x, sub_y, sub_articles_names, sub_article_ids = extract_article_samples_of_category(sub_category_id, data, vectorization_type)
        x.extend(sub_x)
        y.extend(sub_y)
        articles_names.extend(sub_articles_names)
        article_ids.extend(sub_article_ids)

    return x, y, articles_names, article_ids
