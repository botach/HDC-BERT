# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# Utilities
from sklearn.preprocessing import StandardScaler
from DataClasses import RANDOM_SEED

# Keras Neural Network
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import os
    import sys
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')  # this is used to suppress Keras' initialization messages
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasClassifier

    sys.stderr = original_stderr


class GenericClassifier:
    def __init__(self):
        self.classifier = None
        self.name = None


class KNNClassifier(GenericClassifier):
    def __init__(self):
        super().__init__()
        self.classifier = KNeighborsClassifier()
        self.name = 'KNN'
        self.param_grid = \
        {
            'n_neighbors': range(1, 45, 2)
        }


class AdaBoost(GenericClassifier):
    def __init__(self):
        super().__init__()
        self.classifier = AdaBoostClassifier(algorithm='SAMME.R', random_state=RANDOM_SEED)
        self.name = 'AdaBoost'
        self.param_grid = \
        {
            'n_estimators': range(5, 205, 5)
        }


class Id3(GenericClassifier):
    def __init__(self):
        super().__init__()
        self.classifier = DecisionTreeClassifier(criterion="gini", random_state=RANDOM_SEED)
        self.name = 'Id3'
        self.param_grid = \
        {
            'min_samples_split': range(3, 15),
            'min_samples_leaf': range(1, 15)
        }


class RandomForest(GenericClassifier):
    def __init__(self):
        super().__init__()
        self.classifier = RandomForestClassifier(random_state=RANDOM_SEED)
        self.name = 'Random_Forest'
        self.param_grid = \
        {
            'min_samples_split': range(3, 13, 2),
            'min_samples_leaf': range(3, 13, 2),
            'n_estimators': range(5, 155, 25)
        }


class SVM:
    def __init__(self):
        super().__init__()
        self.classifier = svm.SVC(kernel='rbf',random_state=RANDOM_SEED)
        self.name = 'SVM'
        self.param_grid = \
        {
            'C': [1, 10, 50, 100, 300, 500, 750, 1000],
            'gamma': [0.1, 0.3, 0.5, 0.7, 1, 2, 5, 10, 20],
            'kernel': ['rbf', 'linear']
        }


class KerasNeuralNetwork(GenericClassifier):
    def __init__(self):
        super().__init__()
        self.name = "Neural Network"
        self.network = None
        self.labelEncoder = None
        self.categories_num = -1
        self.param_grid = \
        {
            'nn__epochs': range(10, 140, 20),
            'nn__units': range(4, 84, 10),
            'nn__optimizer': ["sgd", "adam"]
        }
        self.__init_classifier()

    def __init_classifier(self):
        estimators = [('ss', StandardScaler()),
                      ('nn', KerasClassifier(build_fn=self.create_model, epochs=10, batch_size=100, verbose=0))]
        self.classifier = Pipeline(estimators)

    def create_model(self, units=64, activation='relu', optimizer='adam'):
        model = Sequential()
        model.add(Dense(units=units, activation=activation, input_shape=(768,)))
        model.add(Dense(units=units, activation=activation))
        model.add(Dense(units=self.categories_num, activation='softmax'))
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',  # Cross-entropy
                      metrics=['accuracy'])  # Accuracy performance metric
        # Return compiled network
        return model
