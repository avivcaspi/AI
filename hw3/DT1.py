import numpy as np
import sklearn as sk
from sklearn import tree


def load_data(train_file, test_file):
    train_set = np.genfromtxt(train_file, delimiter=',', skip_header=1)
    y_train = train_set[:, -1]
    x_train = train_set[:, :-1]
    test_set = np.genfromtxt(test_file, delimiter=',', skip_header=1)
    y_test = test_set[:, -1]
    x_test = test_set[:, :-1]
    features_names = np.genfromtxt(train_file, delimiter=',', dtype=str, skip_footer=train_set.shape[0])[:-1]
    return x_train, y_train, x_test, y_test, features_names


def classify(x_train, y_train, x_test):
    # building the decision tree and training
    classifier = tree.DecisionTreeClassifier("entropy", random_state=2)
    classifier.fit(x_train, y_train)
    # predicting the classes of x_test
    return classifier.predict(x_test)


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, features_names = load_data('train.csv', 'test.csv')
    # train and classify
    y_pred = classify(x_train, y_train, x_test)
    confusion_matrix = sk.metrics.confusion_matrix(y_test, y_pred)
    print(confusion_matrix)
    #  print(f'Error_w = {4 * confusion_matrix[1, 0] + confusion_matrix[0, 1]}')

