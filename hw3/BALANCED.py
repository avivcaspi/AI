from DT1 import load_data
import numpy as np
import sklearn as sk
from sklearn import tree


def classify(x_train, y_train, x_test):
    classifier = tree.DecisionTreeClassifier("entropy", random_state=2)
    classifier.fit(x_train, y_train)
    return classifier.predict(x_test)


def balance(x_train, y_train):
    # number of positive samples
    pos_size = (y_train == 1).sum()
    # indices of all positive samples and pos_size negative samples (first samples)
    neg_indices = np.flatnonzero(y_train == 0)[:pos_size]
    pos_indices = np.flatnonzero(y_train == 1)
    indices = sorted(np.append(neg_indices, pos_indices))
    x = x_train[indices]
    y = y_train[indices]
    return x, y


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, features_names = load_data('train.csv', 'test.csv')
    x_train, y_train = balance(x_train, y_train)
    y_pred = classify(x_train, y_train, x_test)
    confusion_matrix = sk.metrics.confusion_matrix(y_test, y_pred)
    print(confusion_matrix)
    # print(f'Error_w = {4 * confusion_matrix[1, 0] + confusion_matrix[0, 1]}')
