from DT1 import load_data
import sklearn as sk
from sklearn import tree


def classify(x_train, y_train, x_test, d):
    classifier = tree.DecisionTreeClassifier("entropy", min_samples_split=9, class_weight={1: d, 0: 1-d})
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    return y_pred


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, features_names = load_data('train.csv', 'test.csv')
    y_pred = classify(x_train, y_train, x_test)
    confusion_matrix = sk.metrics.confusion_matrix(y_test, y_pred)
    print(confusion_matrix)

