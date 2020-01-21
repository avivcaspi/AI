from DT1 import load_data
import sklearn as sk
from sklearn import tree


def classify(x_train, y_train, x_test, delta):
    classifier = tree.DecisionTreeClassifier("entropy", min_samples_split=9, class_weight={1: delta, 0: 1-delta},
                                             random_state=2)
    classifier.fit(x_train, y_train)
    return classifier.predict(x_test)


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, features_names = load_data('train.csv', 'test.csv')
    y_pred = classify(x_train, y_train, x_test, 0.8)
    confusion_matrix = sk.metrics.confusion_matrix(y_test, y_pred)
    print(confusion_matrix)
    # print(f'Error_w = {4 * confusion_matrix[1, 0] + confusion_matrix[0, 1]}')

