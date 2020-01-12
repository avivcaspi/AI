from KNN1 import *
import itertools


def find_best_features(x_train, y_train, x_test, y_test):
    classifier = KNNClassifier(k=9)
    best_accuracy = 0
    best_subset = None
    for combinations in get_all_subsets(x_test.shape[1]):
        for subset in combinations:
            if len(subset) == 0:
                continue
            x_train_converted, x_test_converted = convert_data_with_features(x_train, x_test, subset)
            classifier.train(x_train_converted, y_train)
            y_pred = classifier.predict(x_test_converted)
            accuracy, _, _, _, _, = get_accuracy(y_pred, y_test)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_subset = subset

    print(f'best subset : {list(best_subset)}\nbest accuracy : {best_accuracy}')


def get_all_subsets(num_features: int):
    features_list = list(range(num_features))
    for r in range(len(features_list) + 1):
        yield itertools.combinations(features_list, r)


def convert_data_with_features(x_train, x_test, features):
    converted_x_train = x_train[:, features]
    converted_x_test = x_test[:, features]
    return converted_x_train, converted_x_test


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, features_names = load_data('train.csv', 'test.csv')
    x_train, x_test = normalize_data(x_train, x_test)
    find_best_features(x_train, y_train, x_test, y_test)
