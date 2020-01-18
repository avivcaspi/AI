from KNN1 import *
from OPT import convert_data_with_features


def sfs(x_train, y_train, x_test, y_test):
    features = []
    features_left = list(range(x_train.shape[1]))
    last_accuracy = -np.inf
    best_accuracy = 0
    classifier = KNNClassifier(k=9)

    while best_accuracy > last_accuracy:
        best_feature = None
        last_accuracy = best_accuracy
        for feature in features_left:
            current_features = sorted([*features, feature])
            x_train_converted, x_test_converted = convert_data_with_features(x_train, x_test, current_features)
            classifier.train(x_train_converted, y_train)
            y_pred = classifier.predict(x_test_converted)
            accuracy, _, _, _, _, = get_accuracy(y_pred, y_test)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_feature = feature
        if best_feature is not None:
            features.append(best_feature)
            features_left.remove(best_feature)

    print(sorted(features))
    # print(f' accuracy: {best_accuracy}')


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, features_names = load_data('train.csv', 'test.csv')
    x_train, x_test = normalize_data(x_train, x_test)
    sfs(x_train, y_train, x_test, y_test)

