from KNN1 import *
from OPT import convert_data_with_features


def sfs(x_train, y_train, x_test, y_test):
    features = []
    features_left = list(range(x_train.shape[1]))
    last_accuracy = -np.inf
    best_accuracy = 0
    classifier = KNNClassifier(k=9)
    # continue until we cant find feature that improve the accuracy
    while best_accuracy > last_accuracy:
        best_feature = None
        last_accuracy = best_accuracy
        # going over all features left
        for feature in features_left:
            # add the next feature to the features until now
            current_features = sorted([*features, feature])
            # convert the data to have the features we want
            x_train_converted, x_test_converted = convert_data_with_features(x_train, x_test, current_features)
            # train and classify
            classifier.train(x_train_converted, y_train)
            y_pred = classifier.predict(x_test_converted)
            accuracy = get_accuracy(y_pred, y_test)
            # update best feature we can add
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_feature = feature
        # if we found an improving feature we add it to features and remove it from the features we have left
        if best_feature is not None:
            features.append(best_feature)
            features_left.remove(best_feature)

    print(sorted(features))
    # print(f' accuracy: {best_accuracy}')


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, features_names = load_data('train.csv', 'test.csv')
    x_train, x_test = normalize_data(x_train, x_test)
    sfs(x_train, y_train, x_test, y_test)

