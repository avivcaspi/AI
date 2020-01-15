from KNN1 import *


class KNN2Classifier(KNNClassifier):
    def predict(self, x_test: np.ndarray):
        # distances matrix (position i,j is train sample i distance from test sample j)
        dist_matrix = euclidean_dist(self.x_train, x_test)

        test_size = x_test.shape[0]
        y_pred = np.zeros(test_size)
        for i in range(test_size):
            nearest_neighbors = np.argpartition(dist_matrix[:, i], self.k)
            nearest_neighbors_labels = self.y_train[nearest_neighbors[:self.k]]
            pos_num = np.sum(nearest_neighbors_labels) * 4
            neg_num = np.sum(nearest_neighbors_labels == 0)
            y_pred[i] = 1 if pos_num > neg_num else 0
        return y_pred


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, features_names = load_data('train.csv', 'test.csv')
    x_train, x_test = normalize_data(x_train, x_test)

    knn_classifier = KNN2Classifier(k=9)
    knn_classifier.train(x_train, y_train)
    y_pred = knn_classifier.predict(x_test)
    confusion_mat = sk.confusion_matrix(y_test, y_pred)
    print(confusion_mat)
    print(f'Error_w = {4 * confusion_mat[1, 0] + confusion_mat[0, 1]}')

