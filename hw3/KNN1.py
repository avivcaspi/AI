import numpy as np
from DT1 import load_data
import sklearn.metrics as sk


def normalize_data(x_train: np.ndarray, x_test: np.ndarray):
    # calculates min and max of training data
    min_train_value = np.min(x_train, axis=0)
    max_train_value = np.max(x_train, axis=0)
    # normalizing train and test set with train max - min values
    x_train_normalized = (x_train - min_train_value) / (max_train_value - min_train_value)
    x_test_normalized = (x_test - min_train_value) / (max_train_value - min_train_value)
    return x_train_normalized, x_test_normalized


class KNNClassifier:
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        # training only includes saving the train data
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test: np.ndarray):
        # distances matrix (position i,j is train sample i distance from test sample j)
        dist_matrix = euclidean_dist(self.x_train, x_test)

        test_size = x_test.shape[0]
        y_pred = np.zeros(test_size)
        for i in range(test_size):
            # finds k nearest neighbors of sample i
            nearest_neighbors = np.argpartition(dist_matrix[:, i], self.k)
            nearest_neighbors_labels = self.y_train[nearest_neighbors[:self.k]]
            # number of positive sample in nearest neighbors
            pos_num = np.sum(nearest_neighbors_labels)
            # determine which class has more labels
            y_pred[i] = 1 if pos_num > self.k - pos_num else 0
        return y_pred


def euclidean_dist(x1: np.ndarray, x2: np.ndarray):
    # calculates L2 norm of x1-x2 , using matrix functions to calculate faster
    # (x1 - x2)^2 = x1^2 - 2x1x2 + x2^2
    x1_squared = (x1 ** 2).sum(axis=1)
    x2_squared = (x2 ** 2).sum(axis=1)
    x1_times_x2 = x1.dot(x2.T)
    dists_squared = x1_squared.reshape(-1, 1) - 2 * x1_times_x2 + x2_squared.reshape(1, -1)
    dists_squared[dists_squared < 0] = 0  # fix numeric errors
    dists = np.sqrt(dists_squared)

    # loops way of calculating
    # dists = np.zeros((x1.shape[0], x2.shape[0]))
    # for i in range(x1.shape[0]):
    #   for j in range(x2.shape[0]):
    #       dists[i, j] = np.linalg.norm(x1[i] - x2[j], ord=2)

    return dists


def get_accuracy(y: np.ndarray, y_pred: np.ndarray):
    # calculates accuracy
    num_correct = (y == y_pred).sum()
    total = y.shape[0]
    accuracy = float(num_correct) / total
    return accuracy


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, features_names = load_data('train.csv', 'test.csv')
    x_train, x_test = normalize_data(x_train, x_test)
    knn_classifier = KNNClassifier(k=9)
    knn_classifier.train(x_train, y_train)
    y_pred = knn_classifier.predict(x_test)
    confusion_mat = sk.confusion_matrix(y_test, y_pred)
    print(confusion_mat)
    # print(f'Error_w = {4*confusion_mat[1,0] + confusion_mat[0,1]}')



