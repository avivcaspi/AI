from DT1 import load_data
import sklearn as sk
from sklearn import tree
from graphviz import Source

def classify(x_train, y_train, x_test, d):
    classifier = tree.DecisionTreeClassifier("entropy", min_samples_split=9, class_weight={1: d, 0: 1-d}, random_state=0)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    '''graph = Source(tree.export_graphviz(classifier, out_file=None, feature_names=features_names,class_names=['-', '+']))
    png_bytes = graph.pipe(format='png')
    with open('tree-12.png', 'wb') as f:
        f.write(png_bytes)'''

    return y_pred


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, features_names = load_data('train.csv', 'test.csv')
    y_pred = classify(x_train, y_train, x_test, 0.8)
    confusion_matrix = sk.metrics.confusion_matrix(y_test, y_pred)
    print(confusion_matrix)
    print(f'Error_w = {4 * confusion_matrix[1, 0] + confusion_matrix[0, 1]}')

