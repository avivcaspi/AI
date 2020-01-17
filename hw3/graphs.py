from DT1 import *


def part9():
    print('part 9-------------------------------------------------------')
    x_train, y_train, x_test, y_test, features_names = load_data('train.csv', 'test.csv')
    y_pred = classify(x_train, y_train, x_test)
    for p in [0.05, 0.1, 0.2]:
        y_pred_new = y_pred
        for i, pred in enumerate(y_pred):
            if pred == 0.0:
                if np.random.choice([True, False], p=[p, 1 - p]):
                    y_pred_new[i] = 1.0
        mat = sk.metrics.confusion_matrix(y_test, y_pred_new)
        print(f'p = {p}\nError_w = {4 * mat[1, 0] + mat[0, 1]}')
        print(mat)


def part3():
    print('part 3------------------------------------------------------')
    x_train, y_train, x_test, y_test, features_names = load_data('train.csv', 'test.csv')
    for x in [3, 9, 27]:
        classifier = tree.DecisionTreeClassifier("entropy", min_samples_split=x)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)

        confusion_matrix = sk.metrics.confusion_matrix(y_test, y_pred)
        print(confusion_matrix)
        print(f'Error_w = {4 * confusion_matrix[1, 0] + confusion_matrix[0, 1]}')


from graphviz import Source


def part4():
    print('part 4 -----------------------------------------')
    x_train, y_train, x_test, y_test, features_names = load_data('train.csv', 'test.csv')
    classifier = tree.DecisionTreeClassifier("entropy", min_samples_split=27)
    classifier.fit(x_train, y_train)
    graph = Source(tree.export_graphviz(classifier, out_file=None, feature_names=features_names, class_names=['-', '+']))
    png_bytes = graph.pipe(format='png')
    with open('dtree_pipe.png', 'wb') as f:
        f.write(png_bytes)


def part7():
    print('part 7------------------------------------------------------')
    x_train, y_train, x_test, y_test, features_names = load_data('train.csv', 'test.csv')
    classifier = tree.DecisionTreeClassifier("entropy")
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    confusion_matrix = sk.metrics.confusion_matrix(y_test, y_pred)
    print('dt1 not trimmed')
    print(f'Error_w = {4 * confusion_matrix[1, 0] + confusion_matrix[0, 1]}')

    classifier = tree.DecisionTreeClassifier("entropy", min_samples_split=27)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    confusion_matrix = sk.metrics.confusion_matrix(y_test, y_pred)
    print('dt1 trimmed')
    print(f'Error_w = {4 * confusion_matrix[1, 0] + confusion_matrix[0, 1]}')


if __name__ == '__main__':
    part3()
    part4()
    part7()
    part9()
