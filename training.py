import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import linear_model
from sklearn import svm
from nn import *

plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def train_and_evaluate_nn(train_x_orig, train_y, test_x_orig, test_y, dev_x, dev_y):
    train_x = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    dev_x = dev_x.reshape(dev_x.shape[0], -1).T
    test_x = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    parameters = L_layer_model(train_x, train_y, layers_dims, dev_x, dev_y, learning_rate=.25, num_iterations=10000,
                               print_cost=True, dynamic_grad_change=False, point_of_change=0.31, second_value=0.05)
    print("\n*NEURAL NET:")
    print("Training", end=" ")
    pred_train = predict(train_x, train_y, parameters)  # ne znam zasto sam stavljao ovo u promjenljive
    print("Validation", end=" ")
    pred_dev = predict(dev_x, dev_y, parameters)
    print("Test", end=" ")
    pred_test = predict(test_x, test_y, parameters)


def train_decision_tree(train_x_orig, train_y):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_x_orig, train_y)

    return clf


def train_lasso(train_x_orig, train_y):
    reg = linear_model.Lasso(alpha=0.1)
    reg.fit(train_x_orig, train_y)

    return reg


def train_svm(train_x_orig, train_y, dfs="ovr"):
    clf = svm.SVC(decision_function_shape=dfs)
    clf.fit(train_x_orig, train_y)

    return clf


def evaluate_linear_model(model, test_x_orig, test_y, name, plot=False):
    predictions = model.predict(test_x_orig)

    passed = 0
    failed = 0

    for i, prediction in enumerate(predictions):
        if (prediction >= 0.5 and test_y[i] >= 0.5) or (prediction < 0.5 and test_y[i] < 0.5):
            passed += 1
        else:
            failed += 1

    print("\n*{} CLASSIFIER:".format(name))
    print("Passed: ", passed)
    print("Failed: ", failed)
    print("Accuracy: ", (passed / len(test_y)) * 100, " %")

    if plot and "tree" in name.lower():
        plt.figure(figsize=(40, 20))  # minimalna rezolucija za iscrtavanje
        _ = tree.plot_tree(model,
                           feature_names=['Age', 'Siblings/Spouses', 'Parents/Children', 'Fare', 'Gender', 'PClass 1',
                                          'PClass 2', 'PClass 3'])
        plt.show()
    elif plot and "vector" in name.lower():
        # TODO: Plot za SVM
        pass
