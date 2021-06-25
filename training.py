import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import VotingClassifier
from nn import *
from naive_bayes import NaiveBayes
import pickle

plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def train_and_evaluate_nn(train_x_orig, train_y, test_x_orig, test_y, dev_x, dev_y, save_weights=False,
                          use_loaded_weights=False):
    train_x = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    dev_x = dev_x.reshape(dev_x.shape[0], -1).T
    test_x = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    if use_loaded_weights:
        print("Loading pre-trained layers...")
        with open("weights", "rb") as infile:
            parameters = pickle.load(infile)
        print("Done.")
    else:
        parameters = L_layer_model(train_x, train_y, layers_dims, dev_x, dev_y, learning_rate=.25, num_iterations=20000,
                                   print_cost=True, dynamic_grad_change=False, point_of_change=0.31, second_value=0.03)

    # parameters = L_layer_model(train_x, train_y, layers_dims, dev_x, dev_y, learning_rate=.25, num_iterations=20000,
    #                           print_cost=True, dynamic_grad_change=False, point_of_change=0.31, second_value=0.02)

    print("\n*NEURAL NET:")
    print("Training", end=" ")
    pred_train = predict(train_x, train_y, parameters)  # ne znam zasto sam stavljao ovo u promjenljive
    print("Validation", end=" ")
    pred_dev = predict(dev_x, dev_y, parameters)
    print("Test", end=" ")
    pred_test = predict(test_x, test_y, parameters)

    if save_weights:
        with open("weights", "wb") as outfile:
            pickle.dump(parameters, outfile)


def train_decision_tree(train_x_orig, train_y):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_x_orig, train_y)

    return clf


def train_lasso(train_x_orig, train_y):
    reg = linear_model.Lasso(alpha=0.1)
    reg.fit(train_x_orig, train_y)

    return reg


def train_svm(train_x_orig, train_y, dfs="ovr"):
    clf = svm.SVC(decision_function_shape=dfs, probability=True)
    clf.fit(train_x_orig, train_y)

    return clf


def train_ensemble(train_x_orig, train_y):
    clf1 = train_decision_tree(train_x_orig, train_y)
    clf2 = train_svm(train_x_orig, train_y)

    eclf = VotingClassifier(estimators=[('dt', clf1), ('svm', clf2)], voting='soft', weights=[1, 4])
    eclf.fit(train_x_orig, train_y)
    return eclf


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


def train_and_evaluate_naive_bayes(train_x_orig, train_y, test_x_orig, test_y, dev_x, dev_y):
    nb = NaiveBayes(train_x_orig, train_y, test_x_orig, test_y, dev_x, dev_y)
    # nb.numeric_col = [0, 1, 2, 3]

    nb.group_col = [[4], [5, 6, 7]]
    # nb.group_col = [[4, 5], [6, 7, 8]]
    nb.learn()

    nb.evaluate_prediction()
