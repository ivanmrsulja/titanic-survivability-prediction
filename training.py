import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import linear_model
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
                                   print_cost=True, dynamic_grad_change=True, low_limit=0.01, point_of_change=0.31,
                                   second_value=0.03)

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


def train_lasso(train_x_orig, train_y, plot=False):
    if plot:
        coefs = []
        n_alphas = 500
        alphas = np.logspace(-10, 0, n_alphas)

        for a in alphas:
            reg = linear_model.Lasso(alpha=a)
            reg.fit(train_x_orig, train_y)
            coefs.append(reg.coef_)

        ax = plt.gca()

        ax.plot(alphas, coefs)
        ax.set_xscale('log')
        ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
        plt.xlabel('alpha')
        plt.ylabel('weights')
        plt.title('Lasso coefficients')
        plt.axis('tight')
        plt.show()

    reg = linear_model.Lasso(alpha=0.0005)
    reg.fit(train_x_orig, train_y)
    return reg


def train_svm(train_x_orig, train_y, dfs="ovr", plot=False):
    clf = svm.SVC(kernel="poly", decision_function_shape=dfs, probability=True)
    clf.fit(train_x_orig, train_y)

    if plot:
        reductor = PCA(n_components=2, random_state=0)
        result = reductor.fit_transform(train_x_orig)

        for_plotting = svm.SVC(kernel="poly", decision_function_shape=dfs, probability=True)
        for_plotting.fit(result, train_y)

        support_vectors = for_plotting.support_vectors_

        plt.scatter(result[:, 0], result[:, 1])
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1], color='red')
        plt.title('SVM')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()

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
                           feature_names=['Age', 'Siblings/Spouses', 'Parents/Children', 'Fare', 'Male', 'Female', 'PClass 1',
                                          'PClass 2', 'PClass 3'])
        plt.show()


def train_and_evaluate_naive_bayes(train_x_orig, train_y, test_x_orig, test_y):
    nb = NaiveBayes(train_x_orig, train_y, test_x_orig, test_y)
    # nb.numeric_col = [0, 1, 2, 3]

    nb.group_col = [[4], [5, 6, 7]]
    # nb.group_col = [[4, 5], [6, 7, 8]]
    nb.learn()

    nb.evaluate_prediction()
