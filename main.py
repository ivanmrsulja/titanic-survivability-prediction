from training import *
from utils import *

if __name__ == '__main__':
    train_x, train_y, test_x, test_y, dev_x, dev_y = get_structured_data(remove_outliers=False)
    train_x_orig, train_y_orig, test_x_orig, test_y_orig, dev_x_orig, dev_y_orig = get_structured_data()

    standardise_data(train_x, test_x, dev_x, print_boxplot=False, print_cor=False, train_y=None)

    # Neural Net: Logistic regression, 4 layers [25->TanH->17->TanH->11->TanH->1->Sigmoid], sigmoid cross-entropy loss)
    # train_and_evaluate_nn(train_x, train_y, test_x, test_y, dev_x, dev_y, save_weights=False, use_loaded_weights=False)

    # Decision tree
    model = train_decision_tree(train_x, train_y)
    evaluate_linear_model(model, test_x, test_y, "DECISION TREE", plot=True)  # works better with non-standardised data

    # Lasso regression
    model = train_lasso(train_x, train_y, plot=True)
    evaluate_linear_model(model, test_x, test_y, "LASSO")

    # Support Vector Machine
    model = train_svm(train_x, train_y, dfs="ovo", plot=True)
    evaluate_linear_model(model, test_x, test_y, "SUPPORT VECTOR MACHINE")

    # Ensemble (DT + SVM)
    model = train_ensemble(train_x, train_y)
    evaluate_linear_model(model, test_x, test_y, "ENSEMBLE")

    # Naive Bayes
    # train_and_evaluate_naive_bayes(train_x_orig, train_y_orig, test_x_orig, test_y_orig, dev_x_orig, dev_y_orig)
