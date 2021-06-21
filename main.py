from training import *
from utils import *

if __name__ == '__main__':
    train_x_orig, train_y, test_x_orig, test_y, dev_x, dev_y = get_structured_data()

    standardise_data(train_x_orig, test_x_orig, dev_x, print_boxplot=False, print_cor=False, train_y=None,
                     remove_outliers=False)

    # Neural Net: Logistic regression, 4 layers [25->TanH->17->TanH->11->TanH->1->Sigmoid], sigmoid cross-entropy loss)
    # train_and_evaluate_nn(train_x_orig, train_y, test_x_orig, test_y, dev_x, dev_y)

    # Decision tree
    model = train_decision_tree(train_x_orig, train_y)
    evaluate_linear_model(model, test_x_orig, test_y, "DECISION TREE", plot=False)

    # Lasso regression
    # model = train_lasso(train_x_orig, train_y)
    # evaluate_linear_model(model, test_x_orig, test_y, "LASSO", plot=False)

    # Support Vector Machine
    model = train_svm(train_x_orig, train_y, dfs="ovo")
    evaluate_linear_model(model, test_x_orig, test_y, "SUPPORT VECTOR MACHINE", plot=True)

    # Ensemble
    model = train_ensemble(train_x_orig, train_y)
    evaluate_linear_model(model, test_x_orig, test_y, "ENSEMBLE", plot=False)
