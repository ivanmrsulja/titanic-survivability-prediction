from nn import *
import pandas as pd

plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def load_data():
    train_set_x_orig = []
    train_set_y_orig = []

    test_set_x_orig = []
    test_set_y_orig = []

    categories = {'1': [1, 0, 0], '2': [0, 1, 0], '3': [0, 0, 1]}
    gender = {'male': [1, 0], 'female': [0, 1]}

    with open('datasets/train.csv') as f:
        lines = f.readlines()
        train_data = lines[1:]

    with open('datasets/test.csv') as f:
        lines = f.readlines()
        test_data = lines[1:]

    with open('datasets/gender_submission.csv') as f:
        lines = f.readlines()
        test_results = lines[1:]

    for line in train_data:
        if "\",," in line:
            continue
        tokens = line.split(",")
        test_case = []
        try:
            test_case.append(float(tokens[-7]))
        except:
            continue
        test_case.append(float(tokens[-6]))
        test_case.append(float(tokens[-5]))
        test_case.append(float(tokens[-3]))
        if tokens[-8] == "male":
            # test_case.extend(gender['male'])
            test_case.append(1.0)
        else:
            # test_case.extend(gender['female'])
            test_case.append(0.0)
        for val in categories[tokens[2].replace("\n", "")]:
            test_case.append(val)
        train_set_x_orig.append(test_case)
        train_set_y_orig.append(float(tokens[1]))

    for i, line in enumerate(test_data):
        if "\",," in line or "e,," in line:
            continue
        tokens = line.split(",")
        test_case = []
        test_case.append(float(tokens[-7]))
        test_case.append(float(tokens[-6]))
        test_case.append(float(tokens[-5]))
        try:
            test_case.append(float(tokens[-3]))
        except:
            continue
        if tokens[-8] == "male":
            # test_case.extend(gender['male'])
            test_case.append(1.0)
        else:
            # test_case.extend(gender['female'])
            test_case.append(0.0)
        for val in categories[tokens[1].replace("\n", "")]:
            test_case.append(val)
        test_set_x_orig.append(test_case)
        test_set_y_orig.append(float(test_results[i].split(",")[1]))
        i += 1

    return np.array(train_set_x_orig), np.array(train_set_y_orig), np.array(test_set_x_orig), np.array(
        test_set_y_orig), [b'died', b'survived']


def get_structured_data():
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

    dev_set_size = (len(train_x_orig) // 5)

    rng = np.random.default_rng(12321)
    indexes = rng.choice(len(train_x_orig), size=dev_set_size, replace=False)

    dev_x = []
    dev_y = []

    for i in indexes:
        dev_x.append(train_x_orig[i])
        dev_y.append(train_y[i])

    np.delete(train_y, indexes)
    np.delete(train_x_orig, indexes)

    dev_x = np.array(dev_x)
    dev_y = np.array(dev_y)

    return train_x_orig, train_y, test_x_orig, test_y, dev_x, dev_y


def standardise_data(train_x_orig, test_x_orig, dev_x, print_cor=False, print_boxplot=False, train_y=None, remove_outliers=False, standardise=True):

    if remove_outliers:
        df = pd.DataFrame(train_x_orig)
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
        train_x_orig = df.to_numpy()

        df = pd.DataFrame(dev_x)
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
        dev_x = df.to_numpy()


    train_arr = []
    dev_arr = []
    test_arr = []

    if standardise:
        for i in range(n_x):
            for el in train_x_orig:
                train_arr.append(el[i])
            for el in dev_x:
                dev_arr.append(el[i])
            for el in test_x_orig:
                test_arr.append(el[i])

            data_train = np.array(train_arr)
            data_dev = np.array(dev_arr)
            data_test = np.array(test_arr)

            mean_train = np.mean(data_train)
            sd_train = np.sqrt(np.var(data_train))
            mean_dev = np.mean(data_dev)
            sd_dev = np.sqrt(np.var(data_dev))
            mean_test = np.mean(data_test)
            sd_test = np.sqrt(np.var(data_test))

            if remove_outliers:
                sd_train += 0.001
                sd_dev += 0.001

            for el in train_x_orig:
                el[i] = (el[i] - mean_train) / sd_train
            for el in dev_x:
                el[i] = (el[i] - mean_dev) / sd_dev
            for el in test_x_orig:
                el[i] = (el[i] - mean_test) / sd_test

            if print_cor:
                print(np.corrcoef(data_dev, train_y))
                print()
            if print_boxplot:
                plt.boxplot(data_dev, notch=None, vert=None, patch_artist=None, widths=None)
                plt.show()

            train_arr.clear()
            dev_arr.clear()
            test_arr.clear()
