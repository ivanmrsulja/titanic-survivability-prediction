import numpy as np
import math


def find_interval_place(intervals, value):
    for i, interval in enumerate(intervals):
        if interval >= value:
            return i


class NaiveBayes:

    def __init__(self, train_x, train_y, test_x, test_y, dev_x, dev_y):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.dev_x = dev_x
        self.dev_y = dev_y

        self.group_col = []
        self.numeric_col = []

        self.min = []
        self.max = []
        self.mean = []

        self.learned = []

        self.intervals = []

        self.data_sum_win = None
        self.data_sum_lose = None

        self.data_win = None
        self.data_lose = None

        self.survival_sum = [0, 0]  # [1 (survived), 0 (died)]

    def calc_win_lose_ratio(self):

        total = len(self.train_y)
        survived = 0
        for i in self.train_y:
            if i > 0:
                survived += 1

        self.survival_sum = [survived, total-survived]

    def init_data_sum(self, intervals):
        self.data_sum_win = []
        self.data_sum_lose = []

        for i in range(len(intervals)):
            interval_size = len(intervals[i])
            self.data_sum_win.append([0]*interval_size)
            self.data_sum_lose.append([0]*interval_size)

        for i in range(len(self.group_col)):
            interval_size = len(self.group_col[i])
            self.data_sum_win.append([0]*interval_size)
            self.data_sum_lose.append([0]*interval_size)

    def learn(self):
        self.calc_win_lose_ratio()

        train_x_np = np.array(self.train_x)
        self.min = train_x_np.min(axis=0)
        self.max = train_x_np.max(axis=0)
        self.mean = train_x_np.mean(axis=0)

        self.intervals = self.get_numeric_intervals()

        self.init_data_sum(self.intervals)

        for row_no, row in enumerate(train_x_np):
            for i, col in enumerate(self.numeric_col):
                place = find_interval_place(self.intervals[i], row[col])

                if self.train_y[row_no] > 0:
                    self.data_sum_win[i][place] += 1
                else:
                    self.data_sum_lose[i][place] += 1

            for i, gc in enumerate(self.group_col):
                i += len(self.numeric_col)
                for j, col in enumerate(gc):
                    if self.train_y[row_no] > 0:
                        self.data_sum_win[i][j] += int(row[col])
                    else:
                        self.data_sum_lose[i][j] += int(row[col])

        self.calc_win_lose_data()

    def calc_win_lose_data(self):
        self.data_win = []
        self.data_lose = []

        for i, data_win in enumerate(self.data_sum_win):
            data = np.array(data_win)
            data = data/self.survival_sum[0]
            self.data_win.append(data.tolist())

        for i, data_lose in enumerate(self.data_sum_lose):
            data = np.array(data_lose)
            data = data/self.survival_sum[1]
            self.data_lose.append(data)

    def get_numeric_intervals(self):
        intervals = []
        for num in self.numeric_col:

            interval_difference = self.max[num] - self.min[num]
            interval_no = min(10, interval_difference)

            step = math.floor(interval_difference / interval_no) + 1
            interval = list(range(math.floor(self.min[num]), math.floor(self.max[num]), step))
            interval.append(9999999)
            intervals.append(interval)

        return intervals

    def predict(self, row):
        win_predict = 1
        lose_predict = 1

        for i, col in enumerate(self.numeric_col):
            place = find_interval_place(self.intervals[i], row[col])
            if self.data_win[i][place] > 0:
                win_predict *= self.data_win[i][place]
            if self.data_lose[i][place] > 0:
                lose_predict *= self.data_lose[i][place]

        for i, gc in enumerate(self.group_col):
            i += len(self.numeric_col)
            for j, col in enumerate(gc):

                # self.data_sum_win[i][j] += int(row[col])
                # self.data_sum_lose[i][j] += int(row[col])
                if row[col] > 0 and self.data_win[i][j] > 0:
                    win_predict *= self.data_win[i][j]
                if row[col] > 0 and self.data_lose[i][j] > 0:
                    lose_predict *= self.data_lose[i][j]

                # In case column is Boolean
                if len(gc) == 1:
                    if row[col] == 0 and self.data_win[i][j] > 0:
                        win_predict *= 1/self.data_win[i][j]
                    if row[col] == 0 and self.data_lose[i][j] > 0:
                        lose_predict *= 1/self.data_lose[i][j]

        if win_predict > lose_predict:
            return 1
        else:
            return 0

    def evaluate_prediction(self):
        # ratio_train = self.evaluate_data(self.train_x, self.train_y)
        ratio_dev = self.evaluate_data(self.dev_x, self.dev_y)
        ratio_test = self.evaluate_data(self.test_x, self.test_y)

        print("\n*NAIVE BAYES:")
        print("Test1: {}%".format(ratio_dev*100))
        print("Test2: {} %".format(ratio_test*100))

    def evaluate_data(self, data, results):
        successful = 0
        unsuccessful = 0

        for i, row in enumerate(data):
            prediction = self.predict(row)
            if prediction == results[i]:
                successful += 1
            else:
                unsuccessful += 1

        return successful / (successful + unsuccessful)
