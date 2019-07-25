import numpy as np
import cv2
import matplotlib.pyplot as plt
from get_dataset import *
import configparser
import pickle

config_path = 'classifier_config/'
curve_path = 'model_and_curve/'

def load_data(train_prop=0.8, neg_prop=0.01):
    data = Data()
    pos_num, neg_num, row, col, pos_data, neg_data = data.load_data()
    pos_data_flatten = pos_data.reshape(pos_data.shape[0], -1).T
    neg_data_flatten = neg_data.reshape(neg_data.shape[0], -1).T
    neg_num = int(neg_num * neg_prop)
    dim = row * col * 3
    # shape of data_set
    data_set = np.zeros((dim + 1, pos_num + neg_num))
    data_set[0: dim, 0:pos_num] = pos_data_flatten
    data_set[dim, 0:pos_num] = 1
    data_set[0: dim, pos_num:pos_num + neg_num] = neg_data_flatten[:, 0:neg_num]
    data_set[dim, pos_num:pos_num + neg_num] = 0
    # shuffle
    index = [i for i in range(data_set.shape[1])]
    np.random.shuffle(index)
    data_set = data_set[:, index]
    # split
    train_num = int(data_set.shape[1] * train_prop)
    test_num = data_set.shape[1] - train_num
    train_set_x = data_set[0:dim, 0:train_num]
    train_set_x = train_set_x/255
    train_set_y = data_set[dim, 0:train_num]
    train_set_y = train_set_y[np.newaxis, :]
    test_set_x = data_set[0:dim, train_num:train_num + test_num]
    test_set_x = test_set_x/255
    test_set_y = data_set[dim, train_num:train_num + test_num]
    test_set_y = test_set_y[np.newaxis, :]

    assert train_set_x.shape == (dim, train_num)
    assert train_set_y.shape == (1, train_num)
    assert test_set_x.shape == (dim, test_num)
    assert test_set_y.shape == (1, test_num)
    return train_set_x, train_set_y, test_set_x, test_set_y, dim


def load_para():
    # cf = configparser.ConfigParser()
    # cf.read("lr_cfg.ini")
    # learning_rate = cf.getfloat("parameters", "learning_rate")
    # num_iterations = cf.getint("parameters", "num_iterations")
    num_iterations = 1000
    learning_rate = 0.1
    threshold = np.linspace(0, 1, 1000)
    return num_iterations, learning_rate, threshold


def sigmoid(z):
    s = 1.0 / (1 + np.exp(-z))
    return s


class Model:
    def __init__(self, num_iterations, learning_rate, dim, threshold,
                 X_train, Y_train, X_test, Y_test):
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.m = X_train.shape[1]
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        # self.w = np.random.rand(dim, 1)
        self.w = np.zeros((dim, 1))
        self.b = 0
        self.dw = 0
        self.db = 0
        self.A = 0
        self.costs = []
        self.cost = 0
        self.threshold = threshold
        # self.acc = []
        # self.prec = []
        self.recall = []
        self.FAR = []
    def load_model(self):
        cf = configparser.ConfigParser()
        cf.read(config_path + "lr_cfg.ini")
        w = cf.get("parameters", "Weights")
        w = w.lstrip('[').rstrip(']').split('][')
        w = np.array(list(map(float, w)))
        w = w.reshape(w.shape[0], -1)
        self.w = w
        b = cf.get("parameters", "Bias")
        self.b = float(b)
        # acc = cf.get("results", "Accuracy")
        # acc = acc.split(' ')
        # acc = list(map(float, acc))
        # self.acc = acc
        # prec = cf.get("results", "Precision")
        # prec = prec.split(' ')
        # prec = list(map(float, prec))
        # self.prec = prec
        # recall = cf.get("results", "Recall")
        # recall = recall.split(' ')
        # recall = list(map(float, recall))
        # self.recall = recall
        # far = cf.get("results", "false Alarm rate")
        # far = far.split(' ')
        # far = list(map(float, far))
        # self.FAR = far
    def propagate(self):
        self.A = sigmoid(np.dot(self.w.T, X_train) + self.b)
        self.cost = -(1 / self.m) * np.sum(np.log(self.A) * self.Y_train + np.log(1 - self.A) * (1 - self.Y_train))
        self.dw = 1 / self.m * np.dot(X_train, (self.A - self.Y_train).T)
        self.db = 1 / self.m * np.sum(self.A - self.Y_train)
    def optimize(self):
        for i in range(self.num_iterations):
            self.propagate()
            if i%100 == 0:
                self.costs.append(self.cost)
            self.w = self.w - self.learning_rate * self.dw
            self.b = self.b - self.learning_rate * self.db
    def predict_ROC(self, X, k):
        m = X.shape[1]
        Y_predict = np.zeros((1, m))
        A = sigmoid(np.dot(self.w.T, X) + self.b)
        for i in range(m):
            if A[0, i] > self.threshold[k]:
                Y_predict[0, i] = 1
            else:
                Y_predict[0, i] = 0
        return Y_predict
    def predict_oneImg(self, imgPath, threshold):
        img = cv2.imread(imgPath)
        hog = cv2.HOGDescriptor()
        descriptor = hog.compute(img)
        descriptor = descriptor.ravel()
        descriptor = descriptor.reshape(descriptor.shape[0], -1)
        A = sigmoid(np.dot(self.w.T, descriptor) + self.b)
        if A[0, 0] > threshold:
            predict = 1
        else:
            predict = 0
        return predict
    def run_model(self):
        self.optimize()
        for k in range(self.threshold.shape[0]):
            Y_predict = self.predict_ROC(self.X_train, k)  # (1,m_test)
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            for i in range(X_train.shape[1]):
                if Y_predict[0, i]-Y_train[0, i] == 0 and Y_train[0, i] == 1:
                    TP = TP + 1
                elif Y_predict[0, i]-Y_train[0, i] == 0 and Y_train[0, i] == 0:
                    TN = TN + 1
                elif Y_predict[0, i]-Y_train[0, i] == 1:
                    FP = FP + 1
                elif Y_predict[0, i] - Y_train[0, i] == -1:
                    FN = FN + 1
                else:
                    pass
            # self.acc.append((TP+TN)/(TP+TN+FP+FN))
            # self.prec.append(TP/(TP+FP))
            self.recall.append(TP/(TP+FN))
            self.FAR.append(FP/(FP+TN))
        # config
        cf = configparser.ConfigParser()
        # cf.add_section("results")
        cf.read(config_path + "lr_cfg.ini")
        # strr = " ".join(list(map(str, self.acc)))
        # cf.set("results", "Accuracy", strr)
        # strr = " ".join(list(map(str, self.prec)))
        # cf.set("results", "Precision", strr)
        # strr = " ".join(list(map(str, self.recall)))
        # cf.set("results", "Recall", strr)
        # strr = " ".join(list(map(str, self.FAR)))
        # cf.set("results", "false Alarm rate", strr)
        strr = "".join(list(map(str, self.w)))
        cf.set("parameters", "Weights", strr)
        strr = str(self.b)
        cf.set("parameters", "Bias", strr)
        cf.write(open("lr_cfg.ini", "w"))
        cur = np.array(self.recall), np.array(self.FAR)
        with open(curve_path + '%scurve.pkl' % "lr", 'wb') as f:
            pickle.dump(cur, f)
    def draw_ROC(self):
        # draw pic
        plt.figure(figsize=(4, 4))
        MPN = 1 - np.array(self.recall)
        plt.plot(MPN, self.FAR, color="red", linewidth=2)
        plt.ylabel("Missed Positive Number")
        plt.xlabel("False Alarm Number")
        plt.title("ROC")
        plt.show()


X_train, Y_train, X_test, Y_test, dim = load_data()
num_iterations, learning_rate, threshold = load_para()
model = Model(num_iterations, learning_rate, dim, threshold,
              X_train, Y_train, X_test, Y_test)
model.run_model()
# model.load_model()
model.draw_ROC()
# predict = model.predict_oneImg("pic1.jpg", 0.1)
