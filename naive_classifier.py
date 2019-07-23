import configparser
import cv2 as cv
import numpy as np
import pickle
from matplotlib import  pyplot as plot
from get_dataset import Data

data = Data()
# data.prepare_data()
pos_num, neg_num, row, col, pos_data, neg_data = data.load_data()
print('...computing HOG')

def cal_curve():
    arr = []
    for i in range(pos_num):
        temp = hog.compute(pos_data[i])
        arr.append(temp.ravel())
    arr = np.array(arr)
    mean = np.mean(np.array(arr), axis=0)

    tot = np.concatenate((pos_data, neg_data))
    tot_num = tot.shape[0]
    arr = []
    for i in range(tot_num):
        temp = hog.compute(tot[i])
        arr.append(temp.ravel())
    arr = np.array(arr)
    dis = []
    for i in range(tot_num):
        dis.append(np.linalg.norm(arr[i] - mean))  # R2


    # mean 4.53
    def get_roc(threshold):
        predict = [x < threshold for x in dis]

        true_pos = np.count_nonzero(predict[:pos_num])
        false_neg = pos_num - true_pos
        false_pos = np.count_nonzero(predict[pos_num:])
        true_neg = neg_num - false_pos

        return false_neg/(true_pos + false_neg), false_pos/(false_pos+true_neg)


    print('...computing ROC curve')
    minn = np.min(dis)
    maxx = np.max(dis)
    threshold = minn
    point_x = []
    point_y = []
    for i in range(300):
        t = get_roc(threshold)
        point_x.append(t[0])
        point_y.append(t[1])
        threshold += (maxx-minn)/300
    return point_x, point_y


plot.title('ROC Curve')
plot.xlabel('missed positive rate')
plot.ylabel('False alarm rate')
hog = cv.HOGDescriptor((row, col), (8, 8), (4, 4), (8, 8), 9, )
points1 = np.array(cal_curve())
hog = cv.HOGDescriptor((row, col), (8, 8), (4, 4), (4, 4), 9, )
points2 = np.array(cal_curve())
plot.plot(points1[0],points1[1],label='1')
plot.plot(points2[0],points2[1],label='2')
plot.show()