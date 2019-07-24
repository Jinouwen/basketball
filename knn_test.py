import configparser
import cv2 as cv
import numpy as np
import pickle
from matplotlib import  pyplot as plot
from get_dataset import Data

data = Data()
# data.prepare_data()
pos_num, neg_num, row, col, pos_data, neg_data = data.load_data()
hog = cv.HOGDescriptor((row, col), (8, 8), (4, 4), (8, 8), 9, )
training_num = int(pos_num * 0.7)
training_label0 = np.ones(training_num, dtype=np.int8)
training_num += int(neg_num * 0.7)
training_label1 = np.zeros(int(neg_num * 0.7), dtype=np.int8)
training_data = np.concatenate((pos_data[:int(pos_num * 0.7)], neg_data[:int(neg_num * 0.7)]))
training_label = np.concatenate((training_label0,training_label1))
test_data = np.concatenate((pos_data[int(pos_num * 0.7):], neg_data[int(neg_num * 0.7):]))
test_label = np.concatenate((np.ones(pos_num - int(pos_num * 0.7)), np.zeros(neg_num - int(neg_num * 0.7))))

training_label =training_label.astype(np.float32)
training_data = training_data.astype(np.float32)
training_data = training_data.reshape(training_data.shape[0],-1)
test_data = test_data.reshape(test_data.shape[0],-1)
test_data = test_data.astype(np.float32)
knn = cv.ml.KNearest_create()
knn.train(training_data,cv.ml.ROW_SAMPLE,training_label)
ret, results, neighbours, dist = knn.findNearest(test_data,5)
print(pos_num-int(pos_num * 0.7))
print(np.count_nonzero(results))
