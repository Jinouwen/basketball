from sklearn import svm
import cv2 as cv
import pickle
from get_dataset import Data
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plot
import configparser
import time
class SVM:

    def __init__(self):
       pass

    def get_data(self,dir,opt):
        '''
        if opt == 0:
            data = Data()
            hog = cv.HOGDescriptor((48, 48), (8, 8), (4, 4), (4, 4), 9, )
            pos_num, neg_num, row, col, pos_data_0, neg_data_0 = data.load_data()
            temp = hog.compute(pos_data_0[0])
            pos_data = np.zeros((pos_num, *temp.shape), dtype=float)
            neg_data = np.zeros((neg_num, *temp.shape), dtype=float)
            for i in range(pos_num):
                pos_data[i] = hog.compute(pos_data_0[i])
            for i in range(neg_num):
                neg_data[i] = hog.compute(neg_data_0[i])
            neg_data = neg_data.reshape(neg_data.shape[0], -1)
            pos_data = pos_data.reshape(pos_data.shape[0], -1)
            test_size = .25
            pos_train_num = int(pos_num * (1 - test_size))
            neg_train_num = int(neg_num * (1 - test_size))
            x_train = np.concatenate((pos_data[:pos_train_num], neg_data[:neg_train_num]))
            x_test = np.concatenate((pos_data[pos_train_num:], neg_data[neg_train_num:]))
            y_train = np.concatenate((np.ones(pos_train_num, dtype=int), np.zeros(neg_train_num, dtype=int)))
            y_test = np.concatenate(
                (np.ones(pos_num - pos_train_num, dtype=int), np.zeros(neg_num - neg_train_num, dtype=int)))
            return x_train,x_test,y_train,y_test

        else:
        '''
        config = configparser.ConfigParser()
        config.read('config.ini')
        neg_data_name = config.get('data_set','data_dir%i' % dir)+config.get('data_set','hog_data%i'% opt )
        with open(neg_data_name,'rb') as f:
            neg_data = pickle.load(f)
        pos_data_name = neg_data_name.replace('neg','pos')
        with open(pos_data_name,'rb') as f:
            pos_data = pickle.load(f)

        pos_num = pos_data.shape[0]
        neg_num = neg_data.shape[0]
        np.random.shuffle(pos_data)
        np.random.shuffle(neg_data)
        test_size = .25
        pos_train_num = int(pos_num * (1 - test_size))
        neg_train_num = int(neg_num * (1 - test_size))
        x_train = np.concatenate((pos_data[:pos_train_num], neg_data[:neg_train_num]))
        x_test = np.concatenate((pos_data[pos_train_num:], neg_data[neg_train_num:]))
        y_train = np.concatenate((np.ones(pos_train_num, dtype=int), np.zeros(neg_train_num, dtype=int)))
        y_test = np.concatenate(
            (np.ones(pos_num - pos_train_num, dtype=int), np.zeros(neg_num - neg_train_num, dtype=int)))
        return x_train,x_test,y_train,y_test

    def train(self,x,y):
        clf = svm.SVC(probability=True, class_weight='balanced', gamma='auto', random_state=np.random.randint(233))
        clf.fit(x,y)
        return clf

    def predict(self,clf,x):
        return clf.predict_proba(x)[:,1].ravel()

    def get_ROC_points(self,y_pro,y_test):
        point_x = []
        point_y = []
        data = []
        threshold = 0
        test_num = y_test.shape[0]
        for i in range(1000):
            y_predict = np.zeros(test_num, dtype=np.int)
            y_predict[y_pro > threshold] = 1
            true_pos = int(np.sum(y_predict + y_test == 2))
            true_neg = int(np.sum(y_predict + y_test == 0))
            false_pos = int(np.sum(y_predict - y_test == 1))
            false_neg = int(np.sum(y_predict - y_test == -1))
            point_x.append(false_pos / (false_pos + true_neg))
            point_y.append(false_neg / (false_neg + true_pos))
            data.append([true_pos,true_neg,false_pos,false_neg])
            threshold += 0.01
        return (point_x,point_y),data

    def all_together(self,dir,opt):
        x_train,x_test,y_train,y_test =self.get_data(dir,opt)
        print('...training')
        clf = self.train(x_train,y_train)
        print('done')
        print('...predict')
        predict = self.predict(clf,x_test)
        print('done')
        return self.get_ROC_points(predict,y_test)

def get_name(dir,index):

    config = configparser.ConfigParser()
    config.read('config.ini')
    return config.get('data_set','data_dir%i' % dir)[17:-1]+'_'+config.get('data_set','hog_data%i'% index)[-15:-4]

if __name__ == '__main__':
    '''
    data = Data()
    hog = cv.HOGDescriptor((48, 48), (8, 8), (4, 4), (4, 4), 9, )
    pos_num, neg_num, row, col, pos_data_0, neg_data_0 = data.load_data()
    temp = hog.compute(pos_data_0[0])
    pos_data = np.zeros((pos_num,*temp.shape),dtype=float)
    neg_data = np.zeros((neg_num,*temp.shape), dtype=float)
    for i in range(pos_num):
        pos_data[i] = hog.compute(pos_data_0[i])
    for i in range(neg_num):
        neg_data[i] = hog.compute(neg_data_0[i])
    neg_data = neg_data.reshape(neg_data.shape[0],-1)
    pos_data = pos_data.reshape(pos_data.shape[0],-1)
    print(neg_data.shape)
    print(neg_data)
    test_size = .25
    pos_train_num = int(pos_num * (1-test_size))
    neg_train_num = int(neg_num * (1-test_size))
    x_train = np.concatenate((pos_data[:pos_train_num],neg_data[:neg_train_num]))
    x_test = np.concatenate((pos_data[pos_train_num:],neg_data[neg_train_num:]))
    y_train = np.concatenate((np.ones(pos_train_num,dtype=int),np.zeros(neg_train_num,dtype=int)))
    y_test = np.concatenate((np.ones(pos_num - pos_train_num,dtype=int),np.zeros(neg_num - neg_train_num,dtype=int)))
    test_num = pos_num + neg_num - pos_train_num - neg_train_num

    threshold = 0
    print(x_train.shape)
    print(y_train.shape)
    print(x_train)
    print(y_train)
    load_opt = 0
    if load_opt:
        with open('model.pkl', 'rb') as f:
            clf = pickle.load(f)
    else:
        print('...training', )
        clf = svm.SVC(probability=True, class_weight='balanced', gamma='auto', random_state=np.random.randint(233))
        clf.fit(x_train, y_train)
        print('done')
        with open('model.pkl','wb') as f:
            pickle.dump(clf,f)
    print('...predict')
    y_pro = clf.predict_proba(x_test)[:,1].ravel()
    print('y_pro',y_pro)
    print('done')
    point_x = []
    point_y = []
    for i in range(1000):
        y_predict = np.zeros(test_num,dtype=np.int)
        y_predict[y_pro > threshold] = 1
        true_pos = int(np.sum(y_predict + y_test == 2))
        true_neg = int(np.sum(y_predict + y_test == 0))
        false_pos = int(np.sum(y_predict - y_test == 1))
        false_neg = int(np.sum(y_predict - y_test == -1))
        print('tp %i  tn %i   fp %i fn %i' % (true_pos,true_neg,false_pos,false_neg))
        point_x.append(false_pos / (false_pos + true_neg))
        point_y.append(false_neg / (false_neg + true_pos))
        threshold += 0.01
    '''
    plot.title('ROC Curve')
    plot.xlabel('False alarm rate')
    plot.ylabel('missed positive rate')
    my_svm = SVM()
    for dir in range(4):
        for i in range(9):
            model_name = get_name(dir,i)
            print('now start:',model_name)
            start_time = time.time()
            points,data = my_svm.all_together(dir,i)
            with open('SVM_result/'+model_name+'.pkl','wb') as f:
                pickle.dump(data,f)
            plot.plot(points[0],points[1],label=model_name)
            print('finished in :',time.time()-start_time)
    plot.legend()
    plot.show()
