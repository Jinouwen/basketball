from keras.models import Model, load_model
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras import optimizers
from keras import losses
from keras.models import Model
from get_dataset import *
import random
import matplotlib.pyplot as plt
import pickle
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def Model_Para():
    epochs = 2
    batch_size = 16
    train_prop = 0.4
    neg_prop = 0.01
    return epochs, batch_size, train_prop, neg_prop


def load_data(train_prop, neg_prop):
    data = Data()
    pos_num, neg_num, row, col, pos_data, neg_data = data.load_data()
    # shuffle neg_data
    neg_index = np.random.randint(0, high=neg_num, size=int(neg_num*neg_prop), dtype='l')
    neg_data = neg_data[neg_index, :, :, :]
    neg_num = int(neg_num*neg_prop)
    # shape of data_set
    data_set_x = np.zeros((pos_num + neg_num, row, col, 3))
    data_set_y = np.zeros((pos_num + neg_num, 1))
    # build data_set
    data_set_x[0: pos_num, :, :, :] = pos_data
    data_set_y[0: pos_num] = 1
    data_set_x[pos_num:pos_num + neg_num, :, :] = neg_data
    data_set_y[pos_num:pos_num + neg_num] = 0
    # shuffle
    index = [i for i in range(len(data_set_y))]
    random.shuffle(index)
    data_set_x = data_set_x[index]
    data_set_y = data_set_y[index]
    # split data
    train_num = int(data_set_x.shape[0] * train_prop)
    test_num = data_set_x.shape[0] - train_num
    train_set_x = data_set_x[0:train_num]
    train_set_y = data_set_y[0:train_num]
    test_set_x = data_set_x[train_num:train_num+test_num]
    test_set_y = data_set_y[train_num:train_num+test_num]

    assert train_set_x.shape == (train_num, row, col, 3)
    assert train_set_y.shape == (train_num, 1)
    assert test_set_x.shape == (test_num, row, col, 3)
    assert test_set_y.shape == (test_num, 1)
    return train_set_x, train_set_y, test_set_x, test_set_y

'''
model定义
'''
def Classifier_Model(input_shape):
    X_input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(32, (3, 3), strides=(1, 1), name='conv0', padding='same')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((2, 2), name='max_pool0')(X)

    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    model = Model(inputs=X_input, outputs=X, name='Classifier_Model')

    return model


epochs, batch_size, train_prop, neg_prop = Model_Para()
X_train, Y_train, X_test, Y_test = load_data(train_prop, neg_prop)
model = Classifier_Model(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
model.compile(
    optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
    loss=losses.binary_crossentropy,
    metrics=["accuracy"])
model.fit(x=X_train, y=Y_train, epochs=epochs, batch_size=batch_size)
loss, acc = model.evaluate(X_test, Y_test)
print("Dev set accuracy = ", acc)

# write the config file
model_count = 0
curve_path = 'model_and_curve/'
config_path = 'classifier_config/'
model_weights = curve_path + str(model_count)+'_classifier_model.h5'
model.save_weights(model_weights)
cf = configparser.ConfigParser()
cf.read(config_path + "cnn_cfg.ini")
cf.add_section("model"+str(model_count))
cf.set("model"+str(model_count), "model_weights", str(model_weights))
cf.set("model"+str(model_count), "epochs", str(epochs))
cf.set("model"+str(model_count), "batch_size", str(batch_size))
cf.set("model"+str(model_count), "train_prop", str(train_prop))
cf.set("model"+str(model_count), "neg_prop", str(neg_prop))
cf.write(open(config_path + "cnn_cfg.ini", "w"))


def predict_ROC(X, predict, threshold):
    m = X.shape[0]
    Y_predict = np.zeros((m, 1))
    for i in range(len(predict)):
        if predict[i] >= threshold:
            Y_predict[i] = 1
        else:
            Y_predict[i] = 0
    return Y_predict


def draw_ROC(recall, FAR):
    plt.figure(figsize=(4, 4))
    MPN = 1 - recall
    plt.plot(MPN, FAR, color="red", linewidth=2)
    plt.ylabel("Missed Positive Number")
    plt.xlabel("False Alarm Number")
    plt.title("ROC")
    plt.show()


def save_ROC(model_count, recall, FAR):
    cur = recall, FAR
    with open(curve_path + '%scurve.pkl' % str(model_count), 'wb') as f:
        pickle.dump(cur, f)


def prepare_ROC(model, model_num, X, Y, threshold):
    # get the predict
    predict = model.predict(X)
    l = len(threshold)
    # acc = np.zeros((l, 1))
    # prec = np.zeros((l, 1))
    recall = np.zeros((l, 1))
    FAR = np.zeros((l, 1))
    # calculate Y
    for k in range(l):
        Y_predict = predict_ROC(X, predict, threshold[k])
        # calculate P/N
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in range(X.shape[0]):
            if Y_predict[i] - Y[i] == 0 and Y[i] == 1:
                TP = TP + 1
            elif Y_predict[i] - Y[i] == 0 and Y[i] == 0:
                TN = TN + 1
            elif Y_predict[i] - Y[i] == 1:
                FP = FP + 1
            elif Y_predict[i] - Y[i] == -1:
                FN = FN + 1
            else:
                pass
        # acc[k] = (TP + TN) / (TP + TN + FP + FN)
        # prec[k] = TP / (TP + FP)
        recall[k] = TP / (TP + FN)
        FAR[k] = FP / (FP + TN)
    draw_ROC(recall, FAR)
    save_ROC(model_num, recall, FAR)


# load model
model.load_weights(curve_path + str(model_count)+'_classifier_model.h5')
threshold = np.linspace(0, 1, 1000)
prepare_ROC(model, model_count, X_test, Y_test, threshold)

