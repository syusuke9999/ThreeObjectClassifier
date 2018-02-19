import keras
from keras.backend import tensorflow_backend as backend
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Activation, Dropout, Flatten, Dense, ActivityRegularization, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import h5py

classes = ["watch","camera","eyeglasses"]
num_classes = len(classes)

batch_size = 32

def main():
    X_train, X_test, y_train, y_test = np.load("./image/3obj.npy")
    X_train = X_train.astype("float") / 256
    X_test = X_test.astype("float") / 256
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    model = model_train(X_train, y_train, X_test, y_test)
    model_eval(model, X_test, y_test)
    backend.clear_session()

def build_model(in_shape):
    model = Sequential()
    model.add(Conv2D(16, (2, 2), padding='same', input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    plot_model(model, to_file='model.png')
    opt = keras.optimizers.RMSprop(lr=0.001, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
        optimizer=opt,metrics=['accuracy'])
    return model

def model_train(X_train, y_train, X_test, y_test):
    model = build_model(X_train.shape[1:])
    es_cb = keras.callbacks.EarlyStopping(monitor='loss', patience = 20, min_delta = 0, verbose=1, mode = 'auto')
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
    checkpointer = ModelCheckpoint(monitor = 'loss',filepath='./image/weights.hdf5', verbose = 1, save_best_only = True)
    history = model.fit(X_train, y_train, batch_size = batch_size, epochs = 200, validation_data = (X_train, y_train), callbacks=[reduce_lr, checkpointer, es_cb])
    hdf5_file = "./image/3obj-model.hdf5"
    model.save(hdf5_file)
    return model

def model_eval(model, X, y):
    pre = model.predict(X, batch_size = batch_size, verbose = 1)
    for i,v in enumerate(pre):
        pre_ans = v.argmax()
        ans = y[i].argmax()
        dat = X[i]
        if ans == pre_ans: continue
        print("[NG]", classes[pre_ans], "!=", classes[ans])
        print(v*[100,100,100])
        fname = "./error/" + str(i) + "-" + classes[pre_ans] + \
            "-ne-" + classes[ans] + ".png"
        dat *= 256
        img = Image.fromarray(np.uint8(dat))
        img.save(fname)
    scores = model.evaluate(X, y, verbose = 1)
    print('Test Loss: ', scores[0])
    print('Test Accuracy: ', scores[1])

def show_layer_image(model, layer_num):
    print('Layer Name: {}'.format(model.layers[layer_num].get_config()['name']))
    config_name = model.layers[layer_num].get_config()['name']
    config_name = config_name[0:6]
    if(config_name=="conv2d"):
        W = model.layers[layer_num].get_weights()[0]
        W = W.transpose(3, 2, 0, 1)
        nb_filter, nb_channel, nb_row, nb_col = W.shape
        for i in range(nb_channel):
            im = W[i , 0]
            # scaling images
            scaler = MinMaxScaler(feature_range = (0, 255))
            im = scaler.fit_transform(im)
            #plt.axis('off')
            plt.imsave("./layer/" + str(layer_num) + "_" + str(i) + ".png", im)

if __name__ == "__main__":
    main()
