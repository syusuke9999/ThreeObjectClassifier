import keras
from keras.layers import Activation, Dropout, Flatten, Dense, ActivityRegularization, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.backend import tensorflow_backend as backend
import three_obj_cnn as three_obj
import sys, os, h5py, glob
from PIL import Image, ImageFilter
import numpy as np
from sklearn.preprocessing import MinMaxScaler

image_size = 128
batch_size = 32

classes = ["腕時計","カメラ","眼鏡"]
classes_english = ["watch","camera","eyeglasses"]
allfiles = []

def main():
    X, Y = CollectFiles()
    model = three_obj.build_model([128,128,3])
    model.load_weights("../image/3obj-model.hdf5")
    Classify(model,X,Y)

def CollectFiles():
    img_dir = "./TestImage/*.*"
    files = glob.glob(img_dir)
    X_Check = []
    X_Filtered = []
    for f in files:
        img = Image.open(f)
        img_rs = img.resize((image_size,image_size),Image.BICUBIC)
        data = np.asarray(img)
        X_Check.append(data)
        img_rs = img_rs.convert("L")
        img_rs = img_rs.convert("RGB")
        filters = ImageFilter.Kernel((3,3), (-2, -1, 0, -1, 1, 1, 0, 1, 2),scale = 3)
        img_filterd = img_rs.filter(filters)
        data2 = np.asarray(img_filterd)
        X_Filtered.append(data2)
    return np.array(X_Check) , np.array(X_Filtered)

def Classify(model, X, Y):
    predicted = model.predict(Y)
    for i,v in enumerate(predicted):
        pre_ans = v.argmax()
        #print("推定オブジェクト名：",classes[pre_ans])
        img = Image.fromarray(np.uint8(X[i]))
        fname = "./ClassifiedImage/" + classes_english[pre_ans] + "/" + str(i) + ".jpg"
        img.save(fname)

if __name__ == "__main__":
    main()
