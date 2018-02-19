import PIL
from PIL import Image
from PIL import ImageEnhance, ImageFilter
import os, glob
import numpy as np
import random, math
import re

classes = ["watch","camera","eyeglasses"]
num_classes = len(classes)
image_size = 128

# 画像の読み込み
#画像データ
X = []
#ラベルデータ
Y = []

def amplify_data(classes,fname, is_train):
    img = Image.open(fname)
    img = img.convert("L")
    img_gary = img.convert("RGB")
    img_resize = img_gary.resize((image_size, image_size),Image.BICUBIC)
    # size 3x3
    # scale default
    f = ImageFilter.Kernel((3,3), (-2, -1, 0, -1, 1, 1, 0, 1, 2),scale = 3)
    img_resize = img_resize.filter(f)
    result = re.sub(r'^.\/image','./filtered',fname)
    img_resize.save(result)
    data = np.asarray(img_resize)
    X.append(data)
    Y.append(classes)
    if not is_train: return
    for angle in range(-20, 20, 5):
        img_r = img_resize.rotate(angle)
        data = np.asarray(img_r)
        X.append(data)
        Y.append(classes)
        img_r45 = img_resize.rotate(45)
        data = np.asarray(img_r45)
        X.append(data)
        Y.append(classes)
        img_r90 = img_resize.transpose(Image.ROTATE_90)
        data = np.asarray(img_r90)
        X.append(data)
        Y.append(classes)
        img_r270 = img_resize.transpose(Image.ROTATE_270)
        data = np.asarray(img_r270)
        X.append(data)
        Y.append(classes)
        tmp = img_resize.transpose(Image.FLIP_LEFT_RIGHT)
        data = np.asarray(tmp)
        X.append(data)
        Y.append(classes)


def make_sample(files, is_train):
    global X, Y
    X = []; Y = []
    for classes, fname in files:
        amplify_data(classes, fname, is_train)
    return np.array(X), np.array(Y)

allfiles = []
for idx, classes in enumerate(classes):
    img_dir = "./image/" + classes + "/*.*"
    files = glob.glob(img_dir)
    for f in files:
        allfiles.append((idx, f))

random.shuffle(allfiles)
th = math.floor(len(allfiles) * 0.6)
train = allfiles[0:th]
test = allfiles[th:]
X_train, y_train = make_sample(train, True)
X_test, y_test = make_sample(test, False)
print("X_train:" + str(X_train.shape) + " X_test:" + str(X_test.shape) + " y_train:" + str(y_train.shape) + " y_test:" + str(y_test.shape))
xy = (X_train, X_test, y_train, y_test)
np.save("./image/3obj.npy",xy)
