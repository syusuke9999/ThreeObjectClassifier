import keras
from keras.layers import Activation, Dropout, Flatten, Dense, ActivityRegularization, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.backend import tensorflow_backend as backend
import three_obj_cnn as three_obj
from three_obj_cnn import show_layer_image
import sys, os
from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import h5py

if len(sys.argv) <=1:
    print("checker.py ファイル名")
    quit()

image_size = 128

classes = ["腕時計","カメラ","眼鏡"]

X = []
files = []
fname = sys.argv[1]
img = Image.open(fname)
img_gray = img.convert("L")
img_not_rgb = img_gray.convert("RGB")
img_s = img_not_rgb.resize((image_size, image_size),Image.LANCZOS)
f = ImageFilter.Kernel((3,3), (-2, -1, 0, -1, 1, 1, 0, 1, 2),scale = 3)
img_s = img_s.filter(f)
in_data = np.asarray(img_s)
X.append(in_data)
files.append(fname)
X = np.array(X)
model = three_obj.build_model(X.shape[1:])
model.load_weights("./image/3obj-model.hdf5")
html = ""
pre = model.predict(X)
for i, v in enumerate(pre):
    print(v)

p = pre.argmax()

print("推定オブジェクト名：",classes[p])


html += """
    <h3>入力</h3>
    <div>
    <p><img src="{0}" width=300></p>
    <br>
    <div>
    <p>推定オブジェクト名:{1}</p>
    </div>
    """.format(files[0],classes[p])

html = "<html><body style='text-align:center;'>" \
    "<style> p { margin:0; padding:0; } </style>" + html + "</body></html>"

with open("result.html", "w") as f:
    f.write(html)

backend.clear_session()
