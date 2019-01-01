from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
from scipy import ndimage
from random import randint

from keras.layers import Conv2D, MaxPooling2D, Dropout, LeakyReLU, Flatten, Dense
from keras.models import Model, load_model, Sequential
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import applications

def l2_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    return np.square(y_true_f - y_pred_f)


def l1_loss(y_true, y_pred):
    return np.abs(y_true - y_pred)


model = load_model('data.hdf5',  custom_objects={'l2_loss': l2_loss, 'l1_loss': l1_loss})



WIDTH = 256
HEIGHT = 256


def fill_raw(data_path):
    path, dirs, files = next(os.walk(data_path))

    sys.stdout.flush()
    i = 0
    for path, subdirs, files in (os.walk(data_path)):
        data = np.array(files)
        
        for name in tqdm(data):
            fullpath = os.path.join(path, name)

            #if not '_05' in fullpath:
                #continue
                
            img = cv2.imread(fullpath)   
            img = cv2.resize(img, (WIDTH, HEIGHT), interpolation = cv2.INTER_AREA)
            img = np.reshape(img, (1, WIDTH, HEIGHT, 3))
            img = img / 255

            output = model.predict(img)[0]
            
            img = img * 255
            img = np.reshape(img, (WIDTH, HEIGHT, 3))
            output = output.astype('int')

            img = cv2.rectangle(np.uint8(img), (output[1], output[0]), (output[3], output[2]), (255, 0, 0), 3, cv2.LINE_8)

            cv2.imwrite('./output/' + str(i) + '.png', img)
            i += 1


fill_raw('data/train_hq')



