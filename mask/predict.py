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

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)



model = load_model('data.hdf5',  custom_objects={'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss})


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
                
            img = cv2.imread(fullpath)   
            img = cv2.resize(img, (WIDTH, HEIGHT), interpolation = cv2.INTER_AREA)
            img = np.reshape(img, (1, WIDTH, HEIGHT, 3))
            img = img / 255

            output = model.predict(img)

            output = np.array((output > .5)).astype('int')
            output = np.reshape(output, (WIDTH, HEIGHT))
            output = output * 255

            masked = np.zeros((WIDTH, HEIGHT, 3))

            for x in range(img.shape[1]):
                for y in range(img.shape[2]):
                    if output[x][y]:
                        masked[x][y] = img[0][x][y]*255
            

            cv2.imwrite('./output/' + str(i) + '.png', np.reshape(masked, (WIDTH, HEIGHT, 3)))
            i += 1


fill_raw('../data/train_hq')



