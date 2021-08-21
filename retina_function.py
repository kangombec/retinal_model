#!/usr/bin/env python
# coding: utf-8
# %%
get_ipython().system('pip3 install natsort')
get_ipython().system('pip install opencv-python')
get_ipython().system('pip install opencv-python-headless')
get_ipython().system('pip install opencv-contrib-python-headless')
get_ipython().system('pip install keras')
get_ipython().system('pip install focal-loss')
get_ipython().system('pip install livelossplot')
get_ipython().system('pip install numba')

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# print(os.getcwd())


# %%
# get_ipython().run_line_magic('matplotlib', 'inline')
import os
from glob import glob
from natsort import natsorted
import cv2 as cv
from tensorflow.keras import models

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
from tensorflow.keras.layers import Conv2D,  Dense, LeakyReLU, Dropout, BatchNormalization, Reshape, Activation, MaxPool2D, Conv2DTranspose, Input, Concatenate, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.metrics import Recall, Precision, Accuracy
import numpy as np
import pandas as pd
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.optimizers import Adam
from skimage import color
from tensorflow.keras import initializers, regularizers, constraints
from skimage.filters import threshold_minimum, threshold_otsu
from livelossplot import PlotLossesKeras
from numba import cuda
from tensorflow.keras.losses import binary_crossentropy




# suppress GPU warning messages.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# suppress all warnings
warnings.filterwarnings('ignore')

"""
This function displays the image
@param axis: represents the axis to be plotted. The axis can be on a specific row or column
@param image: represents the image to be displayed.
@param title: represents the title of the image to be displayed or plotted.
"""
def show_image_roi(axis, image, title):
       axis.imshow(image, cmap='gray')
       axis.axis('off')
       axis.set_title(title)
        

"""
This function get and return the region of interest of the retinal image
@param image_name: represents the retina image of which we want to extract the region of interest.
@return image_ represents the extracted region of the retinal image.
@return ROI shows the image with circle used in identifying the region of interest; detected by the area with brightest spot.

modified from: https://github.com/keko950/Optic-Disc-Segmentation-OpenCV/blob/master/entrega.py
"""       
def get_roi(image_name):
       
       gray = cv.cvtColor(image_name, cv.COLOR_BGR2GRAY)
       canny = cv.Canny(gray, 120, 255, 1)
       
       cnts = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
       

       ROI = image_name
       
       orig = ROI.copy()
       gray = cv.cvtColor(ROI, cv.COLOR_BGR2GRAY)
       
       # clip out dark spaces around the image.
       
       (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(gray)
       
       cv.circle(ROI, maxLoc, 2, (255, 0, 0), 2)
       
       gray = cv.GaussianBlur(gray, (101, 101), 8)
       
       (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(gray)
       image_ = orig.copy()
       
       cv.circle(ROI, maxLoc, 101, (255, 255, 0), 10)
       
       first = int(maxVal)
       second = int(maxLoc[0])
       third = int(maxLoc[1]) 
       iss = image_
           
       if third < 600:
              d = first//2 + 40
              f = first//2 + 40
       else:
              d = 180
              f = 180
       
       # resize the image to only capture the region of interest. we some spaces around the height and width
       # This is tricky, but yea, God made it possible.
       
       image_ = image_[third-f:third+f, second-d:second+d]
       return image_, ROI


"""
This function allocates an image to a given folder
@param: image represents the image
@param: image_name represents the name of the image
@param: directory represents the directory or folder to save the image.
"""
def save_images(image, image_name, directory):
    # change the directory to the image saving directory
    os.chdir(directory) 
    cv.imwrite(image_name, image)


"""
This function displays the image
@param: image represents the image to be displayed.
@param: axis represents the axis to display the image, since we are displaying it in a grid.
@param: title represents the title of the image to be displayed.
"""
def show_image(axis, image, title):
       img = cv.imread(image, cv.IMREAD_COLOR)
       img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
       img = cv.resize(img,(400,400))  
       axis.imshow(img)
       axis.axis('off')
       axis.set_title(title)
        

"""
This function is used to create directories or folders to store files
@param: directory represents the directory name;
"""
def create_directory(directory):
       # check if the directory exists, if it does exist, do not create
        # However, force the directory to be created if it does not exist.
       if not os.path.exists(directory):
              os.makedirs(directory)
                

"""
This function empty a directory, bu=y deleting all files present in the directory.
@param directory: represents the name of the directory
"""                
def empty_directory(directory):
    # check if directory exist.
    # if it does not exist, just create it and do nothing
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        # loop through all files present in the directory, and forcefully delete them.
        for files in glob(directory+"/*"):
            os.remove(files)
    


# %%
"""
This function resizes to width 400 and height 400, and also
converts it to an RGB color space
"""
def resize_rgb(image_bgr):
       resized = cv.resize(image_bgr, (400,400))
       return cv.cvtColor(resized, cv.COLOR_BGR2RGB)


"""
This function returns the optic disc and cup of an annotated or marked retinal image.
@param: image_y represents the marked or annotated image, whose optic disc and cup we wish to mask.
"""
def get_optic_disc_cup_mask(image_y):
       global optic_cup_, first, optic_cup
       image = image_y.copy()
       dark_black1 = (10, 10, 0)
       dark_black2 = (99, 90, 150)
       kernel = np.ones((5,5),np.uint8)

       mask= cv.inRange(image, dark_black1, dark_black2)
       mask_2 = cv.inRange(image, dark_black1, dark_black2)
       mask_2 = cv.dilate(mask_2, kernel, iterations = 1)
       mask_3 = mask.copy()

       medianFiltered = cv.medianBlur(mask,5)
       cnts, hierarchy = cv.findContours(medianFiltered, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
       cntss, hierarchys= cv.findContours(medianFiltered, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
       
       for i in cnts:
              first = cv.drawContours(mask, [i], -1, (255, 255, 255), -10)

       for i in cntss:
              optic_cup = cv.drawContours(mask_2, [i], -1, (0, 0, 0), 10)
       
       new_optic_cup = optic_cup.copy()
       medianFiltered = cv.medianBlur(new_optic_cup,5)
       cnts, hierarchy = cv.findContours(medianFiltered, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
       
       for i in cnts:
              optic_cup_ = cv.drawContours(new_optic_cup, [i], -1, (255, 255, 255), -10)
                
       optic_cup_ = cv.morphologyEx(optic_cup_, cv.MORPH_OPEN, kernel)

              
       return first, optic_cup_



"""
this function saves the masked optic cup to a given directory
@param: image represents a list of images, whose optic cup we wish to extract and save.
@param: directory_name represents the name of the directory, where the optic cup will be saved
"""
def create_save_optic_cup(image, directory_name):       
       """
       destroy and create new directory.
       check if the directory exists, if it exists ignore and write images to the directory.
       modified from: https://www.geeksforgeeks.org/delete-a-directory-or-file-using-python/
       """
       try:
              if not os.path.exists(directory_name):
                  os.makedirs(directory_name)
              else:
                     for files in glob(directory_name+"/*"):
                            os.remove(files)
                             
              for image_path in image:
                     img = cv.imread(image_path)
                     # resize the image to width 224 and height 224.
                     imag, _ = get_roi(img)
                     resized_img = resize_rgb(imag)
                     _, optic_cup = get_optic_disc_cup_mask(resized_img)
       
                     os.chdir(directory_name)
                     cv.imwrite(os.path.basename(image_path), optic_cup)
                  
       except OSError as error:
              print("Directory '% s' can not be removed" % directory_name)
            

"""
this function saves the masked optic disc to a given directory
@param: image represents a list of images, whose optic disc we wish to extract and save.
@param: directory_name represents the name of the directory, where the optic cup will be saved
"""            
def create_save_optic_disc(image, directory_name):       
       """
       destroy and create new directory.
       check if the directory exists, if it exists ignore and write images to the directory.
       modified from: https://www.geeksforgeeks.org/delete-a-directory-or-file-using-python/
       """
       try:
              if not os.path.exists(directory_name):
                  os.makedirs(directory_name)
              else:
                     for files in glob(directory_name+"/*"):
                            os.remove(files)
                             
              for image_path in image:
                     img = cv.imread(image_path)
                     # resize the image to width 224 and height 224.
                     imag, _ = get_roi(img)
                     resized_img = resize_rgb(imag)
                     optic_disc, _ = get_optic_disc_cup_mask(resized_img)
                     os.chdir(directory_name)
                     cv.imwrite(os.path.basename(image_path), optic_disc)
                  
       except OSError as error:
              print("Directory '% s' can not be removed" % directory_name)

    
"""
This function creates new directory and stores the new image in the created folder.   
@param: image represents a list of images.
@param: directory_name represents the directory_name.   
"""
def create_save_img(image, directory_name):       
       """
       destroy and create new directory.
       check if the directory exists, if it exists ignore and write images to the directory.
       modified from: https://www.geeksforgeeks.org/delete-a-directory-or-file-using-python/
       """
       try:
              if not os.path.exists(directory_name):
                  os.makedirs(directory_name)
              else:
                     for files in glob(directory_name+"/*"):
                            os.remove(files)
                             
              for image_path in image:
                     img = cv.imread(image_path, cv.IMREAD_COLOR)
                     imag, _ = get_roi(img)
                    
                     resized_img = resize_rgb(imag)

                     # resize the image to width 224 and height 224.
                     os.chdir(directory_name)
                     cv.imwrite(os.path.basename(image_path), cv.cvtColor(resized_img, cv.COLOR_BGR2RGB))
                  
       except OSError as error:
              print("Directory '% s' can not be removed" % directory_name)

def color_mask(mask):
    global first
    mask_2= cv.inRange(mask, 127, 255)
    medianFiltered = cv.medianBlur(mask_2,5)
    cnts, hierarchy = cv.findContours(medianFiltered, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for i in cnts:
        first = cv.drawContours(mask_2, [i], -1, (255, 255, 255), -10)
    
    return first

        
def get_largest_contour(mask):
    mask = mask.astype(np.uint8)
    # edges = cv.Canny(mask, 50, 200)
    contours, hierachy = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)
        largest_contour = sorted_contours[0]
        pred = cv.drawContours(mask.copy(), largest_contour, -1, (255,255,255), 10)
        
        contour_x, hierachy = cv.findContours(pred.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        pred_ = cv.drawContours(pred.copy(), largest_contour, -1, (255,0,0), -10)
        
    else:
        pred_ = mask
    
    return pred_


# %%
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

"""
This function builds the layers for both encoder and decoder block of the UNET model
@param: input_neuron represents the input neuron to the cnn given shape 224 by 224
@param: num_filt represents the number of kernels

modified from: https://github.com/nikhilroxtomar/Cell-Nuclei-Segmentation-in-TensorFlow-2.0
"""

def conv_block(input_neuron, num_filt):
    x_block = Conv2D(num_filt, 3, padding="same", kernel_initializer="he_normal")(input_neuron)
    x_block = BatchNormalization()(x_block)
    x_block = Activation("relu")(x_block)
    
    x_block = Conv2D(num_filt, 3, padding="same", kernel_initializer="he_normal")(x_block)
    x_block = BatchNormalization()(x_block)
    x_block = Activation("relu")(x_block)
    
    return x_block
    

"""
This function builds the encoder_block otherwise known as the contraction phase of the Unet
@param: input_neuron represents the input neuron to the cnn given shape 224 by 224
@param: num_filt represents the number of kernels

modified from: https://github.com/nikhilroxtomar/Cell-Nuclei-Segmentation-in-TensorFlow-2.0
"""

def enc_block(input_neuron, num_filt, dropout):
    x_block = conv_block(input_neuron, num_filt)
    pool_block = MaxPool2D((2,2))(x_block)
    pool_block = Dropout(dropout)(pool_block)
    return x_block, pool_block
"""
This function builds the decoder_block otherwise known as the expansion phase of the Unet
@param: input_neuron represents the input neuron to the cnn given shape 224 by 224
@param: num_filt represents the number of kernels

modified from: https://github.com/nikhilroxtomar/Cell-Nuclei-Segmentation-in-TensorFlow-2.0
"""

def dec_block(input_neuron, skip_features, num_filt, dropout):
    x_block = Conv2DTranspose(num_filt, (3,3), strides = (2,2), padding="same")(input_neuron)
    x_block = Concatenate()([x_block, skip_features])
    x_block = Dropout(dropout)(x_block)
    x_block = conv_block(x_block, num_filt)
    
    return x_block

"""
This function builds the Unet model
modified from: https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47
"""
def build_unet_model(input_shape):
    input_neuron = Input(input_shape)
    # encoder block 
    s_1, p_1 = enc_block(input_neuron, 32, 0.1)
    s_2, p_2 = enc_block(p_1, 64, 0.1)
    s_3, p_3 = enc_block(p_2, 128, 0.1)
    s_4, p_4 = enc_block(p_3, 256, 0.1)
    
    # bridge.
    bridge = conv_block(p_4, 512)
    
    # decoder block
    d_1 = dec_block(bridge, s_4, 256, 0.1)
    d_2 = dec_block(d_1, s_3, 128, 0.1)
    d_3 = dec_block(d_2, s_2, 64, 0.1)
    d_4 = dec_block(d_3, s_1, 32, 0.1)
    
    # output block
    output = Conv2D(1,(1,1),padding="same", activation ="sigmoid")(d_4)
    model_unet = Model(input_neuron, output, name="ret_u-net")
    
    return model_unet


# ### Model Metrics for Evaluating the performance of model semantic segmentation

# %%
"""
This function computes the intersection over union for the segmented
This is a case of semantic segmentation.
@param: y_true represents the original mask (ground truth mask)
@param: y_pred represents the predicted mask ()

modified from: https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
"""

def iou(y_true, y_pred, smooth=1):
    def f(y_true, y_pred, smooth = 1):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        score = (intersection + smooth) / (union + smooth)
        score = score.astype(np.float32)
        return score
    
    return tf.numpy_function(f, [y_true, y_pred, smooth], tf.float32)

"""
This function computes the dice coefficient of the segmentation.
This is a case of semantic segmentation.
@param: y_true represents the original mask (ground truth mask)
@param: y_pred represents the predicted mask ()

modified from: https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
"""

def dice_coef(y_true, y_pred, smooth = 1):
    y_true_ = Flatten()(y_true)
    y_pred_ = Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_ * y_pred_)
    
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_) + tf.reduce_sum(y_pred_) + smooth)
    
"""
This function computes the dice loss of the segmentation.
This is a case of semantic segmentation.
@param: y_true represents the original mask (ground truth mask)
@param: y_pred represents the predicted mask ()

modified from: https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
"""

def dice_loss(y_true, y_pred, smooth=1):
    return 1.0 - dice_coef(y_true, y_pred, smooth)


"""
This function computes the iou loss of the segmentation.
This is a case of semantic segmentation.
@param: y_true represents the original mask (ground truth mask)
@param: y_pred represents the predicted mask

modified from: https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
"""
def iou_loss(y_true, y_pred, smooth=1):
    return 1.0 - iou(y_true, y_pred, smooth)


"""
The function summarises a list of loss to be compared in this segmentation model.
modified from: https://github.com/shruti-jadon/Traumatic-Brain-Lesions-Segmentation/blob/master/loss_functions.py
"""
def focal_loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    logits = tf.log(y_pred / (1 - y_pred))

    loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)

    return tf.reduce_mean(loss)

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), gamma)



def weighted_bce_loss(y_true, y_pred, weight):
    # avoiding overflow
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    
    # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
    (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)

def weighted_dice_loss(y_true, y_pred, weight=1e-7):
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    loss = 1. - K.sum(score)
    return loss

def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    averaged_mask = K.pool2d(
            y_true, pool_size=(11, 11), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) + \
    weighted_dice_loss(y_true, y_pred, weight)
    return loss

