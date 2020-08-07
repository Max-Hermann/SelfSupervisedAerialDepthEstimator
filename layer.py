from __future__ import division
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Conv2DTranspose, UpSampling2D, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Add, Multiply, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential

import numpy as np

def custom_Conv2DTranspose(output_chn):
    scale = 2
    filter_size = (2 * scale - scale % 2)
    num_channels = 3

    #Create bilinear weights in numpy array
    bilinear_kernel = np.zeros([filter_size, filter_size], dtype=np.float32)
    scale_factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = scale_factor - 1
    else:
        center = scale_factor - 0.5
    for x in range(filter_size):
        for y in range(filter_size):
            bilinear_kernel[x,y] = (1 - abs(x - center) / scale_factor) * \
                                (1 - abs(y - center) / scale_factor)
    weights = np.zeros((filter_size, filter_size, num_channels, num_channels))
    for i in range(num_channels):
        weights[:, :, i, i] = bilinear_kernel

    #assign numpy array to constant_initalizer and pass to get_variable
    bilinear_init = tf.constant_initializer(value=weights, dtype=tf.float32)    

    return Conv2DTranspose(output_chn, (filter_size,filter_size), strides=(scale), padding='SAME',
        kernel_initializer=bilinear_init)


def up_block(x, output_chn, name, short_cut=None, upsample=True,use_conv_transpose=False):
    with tf.name_scope(name):
        x = ReflectionPadding2D()(x)
        x = Conv2D(output_chn, kernel_size=3, strides=(1, 1), padding='valid', activation='elu', name=name+"_conv2d_1")(x)

        if upsample:
            print(x.shape)
            if use_conv_transpose:
                x = Conv2DTranspose(output_chn, (1,1), strides=(2,2))(x)
            else:
                x = UpSampling2D((2,2))(x)
            print(x.shape)

        if short_cut is not None:           
            x = Concatenate(axis=3)([x, short_cut])
                      

        x = ReflectionPadding2D()(x)
        x = Conv2D(output_chn, kernel_size=3, strides=(1, 1), padding='valid', activation='elu', name=name+"_conv2d_2")(x)
    return x


def get_disparity_from_layer(current_layer, name):
    with tf.name_scope(name):
        x = ReflectionPadding2D()(current_layer)
        x = Conv2D(1, kernel_size=3, strides=(1, 1), padding='valid', activation='sigmoid', name=name+"_conv2d_1")(x)        
    return x


def ReflectionPadding2D():   
    return Lambda((lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")))


def res_block(input, output_chn, name, first_flag=False, repetitions=2):
    with tf.name_scope(name):
        if first_flag is False:
            input = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input)
        else:
            input = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(input)

        resblock_output = input
        for i in range(repetitions):
            resblock_output = basic_res_block(input, resblock_output, output_chn, name+"_basic_res_block_"+str(i))
    return resblock_output


def basic_res_block(input, resblock_output, output_chn, name):
    with tf.name_scope(name):
        x = Conv2D(output_chn, kernel_size=3, strides=(1, 1), padding='same', name=name+"_conv2d_1")(resblock_output)
        
     
        x = BatchNormalization(name=name+"_batch_normalization_1")(x)

        x = Activation('relu', name=name+"_relu_1")(x)
        x = Conv2D(output_chn, kernel_size=3, strides=(1, 1), padding='same', name=name+"_conv2d_2")(x)

        x = BatchNormalization(name=name+"_batch_normalization_2")(x)

        residual = Activation('relu', name=name+"_relu_2")(x)
        modified_input = Conv2D(output_chn, kernel_size=3, strides=(1, 1), padding='same', name=name+"_conv2d_3")(input)
        output = Lambda(Add(name=name+"_add"))([modified_input, residual])
    return output


def compute_SSIM_loss(x, y):
    def cal_std(x):
        mu_x = AveragePooling2D((3, 3), 1, 'valid')(x)
        pow_mu_x = mu_x**2
        pow_x = x**2
        pooled_pow_x = AveragePooling2D((3, 3), strides=1, padding='valid')(pow_x)
        sigma_x = pooled_pow_x - pow_mu_x
        return sigma_x, mu_x

    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
    y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    sigma_x, mu_x = cal_std(x)
    sigma_y, mu_y = cal_std(y)
    xy = x*y
    mu_xy = mu_x*mu_y
    sigma_xy = AveragePooling2D((3, 3), 1, 'valid')(xy)
    sigma_xy = sigma_xy - mu_xy
    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x**2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n/SSIM_d
    Last_SSIM = (1-SSIM)/2
    return tf.clip_by_value(Last_SSIM, 0, 1)


def compute_smooth_loss(pred_disp):
    pred_disp = pred_disp / (tf.reduce_mean(pred_disp, axis=[1, 2], keepdims=True) + 1e-7)

    def gradient(pred):
        D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        return D_dx, D_dy

    dx, dy = gradient(pred_disp)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)

    return tf.reduce_mean(tf.abs(dx2)) + tf.reduce_mean(tf.abs(dxdy)) + tf.reduce_mean(tf.abs(dydx)) + tf.reduce_mean(tf.abs(dy2))


def compute_edge_aware_smooth_loss(disp, img):
    disp_mean=tf.reduce_mean(tf.reduce_mean(disp, axis=1, keepdims=True),axis=2,keepdims=True)
    disp = disp / (disp_mean + 1e-7)

    grad_disp_x = tf.abs(disp[:, :, :-1] - disp[:, :, 1:])
    grad_disp_y = tf.abs(disp[:, :-1, :] - disp[:, 1:, :])

    grad_img_x = tf.reduce_mean(tf.abs(img[:, :, :-1] - img[:, :, 1:]), axis=3, keepdims=True)
    grad_img_y = tf.reduce_mean(tf.abs(img[:, :-1, :] - img[:, 1:, :]), axis=3, keepdims=True)

    grad_disp_x *= tf.exp(-grad_img_x)
    grad_disp_y *= tf.exp(-grad_img_y)

    return tf.reduce_mean(grad_disp_x) + tf.reduce_mean(grad_disp_y)


def convert_disp_to_depth(pred_disp, img_height=None, img_width=None, max_depth=100, min_depth=0.1):   
    if img_height is not None and pred_disp.shape[1]!=img_height:
        pred_disp = tf.image.resize_bilinear(pred_disp, (img_height, img_width),align_corners=True)

    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * pred_disp
    depth = 1 / scaled_disp
    return depth
