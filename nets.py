from __future__ import division
import tensorflow as tf
import sys

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Add, Multiply, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential

from layer import *


import numpy as np

global_shared_encoder_model = None
global_shared_dispnet_model = None
global_shared_pose_decoder_model = None




def build_res_net_encoder18(inputs, name):
    
    print("build_res_net_encoder18")
    input_layer = Input(inputs.shape[1:])

    x = Conv2D(32, kernel_size=7, strides=(2, 2), padding='same', name=name+"_conv2d_1")(input_layer)  
    x = BatchNormalization(name=name+"_batch_normalization_1")(x)
    econv1 = Activation('relu', name=name+"_relu_1")(x)

    econv2 = res_block(econv1, 64, "res_0", first_flag=True)
    econv3 = res_block(econv2, 128, "res_1")
    econv4 = res_block(econv3, 256, "res_2")
    econv5 = res_block(econv4, 512, "res_3")

    return Model(inputs=input_layer, outputs=[econv1, econv2, econv3, econv4, econv5])


def shared_encoder(inputs):
    name = "encoder"
    with tf.name_scope(name):
        global global_shared_encoder_model
        if global_shared_encoder_model is not None:
            return global_shared_encoder_model
        print("build_res_net_encoder18")
        model = build_res_net_encoder18(inputs, name)

        global_shared_encoder_model = model
    return model

def build_enet_disp_net(input_image):
    print("inputs",input_image.shape[1:])
    outputs,encoder = get_efficient_unet_b0(input_image.shape[1:], pretrained=True, block_type='upsampling', concat_input=False,input_image=input_image)
    global global_shared_encoder_model
    global_shared_encoder_model=encoder
    return outputs


def build_pose_net_decoder(inputs):
    global global_shared_pose_decoder_model
    if global_shared_pose_decoder_model is not None:
            return global_shared_pose_decoder_model
    
    input_layer = Input(inputs.shape[1:])

    name = "pose_prediction"
    with tf.name_scope(name):
        # DECODING
        with tf.name_scope('pose'):
            cnv6 = Conv2D(256, kernel_size=3, strides=2, padding='same', activation='relu', name=name+"_conv2d_1")(input_layer)
            cnv7 = Conv2D(256, kernel_size=3, strides=2, padding='same', activation='relu', name=name+"_conv2d_2")(cnv6)
            pose_pred = Conv2D(6, kernel_size=1, strides=1, padding='same', name=name+"_conv2d_3")(cnv7)      
            pose_avg = Lambda(lambda x: tf.reduce_mean(x, [1, 2]))(pose_pred)
            pose_reshape = Lambda(lambda x: tf.reshape(x, [-1,1, 6]))(pose_avg)

        global_shared_pose_decoder_model=Model(input_layer,pose_reshape)
        return global_shared_pose_decoder_model

def pose_net_new(image_1, image_2):
    # ENCODING
    _,_,_,_,deph_feature_map_image_1 = shared_encoder(image_1)(image_1)
    _,_,_,_,deph_feature_map_image_2 = shared_encoder(image_2)(image_2)
    inputs = Concatenate(axis=3)([deph_feature_map_image_1,deph_feature_map_image_2])
    return build_pose_net_decoder(inputs)(inputs)

def pose_net(tgt_image, src_image_stack, num_source):
    # ENCODING
    _,_,_,_,deph_feature_map_tgt_image = shared_encoder(tgt_image)(tgt_image)

    deph_feature_src_maps=[]
    for i in range(num_source):    
        src_image = src_image_stack[:, :, :, i*3:(i+1)*3]
        _,_,_,_,deph_feature_map_src_image = shared_encoder(src_image)(src_image)
        deph_feature_src_maps.append(deph_feature_map_src_image)

    name = "pose_prediction"
    with tf.name_scope(name):
        # DECODING
        deph_feature_maps=deph_feature_src_maps[:int(num_source/2)]
        deph_feature_maps.append(deph_feature_map_tgt_image)
        deph_feature_maps.extend(deph_feature_src_maps[int(num_source/2):])

        inputs = Concatenate(axis=3)(deph_feature_maps)
        with tf.name_scope('pose'):
            cnv6 = Conv2D(256, kernel_size=3, strides=2, padding='same', activation='relu', name=name+"_conv2d_1")(inputs)
            cnv7 = Conv2D(256, kernel_size=3, strides=2, padding='same', activation='relu', name=name+"_conv2d_2")(cnv6)
            pose_pred = Conv2D(6*num_source, kernel_size=1, strides=1, padding='same', name=name+"_conv2d_3")(cnv7)
            if data_format is "channels_first":
                pose_avg = Lambda(lambda x: tf.reduce_mean(x, [2, 3]))(pose_pred)
            else:
                pose_avg = Lambda(lambda x: tf.reduce_mean(x, [1, 2]))(pose_pred)
            pose_reshape = Lambda(lambda x: tf.reshape(x, [-1, num_source, 6]))(pose_avg)

        return pose_reshape

def pose_net_old(src_image1, tgt_image, src_image2):
    # ENCODING
    _, _, _, _, deph_feature_map_src_image1 = shared_encoder(src_image1)(src_image1)
    _, _, _, _, deph_feature_map_tgt_image = shared_encoder(tgt_image)(tgt_image)
    _, _, _, _, deph_feature_map_src_image2 = shared_encoder(src_image2)(src_image2)
    name = "pose_prediction"
    with tf.name_scope(name):
        # DECODING
        num_source = 2
        inputs = Concatenate(axis=3)([deph_feature_map_src_image1, deph_feature_map_tgt_image, deph_feature_map_src_image2])
        with tf.name_scope('pose'):
            cnv6 = Conv2D(256, kernel_size=3, strides=2, padding='same', activation='relu', name=name+"_conv2d_1")(inputs)
            cnv7 = Conv2D(256, kernel_size=3, strides=2, padding='same', activation='relu', name=name+"_conv2d_2")(cnv6)
            pose_pred = Conv2D(6*num_source, kernel_size=1, strides=1, padding='same', name=name+"_conv2d_3")(cnv7)
            pose_avg = Lambda(lambda x: tf.reduce_mean(x, [1, 2]))(pose_pred)
            pose_reshape = Lambda(lambda x: tf.reshape(x, [-1, num_source, 6]))(pose_avg)

        return pose_reshape

def disp_net(input_image):
    # ENCODING
    skip1, skip2, skip3, skip4, econv5 = shared_encoder(input_image)(input_image)
    name = "depth_prediction"
    with tf.name_scope("depth_prediction"):
        # DECODING
        with tf.name_scope('decoder'):
            filter = [256, 128, 64, 32, 16]       
            filter = [int(skip4.shape[-1]), int(skip3.shape[-1]), int(skip2.shape[-1]), int(skip1.shape[-1]), 16]

            upconv5 = up_block(econv5, filter[0], name+"_upconv5", skip4)

            upconv4 = up_block(upconv5, filter[1], name+"_upconv4", skip3)
            disp4 = get_disparity_from_layer(upconv4, name+"_disp4")

            upconv3 = up_block(upconv4, filter[2], name+"_upconv3", skip2)
            disp3 = get_disparity_from_layer(upconv3, name+"_disp3")

            upconv2 = up_block(upconv3, filter[3], name+"_upconv2", skip1)
            disp2 = get_disparity_from_layer(upconv2, name+"_disp2")

            upconv1 = up_block(upconv2, filter[4], name+"_upconv1")
            disp1 = get_disparity_from_layer(upconv1, name+"_disp1")
            print("disp1",disp1.shape)


        return [disp1, disp2, disp3, disp4]
