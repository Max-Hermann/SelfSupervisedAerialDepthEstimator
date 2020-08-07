from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np
from AerialDepthEstimator import AerialDepthEstimator
from tensorflow.python.client import device_lib
import os


#flags = tf.compat.v1.flags
flags = tf.flags

flags.DEFINE_string("dataset_dir", "/media/max/Files/formatted_datasets/formatted_aerial_lr_data/", "Dataset directory")
flags.DEFINE_string("checkpoint_dir", "./checkpoints/", "Directory name to save the checkpoints")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate for adam")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
flags.DEFINE_float("smooth_weight", 0.001, "Weight for smoothness")
flags.DEFINE_float("ssim_weight", 1.0, "Weight for ssim")
flags.DEFINE_integer("max_epochs", 100, "Maximum number of training epochs")
flags.DEFINE_integer("epoch_to_reduce_learning_rate", 0, "Epoch after which the lerning rate gets divided by 10. Zero for never")
flags.DEFINE_integer("batch_size", 20, "Size of of a sample batch")
flags.DEFINE_integer("img_height", 96, "Image height")
flags.DEFINE_integer("img_width", 192, "Image width")
flags.DEFINE_integer("seq_length", 3, "Sequence length for each example")
flags.DEFINE_integer("summary_freq", 50, "Logging every log_freq iterations")
flags.DEFINE_integer("save_latest_freq", 10000000, "Save the latest model every save_latest_freq iterations (overwrites the previous latest model)")
flags.DEFINE_boolean("continue_train", False, "Continue training from previous checkpoint")
flags.DEFINE_integer("train_sample_size", 5000*1000000, "Train on subsample of images")
flags.DEFINE_integer("save_epoch", 100, "save every x epochs")

flags.DEFINE_integer("num_source", 2, "number ofsource iamges")
flags.DEFINE_integer("num_scales", 4, "number of scales")
flags.DEFINE_string("gpu_id", "0", "target gpu id")


flags.DEFINE_string("init_checkpoint_file", None, "Specific checkpoint file to initialize from")




FLAGS = flags.FLAGS


def main(_):
   
   
    seed = 8964
    tf.set_random_seed(seed)

    np.random.seed(seed)
    random.seed(seed)

   

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    print(FLAGS.checkpoint_dir)
    print(tf.__version__)

    print("Build with CUDA: ",tf.test.is_built_with_cuda())

    ade = AerialDepthEstimator()
    ade.train(FLAGS)


if __name__ == '__main__':
    tf.app.run()
