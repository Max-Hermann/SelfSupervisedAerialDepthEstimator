from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf

from data_loader import DataLoader
from nets import *
from utils import *
from layer import *


class AerialDepthEstimator(object):
    def __init__(self):
        pass

    def train(self, opt):
        self.opt = opt

        with tf.Graph().as_default() as g:
            self.build_train_graph()
            self.build_summaries()

            config = tf.ConfigProto()

            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True

            with tf.name_scope("parameter_count"):
                parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

            self.saver = tf.train.Saver(max_to_keep=100)

            sv = tf.train.Supervisor(logdir=opt.checkpoint_dir, save_summaries_secs=0, saver=None)


        with sv.managed_session(config=config) as sess:
            print("parameter_count =", sess.run(parameter_count))

            if opt.continue_train:
                if opt.init_checkpoint_file is None:
                    checkpoint = tf.train.latest_checkpoint(opt.checkpoint_dir)
                else:
                    checkpoint = opt.init_checkpoint_file
                print("Resume training from previous checkpoint: %s" % checkpoint)
                self.saver.restore(sess, checkpoint)

            start_time = time.time()
            for step in range(1, opt.max_epochs*self.steps_per_epoch):
                fetches = {
                    "train": self.train_op,
                    "global_step": self.global_step,
                    "incr_global_step": self.incr_global_step
                }
                if step % opt.summary_freq == 0:
                    fetches["loss"] = self.total_loss
                    fetches["mean_inverse_depth"] = self.mean_inverse_depth
                    fetches["smoth_loss"] = self.smooth_loss

                    fetches["summary"] = sv.summary_op

                results = sess.run(fetches)
                gs = results["global_step"]

                if step % opt.summary_freq == 0:
                    sv.summary_writer.add_summary(results["summary"], gs)
                    train_epoch = math.ceil(gs / self.steps_per_epoch)
                    train_step = gs - (train_epoch - 1) * self.steps_per_epoch
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f/it loss: %.3f smooth_loss: %.10f mean inverse depth: %.3f" % (train_epoch, train_step, self.steps_per_epoch,
                                                                                (time.time() - start_time)/opt.summary_freq, results["loss"],
                                                                                results["smoth_loss"],results["mean_inverse_depth"]))
                    start_time = time.time()

                if step % opt.save_latest_freq == 0:
                    print("save1")
                    self.save(sess, opt.checkpoint_dir, 'latest')
                    print("sucess")

                if step % (self.steps_per_epoch*opt.save_epoch) == 0:
                    print("save2")
                    self.save(sess, opt.checkpoint_dir, gs)
                    print("sucess")

            self.save(sess, opt.checkpoint_dir, gs)

    def build_train_graph(self):
        opt = self.opt
        loader = DataLoader(opt.dataset_dir,
                            opt.batch_size,
                            opt.img_height,
                            opt.img_width,
                            opt.num_source,
                            opt.num_scales)

        with tf.device('/cpu:0'):
            with tf.name_scope("data_loading"):
                images, intrinsics = loader.load_train_batch()
                self.steps_per_epoch = loader.steps_per_epoch
                for i in range(len(images)):
                    images[i]=self.preprocess_image( images[i])

                tgt_image = images[int(opt.num_source/2)]
                del images[int(opt.num_source/2)]

                src_image_stack = tf.concat(images, axis=3)
                self.tgt_image = tgt_image
                self.src_image_stack = src_image_stack
        with tf.device('/gpu:'+opt.gpu_id):
            tgt_image_transposed=tgt_image
            src_image_stack_transposed=src_image_stack 

            pred_disp = disp_net(tgt_image_transposed)
            pred_depth = [convert_disp_to_depth(i ,opt.img_height, opt.img_width) for i in pred_disp]

            pred_poses_01=pose_net_new(src_image_stack_transposed[:, :, :, 0:3],tgt_image_transposed)
            pred_poses_12=pose_net_new(src_image_stack_transposed[:, :, :, 3:6],tgt_image_transposed)
            pred_poses=tf.concat([pred_poses_01,pred_poses_12],axis=1)
 



            pred_poses = pred_poses*0.01
            with tf.name_scope("compute_loss"):
                pixel_loss, smooth_loss, mean_inverse_depth = 0, 0, 0
                proj_image_stack=[]
                proj_error_stack=[]
                for s in range(opt.num_scales):
                    mean_inverse_depth += tf.reduce_mean(1/pred_depth[s])

                    #current_downscaled_tgt_image = tf.compat.v1.image.resize_area(tgt_image, [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])
                    current_downscaled_tgt_image = tf.image.resize_area(tgt_image, [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])

                    smooth_loss += opt.smooth_weight / (2**s) * compute_edge_aware_smooth_loss(pred_disp[s], current_downscaled_tgt_image)

                    pixel_loss_stack = []
                    for i in range(opt.num_source):
                        # Inverse warp the source image to the target image frame
                        curr_proj_image, rot_mat_pred = projective_inverse_warp(
                            src_image_stack_transposed[:, :, :, 3*i:3*(i+1)],
                            tf.squeeze(pred_depth[s], axis=3),
                            pred_poses[:, i, :],
                            intrinsics[:, 0, :, :],
                            None)
                        proj_image_stack.append(curr_proj_image)
                        curr_proj_error_l1 = tf.abs(curr_proj_image - tgt_image)
                        curr_proj_error_l2 = tf.square(curr_proj_image - tgt_image)                     
                        curr_proj_error_SSIM = compute_SSIM_loss(curr_proj_image, tgt_image)
                        curr_proj_error_l1_SSIM=(1-opt.ssim_weight)*curr_proj_error_l1+opt.ssim_weight*curr_proj_error_SSIM                        
                        proj_error_stack.append(curr_proj_error_l1_SSIM)
                        pixel_loss_stack.append(curr_proj_error_l1_SSIM)                

                    proj_min_error=tf.reduce_min(pixel_loss_stack, axis=[0])
                    pixel_loss += tf.reduce_mean(proj_min_error)

                total_loss = pixel_loss + smooth_loss

            with tf.name_scope("train_op"):
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.incr_global_step = tf.assign(self.global_step, self.global_step+1)

                do_reduce_learning_rate = tf.greater(self.global_step, self.steps_per_epoch*opt.epoch_to_reduce_learning_rate)
                learning_rate = tf.cond(do_reduce_learning_rate, lambda: opt.learning_rate, lambda: opt.learning_rate/10)
                optim = tf.train.AdamOptimizer(learning_rate, opt.beta1)
                self.train_op = optim.minimize(total_loss)

            self.pred_depth = pred_depth
            self.pred_poses = pred_poses
            self.total_loss = total_loss
            self.smooth_loss = smooth_loss
            self.mean_inverse_depth = mean_inverse_depth
            self.proj_image_stack = proj_image_stack
            self.proj_error_stack = proj_error_stack
            self.proj_min_error= proj_min_error


    def preprocess_image(self, image):
        # Assuming input image is uint8
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image * 2. - 1.

    def deprocess_image(self, image):
        # Assuming input image is float32
        image = (image + 1.)/2.
        return tf.image.convert_image_dtype(image, dtype=tf.uint8)

    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint to %s..." % checkpoint_dir)
        if step == 'latest':
            self.saver.save(sess, os.path.join(checkpoint_dir, model_name + '.latest'))
        else:
            self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def build_summaries(self):
        tf.summary.scalar("smooth_loss", self.smooth_loss)
        tf.summary.scalar("mean_inverse_depth", self.mean_inverse_depth)
        tf.summary.scalar("total_loss", self.total_loss)
        tf.summary.image('scale0_disparity_image', 1./self.pred_depth[0])
        tf.summary.image('scale0_disparity_image_RGB', depth_to_rgb_image_batch(self.pred_depth[0]))
        tf.summary.image('scale0_target_image', self.deprocess_image(self.tgt_image))
        tf.summary.image('scale0_source_image_0', self.deprocess_image(self.src_image_stack[:, :, :, :3]))
        tf.summary.image('scale0_source_image_1', self.deprocess_image(self.src_image_stack[:, :, :, 3*(self.opt.num_source-1):]))
        tf.summary.image('scale0_projected_image_0', self.deprocess_image(self.proj_image_stack[0]))
        tf.summary.image('scale0_projected_image_1', self.deprocess_image(self.proj_image_stack[1]))
        tf.summary.image('scale0__proj_error_image_0',self.deprocess_image(tf.clip_by_value(self.proj_error_stack[0] - 1, -1, 1)))
        tf.summary.image('scale0__proj_error_image_1',self.deprocess_image(tf.clip_by_value(self.proj_error_stack[-1] - 1, -1, 1)))
        tf.summary.image('scale0_proj_min_error', self.deprocess_image(tf.clip_by_value(self.proj_min_error- 1, -1, 1)))


    def get_disp_model(self, input_shape):
        input_layer = Input(input_shape)

        input_uint8 = Lambda(self.preprocess_image)(input_layer)
        disp, _, _, _ = disp_net(input_uint8)

        return Model(inputs=input_layer, outputs=[disp])

    def get_disp_model_for_export(self, input_shape):
        input_layer = Input(input_shape,name="input_layer")
        disp, _, _, _ = disp_net(input_layer)
        return Model(inputs=input_layer, outputs=[disp])

    def get_pose_model(self):
        input_layer = Input(input_shape)

        src_image_1, tgt_image, src_image_2 = Lambda(lambda x: tf.unstack(x, axis=1))(input_layer)

        input_src_image_1_uint8 = Lambda(self.preprocess_image)(src_image_1)
        input_tgt_image_uint8 = Lambda(self.preprocess_image)(tgt_image)
        input_src_image_2_uint8 = Lambda(self.preprocess_image)(src_image_2)

        pred_poses = pose_net_old(input_src_image_1_uint8, input_tgt_image_uint8, input_src_image_2_uint8)

        return Model(inputs=[input_layer], outputs=[pred_poses])
