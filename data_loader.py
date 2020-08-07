from __future__ import division
import os
import random
import tensorflow as tf


class DataLoader(object):
    def __init__(self,
                 dataset_dir=None,
                 batch_size=None,
                 img_height=None,
                 img_width=None,
                 num_source=None,
                 num_scales=None):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.num_source = num_source
        self.num_scales = num_scales   

    def load_train_batch(self):
        """Load a batch of training instances.
        """
        seed = random.randint(0, 2**31 - 1)
        # Load the list of training files into queues
        file_list = self.format_file_list(self.dataset_dir, 'train')


        image_paths_queue = tf.train.string_input_producer(file_list['image_file_list'], seed=seed, shuffle=True)

        cam_paths_queue = tf.train.string_input_producer(file_list['cam_file_list'], seed=seed, shuffle=True)

        self.steps_per_epoch = int(len(file_list['image_file_list'])//self.batch_size)

        # Load images
        img_reader = tf.WholeFileReader()

        _, image_contents = img_reader.read(image_paths_queue)
        image_seq = tf.image.decode_jpeg(image_contents)
        images = self.unpack_image_sequence(image_seq, self.img_height, self.img_width, self.num_source)

        # Load camera intrinsics
        cam_reader = tf.TextLineReader()

        _, raw_cam_contents = cam_reader.read(cam_paths_queue)
        rec_def = []
        for i in range(9):
            rec_def.append([1.])
        raw_cam_vec = tf.decode_csv(raw_cam_contents, record_defaults=rec_def)

        raw_cam_vec = tf.stack(raw_cam_vec)
        intrinsics = tf.reshape(raw_cam_vec, [3, 3])

        images = tf.concat(images, axis=2)

        # Form training batches
        image_all, intrinsics = tf.train.batch([images, intrinsics], batch_size=self.batch_size)

        # Data augmentation
        image_all_aug, intrinsics = self.data_augmentation(image_all, intrinsics, self.img_height, self.img_width, self.num_source)

        images_aug = []
        for i in range(self.num_source+1):
            images_aug.append(image_all_aug[:, :, :, i*3:(i+1)*3])


        intrinsics = self.get_multi_scale_intrinsics(
            intrinsics, self.num_scales)

        return images_aug, intrinsics

    def make_intrinsics_matrix(self, fx, fy, cx, cy):
        # Assumes batch input
        batch_size = fx.get_shape().as_list()[0]
        zeros = tf.zeros_like(fx)
        r1 = tf.stack([fx, zeros, cx], axis=1)
        r2 = tf.stack([zeros, fy, cy], axis=1)
        r3 = tf.constant([0., 0., 1.], shape=[1, 3])
        r3 = tf.tile(r3, [batch_size, 1])
        intrinsics = tf.stack([r1, r2, r3], axis=1)
        return intrinsics

    def data_augmentation(self, im, intrinsics, out_h, out_w, num_source):
        # Random scaling
        def random_scaling(im, intrinsics):
            batch_size, in_h, in_w, _ = im.get_shape().as_list()
            #scaling = tf.random.uniform([2], 1, 1.15)
            scaling = tf.random_uniform([2], 1, 1.15)

            x_scaling = scaling[0]
            y_scaling = scaling[1]
            out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
            out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
            #im = tf.compat.v1.image.resize_area(im, [out_h, out_w])
            im = tf.image.resize_area(im, [out_h, out_w])
            fx = intrinsics[:, 0, 0] * x_scaling
            fy = intrinsics[:, 1, 1] * y_scaling
            cx = intrinsics[:, 0, 2] * x_scaling
            cy = intrinsics[:, 1, 2] * y_scaling
            intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)
            return im, intrinsics

        # Random cropping
        def random_cropping(im, intrinsics, out_h, out_w):
            batch_size, in_h, in_w, _ = tf.unstack(tf.shape(im))
            offset_y = tf.random_uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]

            offset_x = tf.random_uniform([1], 0, in_w - out_w + 1, dtype=tf.int32)[0]

            im = tf.image.crop_to_bounding_box(
                im, offset_y, offset_x, out_h, out_w)
            fx = intrinsics[:, 0, 0]
            fy = intrinsics[:, 1, 1]
            cx = intrinsics[:, 0, 2] - tf.cast(offset_x, dtype=tf.float32)
            cy = intrinsics[:, 1, 2] - tf.cast(offset_y, dtype=tf.float32)
            intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)
            return im, intrinsics

        def color_aug(im,num_source):
            images = []
            for i in range(num_source+1):
                images.append(im[:, :, :, i*3:(i+1)*3])

            im = images

            # randomly shift brightness
            im = tf.image.random_brightness(im, 0.2)

            # contrast
            im = tf.image.random_contrast(im, 0.8, 1.2)

            # saturation
            im = tf.image.random_saturation(im, 0.8, 1.2)

            # hue
            im = tf.image.random_hue(im, 0.1)

            # clip values to RGB range
            im = tf.unstack(im, axis=0)
            im = tf.concat(im, axis=3)
            im = tf.cast(im, dtype=tf.uint8)
            return im

        with tf.device('/cpu:0'):
            im, intrinsics = random_scaling(im, intrinsics)
            im, intrinsics = random_cropping(im, intrinsics, out_h, out_w)
            im = tf.cast(im, dtype=tf.uint8)

        # flip horizontal
        do_flip = tf.random_uniform([]) > 0.5

        im = tf.cond(do_flip, lambda: tf.image.flip_left_right(im), lambda: im)        

        do_color_aug = tf.random_uniform([]) > 0.5

        im = tf.cond(do_color_aug, lambda: color_aug(im,num_source), lambda: im)

        return im, intrinsics

    

    def format_file_list(self, data_root, split):
        with open(data_root + '/%s.txt' % split, 'r') as f:
            frames = f.readlines()
        subfolders = [x.split(' ')[0] for x in frames]
        frame_ids = [x.split(' ')[1][:-1] for x in frames]
        image_file_list = [os.path.join(data_root, subfolders[i],
                                        frame_ids[i] + '.jpg') for i in range(len(frames))]
        cam_file_list = [os.path.join(data_root, subfolders[i],
                                      frame_ids[i] + '_cam.txt') for i in range(len(frames))]
        all_list = {}
        all_list['image_file_list'] = image_file_list
        all_list['cam_file_list'] = cam_file_list
        return all_list

    def unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        
        images = []
        for i in range(num_source+1):
            image = tf.slice(image_seq, [0, img_width*i, 0], [-1, img_width, -1])
            image.set_shape([img_height, img_width, 3])
            images.append(image)

        return images

    

    def get_multi_scale_intrinsics(self, intrinsics, num_scales):
        intrinsics_mscale = []
        # Scale the intrinsics accordingly for each scale
        for s in range(num_scales):
            fx = intrinsics[:, 0, 0]/(2 ** s)
            fy = intrinsics[:, 1, 1]/(2 ** s)
            cx = intrinsics[:, 0, 2]/(2 ** s)
            cy = intrinsics[:, 1, 2]/(2 ** s)
            intrinsics_mscale.append(
                self.make_intrinsics_matrix(fx, fy, cx, cy))
        intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
        return intrinsics_mscale
