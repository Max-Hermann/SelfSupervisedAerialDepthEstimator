from __future__ import division
import numpy as np
from glob import glob
import os
import scipy.misc
import PIL.Image as pil

class aerial_raw_loader(object):
    def __init__(self,
                 dataset_dir,
                 img_height=256,
                 img_width=256,
                 seq_length=5,
                 frame_offset=1):

        self.dataset_dir = dataset_dir
        self.img_height = img_height
        self.img_width = img_width
        self.seq_length = seq_length
        self.frame_offset = frame_offset
        self.collect_train_frames()

    def collect_train_frames(self):
        all_frames = []
        dir_set = os.listdir(self.dataset_dir + '/')
        for dir_name in dir_set:
            if(os.path.isdir(os.path.join(self.dataset_dir + '/'+dir_name))):
                seq_set = os.listdir(self.dataset_dir + '/'+dir_name)
                for seq_name in seq_set:
                    if(os.path.isdir(os.path.join(self.dataset_dir + '/'+dir_name+"/"+seq_name))):
                        img_dir = os.path.join(self.dataset_dir + '/'+dir_name+"/"+seq_name)
                        N = len(glob(img_dir + '/*.png'))
                        for n in range(N-self.frame_offset*(self.seq_length-1)):                            
                            frame_id = '%.4d' % n
                            all_frames.append(dir_name + ' ' + seq_name + ' ' + frame_id)

        self.train_frames = all_frames
        self.num_train = len(self.train_frames)

    def is_valid_sample(self, frames, tgt_idx):
        N = len(frames)
        tgt_drive, cid, _ = frames[tgt_idx].split(' ')
        half_offset = int((self.seq_length - 1)/2)
        min_src_idx = tgt_idx - half_offset
        max_src_idx = tgt_idx + half_offset
        if min_src_idx < 0 or max_src_idx >= N:
            return False
        
        

        min_src_drive, min_src_cid, _ = frames[min_src_idx].split(' ')
        max_src_drive, max_src_cid, _ = frames[max_src_idx].split(' ')
        if tgt_drive == min_src_drive and tgt_drive == max_src_drive and cid == min_src_cid and cid == max_src_cid:
            return True
        return False

    def get_train_example_with_idx(self, tgt_idx):
        if not self.is_valid_sample(self.train_frames, tgt_idx):
            return False
        example = self.load_example(self.train_frames, tgt_idx)
        return example

    def load_image_sequence(self, frames, tgt_idx, seq_length):
        half_offset = int((seq_length - 1)/2)
        image_seq = []
        for o in range(-half_offset, half_offset + 1):
            curr_idx = tgt_idx + o
            dir_name, seq_name, curr_frame_id = frames[curr_idx].split(' ')
            curr_img = self.load_image_raw(dir_name, seq_name, '%.6d' % (int(curr_frame_id)+self.frame_offset*half_offset+o*self.frame_offset))
            if o == 0:
                zoom_x = self.img_width/curr_img.size[0]
                zoom_y = self.img_height/curr_img.size[1]

            curr_img = curr_img.resize((self.img_width, self.img_height), pil.ANTIALIAS)
            image_seq.append(curr_img)
        return image_seq, zoom_x, zoom_y

    def load_example(self, frames, tgt_idx):
        image_seq, zoom_x, zoom_y = self.load_image_sequence(frames, tgt_idx, self.seq_length)
        dir_name, seq_name, curr_frame_id = frames[tgt_idx].split(' ')
        intrinsics = self.load_intrinsics_raw(dir_name, seq_name, curr_frame_id)
        intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)
        example = {}
        example['intrinsics'] = intrinsics
        example['image_seq'] = image_seq
        example['folder_name'] = dir_name + '_'+seq_name
        example['file_name'] = curr_frame_id
        return example

    def load_image_raw(self, dir_name, seq_name, frame_id):
        img_file = os.path.join(self.dataset_dir, dir_name, seq_name, frame_id + '.png')
        fh = open(img_file, 'rb')
        img = pil.open(fh)
        img=np.array(img)
        if img.shape[2]>3:
            img=img[:,:,0:3] 
              
        img=pil.fromarray(img)        
        return img

    def load_intrinsics_raw(self, drive, cid, frame_id):
        date = drive#[:10]
        calib_file = os.path.join(self.dataset_dir, date, 'calib_cam_to_cam.txt')

        filedata = self.read_raw_calib_file(calib_file)
        P_rect = np.reshape(filedata['P_rect_00'], (3, 3))

        intrinsics = P_rect
        return intrinsics

    def read_raw_calib_file(self, filepath):
        # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """Read in a calibration file and parse into a dictionary."""
        data = {}

        with open(filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

    def scale_intrinsics(self, mat, sx, sy):
        out = np.copy(mat)
        out[0, 0] *= sx
        out[0, 2] *= sx
        out[1, 1] *= sy
        out[1, 2] *= sy
        return out
