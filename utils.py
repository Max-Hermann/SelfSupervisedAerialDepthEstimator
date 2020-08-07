from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import math

#from pyquaternion import Quaternion
from math import *


def gray2rgb(im, cmap='gray'):
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgb_img


def normalize_depth_for_display(depth, pc=95, crop_percent=0, normalizer=None, cmap='plasma'):
    # convert to disparity
    depth = 1./(depth + 1e-6)
    if normalizer is not None:
        depth = depth/normalizer
    else:
        depth = depth/(np.percentile(depth, pc) + 1e-6)
    depth = np.clip(depth, 0, 1)
    depth = gray2rgb(depth, cmap=cmap)
    keep_H = int(depth.shape[0] * (1-crop_percent))
    depth = depth[:keep_H]
    depth = depth
    return depth


def depth_to_rgb_image_batch(pred_depth):
    rgb_depth_maps = []
    for i in range(pred_depth.shape[0]):
        slice = pred_depth[i:i+1][0, :, :, 0]
        #rgb_depth_map = tf.compat.v1.py_func(normalize_depth_for_display, [slice], tf.float64, stateful=False)
        rgb_depth_map = tf.py_func(normalize_depth_for_display, [slice], tf.float64, stateful=False)

        rgb_depth_map = tf.expand_dims(rgb_depth_map, axis=0)
        rgb_depth_maps.append(rgb_depth_map)
    rgb_depth_maps = tf.concat(rgb_depth_maps, axis=0)
    return rgb_depth_maps


def euler2mat(z, y, x):
    """Converts euler angles to rotation matrix
     TODO: remove the dimension for 'N' (deprecated for converting all source
           poses altogether)
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        z: rotation angle along z axis (in radians) -- size = [B, N]
        y: rotation angle along y axis (in radians) -- size = [B, N]
        x: rotation angle along x axis (in radians) -- size = [B, N]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
    """
    B = tf.shape(z)[0]
    N = 1
    z = tf.clip_by_value(z, -np.pi, np.pi)
    y = tf.clip_by_value(y, -np.pi, np.pi)
    x = tf.clip_by_value(x, -np.pi, np.pi)

    # Expand to B x N x 1 x 1
    z = tf.expand_dims(tf.expand_dims(z, -1), -1)
    y = tf.expand_dims(tf.expand_dims(y, -1), -1)
    x = tf.expand_dims(tf.expand_dims(x, -1), -1)

    zeros = tf.zeros([B, N, 1, 1])
    ones = tf.ones([B, N, 1, 1])

    cosz = tf.cos(z)
    sinz = tf.sin(z)
    rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
    rotz_2 = tf.concat([sinz,  cosz, zeros], axis=3)
    rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
    zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

    cosy = tf.cos(y)
    siny = tf.sin(y)
    roty_1 = tf.concat([cosy, zeros, siny], axis=3)
    roty_2 = tf.concat([zeros, ones, zeros], axis=3)
    roty_3 = tf.concat([-siny, zeros, cosy], axis=3)
    ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

    cosx = tf.cos(x)
    sinx = tf.sin(x)
    rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
    rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
    rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
    xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)

    rotMat = tf.matmul(tf.matmul(xmat, ymat), zmat)
    return rotMat


def pose_vec2mat(vec, rot_mat=None):
    """Converts 6DoF parameters to transformation matrix
    Args:
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 4, 4]
    """
    batch_size, _ = vec.get_shape().as_list()
    translation = tf.slice(vec, [0, 0], [-1, 3])
    translation = tf.expand_dims(translation, -1)

    if rot_mat is None:
        rx = tf.slice(vec, [0, 3], [-1, 1])
        ry = tf.slice(vec, [0, 4], [-1, 1])
        rz = tf.slice(vec, [0, 5], [-1, 1])
        rot_mat = euler2mat(rz, ry, rx)
        rot_mat = tf.squeeze(rot_mat, axis=[1])

    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [batch_size, 1, 1])
    rot_mat = tf.cast(rot_mat, tf.float32)
    transform_mat = tf.concat([rot_mat, translation], axis=2)
    transform_mat = tf.concat([transform_mat, filler], axis=1)
    return transform_mat, rot_mat, translation


def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
    """Transforms coordinates in the pixel frame to the camera frame.

    Args:
      depth: [batch, height, width]
      pixel_coords: homogeneous pixel coordinates [batch, 3, height, width]
      intrinsics: camera intrinsics [batch, 3, 3]
      is_homogeneous: return in homogeneous coordinates
    Returns:
      Coords in the camera frame [batch, 3 (4 if homogeneous), height, width]
    """
    batch, height, width = depth.get_shape().as_list()
    depth = tf.reshape(depth, [batch, 1, -1])
    pixel_coords = tf.reshape(pixel_coords, [batch, 3, -1])
    cam_coords = tf.matmul(tf.linalg.inv(intrinsics), pixel_coords) * depth
    if is_homogeneous:
        ones = tf.ones([batch, 1, height*width])
        cam_coords = tf.concat([cam_coords, ones], axis=1)
    cam_coords = tf.reshape(cam_coords, [batch, -1, height, width])
    return cam_coords


def cam2pixel(cam_coords, proj):
    """Transforms coordinates in a camera frame to the pixel frame.

    Args:
      cam_coords: [batch, 4, height, width]
      proj: [batch, 4, 4]
    Returns:
      Pixel coordinates projected from the camera frame [batch, height, width, 2]
    """
    batch, _, height, width = cam_coords.get_shape().as_list()
    cam_coords = tf.reshape(cam_coords, [batch, 4, -1])
    unnormalized_pixel_coords = tf.matmul(proj, cam_coords)
    x_u = tf.slice(unnormalized_pixel_coords, [0, 0, 0], [-1, 1, -1])
    y_u = tf.slice(unnormalized_pixel_coords, [0, 1, 0], [-1, 1, -1])
    z_u = tf.slice(unnormalized_pixel_coords, [0, 2, 0], [-1, 1, -1])
    x_n = x_u / (z_u + 1e-10)
    y_n = y_u / (z_u + 1e-10)
    pixel_coords = tf.concat([x_n, y_n], axis=1)
    pixel_coords = tf.reshape(pixel_coords, [batch, 2, height, width])
    return tf.transpose(pixel_coords, perm=[0, 2, 3, 1])


def meshgrid(batch, height, width, is_homogeneous=True):
    """Construct a 2D meshgrid.

    Args:
      batch: batch size
      height: height of the grid
      width: width of the grid
      is_homogeneous: whether to return in homogeneous coordinates
    Returns:
      x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
    """
    x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                    tf.transpose(tf.expand_dims(
                        tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
    y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                    tf.ones(shape=tf.stack([1, width])))
    x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
    y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)
    if is_homogeneous:
        ones = tf.ones_like(x_t)
        coords = tf.stack([x_t, y_t, ones], axis=0)
    else:
        coords = tf.stack([x_t, y_t], axis=0)
    coords = tf.tile(tf.expand_dims(coords, 0), [batch, 1, 1, 1])
    return coords


def projective_inverse_warp(img, depth, pose, intrinsics, rot_mat=None):
    """Inverse warp a source image to the target image plane based on projection.

    Args:
      img: the source image [batch, height_s, width_s, 3]
      depth: depth map of the target image [batch, height_t, width_t]
      pose: target to source camera transformation matrix [batch, 6], in the
            order of tx, ty, tz, rx, ry, rz
      intrinsics: camera intrinsics [batch, 3, 3]
    Returns:
      Source image inverse warped to the target image plane [batch, height_t,
      width_t, 3]
    """
    batch, height, width, _ = img.get_shape().as_list()
    # Convert pose vector to matrix
    pose, rot_mat, translation = pose_vec2mat(pose, rot_mat)
    # Construct pixel grid coordinates
    pixel_coords = meshgrid(batch, height, width)
    # Convert pixel coordinates to the camera frame
    cam_coords = pixel2cam(depth, pixel_coords, intrinsics)
    # Construct a 4x4 intrinsic matrix (TODO: can it be 3x4?)
    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [batch, 1, 1])
    intrinsics = tf.concat([intrinsics, tf.zeros([batch, 3, 1])], axis=2)
    intrinsics = tf.concat([intrinsics, filler], axis=1)
    # Get a 4x4 transformation matrix from 'target' camera frame to 'source'
    # pixel frame.
    proj_tgt_cam_to_src_pixel = tf.matmul(intrinsics, pose)
    src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)
    output_img = bilinear_sampler(img, src_pixel_coords)
    return output_img, rot_mat


def bilinear_sampler(imgs, coords):
    """Construct a new image by bilinear sampling from the input image.

    Points falling outside the source image boundary have value 0.

    Args:
      imgs: source image to be sampled from [batch, height_s, width_s, channels]
      coords: coordinates of source pixels to sample from [batch, height_t,
        width_t, 2]. height_t/width_t correspond to the dimensions of the output
        image (don't need to be the same as height_s/width_s). The two channels
        correspond to x and y coordinates respectively.
    Returns:
      A new sampled image [batch, height_t, width_t, channels]
    """
    def _repeat(x, n_repeats):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([
                n_repeats,
            ])), 1), [1, 0])
        rep = tf.cast(rep, 'float32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    with tf.name_scope('image_sampling'):
        coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
        inp_size = imgs.get_shape()
        coord_size = coords.get_shape()
        out_size = coords.get_shape().as_list()
        out_size[3] = imgs.get_shape().as_list()[3]

        coords_x = tf.cast(coords_x, 'float32')
        coords_y = tf.cast(coords_y, 'float32')

        x0 = tf.floor(coords_x)
        x1 = x0 + 1
        y0 = tf.floor(coords_y)
        y1 = y0 + 1

        y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
        x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
        zero = tf.zeros([1], dtype='float32')

        x0_safe = tf.clip_by_value(x0, zero[0], x_max)
        y0_safe = tf.clip_by_value(y0, zero[0], y_max)
        x1_safe = tf.clip_by_value(x1, zero[0], x_max)
        y1_safe = tf.clip_by_value(y1, zero[0], y_max)

        # bilinear interp weights, with points outside the grid having weight 0
        # wt_x0 = (x1 - coords_x) * tf.cast(tf.equal(x0, x0_safe), 'float32')
        # wt_x1 = (coords_x - x0) * tf.cast(tf.equal(x1, x1_safe), 'float32')
        # wt_y0 = (y1 - coords_y) * tf.cast(tf.equal(y0, y0_safe), 'float32')
        # wt_y1 = (coords_y - y0) * tf.cast(tf.equal(y1, y1_safe), 'float32')

        wt_x0 = x1_safe - coords_x
        wt_x1 = coords_x - x0_safe
        wt_y0 = y1_safe - coords_y
        wt_y1 = coords_y - y0_safe

        # indices in the flat image to sample from
        dim2 = tf.cast(inp_size[2], 'float32')
        dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
        base = tf.reshape(
            _repeat(
                tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
                coord_size[1] * coord_size[2]),
            [out_size[0], out_size[1], out_size[2], 1])

        base_y0 = base + y0_safe * dim2
        base_y1 = base + y1_safe * dim2
        idx00 = tf.reshape(x0_safe + base_y0, [-1])
        idx01 = x0_safe + base_y1
        idx10 = x1_safe + base_y0
        idx11 = x1_safe + base_y1

        # sample from imgs
        imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
        imgs_flat = tf.cast(imgs_flat, 'float32')
        im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')), out_size)
        im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')), out_size)
        im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')), out_size)
        im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')), out_size)

        w00 = wt_x0 * wt_y0
        w01 = wt_x0 * wt_y1
        w10 = wt_x1 * wt_y0
        w11 = wt_x1 * wt_y1

        output = tf.add_n([
            w00 * im00, w01 * im01,
            w10 * im10, w11 * im11
        ])
        return output


def rotation_matrix_from_images_batch(img_0, img_1, camera_Matrix):
    rot_mats = []
    for i in range(img_0.shape[0]):
        img_0_slice = tf.squeeze(img_0[i:i+1])
        img_1_slice = tf.squeeze(img_1[i:i+1])
        camera_Matrix_slice = tf.squeeze(camera_Matrix[i:i+1])
        rot_mat, t = tf.py_func(rotation_matrix_from_images, [img_0_slice, img_1_slice, camera_Matrix_slice], tf.float64, stateful=False)
        rot_mat = tf.expand_dims(rot_mat, axis=0)
        rot_mats.append(rot_mat)
    rot_mats = tf.concat(rot_mats, axis=0)
    return rot_mats


def rotation_matrix_from_images(img_0, img_1, camera_Matrix):
    orb = cv2.ORB_create(100000)

    kp_0, des_0 = orb.detectAndCompute(img_0, None)
    kp_1, des_1 = orb.detectAndCompute(img_1, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    if des_0 is None or des_1 is None:
        return [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]

    matches = bf.match(des_0, des_1)
    matches = sorted(matches, key=lambda x: x.distance)

    kp_0 = [kp_0[m.queryIdx].pt for m in matches]
    kp_1 = [kp_1[m.trainIdx].pt for m in matches]

    kp_0 = np.asarray(kp_0)
    kp_1 = np.asarray(kp_1)

    if len(kp_0) <= 5:
        #print("not enough feature points")
        return [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]], [0., 0., 0.]
    essential_Mat01, mask = cv2.findEssentialMat(kp_0, kp_1, camera_Matrix, method=4)
    retval, R, t, mask = cv2.recoverPose(essential_Mat01, kp_0, kp_1, camera_Matrix, mask=mask)
    return R, t


def calculate_error_of_rot_matrices_batch(m1, m2):
    rot_mats_diffs = 0
    for i in range(m2.shape[0]):
        rot_mat_slice = tf.squeeze(m1[i:i+1])
        rot_mat_pred_slice = tf.squeeze(m2[i:i+1])
        rot_mats_diffs += tf.py_func(calculate_error_of_rot_matrices, [rot_mat_slice, rot_mat_pred_slice, m2.shape[0]], tf.float64, stateful=False)
    return rot_mats_diffs


def calculate_error_of_rot_matrices(m1, m2, batch_size):
    angles1 = euler_angles_from_rotation_matrix(m1)
    angles2 = euler_angles_from_rotation_matrix(m2)

    diff = list(map(abs, angles1-angles2))

    return sum(diff)/len(diff)/batch_size


def euler_angles_from_rotation_matrix(R):
    '''
    From a paper by Gregory G. Slabaugh (undated),
    "Computing Euler angles from a rotation matrix
    '''
    def isclose(x, y, rtol=1.e-5, atol=1.e-8):
        return abs(x-y) <= atol + rtol * abs(y)

    phi = 0.0
    if isclose(R[2, 0], -1.0):
        theta = math.pi/2.0
        psi = math.atan2(R[0, 1], R[0, 2])
    elif isclose(R[2, 0], 1.0):
        theta = -math.pi/2.0
        psi = math.atan2(-R[0, 1], -R[0, 2])
    else:
        theta = -math.asin(R[2, 0])
        cos_theta = math.cos(theta)
        psi = math.atan2(R[2, 1]/cos_theta, R[2, 2]/cos_theta)
        phi = math.atan2(R[1, 0]/cos_theta, R[0, 0]/cos_theta)

    return np.asarray([psi, theta, phi])


def custom_from_matrix():
    @classmethod
    def _from_matrix(cls, matrix):
        """Initialise from matrix representation

        Create a Quaternion by specifying the 3x3 rotation or 4x4 transformation matrix
        (as a numpy array) from which the quaternion's rotation should be created.

        """
        try:
            shape = matrix.shape
        except AttributeError:
            raise TypeError("Invalid matrix type: Input must be a 3x3 or 4x4 numpy array or matrix")

        if shape == (3, 3):
            R = matrix
        elif shape == (4, 4):
            R = matrix[:-1][:, :-1]  # Upper left 3x3 sub-matrix
        else:
            raise ValueError("Invalid matrix shape: Input must be a 3x3 or 4x4 numpy array or matrix")

        # Check matrix properties
        if not np.allclose(np.dot(R, R.conj().transpose()), np.eye(3), rtol=2.e-5, atol=2.e-8):
            #raise ValueError("Matrix must be orthogonal, i.e. its transpose should be its inverse")
            print("Warning: Rotation matrix must be orthogonal, i.e. its transpose should be its inverse")
        if not np.isclose(np.linalg.det(R), 1.0):
            raise ValueError("Matrix must be special orthogonal i.e. its determinant must be +1.0")

        def decomposition_method(matrix):
            """ Method supposedly able to deal with non-orthogonal matrices - NON-FUNCTIONAL!
            Based on this method: http://arc.aiaa.org/doi/abs/10.2514/2.4654
            """
            x, y, z = 0, 1, 2  # indices
            K = np.array([
                [R[x, x]-R[y, y]-R[z, z],  R[y, x]+R[x, y],           R[z, x]+R[x, z],           R[y, z]-R[z, y]],
                [R[y, x]+R[x, y],          R[y, y]-R[x, x]-R[z, z],   R[z, y]+R[y, z],           R[z, x]-R[x, z]],
                [R[z, x]+R[x, z],          R[z, y]+R[y, z],           R[z, z]-R[x, x]-R[y, y],   R[x, y]-R[y, x]],
                [R[y, z]-R[z, y],          R[z, x]-R[x, z],           R[x, y]-R[y, x],           R[x, x]+R[y, y]+R[z, z]]
            ])
            K = K / 3.0

            e_vals, e_vecs = np.linalg.eig(K)
            print('Eigenvalues:', e_vals)
            print('Eigenvectors:', e_vecs)
            max_index = np.argmax(e_vals)
            principal_component = e_vecs[max_index]
            return principal_component

        def trace_method(matrix):
            """
            This code uses a modification of the algorithm described in:
            https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
            which is itself based on the method described here:
            http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/

            Altered to work with the column vector convention instead of row vectors
            """
            m = matrix.conj().transpose()  # This method assumes row-vector and postmultiplication of that vector
            if m[2, 2] < 0:
                if m[0, 0] > m[1, 1]:
                    t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
                    q = [m[1, 2]-m[2, 1],  t,  m[0, 1]+m[1, 0],  m[2, 0]+m[0, 2]]
                else:
                    t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
                    q = [m[2, 0]-m[0, 2],  m[0, 1]+m[1, 0],  t,  m[1, 2]+m[2, 1]]
            else:
                if m[0, 0] < -m[1, 1]:
                    t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
                    q = [m[0, 1]-m[1, 0],  m[2, 0]+m[0, 2],  m[1, 2]+m[2, 1],  t]
                else:
                    t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
                    q = [t,  m[1, 2]-m[2, 1],  m[2, 0]-m[0, 2],  m[0, 1]-m[1, 0]]

            q = np.array(q)
            q *= 0.5 / sqrt(t)
            return q

        return cls(array=trace_method(R))

    return _from_matrix


def euler2quat(z=0, y=0, x=0, isRadian=True):
    ''' Return quaternion corresponding to these Euler angles
    Uses the z, then y, then x convention above
    Parameters
    ----------
    z : scalar
         Rotation angle in radians around z-axis (performed first)
    y : scalar
         Rotation angle in radians around y-axis
    x : scalar
         Rotation angle in radians around x-axis (performed last)
    Returns
    -------
    quat : array shape (4,)
         Quaternion in w, x, y z (real, then vector) format
    Notes
    -----
    We can derive this formula in Sympy using:
    1. Formula giving quaternion corresponding to rotation of theta radians
         about arbitrary axis:
         http://mathworld.wolfram.com/EulerParameters.html
    2. Generated formulae from 1.) for quaternions corresponding to
         theta radians rotations about ``x, y, z`` axes
    3. Apply quaternion multiplication formula -
         http://en.wikipedia.org/wiki/Quaternions#Hamilton_product - to
         formulae from 2.) to give formula for combined rotations.
    '''

    if not isRadian:
        z = ((np.pi)/180.) * z
        y = ((np.pi)/180.) * y
        x = ((np.pi)/180.) * x
    z = z/2.0
    y = y/2.0
    x = x/2.0
    cz = math.cos(z)
    sz = math.sin(z)
    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)
    return np.array([
        cx*cy*cz - sx*sy*sz,
        cx*sy*sz + cy*cz*sx,
        cx*cz*sy - sx*cy*sz,
        cx*cy*sz + sx*cz*sy])
