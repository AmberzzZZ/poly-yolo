import keras.backend as K
import tensorflow as tf
import numpy as np


def mix_loss(args):
    y_true, y_pred = args
    # y_true: [N,H,W,2+1+cls]
    y_true = norm2offset(y_true, y_pred)

    # kp_loss: bce_loss
    kp_loss_ = kp_loss(y_true, y_pred)

    # conf_loss: focal_loss
    conf_loss_ = conf_loss(y_true, y_pred)

    # cls_loss: bce_loss
    cls_loss_ = cls_loss(y_true, y_pred)

    loss = kp_loss_ + conf_loss_ + cls_loss_      # (B,1)

    loss = tf.Print(loss, [K.mean(kp_loss_), K.mean(conf_loss_), K.mean(cls_loss_)], message='  loss:  ')

    # return loss
    return tf.stack([loss, kp_loss_, conf_loss_, cls_loss_])


def norm2offset(y_true, y_pred):
    # offset_value = norm_value * grid_shape - grid_coord
    xy_gt = y_true[...,:2]     # [B,H,W,2]
    conf_gt = y_true[...,2:3]    # [B,H,W,1]
    # grid_shape
    grid_shape = K.shape(y_pred)[1:3]    # h,w
    grid_shape = tf.cast(K.reshape(grid_shape, (1,1,1,2)), tf.float32)
    # grid_coords
    h, w = K.int_shape(y_pred)[1:3]
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
    grid_coords = np.stack([x, y], axis=-1).astype(np.float32)     # [h,w,2]
    grid_coords = K.reshape(grid_coords, (1,h,w,2))
    conf_gt = tf.tile(conf_gt, [1,1,1,2])
    offset_xy_gt = tf.where(conf_gt>0,
                            xy_gt * grid_shape[...,::-1] - grid_coords,
                            tf.zeros_like(xy_gt))
    conf_cls_gt = y_true[...,2:]
    offset_ytrue = K.concatenate([offset_xy_gt, conf_cls_gt], axis=-1)
    return offset_ytrue


# kp_loss: bce_loss
def kp_loss(y_true, y_pred):
    offset_xy_gt = y_true[...,:2]
    conf_gt = y_true[...,2:3]
    offset_xy_pred = y_pred[...,:2]
    kp_loss_ = conf_gt * K.binary_crossentropy(offset_xy_gt, offset_xy_pred, from_logits=True)
    return K.sum(kp_loss_, axis=[1,2,3])


# conf_loss: focal_loss
def conf_loss(y_true, y_pred):
    # gamma = 0.2
    # alpha = 0.75
    # conf_gt = y_true[...,2:3]
    # conf_pred = K.sigmoid(y_pred[...,2:3])
    # epsilon = K.epsilon()
    # pt = 1 - K.abs(conf_gt - conf_pred)
    # pt = K.clip(pt, epsilon, 1-epsilon)
    # alpha_mask = tf.where(conf_gt>0, tf.ones_like(conf_gt)*alpha, tf.ones_like(conf_gt)*(1-alpha))
    # focal_loss_ = -alpha_mask * K.pow(pt, gamma) * K.log(pt)
    # return K.sum(focal_loss_, axis=[1,2,3])
    conf_gt = y_true[...,2:3]
    conf_pred = y_pred[...,2:3]
    conf_loss_ = K.binary_crossentropy(conf_gt, conf_pred, from_logits=True)
    return K.sum(conf_loss_, axis=[1,2,3])


# cls_loss: bce_loss
def cls_loss(y_true, y_pred):
    conf_gt = y_true[...,2:3]
    cls_gt = y_true[...,3:]
    cls_pred = y_pred[...,3:]
    cls_loss_ = conf_gt * K.binary_crossentropy(cls_gt, cls_pred, from_logits=True)
    return K.sum(cls_loss_, axis=[1,2,3])










