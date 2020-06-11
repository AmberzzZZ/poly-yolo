from efficientNet import EfficientNet
from darknet import darknet, Conv_BN
from loss import mix_loss
from keras.layers import Input, UpSampling2D, add, Conv2D, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.multi_gpu_utils import multi_gpu_model
import keras.backend as K
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

###### custom metrics ######
def kp_loss(y_true, y_pred):
    return K.mean(y_pred[:,1])

def conf_loss(y_true, y_pred):
    return K.mean(y_pred[:,2])

def cls_loss(y_true, y_pred):
    return K.mean(y_pred[:,3])

metric_lst = [kp_loss, conf_loss, cls_loss]



def poly_yolo(input_shape=(224,640,1), n_classes=1, backbone='darknet',
              lr=3e-4, decay=5e-6, multi_gpu=False, GPU_COUNT=2, weight_pt=''):

    inpt = Input(input_shape)

    # backbone
    if backbone == 'darknet':
        backmodel = darknet(input_shape=input_shape, n_classes=n_classes, include_top=False, multi_out=True)
    if backbone == 'darknet_se':
        backmodel = darknet(input_shape=input_shape, n_classes=n_classes, n_filters=24, se=True, include_top=False, multi_out=True)
    if backbone == 'efficientNet':
        backmodel = EfficientNet(input_shape=input_shape, n_classes=n_classes, include_top=False, multi_out=True)
    x = backmodel(inpt)

    # feature fusion
    x = feats_fusion(x, 384)

    # head: 1x1
    x = Conv2D(2+1+n_classes, 1, strides=1, padding='same')(x)      # x,y,conf,cls

    # yt: [b,h,w,2+1+cls]
    y_true = Input((input_shape[0]//4, input_shape[1]//4, 2+1+n_classes))

    # loss
    loss = Lambda(mix_loss)([y_true, x])

    # model
    model = Model([inpt, y_true], loss)

    if os.path.exists(weight_pt):
        print("load weight: ", weight_pt)
        model.load_weights(weight_pt, by_name=True, skip_mismatch=True)
    single_model = model
    if multi_gpu:
        model = multi_gpu_model(model, gpus=GPU_COUNT)

    model.compile(Adam(lr=lr, decay=decay),
                  loss=lambda y_true,y_pred: K.mean(y_pred[:,0]),
                  metrics=metric_lst)

    return model, single_model


def feats_fusion(feats, n_filters):
    n_levels = len(feats)
    # 1x1 conv
    feats = [Conv_BN(i, n_filters, kernel_size=1, strides=1, activation=None) for i in feats]
    for i in range(n_levels-1, 0, -1):
        # upsamp
        up = UpSampling2D()(feats[i])
        # add
        feats[i-1] = add([up, feats[i-1]])
    # 3x3 conv
    x = Conv_BN(feats[0], n_filters, 3, strides=1, activation='leaky')
    return x


if __name__ == '__main__':

    model, _ = poly_yolo(input_shape=(128,640,1), n_classes=10, backbone='efficientNet')
    model.summary()




