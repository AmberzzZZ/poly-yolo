from efficientNet import EfficientNet
from darknet import darknet, Conv_BN
from loss import mix_loss
from keras.layers import Input, UpSampling2D, add, Conv2D, Lambda
from keras.models import Model
from keras.optimizers import Adam


def poly_yolo(input_shape=(224,640,1), n_classes=1, backbone='darknet',
              lr=3e-4, decay=5e-6):

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

    # model
    model = Model(inpt, x)
    model.compile(Adam(lr=lr, decay=decay), loss=mix_loss, metrics=None)

    return model


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

    model = poly_yolo(input_shape=(128,640,1), n_classes=1, backbone='efficientNet')
    model.summary()




