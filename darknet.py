from keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, add,  \
                         GlobalAveragePooling2D, Dense, Reshape, Multiply
from keras.models import Model


def darknet(input_tensor=None, input_shape=(416,416,3), n_classes=1000, n_filters=32, se=False,
            include_top=False, multi_out=False):
    if input_tensor is None:
        inpt = Input(input_shape)
    else:
        inpt = input_tensor

    # stem
    x = Conv_BN(inpt, n_filters, 3, strides=1, activation='leaky')
    x = Conv_BN(x, n_filters*2, 3, strides=2, activation='leaky')

    # back
    feats = []
    n_blocks = [1,2,8,8,4]
    n_filters = [n_filters*(2**i) for i in range(1, 6)]
    for level in range(len(n_blocks)):
        # darknet res block
        for i in range(n_blocks[level]):
            x = dark_res_block(x, n_filters[level], se)
        feats.append(x)
        # downsamp
        if level != len(n_blocks)-1:
            x = Conv_BN(x, n_filters[level]*2, 3, strides=2, activation='leaky')

    # head
    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dense(n_classes, activation='softmax')(x)

    if multi_out:
        model = Model(inpt, feats[1:])
    else:
        model = Model(inpt, x)

    return model


def dark_res_block(x, n_filters, se=False):
    inpt = x
    # residual
    x = Conv_BN(x, n_filters//2, 1, strides=1, activation='leaky')
    x = Conv_BN(x, n_filters, 3, strides=1, activation='leaky')
    if se:
        x = se_block(x)

    # shortcut
    if inpt._keras_shape[-1]!=n_filters:
        inpt = Conv_BN(inpt, n_filters, 1, strides=1, activation=None)
    x = add([inpt, x])
    return x


def Conv_BN(x, n_filters, kernel_size, strides, activation=None):
    x = Conv2D(n_filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    if activation:
        x = LeakyReLU(alpha=0.1)(x)
    return x


def se_block(x, se_ratio=0.25):
    inpt = x
    in_channels = x._keras_shape[-1]
    dense_dim = int(in_channels*se_ratio)
    # squeeze
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1,1,in_channels))(x)
    # reduce
    x = Conv2D(dense_dim, 1, strides=1, padding='same', activation='relu')(x)
    # excite
    x = Conv2D(in_channels, 1, strides=1, padding='same', activation='sigmoid')(x)
    # reweight
    x = Multiply()([inpt, x])
    return x


if __name__ == '__main__':

    model = darknet(input_tensor=None, input_shape=(416,416,3), n_classes=1000, include_top=False)
    print(len(model.layers))
    model.summary()
