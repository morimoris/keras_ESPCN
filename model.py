import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, Input, Lambda

def ESPCN(upsampling_scale):
    input_shape = Input((None, None, 1))

    conv2d_0 = Conv2D(filters = 64,
                        kernel_size = (5, 5),
                        padding = "same",
                        activation = "relu",
                        )(input_shape)
    conv2d_1 = Conv2D(filters = 32,
                        kernel_size = (3, 3),
                        padding = "same",
                        activation = "relu",
                        )(conv2d_0)
    conv2d_2 = Conv2D(filters = upsampling_scale ** 2,
                        kernel_size = (3, 3),
                        padding = "same",
                        )(conv2d_1)

    pixel_shuffle = Lambda(lambda z: tf.nn.depth_to_space(z, upsampling_scale))(conv2d_2)

    model = Model(inputs = input_shape, outputs = [pixel_shuffle])

    model.summary()

    return model