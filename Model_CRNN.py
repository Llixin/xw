from keras.layers import *
from keras.models import Model
import keras.backend as K



class CenterLossLayer(Layer):

    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(19, 128),
                                       initializer='uniform',
                                       trainable=False)
        # self.counter = self.add_weight(name='counter',
        #                                shape=(1,),
        #                                initializer='zeros',
        #                                trainable=False)  # just for debugging
        super().build(input_shape)

    def call(self, x, mask=None):
        # x[0] is Nx2, x[1] is Nx10 onehot, self.centers is 10x2
        delta_centers = K.dot(K.transpose(x[1]), (K.dot(x[1], self.centers) - x[0]))  # 10x2
        center_counts = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1  # 10x1
        delta_centers /= center_counts
        new_centers = self.centers - self.alpha * delta_centers
        self.add_update((self.centers, new_centers), x)

        # self.add_update((self.counter, self.counter + 1), x)

        self.result = x[0] - K.dot(x[1], self.centers)
        self.result = K.sum(self.result ** 2, axis=1, keepdims=True)  # / K.dot(x[1], center_counts)
        return self.result  # Nx1

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

def CRNN():
    input1 = Input(shape=(60, 8, 1))
    input2 = Input(shape=(19,))

    #CNN part
    X = Conv2D(filters=64,
               kernel_size=(3, 3),
               padding='same',
               )(input1)
    X = LeakyReLU(0.01)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.3)(X)

    X = Conv2D(filters=64,
               kernel_size=(3, 3),
               padding='same',
               )(X)
    X = LeakyReLU(0.01)(X)
    X = MaxPooling2D((2, 2))(X)   # (bs, 30, 4, 64)
    X = BatchNormalization()(X)
    X = Dropout(0.3)(X)

    X = Conv2D(filters=128,
               kernel_size=(3, 3),
               padding='same',
               )(X)
    X = LeakyReLU(0.01)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.3)(X)

    X = Conv2D(filters=128,
               kernel_size=(3, 3),
               padding='same',
               )(X)
    X = LeakyReLU(0.01)(X)
    X = MaxPooling2D((2, 2))(X)   # (bs, 15, 2, 128)
    X = BatchNormalization()(X)
    X = Dropout(0.3)(X)

    X = Conv2D(filters=256,
               kernel_size=(3, 3),
               padding='same',
               )(X)
    X = LeakyReLU(0.01)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.3)(X)
    X = Conv2D(filters=256,
               kernel_size=(3, 3),
               padding='same',
               )(X)
    X = LeakyReLU(0.01)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.3)(X)
    X = MaxPooling2D((1, 2))(X)   # (bs, 15, 1, 256)
    X = Conv2D(filters=512,
               kernel_size=(3, 3),
               padding='same',
               )(X)
    X = LeakyReLU(0.01)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.3)(X)        # (bs, 15, 1, 512)
    X = Reshape((-1, 512))(X)  # (bs, 15, 512)

    # RNN part
    X = GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(X)
    X = GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(X)


    X = GlobalAveragePooling1D()(X) # (bs, 128)
    X = Dropout(0.5)(X)
    output = Dense(19, activation='softmax', name='behaviour',
                   )(X)

    side = CenterLossLayer(alpha=0.5, name='centerlosslayer')([X, input2])  # Centerloss layer
    return Model([input1, input2], [output, side])