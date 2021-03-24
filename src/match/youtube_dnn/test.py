import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras import backend as K

class SampledSoftmax(Layer):
    def __init__(self, **kwargs):
        super(SampledSoftmax, self).__init__(**kwargs)


    def call(self, inputs):
        """
        The first input should be the model as it were, and the second the
        target (i.e., a repeat of the training data) to compute the labels
        argument

        """
        # the labels input to this function is batch size by 1, where the
        # value at position (i, 1) is the index that is true (not zero)
        # e.g., (0, 0, 1) => (2) or (0, 1, 0, 0) => (1)
        print(inputs)
        print('weights', inputs[0]._keras_history[0].weights[0])
        print('biases', inputs[0]._keras_history[0].bias)
        print('inputs', inputs[0])
        print('labels', tf.reshape(tf.argmax(inputs[1], 1), [-1, 1]))
        return tf.nn.sampled_softmax_loss(weights=inputs[0]._keras_history[0].weights[0],
                                            biases=inputs[0]._keras_history[0].bias,
                                            inputs=inputs[0],
                                            labels=tf.reshape(tf.argmax(inputs[1], 1), [-1, 1]),
                                            num_sampled=1000,
                                            num_classes=200000)

def custom_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)


num_classes = 200000
input = Input(shape=(300,))
target_input = Input(shape=(num_classes,))

dense = Dense(num_classes)

outputs = dense(input)
print(outputs)
print(target_input)
outputs = SampledSoftmax()([outputs, target_input])

model = Model([input, target_input], outputs)
model.compile(optimizer=u'adam', loss=custom_loss)