#####################################################################################################
# Model code where we create the encoder and decoder.
#####################################################################################################


# Loading Libraries
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential, Model


class Layer_learn_end_token(Layer):
    ''' Creating a custom layer to add the learned end token embedding to the encoder model'''

    def __init__(self, **kwargs):
        super(Layer_learn_end_token, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.end = tf.Variable(initial_value = tf.random.uniform(shape = (input_shape[-1],)), trainable = True)
    
    def call(self, inputs):
        end_token = tf.tile(tf.reshape(self.end, shape = (1,1, self.end.shape[0])), [tf.shape(inputs)[0],1,1])
        return tf.keras.layers.concatenate([inputs, end_token], axis = 1)


def Encoder():
    ''' Creating the encoder function that returns the encoder model by using the functionnal API of TensorFlow. '''

    inputs = tf.keras.layers.Input(shape = (13, 128)) # shape of each sample within the english dataset
    h = Layer_learn_end_token()(inputs)
    h = tf.keras.layers.Masking(mask_value = 0.0)(h)
    lstm_output, state_h, state_c = tf.keras.layers.LSTM(units = 512, return_state = True, return_sequences=True)(h)
    model = Model(inputs = inputs, outputs = [state_h, state_c])
    return model

class Decoder(Model):
    ''' Creating the Decoder model using the Model subclassing of TensorFlow '''

    def __init__(self, vocab_size, embedding_dim = 128, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.embedding_layer = tf.keras.layers.Embedding(input_dim = vocab_size + 1, output_dim = embedding_dim, mask_zero = True)
        self.LSTM = tf.keras.layers.LSTM(units = 512, return_sequences = True, return_state= True)
        self.dense = tf.keras.layers.Dense(units = vocab_size + 1)
    
    def call(self, inputs, hidden_state = None, cell_state = None):
        x = self.embedding_layer(inputs)
        x, h , c = self.LSTM(x, initial_state=(hidden_state, cell_state))
        x = self.dense(x)
        return x, h, c

def get_optimizer(learning_rate):
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

    return optimizer


