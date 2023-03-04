#####################################################################################################
# Training loop code that implements a customizing training loop.
#####################################################################################################


# Loading Libraries
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential, Model
from matplotlib import pyplot as plt
import numpy as np

def german_to_decoder(german_batch) -> tuple:
    """ This function returns a tuple containing German inputs and outputs for the decoder model.

    Args:
        german_batch (tf.Tensor): Tokenized german sentence

    Returns:
        tuple: tuple containing German inputs and outputs for the decoder model.
    """

    decoder_input = tf.where(german_batch == 2, 0, german_batch)
    decoder_ouput = tf.concat([german_batch[:,1:], tf.zeros([tf.shape(german_batch)[0], 1], tf.int32)], 1)
    return (decoder_input, decoder_ouput)


@tf.function
def forward_backward_pass(encoder, 
                          decoder, 
                          english_input, 
                          german_input, 
                          german_output):
    """ Compute the forward pass and backward pass for the model

    Args:
        encoder (tf.model): encoder of the model
        decoder (tf.model): decoder of the model
        english_input (tf.Tensor): tokenized english sentence
        german_input (tf.Tensor): tokenized german input
        german_output (tf.Tensor): tokenized german output

    Returns:
        float, tf.Tensor: This function returns the value of the cost function and the gradients of the parameters.
    """

    with tf.GradientTape() as tape:
        state_h_encoder, state_c_encoder = encoder(english_input)
        decoder_outputs, _, _ = decoder(german_input, hidden_state = state_h_encoder, cell_state = state_c_encoder)

        loss = tf.keras.metrics.sparse_categorical_crossentropy(german_output, decoder_outputs, from_logits=True)
        gradients = tape.gradient(loss, encoder.trainable_variables + decoder.trainable_variables)
    
    return loss, gradients

@tf.function
def training_step(encoder, 
                  decoder, 
                  optimizer, 
                  english_input, 
                  german_input, 
                  german_output):
    """ This functions apply the gradient of the cost function as respect to the training parameters of the model.

    Args:
        encoder (tf.model): encoder of the model
        decoder (tf.model): decoder of the model
        optimizer (tf.optimizer): optimizer used.
        english_input (tf.Tensor): tokenized english sentence
        german_input (tf.Tensor): tokenized german input
        german_output (tf.Tensor): tokenized german output

    Returns:
        float: return the cost function value.
    """

    loss, gradients = forward_backward_pass(encoder, decoder, english_input, german_input, german_output)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + decoder.trainable_variables))
    return loss

@tf.function
def validation_step(encoder, 
                    decoder, 
                    english_input, 
                    german_input, 
                    german_output):
    """ This functions apply the gradient of the cost function as respect to the validation parameters of the model.

    Args:
        encoder (tf.model): encoder of the model
        decoder (tf.model): decoder of the model
        english_input (tf.Tensor): tokenized english sentence
        german_input (tf.Tensor): tokenized german input
        german_output (tf.Tensor): tokenized german output

    Returns:
        float: return the cost function value.
    """

    state_h_encoder, state_c_encoder = encoder(english_input)
    decoder_outputs, _, _ = decoder(german_input, hidden_state = state_h_encoder, cell_state = state_c_encoder)
    loss = tf.keras.metrics.sparse_categorical_crossentropy(german_output, decoder_outputs, from_logits=True)
    return loss


def training_run(encoder, 
                 decoder, 
                 optimizer, 
                 training_dataset):
    """ Compute one iteration of the training run.

    Args:
        encoder (tf.model): encoder of the model
        decoder (tf.model): decoder of the model
        optimizer (tf.optimizer): optimizer used.
        training_dataset (tf.dataset): training dataset

    Returns:
        list: list of batch training loss.
    """

    batch_training_loss = []

    for english_input, german_input_raw in training_dataset:
        german_decoder_input, german_decoder_output = german_to_decoder(german_input_raw)

        training_loss = training_step(encoder, decoder, optimizer, english_input, german_decoder_input, german_decoder_output)

        batch_training_loss.append(training_loss)
    return batch_training_loss


def validation_run(encoder, 
                   decoder, 
                   validation_dataset):
    """ Compute one iteration of the validation run.

    Args:
        encoder (tf.model): encoder of the model
        decoder (tf.model): decoder of the model
        validation_dataset (tf.dataset): validation dataset

    Returns:
        list: list of batch validation loss.
    """

    batch_validation_loss = []
    for english_input, german_input_raw in validation_dataset:
        german_decoder_input, german_decoder_output = german_to_decoder(german_input_raw)
        validation_loss = validation_step(encoder, decoder, english_input, german_decoder_input, german_decoder_output)
        batch_validation_loss.append(validation_loss)
    
    return batch_validation_loss

def display_info_learning(avg_training_loss_per_epoch, 
                          avg_validation_loss_per_epoch):
    """ Display the training and validation graphs w.r.t epochs.

    Args:
        avg_training_loss_per_epoch (list): list of batch training loss.
        avg_validation_loss_per_epoch (list): list of batch validation loss.
    """

    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.plot(avg_training_loss_per_epoch)
    plt.title('Training Loss vs. epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.subplot(122)
    plt.plot(avg_validation_loss_per_epoch)
    plt.title('Validation Loss vs. epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.savefig("results/display_info_learning.png")


def test_model(tokenizer, embedding_layer, English_sentences, encoder, decoder):
    """ This function will test the model for 5 random sentences

    Args:
        tokenizer (tf.tokenizer): tokenizer object used for tokenizing the English/German sentences
        embedding_layer (tf.Tensor): tensor that contains embedding english words already trained.
        English_sentences (tf.Tensor): tokenized english sentences
        encoder (tf.model): encoder of the model
        decoder (tf.model): decoder of the model
    """

    start = tokenizer.word_index["<start>"]
    end = tokenizer.word_index["<end>"]

    for i in np.random.choice(len(English_sentences), size = 5):
        english_sentence = English_sentences[i]
        s =  tf.strings.split(english_sentence)
        s = embedding_layer(s)
        s = tf.pad(s, tf.convert_to_tensor([[13-tf.shape(s)[0], 0], [0, 0]]), "CONSTANT")

        hidden_state, cell_state = encoder(s[None, ...])

        translated_sentence = []

        token_index = start
        while True:
            token = tf.Variable([[token_index]])
            output, hidden_state, cell_state = decoder(token, hidden_state, cell_state)

            token_index = np.argmax(output[0][0].numpy())
            if token_index == end:
                break

            german_word = tokenizer.index_word[token_index]
            translated_sentence.append(german_word)

        translated_sentence = " ".join(translated_sentence)
        with open('results/translations.txt', 'a') as file:
            file.write("English sentence:\t\t" + english_sentence + '\n')
            file.write("German translation:\t\t" + translated_sentence + '\n')
            file.write('\n')
