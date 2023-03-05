#####################################################################################################
# Main code
#####################################################################################################


# Loading Libraries
import tensorflow as tf
import tensorflow_hub as hub
import unicodedata
import re
import numpy as np
import json
from tensorflow.keras.preprocessing.text import Tokenizer
import random
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential, Model
from time import time
from matplotlib import pyplot as plt
from utils.Preprocessing_data import Dataset_Preprocessing
from utils import Model as md
from utils import Learning_part as lp
import argparse

def init_parser():
    """ Here we define the parameters that we can pass through the terminal to run the code.

    Returns:
        dict: dictionnary of parameters.
    """
    parser = argparse.ArgumentParser(description='Neural Translation quick training script')
    parser.add_argument('--data_number', default=20000, type=int, metavar='N',
                            help='number of total pair of sentences to extract')

    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                            help='number of total epochs to run')
    
    parser.add_argument('--lr', default=0.001, type=float,
                        help='initial learning rate')

    parser.add_argument('--batch_size', default=16, type=int,
                        help='batch size')
    return parser

def main():
    parser = init_parser()
    args = parser.parse_args()
    #                             -------------------------- BEGINING OF DATA PIPELINE --------------------------

    Dataset_object = Dataset_Preprocessing(pair_of_sentences = args.data_number)

    data_examples = Dataset_object.get_dataset()

    English_sentences,German_sentences = Dataset_object.split_German_English_sentences(data_examples)

    tokenizer, German_sentence_seq, (index_word, word_index) = Dataset_object.tokenize_German_text(German_sentences)

    Dataset_object.print_random_examples(English_sentences, German_sentences, German_sentence_seq)

    padded_German_sequences = Dataset_object.make_padded_dataset(German_sentence_seq)

    embedding_layer = Dataset_object.get_english_embeddings()

    (X_train, y_train, X_valid, y_valid) = Dataset_object.create_train_valid_dataset(English_sentences, padded_German_sequences)

    training_dataset, validation_dataset  = Dataset_object.make_Dataset(X_train, X_valid, y_train, y_valid)

    training_dataset, validation_dataset = Dataset_object.split_input_english(training_dataset), \
                                           Dataset_object.split_input_english(validation_dataset)


    training_dataset, validation_dataset = Dataset_object.embedding_sentence(training_dataset, embedding_layer), \
                                           Dataset_object.embedding_sentence(validation_dataset, embedding_layer)

    training_dataset, validation_dataset = Dataset_object.filter_length(training_dataset), \
                                           Dataset_object.filter_length(validation_dataset)

    training_dataset, validation_dataset = Dataset_object.padd_english_sentence(training_dataset), \
                                           Dataset_object.padd_english_sentence(validation_dataset)

    training_dataset, validation_dataset = Dataset_object.batched_dataset(training_dataset, batch_size = args.batch_size), \
                                           Dataset_object.batched_dataset(validation_dataset, batch_size = args.batch_size)

    
    #                              ---------------------------- END OF DATA PIPELINE ----------------------------

    #                              ---------------------------- BEGINING OF LEARNING ----------------------------
    
    VOCAB_SIZE = int(list(index_word.keys())[-1])
    optimizer = md.get_optimizer(learning_rate = args.lr)
    encoder = md.Encoder()
    decoder = md.Decoder(VOCAB_SIZE)
    avg_training_loss_per_epoch = []
    avg_validation_loss_per_epoch = []
    start = time()

    for epoch in range(args.epochs):
        batch_training_loss = lp.training_run(encoder, decoder, optimizer, training_dataset)
        avg_training_loss_per_epoch.append(np.mean(batch_training_loss))

        # Saving the weight of the training session
        encoder.save_weights("checkpoints/encoder.{}".format(epoch))
        decoder.save_weights("checkpoints/decoder.{}".format(epoch))

        batch_validation_loss = lp.validation_run(encoder, decoder, validation_dataset)
        avg_validation_loss_per_epoch.append(np.mean(batch_validation_loss))

        print(" Epoch {:03d}: Training Loss: {:.3f}, Validation Loss: {:.3f}".format(epoch+1, avg_training_loss_per_epoch[epoch],
                                                            avg_validation_loss_per_epoch[epoch]))

    end = time()
    print(f"The model took {end - start} seconds")
    lp.display_info_learning(avg_training_loss_per_epoch, avg_validation_loss_per_epoch)

    #                             -------------------------- END OF LEARNING --------------------------

    #                             ------------------------ BEGINING OF TESTING ------------------------

    lp.test_model(tokenizer, embedding_layer, English_sentences, encoder, decoder)


if __name__ == '__main__':
    main()





