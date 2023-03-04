import tensorflow as tf
import tensorflow_hub as hub
import unicodedata
import re
import numpy as np
import json
from tensorflow.keras.preprocessing.text import Tokenizer
import random



class Dataset_Preprocessing():
    def __init__(self, dataset_path: str = 'data/deu.txt', pair_of_sentences: int = 200000):
        self.dataset_path = dataset_path
        self.pair_of_sentences = pair_of_sentences
    
    def get_dataset(self) -> list:
        """ we get the dataset from the path.

        Returns:
            list: list en german and english sentences.
        """

        data_examples = []
        with open(self.dataset_path, 'r', encoding='utf8') as f:
            for line in f.readlines():
                if len(data_examples) < self.pair_of_sentences:
                    data_examples.append(line)
                else:
                    break
        
        return data_examples

    def unicode_to_ascii(self, s: str) -> str:
        """ Reconstitute the given string to ascii encodage.

        Args:
            s (str): sentence to convert to ascii encoding.

        Returns:
            str: sentence convering to ascii encoding.
        """

        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


    def preprocess_sentence(self, sentence: str) -> str:
        """  Preprocessing English and German sentences

        Args:
            sentence (str): english sentence to preprocess.

        Returns:
            str: english sentence that is preprocessed.
        """

        sentence = sentence.lower().strip()
        sentence = re.sub(r"ü", 'ue', sentence)
        sentence = re.sub(r"ä", 'ae', sentence)
        sentence = re.sub(r"ö", 'oe', sentence)
        sentence = re.sub(r'ß', 'ss', sentence)
        
        sentence = Dataset_Preprocessing.unicode_to_ascii(self, sentence)
        sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
        sentence = re.sub(r"[^a-z?.!,']+", " ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        
        return sentence.strip()

    def split_German_English_sentences(self, data: list) -> (list, list):
        """ This function splits a list of sentences into a list of english sentences and german sentences.

        Args:
            data (list): list of sentences to split.

        Returns:
            tuple: return a tuple of lists of sentences in english and german.
        """

        English_sentences = list()
        German_sentences = list()

        for sentence in data:
            words = sentence.split('\t')
            English_sentences.append(Dataset_Preprocessing.preprocess_sentence(self, words[0]))
            # Here we add a special "<start>" and "<end>" token to the beginning and end of every German sentence.
            German_sentences.append('<start> ' + Dataset_Preprocessing.preprocess_sentence(self, words[1]) + ' <end>')
        
        return (English_sentences, German_sentences)

    
    def tokenize_German_text(self, German_sentences: list):
        """ Tokenize the german sentences

        Args:
            German_sentences (list): list of german sentences.

        Returns:
            tensorflow.tokenizer: a Tensor of tokenize german words.
        """
        tokenizer = Tokenizer(num_words=None, 
                            filters='',
                            lower=True,
                            split=' ',
                            char_level=False,
                            oov_token=None)

        tokenizer.fit_on_texts(German_sentences)
        tokenizer_config = tokenizer.get_config()

        # Save word_index and index_word as python dictionaries

        index_word = json.loads(tokenizer_config['index_word'])
        word_index = json.loads(tokenizer_config['word_index'])

        sentence_seq = tokenizer.texts_to_sequences(German_sentences)

        return tokenizer, sentence_seq, (index_word, word_index)
    
    def print_random_examples(self, English_sentences, 
                              German_sentences, 
                              German_sentence_seq, 
                              mum_sentences = 5):
        """ Printing some random sentences in english, german and the tokenize german sentences associated.

        Args:
            English_sentences (list): list of English sentences
            German_sentences (list): list en German sentences
            German_sentence_seq (tensorflow.tokenizer): Tensor of tokenize german sentences
            mum_sentences (int, optional): Number of sentences that we want to display. Defaults to 5.
        """

        inx = np.random.choice(len(German_sentences), mum_sentences)
        for i in inx:
            print(f"English sentence: {English_sentences[i]}")
            print(f"German sentence: {German_sentences[i]}")
            print(f"The tokenize German sentence: {German_sentence_seq[i]}")
    
    def make_padded_dataset(self, German_sentence_seq):
        """ This function padds the German tokenizer dataset at the end of each sentence.

        Args:
            German_sentence_seq (tensorflow.tokenizer): Tensor of tokenize german sentences

        Returns:
            tf.Tensor: Tensor of tokenizer german sentences that are padded. 
        """

        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(German_sentence_seq, padding='post', value=0)
        return padded_sequences
    
    def get_english_embeddings(self):
        """ We load some english embeddings from a pre-trained English word embedding module from TensorFlow Hub.
        The URL for the module is https://tfhub.dev/google/tf2-preview/nnlm-en-dim128-with-normalization/1. 
        This module has also been made available as a complete saved model in the folder `'./models/tf2-preview_nnlm-en-dim128_1'`. 
        This embedding takes a batch of text tokens in a 1-D tensor of strings as input. It then embeds the separate tokens into a 128-dimensional space. 

        This model can also be used as a sentence embedding module. The module will process each token by removing punctuation and splitting on spaces.
        It then averages the word embeddings over a sentence to give a single embedding vector.
        However, we will use it only as a word embedding module, and will pass each word in the input sentence as a separate token.

        Returns:
            tf.Tensor: embedding vector of english words.
        """

        embedding_layer = hub.KerasLayer("./models/tf2-preview_nnlm-en-dim128_1", 
                                    output_shape=[128], input_shape=[], dtype=tf.string)
        return embedding_layer
    
    def create_train_valid_dataset(self, English_sentences, padded_German_sequences):
        """ We split the dataset into training and validation datasets.

        Returns:
            tuple: list of training and validation inputs and outputs.
        """

        valid_split_size = int(0.2 * self.pair_of_sentences)

        # Shuffle the dataset
        temp = list(zip(English_sentences, padded_German_sequences))
        random.shuffle(temp)
        English_sentences, padded_German_sequences = zip(*temp)
        English_sentences, padded_German_sequences = list(English_sentences), np.array(padded_German_sequences)

        X_train = English_sentences[:valid_split_size]
        X_valid = English_sentences[self.pair_of_sentences - valid_split_size:]

        y_train = padded_German_sequences[:valid_split_size, :]
        y_valid = padded_German_sequences[self.pair_of_sentences - valid_split_size:, :]

        return (X_train, y_train, X_valid, y_valid)
    
    def make_Dataset(self, X_train,
                     X_valid, 
                     y_train, 
                     y_valid):
        """ We create two tensorflow datasets (for training and validation).

        Args:
            X_train (list): list of english dataset for training
            X_valid (list): list of english dataset for validation
            y_train (tf.Tensor): Tensor of tokenize german sentences for training
            y_valid (tf.Tensor): Tensor of tokenize german senetences for validation

        Returns:
            tuple: tuple of training and validation tensorflow datasets.
        """

        training_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        validation_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
        return (training_dataset, validation_dataset)
    
    def split_input_english(self, dataset):
        """ split input english sentences into words.

        Args:
            dataset (tf.dataset): English dataset.
        """
        def split_word(input, label):
            input_words = tf.strings.split(input)
            return (input_words, label)
        dataset = dataset.map(split_word)

        return dataset
    
    def embedding_sentence(self, dataset, embedding_layer):
        """ Embedding of the english words by using the embedding_layer

        Args:
            dataset (tf.dataset): english dataset
            embedding_layer (tf.dataset): english dataset where english words are embedded.
        """

        def embedded(input, label):
            embedded_word = embedding_layer(input)
            return (embedded_word, label)
        
        dataset = dataset.map(embedded)

        return dataset
    
    def filter_length(self, dataset):
        """ filetring the length of the english input. if above 13 we 
        dont take it, otherwise we take it.

        Args:
            dataset (tf.dataset): english dataset.
        """

        def filtering(input, label):
            if tf.shape(input)[0] > 13:
                return False
            else:
                return True

        dataset = dataset.filter(filtering)

        return dataset
    
    def padd_english_sentence(self, dataset):
        """ We padd the english dataset at the end of zeros.

        Args:
            dataset (tf.dataset): englidsh dataset
        """

        def padding(input, label):
            shape_sentence = tf.shape(input)[0]
            if shape_sentence < 13:
                paddings = tf.convert_to_tensor([[13-tf.shape(input)[0], 0], [0, 0]])
                input_padded = tf.pad(input, paddings, "CONSTANT")
                return (input_padded, label)
            else:
                return (input, label)
        
        dataset = dataset.map(padding)

        return dataset
    
    def batched_dataset(self, dataset, batch_size = 16):
        """ Creating a batch of training and validation datasets.

        Args:
            dataset (tf.dataset): dataset to be batched.
            bbatch_size (int): batch size

        Returns:
            tf.dataset: dataset batched.
        """
        
        dataset = dataset.batch(batch_size)

        return dataset