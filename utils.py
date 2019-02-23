import tensorflow as tf
import csv
import io
import math
import random
import numpy as np
from keras_preprocessing.text import tokenizer_from_json,\
                                                text_to_word_sequence
from keras_preprocessing.sequence import pad_sequences
from config import TOKENIZER_JSON_PATH, MAX_DOC1_LENGTH, MAX_DOC2_LENGTH


class MSAIDataLoader:
    def __init__(self, tokenizer_json_file=TOKENIZER_JSON_PATH,
                        max_query_length=15,
                        max_document_length=70):

        self.tokenizer = tokenizer_from_json(io.open(tokenizer_json_file).read())
        self.text_processor = lambda x : self.tokenizer.texts_to_sequences(x)
        self.raw_text_processor = lambda x : text_to_word_sequence(x)
        self.max_query_length = max_query_length
        self.max_document_length = max_document_length


    def data_iterator(self, data_file, mode='train', raw_fields=False,
                            batch_size=64, neg_prob=1):
        """
            File must be in the format as Microsft AI challenge data
            TSV file with no header

            :params:
                - file
                - tokenizer: Keras tokenizer
        """
        while True:
            with open(data_file, 'r') as csvfile:
                data_reader = csv.reader(csvfile, delimiter='\t')
                current_query_id = None
                target = None
                example = {}
                x_batch = []
                y_batch = []
                for row in data_reader:
                    if mode == 'train':
                        doc1_id, doc1, doc2, target, doc2_id = row
                    elif mode == 'valid':
                        doc1_id, doc1, doc2, _, doc2_id = row
                    elif mode == 'test':
                        doc1_id, doc1, doc2, doc2_id = row
                    else:
                        assert False, "{} Mode not supported ".format(mode)

                    if int(target) == 0 and random.random() > neg_prob:
                        continue
                    processed_doc1 = self.text_processor([doc1])
                    processed_doc2 = self.text_processor([doc2])
                    processed_doc1 = pad_sequences(processed_doc1,
                                                    maxlen=self.max_query_length)
                    processed_doc2 = pad_sequences(processed_doc2,
                                                    maxlen=self.max_document_length)

                    example  = [ processed_doc1[0], processed_doc2[0]]
                    if raw_fields:
                        raw_doc1 = self.raw_text_processor(doc1)
                        raw_doc2 = self.raw_text_processor(doc2)

                        raw_doc1 = pad_sequences([raw_doc1],
                                                      maxlen=self.max_query_length,
                                                      dtype=object,
                                                      value='-PAD-')
                        raw_doc2 = pad_sequences([raw_doc2],
                                                      maxlen=self.max_document_length,
                                                      dtype=object,
                                                      value='-PAD-')
                        example.extend([raw_doc1[0], raw_doc2[0]])
                    if target != None:
                        yield (tuple(example), [int(target)])
                    else:
                        print("Not ok")
                        yield tuple(example)


    def batch_data_generator(self, data_file, mode='train', raw_fields=False,
                            batch_size=64, neg_prob=1):
        generator = self.data_iterator(data_file,
                                         mode=mode,
                                         raw_fields=raw_fields,
                                         neg_prob=neg_prob,
                                         batch_size=batch_size)
        while True:
            x_batch, y_batch = [], []
            data = next(generator)
            if type(data) == tuple:
                x_batch.append(data[0])
                y_batch.append(data[1])
            else:
                x_batch.append(data)

            x_batch_final = x_batch
            if y_batch:
                y_batch_final = y_batch
            x_batch, y_batch = [], []
            x_batch_final = zip(*x_batch_final)
            x_batch_final = tuple(map(np.array, x_batch_final))
            if y_batch:
                yield x_batch_final, y_batch_final
            else:
                yield x_batch_final


    def tf_data_iterator(self, data_file, mode='train', raw_fields=False,
                                        batch_size=32, neg_prob=1,\
                                        shuffle=True):
        train_generator = lambda : self.data_iterator(data_file,
                                                             mode=mode,
                                                             raw_fields=raw_fields,
                                                             neg_prob=neg_prob,
                                                             batch_size=batch_size)
        train_generator_dtypes = ( tf.int32, tf.int32)


        train_generator_shapes = (tf.TensorShape([self.max_query_length]),
                                 tf.TensorShape([self.max_document_length]))

        if raw_fields:
            train_generator_dtypes = ( tf.int32, tf.int32, tf.string, tf.string)
            train_generator_shapes = (tf.TensorShape([self.max_query_length]),
                                    tf.TensorShape([self.max_document_length]),
                                      tf.TensorShape([self.max_query_length]),
                                    tf.TensorShape([self.max_document_length]))

        if mode != "test":
            train_generator_dtypes = (train_generator_dtypes, tf.int32)
            train_generator_shapes = (train_generator_shapes, tf.TensorShape([1]))

        dataset = tf.data.Dataset.from_generator(train_generator,
                                             output_types=train_generator_dtypes,
                                             output_shapes=train_generator_shapes)
        if shuffle :
            dataset = dataset.shuffle(10000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        return dataset

    def get_number_of_steps(self, data_file, batch_size):
        data_file = io.open(data_file)
        number_of_lines = 0
        for line in data_file:
            number_of_lines += 1
        return math.ceil(number_of_lines/batch_size)


if __name__ == '__main__':
    data_loader = MSAIDataLoader()
    # iterator = data_loader.data_iterator('data/msai/data.tsv')
    iterator = data_loader.tf_data_iterator('data/msai/data.tsv')

    # print(next(iterator))
