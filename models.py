from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Concatenate, \
                                    CuDNNGRU, Bidirectional, Dense, \
                                    GlobalAveragePooling1D, GlobalMaxPooling1D,\
                                    Conv1D, LSTM, Add, BatchNormalization,\
                                    Activation, CuDNNLSTM, Dropout, Reshape,\
                                    Conv2D, MaxPooling2D, Flatten, Subtract
import numpy as np

from custom_layers import Match, ElmoEmbeddingLayer
from data_loader import _load_msai_data, PairGenerator, load_embedding
from ranking_losses import rank_hinge_loss

def get_model_v1(max_query_length,
          max_response_length,
          max_vocab_size,
          embedding_dim=300,
          embedding_weight=None):

    query = Input(shape=(max_query_length,) )
    doc = Input(shape=(max_response_length,) )

    embedding = Embedding(max_vocab_size, 300,weights=[embedding_weight] if embedding_weight is not None else None,
                            trainable=False)
    q_embed = embedding(query)
    d_embed = embedding(doc)

    rnn = Bidirectional(CuDNNGRU(50,return_sequences=True))

    q_conv1 = rnn(q_embed)
    d_conv1 = rnn(d_embed)

    cross = Match(match_type='dot')([q_conv1, d_conv1])

    z = Reshape((15, 50, 1))(cross)
    z = Conv2D(filters=50, kernel_size=(3,3), padding='same', activation='relu')(z)
    z = Conv2D(filters=25, kernel_size=(3,3), padding='same', activation='relu')(z)
    z = MaxPooling2D(pool_size=(3,3))(z)
    z = Conv2D(filters=10, kernel_size=(3,3), padding='same', activation='relu')(z)
    z = MaxPooling2D(pool_size=(3,3))(z)

    pool1_flat = Flatten()(z)
    pool1_flat_drop = Dropout(rate=0.5)(pool1_flat)
    out_ = Dense(1)(pool1_flat_drop)

    model = Model(inputs=[query,doc],outputs=out_)
    model.compile(optimizer='adadelta',loss=rank_hinge_loss())
    return model


def get_model_v2(max_query_length,
                  max_response_length,
                  max_vocab_size,
                  embedding_dim=300,
                  embedding_weight=None,\
                  pairwise_loss=False):

    query = Input(shape=(max_query_length, ) )
    doc = Input(shape=(max_response_length, ) )

    embedding = Embedding(max_vocab_size, 300, weights=[embedding_weight] if embedding_weight is not None else None,
                            trainable=False)
    q_embed = embedding(query)
    d_embed = embedding(doc)

    q_embed = Dropout(rate=0.2)(q_embed)
    d_embed = Dropout(rate=0.2)(d_embed)


    rnn = Bidirectional(CuDNNLSTM(100, return_sequences=True))

    q_conv1 = rnn(q_embed)
    d_conv1 = rnn(d_embed)

    cross = Match(match_type='dot')([q_conv1, d_conv1])

    z = Reshape((max_query_length, max_response_length, 1))(cross)
    pool1_flat = Flatten()(z)
    pool1_flat = Dense(50,activation='relu')(pool1_flat)
    pool1_flat_drop = Dropout(rate=0.2)(pool1_flat)
    out_ = Dense(1,activation='sigmoid' if pairwise_loss else None)(pool1_flat_drop)

    model = Model(inputs=[query,doc], outputs=out_)
    model.compile(optimizer='adam', loss="binary_crossentropy" if pairwise_loss\
                                        else rank_hinge_loss())
    return model



def get_model_with_elmo(max_query_length,
                  max_response_length,
                  max_vocab_size,
                  embedding_dim=300,
                  embedding_weight=None,\
                  pairwise_loss=False):

    query = Input(shape=(max_query_length, ) )
    doc = Input(shape=(max_response_length, ) )

    query_raw = Input(shape=(max_query_length, ), dtype="string" )
    doc_raw = Input(shape=(max_response_length, ), dtype="string" )

    embedding = Embedding(max_vocab_size, 300, weights=[embedding_weight] if embedding_weight is not None else None,
                            trainable=False)
    elmo_embedding = ElmoEmbeddingLayer()

    q_embed = embedding(query)
    d_embed = embedding(doc)
    q_embed_elmo = elmo_embedding(query_raw)
    d_embed_elmo = elmo_embedding(doc_raw)

    q_embed = Concatenate(axis=-1)([q_embed, q_embed_elmo])
    d_embed = Concatenate(axis=-1)([d_embed, d_embed_elmo])

    q_embed = Dropout(rate=0.2)(q_embed)
    d_embed = Dropout(rate=0.2)(d_embed)


    rnn = Bidirectional(CuDNNLSTM(100, return_sequences=True))

    q_conv1 = rnn(q_embed)
    d_conv1 = rnn(d_embed)

    cross = Match(match_type='dot')([q_conv1, d_conv1])

    z = Reshape((max_query_length, max_response_length, 1))(cross)
    pool1_flat = Flatten()(z)
    pool1_flat = Dense(50,activation='relu')(pool1_flat)
    pool1_flat_drop = Dropout(rate=0.2)(pool1_flat)
    out_ = Dense(1,activation='sigmoid' if pairwise_loss else None)(pool1_flat_drop)

    model = Model(inputs=[query,doc, query_raw, doc_raw], outputs=out_)
    model.compile(optimizer='adam', loss="binary_crossentropy" if pairwise_loss\
                                        else rank_hinge_loss())
    return model
