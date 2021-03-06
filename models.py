from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, concatenate, \
                                    CuDNNGRU, Bidirectional, \
                                    GlobalAveragePooling1D, GlobalMaxPooling1D,\
                                    Conv1D, LSTM, Add, BatchNormalization,\
                                    Activation, CuDNNLSTM, Dropout, Reshape,\
                                    Conv2D, MaxPooling2D, Flatten
import numpy as np

from custom_layers import Match
from data_loader import _load_msai_data, PairGenerator, load_embedding
from eval import eval_map
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

    cross = Match(match_type='plus')([q_conv1, d_conv1])

    z = Reshape((15, 50, 100))(cross)
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
                  embedding_weight=None):

    query = Input(shape=(max_query_length,) )
    doc = Input(shape=(max_response_length,) )

    embedding = Embedding(max_vocab_size, 300,weights=[embedding_weight] if embedding_weight is not None else None,
                            trainable=False)
    q_embed = embedding(query)
    d_embed = embedding(doc)

    rnn = Bidirectional(CuDNNLSTM(50,return_sequences=True))

#     q_conv1 = Conv1D(20, 3, padding='same',activation='relu') (rnn(q_embed))
#     q_conv1 = Dropout(0.75)(q_conv1)
#     d_conv1 = Conv1D(20, 3, padding='same',activation='relu') (rnn(d_embed))
#     d_conv1 = Dropout(0.75)(d_conv1)


    q_conv1 = rnn(q_embed)
    d_conv1 = rnn(d_embed)

    cross = Match(match_type='plus')([q_conv1, d_conv1])

    z = Reshape((15, 50, 100))(cross)
#     z = Conv2D(filters=50, kernel_size=(3,3), padding='same', activation='relu')(z)
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
