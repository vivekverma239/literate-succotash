import os
import pickle
import fire
import numpy as np

from data_loader import _load_msai_data, PairGenerator, load_embedding, PairGeneratorWithRaw
from evaluation import eval_map
# from tf.keras.callbacks import LambdaCallback
# from tf.keras.backend import K
# from models import get_model_v2 as get_model
from models import get_model_with_elmo as get_model
from utils import MSAIDataLoader

# python main.py --train_tsv_file data/msai/data.tsv --test_tsv_file eval1_unlabelled.tsv --embedding_file_path /home/vivek/Projects/research/text_classification/data/glove/glove.840B.300d.txt



def main(train_tsv_file,test_tsv_file,
         epochs=20,
         max_vocab_size=50000,
         embedding_file_path='data/glove/glove.6B.300d.txt',
         max_query_length=15,
         max_response_length=70,
         sample_queries=50,
         embedding_dim=300,
         validation_split=0.05,
         data_pickle_file='data/data.pkl',
         use_pickled_data=True,
         pairwise_loss=False
         ):
    """
        Function for model training

        :params:
            - train_tsv_file: Training CSV File
            - test_tsv_file: Test CSV File, Predictions will be done on these
            - max_vocab_size: Maximum Vocab Size to keep
            - embedding_file_path: Glove Embedding File
            - max_query_length: Max Query Length to consider, words will be trimmed after max_query_length
            - max_response_length: Max Response Length to consider, words will be trimmed after max_response_length
            - embedding_dim: Dimension of the embedding, should match with embedding file
            - validation_split: Fraction of training data to take as validation set
            - data_pickle_file: Processed data will be cached in this file (enables fast loading)
            - use_pickled_data: Whether to use cached data or process from scratch
            - pairwise_loss: Use pairwise_loss rather than ranking loss
    """

    word_index = None
    data_loader = MSAIDataLoader(max_query_length=max_query_length, max_document_length=max_response_length )
    word_index = data_loader.tokenizer.word_index

    # Limit word Vocab to Max Vocab
    word_index = {k:v for k,v in word_index.items() if v < max_vocab_size}

    # Case to handle when len(word_index) < max_vocab_size
    max_vocab_size = len(word_index)+1

    # Embedding Loader
    embedding_weight = load_embedding(embedding_file_path,
                                      word_index,
                                      embedding_dim)

    # Define the model
    model = get_model(max_query_length=max_query_length,
                      max_response_length=max_response_length,
                      max_vocab_size=max_vocab_size,
                      embedding_dim=300,
                      embedding_weight=embedding_weight)

    # train_generator = data_loader.data_iterator('data/train.tsv', raw_fields=False, neg_prob=0.3, batch_size=128)
    # valid_generator = data_loader.data_iterator('data/valid.tsv', raw_fields=False, batch_size=10)

    train_generator = data_loader.tf_data_iterator('data/train.tsv', raw_fields=True, neg_prob=0.3, batch_size=128)
    valid_batch_size = 1024
    valid_generator = data_loader.tf_data_iterator('data/valid.tsv', raw_fields=True, batch_size=valid_batch_size, shuffle=False)
    from tqdm import tqdm
    temp_iterator = data_loader.data_iterator('data/valid.tsv', raw_fields=False)
    valid_steps = data_loader.get_number_of_steps('data/valid.tsv', batch_size=1)
    y_valid = []
    for y_ in tqdm(temp_iterator):
        if len(y_valid) == valid_steps:
            break
        y_valid.append(y_[-1])

    # TODO Add some saver object and calleback also for evaluation
    # init_callback = LambdaCallback(
    #                 on_train_start=lambda logs : K.default_session() )
    # callbacks = []
    for i in range(epochs):
    #   model.fit_generator(train_generator,\
    #                       epochs=1,
    #                       steps_per_epoch=10000,
    #                       validation_data=valid_generator,
    #                       validation_steps=1000)

      # Calculating MAP of Validation set
    #   metrics = []
    #   count = 0
    #   for x,y in valid_generator:
    #     if count == 10000:
    #       break
    #     y_pred = model.predict(x)
    #     metrics.append(eval_map(y,y_pred))
    #     count += 1
    #   print('MAP : {}'.format(np.mean(metrics)))

        model.fit(train_generator,\
                          epochs=1,
                          steps_per_epoch=10000,
                          validation_data=valid_generator,
                          validation_steps=1000)
        import math
        y_valid_pred = model.predict(valid_generator, steps=math.ceil(valid_steps/valid_batch_size), verbose=1)
      # Calculating MAP of Validation set
        metrics = []
        count = 0
        for idx in range(int(valid_steps/10)):
            last_idx = int(idx)*10
            next_idx = (int(idx) +1) *10
            metrics.append(eval_map(y_valid[last_idx: next_idx],y_valid_pred[last_idx: next_idx]))

        print('MAP : {}'.format(np.mean(metrics)))
    # # Load best model and predict
    # scores = model.predict([test_queries,test_responses])
    # scores = np.squeeze(scores)
    # scores = list(map(str, scores))
    # results_file = open("{}_results.tsv".format(test_tsv_file.split('.tsv')[0]), "w")
    # for idx, test_id in enumerate(test_query_id):
    #     if idx % 10 == 0:
    #         query_scores = "\t".join(scores[idx:idx+10])
    #         results_file.write("{}\t{}\n".format(test_id, query_scores))
    # results_file.close()

if __name__ == '__main__':
    fire.Fire(main)
