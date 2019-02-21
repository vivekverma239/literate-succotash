import os
import pickle
import fire
import numpy as np

from data_loader import _load_msai_data, PairGenerator, load_embedding, PairGeneratorWithRaw
from evaluation import eval_map
# from models import get_model_v2 as get_model
from models import get_model_with_elmo as get_model

# python main.py --train_tsv_file data/msai/data.tsv --test_tsv_file eval1_unlabelled.tsv --embedding_file_path /home/vivek/Projects/research/text_classification/data/glove/glove.840B.300d.txt



def main(train_tsv_file,test_tsv_file,
         epochs=20,
         max_vocab_size=50000,
         embedding_file_path='data/glove/glove.6B.300d.txt',
         max_query_length=15,
         max_response_length=50,
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
    if use_pickled_data and os.path.exists(data_pickle_file):
        print("Loading Pickled Data...")
        data = pickle.load(open(data_pickle_file,'rb'))
        word_index, train_query_id, train_queries,\
        train_responses, y_train,\
        val_query_id, val_queries, val_responses, y_val,\
        test_query_id, test_queries, test_responses, \
        train_queries_raw, train_responses_raw,\
        val_queries_raw, val_responses_raw,\
        test_queries_raw, test_responses_raw = data

    else:
        # Load and process all the data
        word_index, train_query_id, train_queries,\
        train_responses, y_train,\
        val_query_id, val_queries, val_responses, y_val,\
        test_query_id, test_queries, test_responses, \
        train_queries_raw, train_responses_raw,\
        val_queries_raw, val_responses_raw,\
        test_queries_raw, test_responses_raw = _load_msai_data(
                                            train_tsv_file=train_tsv_file,
                                            test_tsv_file=test_tsv_file,
                                            max_query_length=max_query_length,
                                            max_response_length=max_response_length,
                                            max_vocab_size=max_vocab_size,
                                            validation_split=validation_split,
                                            )

        pickle.dump(
                    [word_index, train_query_id, train_queries,\
                     train_responses, y_train,\
                     val_query_id, val_queries, val_responses, y_val,\
                     test_query_id, test_queries, test_responses, \
                     train_queries_raw, train_responses_raw,\
                     val_queries_raw, val_responses_raw,\
                     test_queries_raw, test_responses_raw],
                     open(data_pickle_file,"wb")
                    )

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
                      embedding_weight=embedding_weight,
                      pairwise_loss=pairwise_loss)

    # Making data generators
    train_generator = PairGeneratorWithRaw(doc1=train_queries,
                                     doc2=train_responses,
                                     doc1_raw=train_queries_raw,
                                     doc2_raw=train_responses_raw,
                                     doc1_raw_max_len=max_query_length,
                                     doc2_raw_max_len=max_response_length,
                                     y_true=y_train,
                                     query_id=train_query_id)

    if pairwise_loss:
        train_generator = train_generator.get_iterator(sample_queries)
    else:
        train_generator = train_generator.get_classification_iterator()


    valid_generator = PairGeneratorWithRaw(doc1=val_queries,
                                     doc2=val_responses,
                                     doc1_raw=val_queries_raw,
                                     doc2_raw=val_responses_raw,
                                      doc1_raw_max_len=max_query_length,
                                      doc2_raw_max_len=max_response_length,
                                     y_true=y_val,
                                     query_id=val_query_id).get_iterator_test()

    # TODO Add some saver object and calleback also for evaluation
    for i in range(epochs):
      model.fit_generator(train_generator,\
                          epochs=1,
                          steps_per_epoch=10000,
                          validation_data=valid_generator,
                          validation_steps=1000)
      # Calculating MAP of Validation set
      metrics = []
      count = 0
      for x,y in valid_generator:
        if count == 10000:
          break
        y_pred = model.predict(x)
        metrics.append(eval_map(y,y_pred))
        count += 1
      print('MAP : {}'.format(np.mean(metrics)))

    # Load best model and predict
    scores = model.predict([test_queries,test_responses])
    scores = np.squeeze(scores)
    scores = list(map(str, scores))
    results_file = open("{}_results.tsv".format(test_tsv_file.split('.tsv')[0]), "w")
    for idx, test_id in enumerate(test_query_id):
        if idx % 10 == 0:
            query_scores = "\t".join(scores[idx:idx+10])
            results_file.write("{}\t{}\n".format(test_id, query_scores))
    results_file.close()

if __name__ == '__main__':
    fire.Fire(main)
