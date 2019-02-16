import os
import pickle
import fire
import numpy as np

from data_loader import _load_msai_data, PairGenerator, load_embedding
from eval import eval_map
# from models import get_model_v1 as get_model
from models import get_model_v2 as get_model




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
         use_pickled_data=True
         ):
    """
        Function for model training

        :params:
            - train_tsv_file
            - test_tsv_file
            - max_vocab_size
            - embedding_file_path
            - max_query_length
            - max_response_length
            - embedding_dim
            - validation_split
    """

    word_index = None
    if use_pickled_data and os.path.exists(data_pickle_file):
        print("Loading Pickled Data...")
        word_index, train_query_id, train_queries,\
        train_responses, y_train, val_query_id,\
        val_queries, val_responses, y_val,\
        test_query_id, test_queries, test_responses = pickle.load(open(data_pickle_file,'rb'))

    else:
        # Load and process all the data
        word_index, train_query_id,\
        train_queries, train_responses,\
        y_train, val_query_id,\
        val_queries, val_responses,\
        y_val, test_query_id,\
        test_queries, test_responses = _load_msai_data(
                                            train_tsv_file=train_tsv_file,
                                            test_tsv_file=test_tsv_file,
                                            max_query_length=max_query_length,
                                            max_response_length=max_response_length,
                                            max_vocab_size=max_vocab_size,
                                            validation_split=validation_split
                                            )

        pickle.dump(
                     [word_index, train_query_id, train_queries,
                     train_responses, y_train, val_query_id,
                     val_queries, val_responses, y_val,\
                     test_query_id, test_queries, test_responses],
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
                      embedding_weight=embedding_weight)

    # Making data generators
    train_generator = PairGenerator(doc1=train_queries,
                                     doc2=train_responses,
                                     y_true=y_train,
                                     query_id=train_query_id).get_iterator(sample_queries)

    valid_generator = PairGenerator(doc1=val_queries,
                                     doc2=val_responses,
                                     y_true=y_val,
                                     query_id=val_query_id).get_iterator_test()

    for i in range(epochs):
      model.fit_generator(train_generator,\
                          epochs=1,
                          steps_per_epoch=10000,
                          validation_data=valid_generator,
                          validation_steps=1000)
      metrics = []
      count = 0
      for x,y in valid_generator:
        if count == 10000:
          break
        y_pred = model.predict(x)
        metrics.append(eval_map(y,y_pred))
        count += 1
      print('MAP : {}'.format(np.mean(metrics)))

if __name__ == '__main__':
    fire.Fire(main)
