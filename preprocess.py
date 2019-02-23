import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
import sys
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.preprocessing.text import text_to_word_sequence


def split_data(train_tsv_file,
               validation_split=0.2,
               train_file_path='data/train.tsv',\
               valid_file_path='data/valid.tsv'):
    """
        Load Microsoft AI Challenge Dataset from TSV file

        :params:
            - tsv_file: TSV data file provided by Microsoft
            - max_query_length: Max Sequence Length of Query, Queries data will be
                                truncated upto this length
            - max_response_length: Max Reponse Length, Responses will be truncated
                                    upto this length
    """
    # Read data file and assign column names
    print("Reading train data...")
    data = pd.read_csv(train_tsv_file, header=None, sep='\t')
    data.columns = columns= ['query_id', 'query', 'response', 'target', 'response_id']

    # Sample Validation Set
    query_ids = list(set(data['query_id'].tolist()))
    val_ids = random.sample(query_ids, int(len(query_ids)*validation_split))
    train_ids = list(set(query_ids) - set(val_ids))
    sample = {i:True for i in train_ids}
    sample_val = {i:True for i in val_ids}
    flag_train = [i in  sample for idx, i in enumerate(data['query_id'].tolist())]
    train_data = data[flag_train]
    flag_val = [i in  sample_val for idx, i in enumerate(data['query_id'].tolist())]
    val_data = data[flag_val]
    train_data.to_csv(train_file_path, sep='\t', index=False, header=None)
    val_data.to_csv(valid_file_path, sep='\t', index=False, header=None)

if __name__ == '__main__':
       split_data("data/msai/data.tsv", validation_split=0.05)
