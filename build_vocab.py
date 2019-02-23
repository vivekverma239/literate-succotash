"""
    Initialize tokenizer and save in folder
"""
import io
import fire
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from config import TOKENIZER_JSON_PATH

def build_vocab(train_tsv_file,\
                    test_tsv_file=None,
                    max_vocab_size=30000):
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
    train_data = pd.read_csv(train_tsv_file, header=None, sep='\t')
    train_data.columns =  ['query_id', 'query', 'response', 'target', 'response_id']

    train_queries = train_data['query'].tolist()
    train_responses = train_data['response'].tolist()

    if test_tsv_file:
        test_data = pd.read_csv(test_tsv_file, header=None, sep='\t')
        test_data.columns = ['query_id', 'query', 'response', 'response_id']
        train_queries += test_data['query'].tolist()
        train_responses += test_data['response'].tolist()

    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token=True)
    tokenizer.fit_on_texts(train_queries + train_responses )
    io.open(TOKENIZER_JSON_PATH, 'w').write(tokenizer.to_json())

if __name__ == '__main__':
    fire.Fire(build_vocab)
    # python build_vocab.py --train_tsv_file 'data/msai/data.tsv' --test_tsv_file data/msai/eval1_unlabelled.tsv
