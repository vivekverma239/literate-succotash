import random
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.preprocessing.text import text_to_word_sequence


def _load_msai_data(train_tsv_file,\
                    test_tsv_file,\
                    max_query_length=15,
                    max_response_length=50,
                    max_vocab_size=30000,
                    validation_split=0.2):
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
    data = pd.read_csv(train_tsv_file, header=None, sep='\t')
    data.columns = columns= ['query_id', 'query', 'response', 'target', 'response_id']

    test_data = pd.read_csv(test_tsv_file, header=None, sep='\t' )
    test_data.columns = ['query_id', 'query', 'response', 'response_id']
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

    train_queries = train_data['query'].tolist()
    train_query_id = train_data['query_id'].tolist()
    train_responses = train_data['response'].tolist()
    y_train = train_data['target']

    test_queries = test_data['query'].tolist()
    test_query_id = test_data['query_id'].tolist()
    test_responses = test_data['response'].tolist()


    val_queries = val_data['query'].tolist()
    val_query_id = val_data['query_id'].tolist()
    val_responses = val_data['response'].tolist()
    y_val = val_data['target']

    train_queries_raw = sequence.pad_sequences([text_to_word_sequence(i) for i in train_queries], dtype=object, value="")
    train_responses_raw = sequence.pad_sequences([text_to_word_sequence(i) for i in train_queries], dtype=object, value="")
    test_queries_raw = sequence.pad_sequences([text_to_word_sequence(i) for i in train_queries], dtype=object, value="")
    test_responses_raw = sequence.pad_sequences([text_to_word_sequence(i) for i in train_queries], dtype=object, value="")
    val_queries_raw = sequence.pad_sequences([text_to_word_sequence(i) for i in train_queries], dtype=object, value="")
    val_responses_raw = sequence.pad_sequences([text_to_word_sequence(i) for i in train_queries], dtype=object, value="")

    tokenizer = text.Tokenizer(num_words=max_vocab_size)
    tokenizer.fit_on_texts(train_queries + train_responses )

    # Processing Training Data
    train_queries = tokenizer.texts_to_sequences(train_queries)
    train_responses = tokenizer.texts_to_sequences(train_responses)
    train_queries = sequence.pad_sequences(train_queries, maxlen=max_query_length)
    train_responses = sequence.pad_sequences(train_responses, maxlen=max_response_length)

    # Processing Validation Data
    val_queries = tokenizer.texts_to_sequences(val_queries)
    val_responses = tokenizer.texts_to_sequences(val_responses)
    val_queries = sequence.pad_sequences(val_queries, maxlen=max_query_length)
    val_responses = sequence.pad_sequences(val_responses, maxlen=max_response_length)

    # Processing Test Data
    test_queries = tokenizer.texts_to_sequences(test_queries)
    test_responses = tokenizer.texts_to_sequences(test_responses)
    test_queries = sequence.pad_sequences(test_queries, maxlen=max_query_length)
    test_responses = sequence.pad_sequences(test_responses, maxlen=max_response_length)



    return tokenizer.word_index, train_query_id, train_queries, train_responses, y_train,\
            val_query_id, val_queries, val_responses, y_val,\
            test_query_id, test_queries, test_responses, \
            train_queries_raw, train_responses_raw,  val_queries_raw, val_responses_raw,\
            test_queries_raw, test_responses_raw








class PairGenerator():
    """
        Helper Class for pairwise training of query , response documents
    """
    def __init__(self, doc1, doc2, y_true, query_id):
        """

        """
        self.num_negative = 6
        self.query_dict = {}
        for query, x1, x2, y in zip(query_id,doc1,doc2,y_true):
          item = self.query_dict.get(query,{})
          if y > 0:
            temp = item.get('pos',[])
            temp.append((x1,x2))
            item['pos'] = temp
          else:
            temp = item.get('neg',[])
            temp.append((x1,x2))
            item['neg'] = temp
          self.query_dict[query] = item

    def get_iterator(self,sample_queries=20):
        while True:
          sampled_queries = random.sample(self.query_dict.keys(),sample_queries)
          x = []
          for key in sampled_queries:
            item = self.query_dict[key]
            pos = item.get('pos',[])
            neg = item.get('neg',[])
            if pos and len(neg) > self.num_negative:
              x.extend(random.sample(pos,1))
              x.extend(random.sample(neg,self.num_negative))
          temp =  list(zip(*x))
          temp_y = np.array([1]* len(temp[0]))
          yield [np.array(temp[0]), np.array(temp[1])] , temp_y

    def get_iterator_test(self):
        while True:
          x = []
          for key in self.query_dict.keys():
            item = self.query_dict[key]
            x = item.get('pos',[])  + item.get('neg',[])
            y = [1 for _ in  item.get('pos',[])] + [0 for _ in  item.get('neg',[])]
            temp =  list(zip(*x))
            yield [np.array(temp[0]), np.array(temp[1])] , y


    def get_classification_iterator(self,sample_queries=50):
        while True:
          sampled_queries = random.sample(self.query_dict.keys(),sample_queries)
          x = []
          y = []
          for key in sampled_queries:
            item = self.query_dict[key]
            pos = item.get('pos',[])
            neg = item.get('neg',[])
            if pos and len(neg) > self.num_negative:
              x.extend(random.sample(pos,1))
              y.append(1)
              x.extend(random.sample(neg,self.num_negative))
              y.extend([0 for _ in range(self.num_negative)])
          temp =  list(zip(*x))
          temp_y = np.array([1]* len(temp[0]))
          yield [np.array(temp[0]), np.array(temp[1])] , y


class PairGeneratorWithRaw():
    """
        Helper Class for pairwise training of query , response documents
    """
    def __init__(self, doc1, doc2, y_true, query_id, doc1_raw, doc2_raw):
        """

        """
        self.num_negative = 6
        self.query_dict = {}

        zipped_items = zip(query_id,doc1,doc2,y_true,doc1_raw, doc2_raw)
        for query, x1, x2, y, x1_raw, x2_raw in zipped_items:
          item = self.query_dict.get(query,{})
          if y > 0:
            temp = item.get('pos',[])
            temp.append((x1, x2, x1_raw, x2_raw))
            item['pos'] = temp
          else:
            temp = item.get('neg',[])
            temp.append((x1, x2, x1_raw, x2_raw))
            item['neg'] = temp
          self.query_dict[query] = item

    def get_iterator(self,sample_queries=20):
        while True:
          sampled_queries = random.sample(self.query_dict.keys(),sample_queries)
          x = []
          for key in sampled_queries:
            item = self.query_dict[key]
            pos = item.get('pos',[])
            neg = item.get('neg',[])
            if pos and len(neg) > self.num_negative:
              x.extend(random.sample(pos,1))
              x.extend(random.sample(neg,self.num_negative))
          temp =  list(zip(*x))
          temp_y = np.array([1]* len(temp[0]))
          yield [np.array(temp[0]), np.array(temp[1])] , temp_y

    def get_iterator_test(self):
        while True:
          x = []
          for key in self.query_dict.keys():
            item = self.query_dict[key]
            x = item.get('pos',[])  + item.get('neg',[])
            y = [1 for _ in  item.get('pos',[])] + [0 for _ in  item.get('neg',[])]
            temp =  list(zip(*x))
            yield [np.array(temp[0]), np.array(temp[1])] , y


    def get_classification_iterator(self,sample_queries=50):
        while True:
          sampled_queries = random.sample(self.query_dict.keys(),sample_queries)
          x = []
          y = []
          for key in sampled_queries:
            item = self.query_dict[key]
            pos = item.get('pos',[])
            neg = item.get('neg',[])
            if pos and len(neg) > self.num_negative:
              x.extend(random.sample(pos,1))
              y.append(1)
              x.extend(random.sample(neg,self.num_negative))
              y.extend([0 for _ in range(self.num_negative)])
          temp =  list(zip(*x))
          temp_y = np.array([1]* len(temp[0]))
          yield [np.array(temp[0]), np.array(temp[1])] , y



def load_embedding(embedding_file_path, word_index, embedding_dim):
    """
        Load Embeddings from a text file

        :params:
            - embedding_file_path
            - word_index
            - embedding_dim
    """
    # Create a Numpy Placeholder for Embedding
    max_features = len(word_index)+1
    embedding_weights = np.random.random([max_features,embedding_dim])
    count = 0
    glove_file = open(embedding_file_path)
    for line in glove_file:
      word, vector = line.split(' ')[0], line.split(' ')[1:]
      if word in word_index and word_index[word] <= max_features:
        count += 1
        vector = list(map(float,vector))
        embedding_weights[word_index[word]] = [float(i) for i in vector]

    print('Fraction found in glove {}'.format(count/len(embedding_weights)))
    return embedding_weights
