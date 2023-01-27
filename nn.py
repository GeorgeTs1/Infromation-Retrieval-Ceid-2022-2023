import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#import spacy
#from nltk.corpus import stopwords
#from wordcloud import WordCloud,STOPWORDS
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from keras.layers import Bidirectional,Embedding,Flatten
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.optimizers import Adam
import warnings
import timeit

warnings.filterwarnings("ignore")


def uploadAllRatings(df):
    for row in df.itertuples():
        yield{
            "_index" : "neural_net",
            "_source" : {
                "isbn" : str(row.isbn),
                "uid": str(row.uid),
                "rating" : float(row.rating),
                "cluster": str(row.cluster)
            }
        }


def get_all_ratings(cluster):
    isbn_ratings = helpers.scan(es,query={
    
    "query":{
    
    "terms":{"cluster": [cluster]}  
    
  },
  "fields": [
    "isbn",
    "rating"
  ],
  
  "_source": False

    },index='ratings_of_all_users')
    
    isbns = []
    ratings = []

    for i in list(isbn_ratings):
        isbns.append(i['fields']['isbn'][0])
        ratings.append(i['fields']['rating'][0])

    data =  {'isbn':isbns , 'rating' : ratings}

    tmp = pd.DataFrame(data=data)

    summaries = helpers.scan(es,query={
                  "query" : {
                    "bool" : {
                     "must" : {
                        "match_all": {}
                    },
                    "filter": {"terms": {"isbn.keyword": isbns}}}},   
                    "fields": [
                        "summary",
                        "isbn"
                    ],
            "_source": False},index='books')
    
    not_rated_books = helpers.scan(es,query={

          "query":
          {
             "bool":{
                 "must":{
                        "match_all": {}
                    
                 },

                 "must_not" : {

                      "terms" : {"isbn" : isbns}

                 }
             }
          },
        "fields" : [ "summary",'isbn']
        ,
        "_source" : False},index='books')


    summaries_l = []
    isbn_sum = []
    summaries_not_rated = []
    isbns_not_rated = []


    for i in list(summaries):
        summaries_l.append(i['fields']['summary'][0])
        isbn_sum.append(i['fields']['isbn'][0])
    
    for i in list(not_rated_books):
        summaries_not_rated.append(i['fields']['summary'][0])
        isbns_not_rated.append(i['fields']['isbn'][0])

    data2 = {'isbn' : isbn_sum , 'summary' : summaries_l}

    data3 = {'isbn' : isbns_not_rated, 'summary': summaries_not_rated}

    tmp2 = pd.DataFrame(data=data2)

    not_rated_summaries = pd.DataFrame(data=data3)

    ratings_summaries = tmp2.merge(tmp,on='isbn',how='inner')

    return ratings_summaries , not_rated_summaries


def tokenizer_X(X):
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(X)
  X = tokenizer.texts_to_sequences(X)
  max_length = max([len(x) for x in X_train])
  vocab_size = len(tokenizer.word_index)+1 #add 1 to account for unknown word
  #print("Vocabulary size: {}".format(vocab_size))
  #print("Max length of sentence: {}".format(max_length))
  X = pad_sequences(X, max_length ,padding='post')
  return X,vocab_size,max_length



def tokenizer_unrated(X,size):
  tokenizer = Tokenizer(num_words=size)
  tokenizer.fit_on_texts(X)
  X = tokenizer.texts_to_sequences(X)
  max_length = max([len(x) for x in X_train])
  vocab_size = size 
  #print("Vocabulary size: {}".format(vocab_size))
  #print("Max length of sentence: {}".format(max_length))
  X = pad_sequences(X, max_length ,padding='post')
  return X,vocab_size,max_length



def nn(vocab,emb_vector,X):
  model = Sequential()
  model.add(Embedding(vocab,emb_vector,input_length=X.shape[1]))
  model.add(Bidirectional(LSTM(64,return_sequences=True)))
  model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(Dense(32,activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(16,activation='relu'))
  model.add(Dense(1,activation='sigmoid'))
  model.compile(loss='mse', optimizer=Adam(learning_rate=0.0001), metrics=['mean_absolute_error'])
  callbacks = [EarlyStopping(monitor='val_loss', patience=2),
              ModelCheckpoint('../model/model.h5', save_best_only=True, 
                              save_weights_only=False)]
  model.summary()

  return model,callbacks



if __name__ == '__main__':
  es = Elasticsearch(host='localhost', port='9200',http_auth=("marios","11111111"))
  embedding_vector_length=32
  tokenizer = Tokenizer()
  scaler = MinMaxScaler(feature_range=(0,1))

  print("creating index neural net...")

  if not es.indices.exists(index = 'neural_net'):
    es.indices.create(index='neural_net')

  es.indices.put_settings(index='neural_net',body={
    "refresh_interval" : "30s" 
    })

  # es.indices.put_settings(index='neural_net',body={
  #   "refresh_interval" : "-1" 
  # })




  t_0 = timeit.default_timer()
  for i in range(64):
    data = pd.DataFrame(columns=['isbn','summary','rating'])
    not_rated = pd.DataFrame(columns=['isbn','summary'])
    final_data = pd.DataFrame(columns=['cluster','isbn','rating'])

    data,not_rated = get_all_ratings(i)
    X = data['summary'].copy()
    y = data['rating'].copy()

    y = scaler.fit_transform(y.values.reshape(-1,1))

    X_train, X_val, y_train, y_val = train_test_split(X, y,random_state=1,test_size=0.335)

    X_train,vocab_size,max_length = tokenizer_X(X_train)
    X_val,vocab_validate,max_validate = tokenizer_X(X_val)
    model,callbacks = nn(vocab_size,embedding_vector_length,X_train)

    history  = model.fit(X_train, y_train, validation_data=(X_val,y_val), 
                    epochs=3, batch_size=32, verbose=1,shuffle=True,
                    callbacks=callbacks)

    X_test_summaries,v,max_length_unrated = tokenizer_unrated(not_rated['summary'],vocab_size)

    pred = model.predict(X_test_summaries)

    rounded = [np.around(x,decimals=1)  for x in pred]

    
    rounded = np.vstack(rounded)

    predicted_ratings = rounded


    final_data['isbn'] = not_rated['isbn']
    final_data['rating'] = predicted_ratings

    final_data['cluster'] = [i] * len(final_data.axes[0])
    print(final_data.head())
    print(f'Uploading cluster {i} to elasticsearch...')

    '''for ok,response in helpers.streaming_bulk(es,uploadAllRatings(final_data),chunk_size=10000):
      if not ok:
        print(response)'''

    


    print(f'cluster_ratings {i} is uploaded to elastic')

    
  t_1 = timeit.default_timer()
  elapsed_time = round((t_1 - t_0), 3)

  print(f'Loop elapsed time in seconds is: {elapsed_time}')


