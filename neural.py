from keras.layers import LSTM, Dense, Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import json

es = Elasticsearch(host='localhost', port='9200', http_auth=("marios","11111111"), http_compress=True)

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

    print(json.dumps(isbns))


    summaries = helpers.scan(es,query={
       "query":
            {
                "terms": {"isbn": isbns}
            },
            "fields": [
                "summary",
                "isbn"
            ],

            "_source": False},index='books')
            
    summaries_l = []
    isbn_sum = []


    for i in list(summaries):
        summaries_l.append(i['fields']['summary'][0])
        isbn_sum.append(i['fields']['isbn'][0])
    print(len(isbn_sum))


    data2 = {'isbn' : isbn_sum , 'summary' : summaries_l}
    tmp2 = pd.DataFrame(data=data2)
    ratings_summaries = tmp2.merge(tmp,on='isbn',how='inner')
    print(ratings_summaries)

for i in range(64):
    tmp = pd.DataFrame(columns=['summary','rating'])
    get_all_ratings(i)