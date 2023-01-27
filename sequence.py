from collections import deque
import ctypes
from multiprocessing import Manager, Process, Value, freeze_support,Lock
import os
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from elasticsearch_dsl import connections
import requests
import pandas as pd
import json
from elasticsearch.helpers import parallel_bulk,bulk,streaming_bulk
import sys
import fileinput
import numpy as np
import polars as pl
import time
import multiprocessing
from itertools import repeat
from joblib import Parallel, delayed
import urllib3
import warnings
import warnings
from elasticsearch.exceptions import ElasticsearchWarning



es = Elasticsearch(host='localhost', port='9200',http_auth=("elastic","Altair1453"), http_compress=True)

warnings.simplefilter('ignore', ElasticsearchWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 



def isbn_rated(uid):

  isbn_rated_list = []

  result = es.search(body={
    "query":{
      "match" : { "uid" : uid}},
    "sort": [
    {"_id": "desc"}
    ]      
        
      },index='ratings',size=10000)

  if len(result['hits']['hits']):
    books_rated = result['hits']['hits']
    tmp = result['hits']['hits'][-1]['sort'][0]
    while 1:
      next_result = es.search(body={
      "query": {
          "match": {
              "uid": uid
          }
      },
      "search_after": [tmp],
      "sort": [
        {"_id": "desc"}
      ]
      },index="ratings",size=10000)
      
      if len(next_result['hits']['hits'])==0:
        break
      tmp = next_result['hits']['hits'][-1]['sort'][0]
      books_rated.extend(next_result['hits']['hits'])
    for i in books_rated:
      isbn_rated_list.append(str(i['_source']['isbn']))
    return isbn_rated_list
  else:
    return isbn_rated_list

def isbn_unrated(isbn):

    isbn_unrated_list = []

    result = helpers.scan(es,body={"query": {
      "bool" : {
        "must" : {
          
          "match_all": {}
          
        },
        "must_not":[ 
            {"terms" : {
              "isbn" : isbn
              }
          }
        ]}},
        "sort": [
          {"_id": "desc"}
        ]      
      },index="books",scroll='1d')

    result = list(result)

    for j in result:
      isbn_unrated_list.append(j['_source']['isbn'])
      

    # if len(result['hits']['hits']):
    #   books_unrated = result['hits']['hits']
    #   tmp = result['hits']['hits'][-1]['sort'][0]
    #   while 1:
    #     next_result = es.search(body={
    #       "query": {
    #           "bool" : {
    #             "must" : {
    #               "match_all": {}
    #             },
    #             "must_not":[ 
    #                 {"terms" : {
    #                   "isbn" : isbn
    #                   }}]}},
    #         "search_after": [tmp],
    #         "sort": [
    #           {"_id": "desc"}
    #       ]
    #       },index="books",size=10000)
    #     if len(next_result['hits']['hits'])==0:
    #       break
    #     tmp = next_result['hits']['hits'][-1]['sort'][0]
    #     books_unrated.extend(next_result['hits']['hits'])
    #   for j in books_unrated:   
    #     isbn_unrated_list.append(str(j['_source']['isbn']))
    #   return isbn_unrated_list
    # else:
    #   return

    return isbn_unrated_list



def get_user_cluster(uid):
    cluster = es.search(query = {"term":{"uid" : uid}}, index = "user_clusters", size = 1)
    return cluster['hits']['hits']

def get_users_same_cluster(cluster,uid):

  users_same_cluster = []
  
  result = helpers.scan(es,index="user_clusters",body={
    "query":{
    
      "bool" :{
      
      "must" : {
        "match" : { "cluster" : cluster }
      },
      "must_not": {     
              "match": {"uid": uid }
          }
        }},

      },scroll='1d')

  result = list(result)

  for j in result:
      users_same_cluster.append(j['_source']['uid'])

  return users_same_cluster
      



  # if len(result['hits']['hits']):
  #     users_gen = result['hits']['hits']
  #     tmp = result['hits']['hits'][-1]['sort'][0]
  #     while 1:
  #       next_result = es.search(body={
  #         "query":{
  #             "bool" : {
  #             "must" : {
  #               "match" : { "cluster" : cluster }
  #             }}},
  #           "search_after": [tmp],
  #           "sort": [
  #             {"_id": "desc"}
  #         ]
  #         },index="user_clusters",size=10000)
  #       if len(next_result['hits']['hits'])==0:
  #         break
  #       tmp = next_result['hits']['hits'][-1]['sort'][0]
  #       users_gen.extend(next_result['hits']['hits'])
  #     for k in users_gen:
  #         users_same_cluster.append(int(k['_source']['uid']))
  #     return users_same_cluster
  # else:
  #   return users_same_cluster



  
    

        
def get_avg4books(isbn,uid,cluster):
   
   isbn_cluster =  []
   avg_rating = []
   users_list = get_users_same_cluster(cluster,uid)

   if len(users_list):
      ratings_4_book_i_query = es.search(index="ratings",query={
                  "bool" : {
                    
                    "must" : [
                      {"terms" : { "uid" : users_list }},
                      {"terms" : { "isbn" : isbn }}
                    ],
                    
                    "must_not": {
              
                        "match": {"uid": uid }
            
                      }
                }},aggregations={
                  "group_avg":{
                    "terms" : {"field" : "isbn.keyword"},
                  "aggregations" :{
                  "avg_grade" : {"avg" : {"field" : "rating"}},
                  }}},size=0)


      for i in ratings_4_book_i_query['aggregations']['group_avg']['buckets']:
          isbn_cluster.append(i['key'])
          avg_rating.append(i['avg_grade']['value'])
      
      return isbn_cluster,avg_rating   
   else:
      return [] , []
 


  
  
  
def upload_users_ratings(ratings):
  
  uid = ratings['uid'].tolist()
  isbn = ratings['isbn'].tolist()
  rating = ratings['rating'].tolist()

  for index in range(len(uid)):
    yield{ 
             "_index" : "ratings2",
             "_source" : {
                "uid" : int(uid[index]),
                "isbn" : str(isbn[index]),
                "rating" : float(rating[index])
                }
            }


            



df = None
lock  = None




def init(dataf,l):
    ''' store the counter for later use '''
    global df,lock
    df = dataf
    lock = l


def get_users_ratings_books(users):
  # books = helpers.scan(es, index = 'books', scroll = '1d', size = 100)
  # ratings = helpers.scan(es, index = 'ratings', scroll = '1d', size = 100)
  # numUsersInEs = es.count(index='users', body={'query': {'match_all': {}}})["count"] # getting the number of users from elastic
  tmp_df = pd.DataFrame(columns=['uid','isbn','rating'])
  cnt = 1

  for uid in users:

    cluster = get_user_cluster(uid)

    if len(cluster):

      isbn_cluster = []
      avg_rating = []

      books_rated = []
      books_unrated = []

      books_rated += isbn_rated(uid)


      books_unrated += isbn_unrated(books_rated)


      isbn_cluster , avg_rating= get_avg4books(books_unrated,uid,cluster[0]['_source']['cluster'])

      if len(isbn_cluster) and len(avg_rating):
        tmp_df['uid'] = [uid] * len(isbn_cluster)
        tmp_df['isbn'] = isbn_cluster
        tmp_df['rating'] = avg_rating

        #df = pd.concat([df,tmp_df],ignore_index=True)
        pb = parallel_bulk(es, upload_users_ratings(tmp_df), chunk_size=500, thread_count=8, queue_size=8)
        deque(pb, maxlen = 0)
        tmp_df = tmp_df[0:0]

      cnt = cnt +1

      print(cnt)
          
        

    
  





  

    



# get_users_ratings_books()


if __name__ == '__main__':



  users = helpers.scan(es, index = 'users', scroll = '1d', size = 100)

  books_not_rated_df = pd.DataFrame(columns=['uid','isbn','rating'])
  
  uids = [user['_source']['uid'] for user in users]

  mgr = Manager()
  ns = mgr.Namespace()
  ns.df = books_not_rated_df

  l = Lock()

  start = time.time()

  get_users_ratings_books(uids)

  end = time.time()

  mins = (end-start)//60

  print("i am done")

  print(mins)


  
  # print("let shall be goal")

  
  
  # with multiprocessing.Pool(initializer=init,initargs=(ns.df,l)) as pool:
  #   pool.map(get_users_ratings_books,uids)

  # end = time.time()

  # mins = (end-start)//60

  # print("i am done")

  # print(mins)

  

# ratings_4_book_i_query = helpers.scan(es,query={

#               "query":{
              
#               "bool" : {
                
#                 "must" : [
#                   {"terms" : { "uid" : 18131 }},
#                   {"terms" : { "isbn" : '0895261715' }}
#                 ],
                
#                 "must_not": {
          
#                     "match": {"uid": 18163 }
        
#                   }
#             }}},aggregations={
#               "group_avg":{
#                 "terms" : {"field" : "isbn.keyword"},
#               "aggregations" :{
#               "avg_grade" : {"avg" : {"field" : "rating"}},
#               }}},index='ratings',scroll='1d')


  

# for i in ratings_4_book_i_query:
#   print(i)
#   break

#get_users_ratings_books()        

# books_df = pd.read_csv("BX-Books.csv") #reading data from csv file
# books_ratings_df = pd.read_csv("BX-Book-Ratings.csv")

# users = pd.DataFrame()

# users=get_users_same_cluster(7)


# users_ratings = users.merge(books_ratings_df,on=['uid'],how='left')


# users_rati_ok = users_ratings.dropna()


# avg_ratings = pd.DataFrame()

# avg_ratings = users_rati_ok.groupby(['isbn']).mean().reset_index()


# avg_ratings = avg_ratings.drop(columns=['uid', 'cluster'])

# users_ratings_nan = users_ratings.drop(users_rati_ok.index)
# users_ratings_nan = users_ratings_nan.drop(columns=['isbn', 'rating','cluster'])

# test = users_ratings_nan.merge(avg_ratings,how='cross')

# print(test)

# books_df = pl.read_csv("BX-Books.csv") #reading data from csv file
# books_ratings_df = pl.read_csv("BX-Book-Ratings.csv")








"""t1=np.array([1,2,3])
t2=np.array([3,4,5])

dt1 = pd.DataFrame(t1,columns=['t1'])
dt2 = pd.DataFrame(t2,columns=['t2'])


df1 = dd.from_pandas(dt1,npartitions=10)
df2 = dd.from_pandas(dt2,npartitions=10)

df1['key']=1
df2['key']=1

df3 = dd.merge(df1,df2,on='key')

print(df3.head())
print(len(df3.index))

print(df3.size)
"""


# def generate_parallel_obj(data,index_name):
#     for index, document in data.iterrows():
#        yield{
#           "_index" : index_name,
#               "_source" : {
#                   "uid" : int(document["uid"]),
#                   "isbn" : str(document["isbn"]),
#                   "rating" : float(document["rating"])
#                   }
#             }

# def toEs(data,index):
#     print(data)
#     print("hello world")
#     bulk(es,data.map_partitions(generate_parallel_obj,'index'))

#     return 'hello cruel world'
    
    
  


# if __name__ == '__main__': 


#   es.indices.delete(index='test1', ignore=[400, 404])  

#   books_df = books_df.drop(columns=['book_title','book_author','year_of_publication','publisher','summary','category'])

#   users_ratings_meta = pd.DataFrame({'uid': pd.Series(dtype='int'),
#                    'isbn': pd.Series(dtype='str'),
#                    'rating': pd.Series(dtype='float')})
  
#   st = time.time()

#   for i in range(1,9):
    
#     users_unrated = pd.DataFrame() 
  
#     users=get_users_same_cluster(i)

#     users = pl.DataFrame(users)

#     users = users.drop('cluster')

#     users_ratings = users.join(books_ratings_df,on='uid',how='left')

    
#     users_rati_ok = users_ratings.drop_nulls()

#     print(users_rati_ok)

#     avg_ratings = users_rati_ok.groupby("isbn").agg([pl.mean(column='rating')])


#     books_df = books_df.with_column(pl.lit(1).alias("key"))

#     users_rati_ok = users_rati_ok.with_column(pl.lit(1).alias("key"))

#     users = users.with_column(pl.lit(1).alias("key"))

#     users_books = users.join(books_df,on='key')

#     merged = users_books.join(users_rati_ok,how='left')


#     users_unrated = merged[merged['_merge']=='left_only']

#     users_unrated_dask = users_unrated.drop(columns=['key','rating','_merge'])

#     users_rati_final = users_unrated_dask.join(avg_ratings,on='isbn',how='inner')

#     print(users_rati_final.describe())

#     print("i am done")

#     break

#   et= time.time()

#   elapsed_time = et - st
#   print('Execution time:', elapsed_time, 'seconds')

""" rated = []
rated += isbn_rated(3.0)
unrated = []
unrated+=isbn_unrated(rated)
users_same_cluster = helpers.scan(es,index="user_clusters",query={"query":{
    
      "bool" : {
      
      "must" : {
        "match" : { "cluster" : "5.0" }
      },
      
      "must_not": {
          
        "match": {"uid": "3.0" }
        
          }
       }
      }})
users_list = []
for j in users_same_cluster:
  users_list.append(j['_source']['uid']) 
ratings_4_book_i_query = es.search(index="ratings",body={
            "query":
            {
              
              "bool" : {
                
                "must" : [
                  {"terms" : { "uid" : ["9971"] }},
                  {"terms" : { "isbn" : unrated }}
                ],
                
                "must_not": {
          
                    "match": {"uid": "3.0" }
        
                  }
            }},"aggs":
              {
              "tags":{
              "terms" : { "field" : "isbn.keyword" },
               "aggs": { "avg_grade" : {"avg" : {"field" : "rating"}},
              }}}} 
              ,size=10000) 
print(ratings_4_book_i_query['aggregations']['tags']['buckets']) """