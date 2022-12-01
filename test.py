from elasticsearch import Elasticsearch
from elasticsearch import helpers
from elasticsearch_dsl import connections
import requests
import pandas as pd
import json
from elasticsearch.helpers import bulk
import sys
import fileinput
import numpy as np
 
es = Elasticsearch(host='localhost', port='9200',http_auth=("elastic","Altair1453"), http_compress=True)


def isbn_rated(uid):
  books_rated = helpers.scan(es,query={"query":{"term" : { "uid" : uid}},
      "fields": ["isbn"]},index="ratings")
  for num, doc in enumerate(books_rated):   
        yield str(doc['_source']['isbn'])

def isbn_unrated(isbn):
    books_unrated = helpers.scan(es,query={"query": {
      "bool" : {
        "must" : {
          
          "match_all": {}
          
        },
        "must_not":[ 
            {"terms" : {
              "isbn" : isbn
              }
          }
        ]
      }
      }},index="books")
    for num, doc in enumerate(books_unrated):
        yield str(doc['_source']['isbn'])


def get_user_cluster(uid):
    cluster = es.search(query = {"term":{"uid" : uid}}, index = "user_clusters", size = 1)
    return cluster['hits']['hits'][0]['_source']['cluster']


        
def get_avg4books(isbn,uid):
   cluster = get_user_cluster(uid)
   users_same_cluster = []
   ratings = {}
   

   users_same_cluster += helpers.scan(es,index="user_clusters",query={"query":{
    
      "bool" : {
      
      "must" : {
        "match" : { "cluster" : cluster }
      },
      
      "must_not": {
          
        "match": {"uid": uid }
        
          }
       }
      }})

   users_list = []

   for j in users_same_cluster:
     users_list.append(j['_source']['uid']) 

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
              "tags":{
              "terms" : { "field" : "isbn.keyword" },
               "aggs": { "avg_grade" : {"avg" : {"field" : "rating"}},
              }}},scroll='3m')
    
    
   ratings_l = ratings_4_book_i_query['aggregations']['tags']['buckets']

   for i in ratings_l:
       ratings[i['key']] = i['avg_grade']['value']   
   
   return ratings   
    

   
      
         
   
           



def get_users_ratings_books():
  users = helpers.scan(es, index = 'users', scroll = '3m', size = 100)
  books = helpers.scan(es, index = 'books', scroll = '3m', size = 100)
  ratings = helpers.scan(es, index = 'ratings', scroll = '3m', size = 100)
  numUsersInEs = es.count(index='users', body={'query': {'match_all': {}}})["count"] # getting the number of users from elastic
  books_not_rated_df = pd.DataFrame() 
  

  for i in users:
    books_rated = []
    books_unrated = []
    print(i['_source']['uid'])
    
    books_rated += isbn_rated(i['_source']['uid'])
    books_unrated += isbn_unrated(books_rated)

    books_not_rated_df['uid'] = i['_source']['uid']
    books_not_rated_df['rating'] = get_avg4books(books_unrated,i['_source']['uid'])

  print(books_not_rated_df)
  


  

#get_users_ratings_books()        



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