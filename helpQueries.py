from elasticsearch import Elasticsearch
import pandas as pd
from numpy import nan as NaN

def tenPercentMostRelevantBooks(es, cMetric=0):
    
    #uid = int(input('Insert a user id: '))
    #word = str(input('Now insert a string to query: '))

    #to avoid warnings search queries in es 7.15.0 and higher should be written as follows:
    #respUid = es.search(query = {"match_all": {}}, index = "ratings_of_all_users")

    #respUid = es.search(query = {"query_string": {'query': f"*_{uid}.0*", 'default_field': "uid"}}, index = "ratings_of_all_users")
        #query = {"query_string":{"query" : f"*,{uid},*", "default_field": "uid"}}, index = "ratings_of_all_users", size = 10000)
    #respBooks = es.search(query = {"match":{"book_title": f"{word}"}}, index = 'books', size = 10000)
    #print(respUid)
    print(es.count(index='users', body={'query': {'match_all': {}}})["count"])
    print(es.count(index='user_clusters', body={'query': {'match_all': {}}})["count"])

es = Elasticsearch(host='localhost', port='9200',http_auth=("marios","11111111"), http_compress=True)

metrics = tenPercentMostRelevantBooks(es)
