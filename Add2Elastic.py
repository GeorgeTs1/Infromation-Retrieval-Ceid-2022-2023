from elasticsearch import Elasticsearch
from elasticsearch import helpers
from elasticsearch_dsl import connections
import requests
import pandas as pd
import json
from elasticsearch.helpers import bulk



#isbn,book_title,book_author,year_of_publication,publisher,summary,category
#uid,isbn,rating
#uid,location,age

books_df = pd.read_csv("BX-Books.csv") #reading data from csv file
books_ratings_df = pd.read_csv("BX-Book-Ratings.csv")
books_users_df = pd.read_csv("BX-Users.csv")



es = Elasticsearch(host='localhost', port='9200',http_auth=("elastic","Altair1453"))


def books_csv_to_elastic(df):
    for index,row in df.iterrows():
        print(f'Adding the object {index} to elastic search')
        yield{ 
             "_index" : "books",
             "_source" : {
                "isbn" : str(row["isbn"]),
                'book_title' : str(row['book_title']),
                'book_author' : str(row['book_author']),
                'year_of_publication' : int(row['year_of_publication']),
                'publisher' : str(row['publisher']),
                'summary' : str(row['summary']),
                'category' : str(row['category'])
                }

            }

        
def books_ratings_csv_to_elastic(df):
    for index,row in df.iterrows():
        print(f'Adding the object {index} to elastic search')
        yield{ 
             "_index" : "ratings",
             "_source" : {
                "uid" : int(row["uid"]),
                "isbn" : str(row["isbn"]),
                "rating" : float(row["rating"])
                }

            }



def books_users_csv_to_elastic(df):
    for index,row in df.iterrows():
        print(f'Adding the object {index} to elastic search')
        yield{ 
             "_index" : "users",
             "_source" : {
                "uid" : int(row["uid"]),
                "location" : str(row["location"]),
                "age" : str(row["age"])
                }

            }

bulk(es,books_csv_to_elastic(books_df))
bulk(es,books_ratings_csv_to_elastic(books_ratings_df))
bulk(es,books_users_csv_to_elastic(books_users_df))







