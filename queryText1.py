from elasticsearch import Elasticsearch
import pandas as pd
from numpy import nan as NaN

def tenPercentMostRelevantBooks(es, cMetric):
    
    uid = int(input('Insert a user id: '))
    word = str(input('Now insert a string to query: '))

    #to avoid warnings search queries in es 7.15.0 and higher should be written as follows:
    respUid = es.search(query = {"match":{"uid" : f"{uid}"}}, index = "ratings", size = 10000)
    respBooks = es.search(query = {"match":{"book_title": f"{word}"}}, index = 'books', size = 10000)

    #create dataframes to use
    uidDf = pd.DataFrame({'isbn': [], f'ratingOfUser{uid}': []})
    similaritiesDf = pd.DataFrame({'isbn': [], 'similarity': []})
    allMetricsDf = pd.DataFrame() #isbn | rating | similarity
    finalMetricDf = pd.DataFrame({'isbn': [], 'customMetric': []})

    for i in range(len(respUid["hits"]['hits'])):
        uidDf = uidDf.append(
            {'isbn':respUid["hits"]['hits'][i]['_source']['isbn'],
            'rating':respUid["hits"]['hits'][i]['_source']['rating']}, ignore_index=True
            )

    for i in range(len(respBooks['hits']['hits'])):
        similaritiesDf = similaritiesDf.append(
            {'isbn': respBooks["hits"]['hits'][i]['_source']['isbn'],
            'similarity': respBooks["hits"]['hits'][i]['_score']}, ignore_index=True
        )

    allMetricsDf = pd.merge(uidDf, similaritiesDf, on='isbn', how='outer') #form: isbn | rating | similarity

    allMetricsDf = allMetricsDf.replace(NaN, 0)  #replace NaN values with 0

    for index, row in allMetricsDf.iterrows():
        finalMetricDf = finalMetricDf.append({'isbn': row['isbn'], 'customMetric':
        cMetric * row['similarity'] + (1.0-cMetric) * int(row['rating'])}, ignore_index=True 
        ) 

    finalMetricDf.sort_values(by=['customMetric'], inplace=True, ascending=False)

    numBooksToShow = len(finalMetricDf.index) // 10
    #drop all rows with zero custom metric
    finalMetricDf = finalMetricDf[finalMetricDf['customMetric'] > 0]
    
    #drop all rows except 10% with highest final score 
    finalMetricDf = finalMetricDf.head(numBooksToShow)

    titles = []
    for index, row in finalMetricDf.iterrows():
        titles.append(es.search(query = {"match":{"isbn": f"{row['isbn']}"}},
        index = 'books')['hits']['hits'][0]['_source']['book_title'])
    finalMetricDf.insert(1, "title", titles, True) #adding new column containing titles
    return finalMetricDf                           #isbn | title | customMetric


if __name__ == '__main__':

    es = Elasticsearch(host='localhost', port='9200',http_auth=("marios","11111111"), http_compress=True)
    customMetricFactor = 0.6
    metrics = tenPercentMostRelevantBooks(es, customMetricFactor)
    print(metrics.head())