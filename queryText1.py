from elasticsearch import Elasticsearch, helpers
import pandas as pd
from numpy import nan as NaN

def getAllratingsOfNNRating(es, cluster):
    resp = helpers.scan(es, index = 'neural_net', scroll = '3m', size = 100, query = {'query': {'match': {'cluster' : f'{cluster}'}}})
    for num, doc in enumerate(resp):   
        yield [
                str(doc['_source']['isbn']),
                float(doc['_source']['rating'])]

def tenPercentMostRelevantBooks(es, cMetric):
    
    uid = int(input('Insert a user id: '))
    word = str(input('Now insert a string to query: '))

    respUid2 = []
    #to avoid warnings search queries in es 7.15.0 and higher should be written as follows:
    x = int(input("Do you want to search given ratings, or all ratings? (1 = given, 2 = all, 3 = after neural network):"))
    if x == 1:
        respUid = es.search(query = {"match":{"uid" : f"{uid}"}}, index = "ratings", size = 10000)
    elif x == 2:
        respUid = es.search(query = {"query_string": {'query': f"*_{uid}.0*", 'default_field': "uid"}}, index = "ratings_of_all_users", size = 10000)
    elif x == 3:
        respUid = es.search(query = {"query_string": {'query': f"*_{uid}.0*", 'default_field': "uid"}}, index = "ratings_of_all_users", size = 10000)
        userCluster = int(float(es.search(query = {"match": {'uid': f'{uid}.0'}}, index = 'user_clusters', size = 1)['hits']['hits'][0]['_source']['cluster']))
        respUid2 += getAllratingsOfNNRating(es, userCluster)
    respBooks = es.search(query = {"match":{"book_title": f"{word}"}}, index = 'books', size = 10000)

    #create dataframes to use
    uidDf = pd.DataFrame({'isbn': [], f'ratingOfUser{uid}': []})
    similaritiesDf = pd.DataFrame({'isbn': [], 'similarity': []})
    allMetricsDf = pd.DataFrame() #isbn | rating | similarity
    finalMetricDf = pd.DataFrame({'isbn': [], 'title': [], 'customMetric': []})

    for i in range(len(respUid["hits"]['hits'])):
        uidDf = uidDf.append(
            {'isbn':respUid["hits"]['hits'][i]['_source']['isbn'],
            'rating':respUid["hits"]['hits'][i]['_source']['rating']}, ignore_index=True
            )
    uidDf = uidDf.append(pd.DataFrame(respUid2, columns = ['isbn', 'rating']))

    for i in range(len(respBooks['hits']['hits'])):
        similaritiesDf = similaritiesDf.append(
            {'isbn': respBooks["hits"]['hits'][i]['_source']['isbn'],
            'title': respBooks["hits"]['hits'][i]['_source']['book_title'],
            'similarity': respBooks["hits"]['hits'][i]['_score']}, ignore_index=True
        )
    
    allMetricsDf = pd.merge(similaritiesDf, uidDf, on='isbn', how='inner') #form: isbn | rating | similarity

    allMetricsDf = allMetricsDf.replace(NaN, 0)  #replace NaN values with 0

    for row in allMetricsDf.itertuples():
        finalMetricDf = finalMetricDf.append({'isbn': row.isbn, 'title': row.title, 'customMetric':
        cMetric * row.similarity + (1.0-cMetric) * int(row.rating)}, ignore_index=True 
        ) 

    finalMetricDf.sort_values(by=['customMetric'], inplace=True, ascending=False)
    #drop all rows with zero custom metric
    finalMetricDf = finalMetricDf[finalMetricDf['customMetric'] > 0]
    numBooksToShow = finalMetricDf.shape[0] // 10 

    #drop all rows except 10% with highest final score 
    finalMetricDf = finalMetricDf.head(numBooksToShow)

    return finalMetricDf                           #isbn | title | customMetric


if __name__ == '__main__':

    es = Elasticsearch(host='localhost', port='9200',http_auth=("elastic","Altair1453"), http_compress=True)
    customMetricFactor = 0.6
    metrics = tenPercentMostRelevantBooks(es, customMetricFactor)
    print(metrics.head(metrics.shape[0]))
