from elasticsearch import helpers
import pandas as pd
import time
from datetime import timedelta
from elasticsearch import Elasticsearch


pd.options.mode.chained_assignment = None #get rid of warning

def getAllUsersOfClusters(es):
    resp = helpers.scan(es, index = 'user_clusters', scroll = '3m', size = 100)
    for num, doc in enumerate(resp):   
        yield [ float(doc['_source']['uid']), float(doc['_source']['cluster'])]

def getAllRatings(es):
    resp = helpers.scan(es, index = 'ratings', scroll = '3m', size = 100)
    for num, doc in enumerate(resp):   
        yield [ float(doc['_source']['uid']), str(doc['_source']['isbn']), float(doc['_source']['rating'])]

def getAllBooks(es):
    resp = helpers.scan(es, index = 'books', scroll = '3m', size = 100)
    for num, doc in enumerate(resp):   
        yield [ str(doc['_source']['isbn'])]

def uploadAvgsOfClusters(allAvgs):
    for row in allAvgs.itertuples():
        yield{
            "_index" : f"ratings_of_cluster",
            "_source" : {
                "isbn" : str(row.isbn),
                "av_rating" : float(row.av_rating)
            }
        }

def uploadAllRatings(df):
    for row in df.itertuples():
        yield{
            "_index" : "ratings_of_all_users",
            "_source" : {
                "isbn" : str(row.isbn),
                "uid": str(row.uid),
                "rating" : float(row.rating),
                "cluster": str(row.cluster)
            }
        }


if __name__ == '__main__':
    es = Elasticsearch(
    "http://localhost:9200",
    http_auth=("elastic","putyourpasswordhere"),timeout=3600)
    #print(es.count(index='ratings_of_all_users', body={'query': {'match_all': {}}})["count"])
    numOfClusters = 64

    clUsers = []
    allRatings = []
    allBooks = []

    clUsers += getAllUsersOfClusters(es) #uid | cluster
    allRatings += getAllRatings(es)
    allBooks += getAllBooks(es)

    clUsersDf = pd.DataFrame(clUsers, columns = ['uid', 'cluster'])
    allRatingsDf = pd.DataFrame(allRatings, columns = ['uid', 'isbn', 'rating'])
    allBooksDf = pd.DataFrame(allBooks, columns = ['isbn'])

    allAvgs = pd.merge(allRatingsDf,clUsersDf, how="left",
    on=["uid"]).dropna().groupby(['isbn', 'cluster'])['rating'].mean().add_suffix('').reset_index()
    allAvgs.rename(columns = {'rating': 'av_rating'}, inplace = True)

    input('Press any button to start uploading cluster averages...')
    es.indices.delete(index = 'ratings_of_cluster', ignore = [400, 404])
    helpers.bulk(es, uploadAvgsOfClusters(allAvgs))
    print('Cluster averages uploaded')

    st = time.time()

    allAvgs['cluster'] = allAvgs['cluster'].astype(float).astype(int)
    allAvgs['isbn'] = allAvgs['isbn'].astype(str)
    allRatingsDf['uid'] = allRatingsDf['uid'].astype(str)

    es.indices.delete(index='ratings_of_all_users', ignore=[400, 404])
    
    for i in range(numOfClusters):
        ratingsPerCluster = allAvgs.loc[allAvgs['cluster'] == i]
        usersPerCluster = clUsersDf.loc[clUsersDf['cluster'] == i]

        lp1, lp2 = pd.core.reshape.util.cartesian_product(
            [usersPerCluster['uid'].to_list(), ratingsPerCluster['isbn'].to_list()])
        finalDf = pd.DataFrame(dict(uid = lp1, isbn = lp2))
        finalDf['uid'] = finalDf['uid'].astype(str)
        finalDf = pd.merge(finalDf, allRatingsDf, how = 'left', on = ['uid', 'isbn'])
        finalDf = pd.merge(finalDf, ratingsPerCluster, how = 'left', on = ['isbn'])
        #finalDf: uid | isbn | rating | av_rating
        finalDf.rating.fillna(finalDf.av_rating, inplace=True)
        del finalDf['av_rating']    #finalDf: uid | isbn | rating 
        finalDf = finalDf.groupby(['isbn', 'rating'])['uid'].apply(' _'.join).reset_index()
        finalDf['cluster'] = i
        helpers.bulk(es, uploadAllRatings(finalDf))
        print(i)
    et = time.time()
    print('That took: ', str(timedelta(seconds = et - st)))
