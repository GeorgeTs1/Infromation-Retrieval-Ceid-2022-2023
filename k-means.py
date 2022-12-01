from elasticsearch import Elasticsearch, helpers
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

def getAllUsers(es):
    resp = helpers.scan(es, index = 'users', scroll = '3m', size = 100)

    for num, doc in enumerate(resp):   
        yield [ int(doc['_source']['uid']),
                str(doc['_source']['location']).rsplit(',')[-1].replace(' ', '', 1).replace('"', '').replace('_ ',''),
                float(doc['_source']['age'])]

def userClustersToElastic(df):
    print('Starting upload to index: user_clusters')
    for index,row in df.iterrows():
        print(f"Uplaoding {row['uid']} user's clustering data")
        yield{ 
            "_index" : "user_clusters",
            "_source" : {
                "uid" : int(row["uid"]),
                "cluster" : int(row["Cluster"])
                }
            }

def makeUserInfoDf(es):
    userData = []
    userData += getAllUsers(es)
    usersDf = pd.DataFrame(userData, columns = ['uid', 'Country', 'Age'])
    numUsersInEs = es.count(index='users', body={'query': {'match_all': {}}})["count"]
    if not numUsersInEs == len(usersDf.index):
        print('ERROR: Not equal num of users returned')
    return usersDf

def classifyUsers(entrydf):
    df = entrydf.copy()  
    df['Country'] = abs(df['Country'].apply(hash)) % (10 ** 3)
    #df = df.dropna()
    #To give random ages to users with age == NaN delete line above and uncomment line below:
    df.loc[df['Age'].isnull(),'Age'] = 18 + (np.trunc(abs(np.random.normal(0,50, len(df.loc[df['Age'].isnull()]))))) % 100    
    return df



if __name__ == "__main__":
    es = Elasticsearch(host='localhost', port='9200', http_auth=("elastic","Altair1453"),http_compress=True)
    usersDf = makeUserInfoDf(es)
    classifiedUsersDf = classifyUsers(usersDf)
    print(usersDf) 
    
    z = classifiedUsersDf['Country']
    y = classifiedUsersDf['Age']

    data = list(zip(z, y))

    kmeans = KMeans(n_clusters = 8)
    kmeans.fit(data)

    clusterByUser = kmeans.fit_predict(data, y=None, sample_weight=None)
    classifiedUsersDf.insert(2, "Cluster", clusterByUser, True)
    print(classifiedUsersDf)

    figure, axis = plt.subplots(3, 1)

    axis[0].scatter(z, y)
    axis[0].set_xlabel('Country num')
    axis[0].set_ylabel('Age')
    axis[0].set_title('Starting Data')

    axis[1].scatter(z, y, c=kmeans.labels_)
    axis[1].set_ylabel('Age')
    axis[1].set_xlabel('country')
    axis[1].set_title('Clusters colored')

    inertias = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    axis[2].plot(range(1, 11), inertias, marker='o')
    axis[2].set_title('Elbow method')
    axis[2].set_xlabel('Number of clusters')
    axis[2].set_label('Inertia')

    plt.show()

    input('Press any button to upload Users with clusters.')
    #es 8+ replace line below with: es.options(ignore_status=[400,404]).indices.delete(index='userClusters')
    es.indices.delete(index='user_clusters', ignore=[400, 404])
    helpers.bulk(es, userClustersToElastic(classifiedUsersDf))
