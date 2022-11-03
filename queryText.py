from elasticsearch import Elasticsearch, helpers
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from elasticsearch.helpers import bulk
import json

def getAllBooks(resp):
    for num, doc in enumerate(resp):   
        yield [doc['_source']['isbn'], doc['_source']['book_title'], doc['_source']['summary']]

def similarities_to_elastic(chunkList):
    for i in range(len(chunkList)):
        print(f'Uploadinfg chunk no. {i+1} ')
        for index, document in chunkList[i].iterrows():
            index.todict()
            yield{
                "_index" : f"similarity_numbers",
                "_source": document
            }

es = Elasticsearch(host='localhost', port='9200',http_auth=("marios","11111111"), http_compress=True)
y = 0
books = []

resp = helpers.scan(
        es,
        index = 'books',
        scroll = '3m',
        size = 10,
    )

books += getAllBooks(resp)

stopwords = nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

iterations = 32
tfidfs = []
st = time.time()
for j in range(iterations):
    docs = []
    x = j * len(books)//iterations + 1
    if j == 0:
        x = 0
    for i in range(x, (j+1)*len(books)//iterations):
        doc = re.sub('[^a-zA-Z]', ' ', books[i][2])
        doc = str(doc).lower()
        doc = doc.split()
        doc = [lemmatizer.lemmatize(word) for word in doc if not word in set(stopwords)]    
        doc = ' '.join(doc)
        docs.append(doc)

    vectorizer = TfidfVectorizer() 
    vectors = vectorizer.fit_transform(docs)
    tf_idf = pd.DataFrame(vectors.todense())
    tf_idf.columns = vectorizer.get_feature_names_out()
    tfidf_matrix = tf_idf.T
    tfidf_matrix.columns = ['isbn: '+str(books[i][0])+'  '+'title: '+str(books[i][1]) for i in range(0, len(docs))]
    tfidfs.append(tfidf_matrix.T)
    print(f"Chunk no. {j+1} ready!")

print(f'All chunks ready. Uploading {iterations} chunks to elasticsearch: ')

j = 1
for i in tfidfs: 
    print(type(i))
'''    result = i.to_json(orient="split")
    parsed = json.loads(result)
    result = json.dumps(parsed, indent=4)  
    docket_content = result.read()
    es.index(index='similarities', ignore=400, doc_type='docket', 
    id=j, body=json.loads(docket_content))
    j = j + 1
    print(f'Chunk {j} uploaded!')'''
et = time.time()

print('time: ',et-st)
