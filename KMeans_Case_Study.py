
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as scs
import statsmodels.api as sm
from collections import Counter
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from pandas import DataFrame
from scipy import cluster
get_ipython().magic(u'matplotlib inline')


# In[2]:

test_data = pd.read_csv('Train.csv')
y = test_data.pop('SalePrice')
X = test_data.set_index('SalesID')

def yearbin(x):
    if x < 1900:
        return "None"
    if x > 1899 and x < 1910:
        return 1900
    elif x >= 1910 and x < 1920:
        return 1910
    elif x >= 1920 and x < 1930:
        return 1920
    elif x >= 1930 and x < 1940:
        return 1930
    elif x >= 1940 and x < 1950:
        return 1940
    elif x >= 1950 and x < 1960:
        return 1950
    elif x >= 1960 and x < 1970:
        return 1960
    elif x >= 1970 and x < 1980:
        return 1970
    elif x >= 1980 and x < 1990:
        return 1980
    elif x >= 1990 and x < 2000:
        return 1990
    elif x >= 2000 and x < 2010:
        return 2000
    elif x >= 2010 and x < 2020:
        return 2010

yearbins = X['YearMade'].apply(yearbin)
X['YearBin'] = yearbins


# In[3]:

X.drop(X.columns[[0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,17, 37, 38, 39]], axis=1, inplace=True)
X['Enclosure'] = X['Enclosure'].map({'EROPS AC': 'EROPS w AC', 'EROPS w AC': 'EROPS w AC', 'EROPS': 'EROPS', 'NO ROPS' : 'NO ROPS', 'OROPS': 'OROPS',                                     'None or Unspecified': 'None or Unspecified'})


# In[4]:

X.info()


# In[5]:

for col in X.columns:
    dummy_var = pd.get_dummies(X[col])
    X[list(Counter(X[col]))[1:]] = dummy_var[list(Counter(X[col]))[1:]]
    del X[col]


# In[6]:

kmeans = KMeans(n_clusters =3)
kmeans.fit(X)


# In[7]:

top_centroids = kmeans.cluster_centers_.argsort()[:,-1:-11:-1]
print "top features for each cluster:"
for num, centroid in enumerate(top_centroids):
    print "%d: %s" % (num, ", ".join(str(X.columns[i]) for i in centroid))


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



