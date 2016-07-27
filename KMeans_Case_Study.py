
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
year_lambda = lambda x: x-1960
from1960 = year_lambda(X['YearMade']).map(lambda x: 0 if x <0 else x)
X['YearMade'] = from1960


# In[3]:

X.drop(X.columns[[0,1,2,3,5,6,7,8,9,10,11,12,14,15,17, 37, 38, 39]], axis=1, inplace=True)
X['Enclosure'] = X['Enclosure'].map({'EROPS AC': 'EROPS w AC', 'EROPS w AC': 'EROPS w AC', 'EROPS': 'EROPS', 'NO ROPS' : 'NO ROPS', 'OROPS': 'OROPS',                                     'None or Unspecified': 'None or Unspecified'})


# In[4]:

for col in X.columns[1:]:
    dummy_var = pd.get_dummies(X[col])
    X[list(Counter(X[col]))[1:]] = dummy_var[list(Counter(X[col]))[1:]]
    del X[col]


# In[5]:

print "top features for each cluster:"
for num, centroid in enumerate(top_centroids):
    print "%d: %s" % (num, ", ".join(X.columns[i] for i in centroid))


# In[ ]:

initial = [cluster.vq.kmeans(X,i) for i in xrange(2,5)]
plt.plot([var for (cent,var) in initial])


# In[ ]:

cent, var = initial[]
assignment, cdist = cluster.vq.vq(X, cent)
plt.scatter(X[:,0], tests[:,1] c=assignment)


# In[ ]:




# In[ ]:



