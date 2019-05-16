#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
os.chdir('C:\\Analytics\\Deep Learning\\image dimesionality reduction')


# In[2]:


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.decomposition import PCACA


# In[3]:


X = np.load('X.npy')
Y = np.load('Y.npy')
X.shape


# It has 2062 images each has 64x64 piexel

# In[4]:


plt.imshow(X[0])


# In[7]:


9- np.argmax(Y[0])


# In[8]:


X_flat = np.array(X).reshape((2062,64*64))

X_train,X_test,y_train,y_test = train_test_split(X_flat,Y,test_size=0.3,random_state=42)


# In[9]:


clf = MLPClassifier(solver='adam',alpha=1e-5,hidden_layer_sizes=(20,20,20),random_state=1)
clf.fit(X_train,y_train)


# In[10]:


y_hat = clf.predict(X_test)
print('accuracy:' + str(accuracy_score(y_test,y_hat)))


# As we can see pretty poor model. now going to reduce the dimension, but before
# that we need to decide what we want to reduce it. So we are going to try and find
# the number of dimensions that keeps 95% of variance of the original images

# In[11]:


pca_dims = PCA()
pca_dims.fit(X_train)
cumsum = np.cumsum(pca_dims.explained_variance_ratio_)
d = np.argmax(cumsum>=0.95) + 1


# In[12]:


d


# See we have gone the dimesion from 4096 to 292

# In[13]:


pca = PCA(n_components=d)
X_reduced = pca.fit_transform(X_train)
X_recovered = pca.inverse_transform(X_reduced)


# In[16]:


print('reduced shape:'+str(X_reduced.shape))
print('recovered shape:'+str(X_recovered.shape))


# In[18]:


f = plt.figure()
f.add_subplot(1,2, 1)
plt.title("original")
plt.imshow(X_train[0].reshape((64,64)))
f.add_subplot(1,2, 2)

plt.title("PCA compressed")
plt.imshow(X_recovered[0].reshape((64,64)))
plt.show(block=True)


# In[19]:


# now again lets check the accuracy
clf_reduced = MLPClassifier(solver='adam',alpha=1e-5,hidden_layer_sizes=(20,20,20))
clf_reduced.fit(X_reduced,y_train)


# In[20]:


X_test_reduced = pca.transform(X_test)
y_hat_reduced = clf_reduced.predict(X_test_reduced)
print('accuracy:'+str(accuracy_score(y_hat,y_hat_reduced)))


# In[ ]:


as you can see the accuracy little bit up after reductin the dimension

