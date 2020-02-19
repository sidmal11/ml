#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pickle
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.neighbors import KNeighborsClassifier



# In[3]:


def create_knn(fold_num,max_neighbors):
    train_X,train_y = read_train_csv(fold_num)
    for num in range(0,max_neighbors):
        name_1= "knn/knn_model_"+str(fold_num)+"k="+str(num)+".txt"
        knn = KNeighborsClassifier(n_neighbors = num+1)
        knn.fit(train_X, train_y)
        with open(name_1, 'wb') as f1:
            pickle.dump(knn, f1)
        f1.close()


    # knn = KNeighborsClassifier(n_neighbors=n)
    # knn.fit(train_X,train_y)
    # name_1= "knn/knn_model_"+str(fold_num)+"n="+str()+".txt"
    # with open(name_1, 'wb') as f1:
    #     pickle.dump(knn, f1)
    # f1.close()


# In[1]:


def create_dtc(fold_num):
    train_X,train_y = read_train_csv(fold_num)
    dtc = DecisionTreeClassifier()
    dtc.fit(train_X, train_y)
    name_2 = "dtc/dtc_model_"+str(fold_num)+".txt"
    with open(name_2, 'wb') as f2:
        pickle.dump(dtc, f2)
    f2.close()


# In[ ]:
def read_train_csv(num):
    train_X = np.genfromtxt('train_csv/train_X_'+str(num)+'.csv',delimiter=',')
    train_y = np.genfromtxt('train_csv/train_y_'+str(num)+'.csv',delimiter=',')
    return train_X,train_y





# In[ ]:





# In[ ]:




