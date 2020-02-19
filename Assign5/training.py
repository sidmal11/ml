#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier



# In[3]:


def create_knn(train_X,train_y,fold_num,max_neighbors):
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


def create_dtc(train_X,train_y,fold_num):
    dtc = DecisionTreeClassifier()
    dtc.fit(train_X, train_y)
    name_2 = "dtc/dtc_model_"+str(fold_num)+".txt"
    with open(name_2, 'wb') as f2:
        pickle.dump(dtc, f2)
    f2.close()


# In[ ]:





# In[ ]:





# In[ ]:




