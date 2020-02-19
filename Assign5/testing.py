#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np


# In[7]:


def run_knn(fold_num,max_neighbors):
    test_X,test_y = read_test_csv(fold_num)
    accuracies = np.zeros(max_neighbors)
    accuracies = accuracies.astype(np.float)
    for num in range(0,max_neighbors):
        name_1 = "knn/knn_model_"+str(fold_num)+"k="+str(num)+".txt"
        with open(name_1, 'rb') as f3:
            loaded_model = pickle.load(f3)
        f3.close()
        pred_y = loaded_model.predict(test_X)
        accuracy =  accuracy_score(test_y,pred_y)
        accuracies[num] = accuracy.astype(np.float64) 
        file_object = open('accuracy/accuracy_knn.txt', 'a')
        file_object.write("Accuracy For knn for fold " +str(fold_num) +" and k = "+ str(num+1) +" is : "+ str(accuracy) + "\n")
        file_object.close()
    return accuracies


# In[8]:


def run_dtc(fold_num):
    test_X,test_y = read_test_csv(fold_num)
    name_2 = "dtc/dtc_model_"+str(fold_num)+".txt"
    with open(name_2, 'rb') as f4:
        loaded_model = pickle.load(f4) 
    f4.close()
    pred_y = loaded_model.predict(test_X)
    accuracy = accuracy_score(test_y, pred_y)
    file_object = open('accuracy/accuracy_dtc.txt', 'a')
    file_object.write("Accuracy For dtc for fold " +str(fold_num) +" is : "+ str(accuracy) + "\n")
    file_object.close()
    return accuracy 


# In[ ]:
def read_test_csv(num):
    test_X = np.genfromtxt('test_csv/test_X_'+str(num)+'.csv',delimiter=',')
    test_y = np.genfromtxt('test_csv/test_y_'+str(num)+'.csv',delimiter=',')
    return test_X,test_y





