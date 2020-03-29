#!/usr/bin/env python
# coding: utf-8

# In[1]:


from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import Dataset
import os
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from surprise import KNNBasic
from surprise import NMF
import surprise.accuracy
import pandas as pd
import numpy as np


# In[2]:


#load data from a file 
file_path = os.path.expanduser('restaurant_ratings.txt')
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader)


# In[3]:


algo = SVD()
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)


# In[4]:


# PMF
algo2 = SVD(biased=False)
cross_validate(algo2, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)


# In[ ]:





# In[5]:


algo3 = NMF()
cross_validate(algo3, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)


# In[ ]:





# In[6]:


# User based
algo4 = KNNBasic(sim_options = {'user_based': True })
cross_validate(algo4, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)


# In[7]:


# Item based
algo5 = KNNBasic(sim_options = {'user_based': False })
cross_validate(algo5, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)


# In[ ]:





# In[8]:


kf = KFold(n_splits=3)


# for  i in range(4):
#     exec(f'RMSE_{i}=[]')
#     exec(f'MAE_{i}=[]')
RMSE = []
MAE = []


algo_list = []
algo_list.append(SVD())
algo_list.append(SVD(biased=False))
algo_list.append(NMF())
algo_list.append(KNNBasic(sim_options = {'user_based': True}))
algo_list.append(KNNBasic(sim_options = {'user_based': False}))


# In[9]:


i = 0
for trainset, testset in kf.split(data):
    i += 1
    # train and test algorithm.
    
    for algo in algo_list:
        algo.fit(trainset)
        predictions = algo.test(testset)
        
        RMSE.append(surprise.accuracy.rmse(predictions, verbose=False))
        MAE.append(surprise.accuracy.mae(predictions, verbose=False))


# In[63]:


RMSE_mat = np.asarray(RMSE)
MAE_mat = np.asarray(MAE)
RMSE_mat = RMSE_mat.reshape((3,5))
MAE_mat = MAE_mat.reshape((3,5))

algolist=["  SVD  ", "  PMF  ", "  NMF  ","User based","Item based"]


# print(MAE_mat)
for i in range(3):
    print("\n Rmse for fold : "+str(i+1)+" = ")
    print(algolist)
    [print(RMSE_mat[i])]
    print("\n Mae for fold : "+str(i+1)+" = ")
    print(algolist)

    print(RMSE_mat[i])

avg_RMSE = np.mean(RMSE_mat, axis=0)
avg_MAE = np.mean(MAE_mat, axis=0)


print()
print("              ",algolist)
print ("Average RSME is :",avg_RMSE)
print ("Average MAE is  :",avg_MAE)


# In[11]:


metrics =['MSD','cosine','pearson']   
for metric in metrics:
    algo = KNNBasic(sim_options = {'name':metric,'user_based': True })
    exec(f'perf_user_{metric} = cross_validate(algo, data, measures=["RMSE","MAE"], cv=3, verbose=False)')
    algo2 = KNNBasic(sim_options = {'name':metric,'user_based': False })
    exec(f'perf_item_{metric} = cross_validate(algo, data, measures=["RMSE","MAE"], cv=3, verbose=False)')


# In[12]:


print(perf_user_MSD)
import seaborn as sns
import matplotlib.pyplot as plt


# In[13]:


metrics = np.array(['MSD', 'cosine', 'pearson'])
def fun(err):
    vals = [np.mean(perf_user_MSD['test_{}'.format(err)]), np.mean(perf_user_cosine['test_{}'.format(err)]), np.mean(perf_user_pearson['test_{}'.format(err)])]
    series = pd.Series(name='{}'.format(err), data=vals)
    ax = sns.barplot(metrics, series.values)
    ax.set_title("Comparision of metric:{} on User Based".format(err))
    ax.set_ylabel('{} values'.format(err))
    ax.set_xlabel('Metrics')
    ax.set_ylim(.7, 1.1)
    plt.show()
    print(vals)
    
    vals = [np.mean(perf_item_MSD['test_{}'.format(err)]), np.mean(perf_item_cosine['test_{}'.format(err)]), np.mean(perf_item_pearson['test_{}'.format(err)])]
    series = pd.Series(name='{}'.format(err), data=vals)
    ax = sns.barplot(metrics, series.values)
    ax.set_title("Comparision of metric:{} on Item Based ".format(err))
    ax.set_ylabel('{} values'.format(err))
    ax.set_xlabel('Metrics')
    ax.set_ylim(.7, 1.1)
    plt.show()
    print(vals)
    
error= ['mae','rmse']
for err in error:
    fun(err)


# In[31]:


user_Based_RMSE_of_different_k = []
item_Based_RMSE_of_different_k = []
for i in range(1,40):
    algo = KNNBasic(k=i, sim_options = {
            'name': 'MSD',
            'user_based': True 
            })
    algo2 =KNNBasic(k=i, sim_options = {
            'name': 'MSD',
            'user_based': False 
            })
      
    
    perf_UserBased_MSD = cross_validate(algo, data, measures=['RMSE'],cv=3, verbose=True)
    perf_ItemBased_MSD = cross_validate(algo2, data, measures=['RMSE'],cv=3, verbose=True)
    
    print("________________________________")
    print("K= ", i)
    user_Based_RMSE_of_different_k.append(np.mean(perf_UserBased_MSD['test_rmse']))
    item_Based_RMSE_of_different_k.append(np.mean(perf_ItemBased_MSD['test_rmse']))    


# In[53]:


based = ['user','item']
def fun2(i,j):
    us = [user_Based_RMSE_of_different_k[1:j],item_Based_RMSE_of_different_k[1:j]]
    min_RMSE_index = np.argmin(us[i])
    print("Best K= ", min_RMSE_index)
    print("Best RMSE= ", us[i][min_RMSE_index])

    series = pd.Series(name='rmse', data=us[i])

    ax = sns.barplot(series.index, series.values)
    ax.set_ylabel('RMSE values')
    ax.set_xlabel('Number of K')
    ax.set_xlim(0, j)
    if i==0:
        ax.set_title("User based")
    else:
        ax.set_title("Item based")
        
    ax.set_ylim(.7, 1.2)
    plt.show()

for i,base in enumerate(based):
    fun2(i,20)


# In[54]:


for i,base in enumerate(based):
    fun2(i,30)


# In[55]:


for i,base in enumerate(based):
    fun2(i,40)


# In[ ]:




