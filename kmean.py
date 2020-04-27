#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


file=open('EastWestAirlinesCluster.csv')


# In[3]:


def juli(a,b): #定义数据之间距离（这里用欧式距离）
    return np.sqrt(np.sum((a-b)**2))


# In[4]:


def chushi(d,k): #随机取初始k个质心
    sj=[]
    for i in range(k):
        sj.append(d[int(len(d)*np.random.rand()//1)])
    return np.array(sj)


# In[5]:


data=[]   #将文件内容储存为列表
for i in file.readlines():
    data.append(i)


# In[6]:


dataset=[]  #将列表中的内容转为float
for i in data[1:]:
    dataset.append(list(map(float,i.strip().split(','))))


# In[7]:


dataset=np.array(dataset)  #将列表转为数组


# In[8]:


dataset=dataset[:,1:-1] #将数据集中的第一列和最后一列去掉


# In[9]:


def Kmean(dataset,k,num):
    zhixin=chushi(dataset,k) #储存每个质心的坐标
    m=len(dataset)
    n=len(dataset[0])
    xinxi=np.zeros((m,2))  #储存每条数据属于哪个簇以及数据到质心的 距离
    index=0
    while index<num:
        index=index+1
        for i in range(m):
            mindis=np.inf
            mininx=-1
            for j in range(k):
                dis=juli(dataset[i],zhixin[j])
                if dis<mindis:
                    mindis=dis
                    mininx=j
            xinxi[i]=mininx,mindis
        for i in range(k):
            fz=np.nonzero(xinxi[:,0]==i)[0]
            zhixin[i]=np.mean(dataset[fz],axis=0)
    return zhixin,xinxi


# In[10]:


z,x=Kmean(dataset,7,500)


# In[ ]:




