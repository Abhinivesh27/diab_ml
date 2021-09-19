#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('https://raw.githubusercontent.com/KamaleshKarthi14/Diabetes/main/diabetes.csv')


# In[3]:


df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


from sklearn.model_selection import train_test_split 


# In[10]:


x = df.iloc[:,df.columns != 'Outcome']
y = df.iloc[:,df.columns == 'Outcome']


# In[11]:


xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.4)


# In[12]:


from sklearn.ensemble import RandomForestClassifier


# In[14]:


model = RandomForestClassifier()


# In[15]:


model.fit(xtrain,ytrain.values.ravel())


# In[16]:


predict_output = model.predict(xtest)


# In[17]:


from sklearn.metrics import accuracy_score


# In[18]:


ac = accuracy_score(predict_output,ytest)


# In[20]:


print("Accuracy of this model: ",ac)


# In[ ]:




