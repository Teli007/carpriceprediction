#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import pickle


# In[2]:


A=pd.read_csv("car data.csv")


# In[3]:


A.head()


# In[4]:


A.info()


# In[5]:


A.isnull().sum()


# In[6]:


A.describe()


# In[7]:


A.columns


# In[8]:


print(A["Fuel_Type"].value_counts())
print("*****************************")
print(A["Seller_Type"].value_counts())
print("*****************************")
print(A["Transmission"].value_counts())
print("*****************************")
print(A["Owner"].value_counts())


# In[9]:


A.corr()


# In[10]:


A=A.drop(columns=["Car_Name"])


# In[11]:


A["current_year"]=2020


# In[12]:


A["no_years"]=A["current_year"]-A["Year"]


# In[13]:


A=A.drop(columns=["Year"])


# In[14]:


A.head()


# In[19]:


A=pd.get_dummies(A,drop_first=True)
A.head()


# In[20]:


x=A.drop(columns=["Selling_Price"])
y=A.Selling_Price


# In[21]:


from sklearn.ensemble import RandomForestRegressor
Rr=RandomForestRegressor(max_depth=25)
model=Rr.fit(x,y)


# In[22]:


filename = 'car_price_pridiction.pkl'
pickle.dump(model, open(filename, 'wb'))


# In[ ]:





# In[ ]:





# In[ ]:




