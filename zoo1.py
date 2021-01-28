#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib,pylab as plt


# In[3]:


z=pd.read_csv("zoo.csv")
z.head()


# In[4]:


df=pd.DataFrame(z)


# In[5]:


row_indexes=df[df['type']==1].index


# In[6]:


df.loc[row_indexes,'Type']="mammal"


# In[7]:


df.head()


# In[9]:


row_indexes=df[df['type']==2].index
df.loc[row_indexes,'Type']="bird"


# In[11]:


row_indexes=df[df['type']==3].index
df.loc[row_indexes,'Type']="reptile"


# In[13]:


row_indexes=df[df['type']==4].index
df.loc[row_indexes,'Type']="fish"


# In[15]:


row_indexes=df[df['type']==5].index
df.loc[row_indexes,'Type']="amphibian"


# In[17]:


row_indexes=df[df['type']==6].index
df.loc[row_indexes,'Type']="insect"


# In[19]:


row_indexes=df[df['type']==7].index
df.loc[row_indexes,'Type']="crustacean"


# In[20]:


df.head()


# In[27]:


#Rename colunm name 
df.rename(columns={"animal name":"animalname"},inplace=True)

#drop type and animalname column


# In[32]:


df.head()


# ### Training and testing

# In[33]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNC
train,test = train_test_split(df,test_size = 0.2)


# In[34]:


neigh = KNC(n_neighbors= 10)


# In[35]:


neigh.fit(train.iloc[:,0:16],train.iloc[:,16])


# In[36]:


train_acc = np.mean(neigh.predict(train.iloc[:,0:16])==train.iloc[:,16])


# In[37]:


test_acc = np.mean(neigh.predict(test.iloc[:,0:16])==test.iloc[:,16])
test_acc


# ###### Accuracy achieved as choosing n=10 is 71.42% 

# ##### Considering neighbours,n=4

# In[38]:


neigh = KNC(n_neighbors= 4)


# In[39]:


neigh.fit(train.iloc[:,0:16],train.iloc[:,16])


# In[40]:


train_acc = np.mean(neigh.predict(train.iloc[:,0:16])==train.iloc[:,16])


# In[43]:


test_acc = np.mean(neigh.predict(test.iloc[:,0:16])==test.iloc[:,16])
test_acc


# In[49]:


acc = []
for i in range(4,50,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,0:16],train.iloc[:,16])
    train_acc = np.mean(neigh.predict(train.iloc[:,0:16])==train.iloc[:,16])
    test_acc = np.mean(neigh.predict(test.iloc[:,0:16])==test.iloc[:,16])
    acc.append([train_acc,test_acc])


# In[50]:


plt.plot(np.arange(4,50,2),[i[0] for i in acc],"ro-")


# In[46]:


plt.plot(np.arange(4,50,2),[i[1] for i in acc],"bo-")


# In[ ]:




