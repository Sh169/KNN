#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNC
import warnings 
warnings.filterwarnings("ignore")


# In[37]:


zoo=pd.read_csv("zoo.csv")


# In[40]:


train,test=train_test_split(zoo,test_size=0.2)

neigh=KNC(n_neighbors=3)
neigh.fit(train.iloc[:,1:17],train.iloc[:,17])


# In[41]:


#To find train accuracy

train_acc=np.mean(neigh.predict(train.iloc[:,1:17])==train.iloc[:,17])


# In[42]:


#To find test accuracy

test_acc=np.mean(neigh.predict(test.iloc[:,1:17])==test.iloc[:,17])


# In[43]:


#Similarly for n_neighbors=7

neigh=KNC(n_neighbors=7)
neigh.fit(train.iloc[:,1:17],train.iloc[:,17])


# In[44]:


#To find train accuracy

train_acc=np.mean(neigh.predict(train.iloc[:,1:17])==train.iloc[:,17])


# In[45]:


#To find test accuracy

test_acc=np.mean(neigh.predict(test.iloc[:,1:17])==test.iloc[:,17])


# In[46]:


acc=[]

for i in range(3,50,2):
    neigh=KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,1:17],train.iloc[:,17])
    train_acc=np.mean(neigh.predict(train.iloc[:,1:17])==train.iloc[:,17])
    test_acc=np.mean(neigh.predict(test.iloc[:,1:17])==test.iloc[:,17])
    acc.append([train_acc,test_acc])
    


# In[47]:


import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")


# In[48]:


# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")


# ### OR

# In[49]:


X=zoo.iloc[:,1:17]
y=zoo.iloc[:,17]


# In[50]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

from sklearn.preprocessing import scale

X_train=scale(X_train)
X_test=scale(X_test)


# In[51]:


neigh=KNC(n_neighbors=7)
neigh.fit(X_train,y_train)


# In[52]:


y_pred=neigh.predict(X_test)


# In[53]:


from sklearn.metrics import confusion_matrix , classification_report


# In[54]:


print(confusion_matrix(y_test,y_pred))
pd.crosstab(y_test.values.flatten(),y_pred)


# In[55]:


print(classification_report(y_test,y_pred))

