#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


data = pd.read_csv(r"E:\tennis.csv")


# In[11]:


data.head(16)


# In[20]:


from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


# In[21]:


data_train = pd.read_csv(r"E:\Assignemnt\assets\play_tennis_train.csv")
data_test = pd.read_csv(r"E:\Assignemnt\assets\play_tennis_test.csv")


# In[22]:


le = preprocessing.LabelEncoder()
data_train_df = pd.DataFrame(data_train)
data_train_df_encoded = data_train_df.apply(le.fit_transform)

data_test_df = pd.DataFrame(data_test)
data_test_df_encoded = data_test_df.apply(le.fit_transform)


# In[23]:


x_train = data_train_df_encoded.drop(['play'],axis=1)
y_train = data_train_df_encoded['play']

x_test = data_test_df_encoded.drop(['play'],axis=1)
y_test = data_test_df_encoded['play']


# In[24]:


model = GaussianNB()
nbtrain = model.fit(x_train, y_train)

y_pred = nbtrain.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[25]:


disp = plot_confusion_matrix(nbtrain, x_test, y_test,
                                 display_labels=['No','Yes'],
                                 cmap=plt.cm.Blues)
disp.ax_.set_title('Confusion Matrix')

print('Confusion Matrix')
print(disp.confusion_matrix)

plt.show()
# confusion_


# In[26]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, plot_confusion_matrix


# In[ ]:




