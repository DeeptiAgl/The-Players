
# coding: utf-8

# In[107]:

import pandas as pd
import numpy as np


# In[108]:

df=pd.read_csv("/home/rsingla1/project_data_latest.csv")
df1=pd.read_csv("/home/rsingla1/project_data_latest.csv")




# In[109]:

bins=[-1,3,6,9,11]
group_names=['0','1','2','3']
df['performance']=pd.cut(df['playoff_wins'],bins,labels=group_names)


# In[110]:

df1.columns


# In[245]:

#df2=df[['playoff_wins','performance']]

df1.drop(["Tm"], axis=1, inplace=True)
df2=df[['performance']]

#df3=df1["performance",{"excellent":4,"good":3,"average":2,"low":1}]
#df3
#df['performance']=df['performance'].convert_objects(convert_numeric=True).df3


# In[352]:

X,y=df1,df2
b=np.array(y)
c=b.ravel()
df3=df1.as_matrix()
df3


# In[116]:

#X_new=SelectKBest(chi2, k=2).fit_transform(X, c)


# In[244]:

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
name=df1.columns.values
name


# In[259]:

#use linear regression as the model
lr = LinearRegression()
#rank all features, i.e continue the elimination until the last one
rfe = RFE(lr, n_features_to_select=1)
rfe.fit(df3,c)
print ("Features sorted by their rank:")
print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_),name)))


# In[263]:

# Feature Importance
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
# load the iris datasets

# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(df3,c)
# display the relative importance of each attribute
print(model.feature_importances_)


# In[271]:

#Applying K- Nearest Neighbour Algorithm on our top 10 features
from sklearn.neighbors import KNeighborsClassifier


# In[265]:

knn=KNeighborsClassifier(n_neighbors=1)


# In[267]:

print(knn)


# In[269]:

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df3,c,test_size=0.4)


# In[292]:

from sklearn import metrics
knn=KNeighborsClassifier(n_neighbors=13)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
print(metrics.accuracy_score(y_test,y_pred))


# In[288]:

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)
print(metrics.accuracy_score(y_test,y_pred))


# In[290]:

k_range=range(1,26)
scores=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_test)
    scores.append(metrics.accuracy_score(y_test,y_pred))
    
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.plot(k_range,scores)
plt.xlabel('value of k for knn')
plt.ylabel('Testing accuracy')


# In[349]:

## from sklearn import metrics
from sklearn.cross_validation import cross_val_score
knn=KNeighborsClassifier(n_neighbors=5)
print (cross_val_score(knn,df3,c,cv=10,scoring="accuracy").mean())


# In[302]:

k_range=range(1,120)
k_scores=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(knn,df3,c,cv=10,scoring="accuracy")   
    k_scores.append(scores.mean())
    
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.plot(k_range,k_scores)
plt.xlabel('value of k for knn')
plt.ylabel('Testing accuracy')


# In[ ]:



