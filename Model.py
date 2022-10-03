import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

dataset=pd.read_csv("C:\\Users\\SowmyaEppalpalli\\Downloads\\insurance.csv")

dataset['sex']=dataset['sex'].map({'female':0,'male':1})
dataset['smoker']=dataset['smoker'].map({'yes':0,'no':1})
data= pd.get_dummies(dataset['region'],drop_first = True)
dataset=pd.concat([dataset,data],axis=1)
dataset.drop('region',axis=1,inplace=True)

X=dataset.drop(['charges'],axis=1)
y=dataset['charges']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)

from sklearn.linear_model import LinearRegression
lg=LinearRegression()
lg.fit(X_train,y_train)

pickle.dump(lg,open("Model.pkl","wb"))
model=pickle.load(open("Model.pkl","rb"))





