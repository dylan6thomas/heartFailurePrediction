'''From Heart Failure Prediction dataset by Larxel on Kaggle'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras

ds=pd.read_csv('HeartFailurePrediction.csv')
x=ds.iloc[:,:-1].values
y=ds.iloc[:,-1].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.transform(x_test)

parameters=[{'C':[.001,.003,.01,.03,.1,.3,1,3,10,30,100,300,1000],'kernel':['rbf','poly','sigmoid'],'degree':[1,2,3,4,5,6,7,8,9]}]
model=GridSearchCV(SVC(),parameters,scoring='accuracy',verbose=2,cv=10,n_jobs=1)
model.fit(x_train,y_train)
pred=model.predict(x_test)
print(model.best_params_)
print(model.best_score_)
print(accuracy_score(y_test,pred))
print(confusion_matrix(y_test,pred))

parameters=[{'n_estimators':[3,10,30,100,300]}]
model=GridSearchCV(RandomForestClassifier(),parameters,scoring='accuracy',verbose=2,cv=10,n_jobs=1)
model.fit(x_train,y_train)
pred=model.predict(x_test)
print(model.best_params_)
print(model.best_score_)
print(accuracy_score(y_test,pred))
print(confusion_matrix(y_test,pred))

model=keras.models.Sequential([
    keras.layers.Dense(32,activation='relu'),
    keras.layers.Dense(16,activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')
    ])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10)
pred=model.predict(x_test).round()
print(accuracy_score(y_test,pred))
print(confusion_matrix(y_test,pred))