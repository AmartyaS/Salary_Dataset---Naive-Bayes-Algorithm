# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 17:34:05 2021

@author: ASUS
"""

#Importing all the neccessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

#Loading the data-file
train = pd.read_csv(r"D:\Data Science Assignments\R-Assignment\Naive_Bayes\SalaryData_Train.csv")
test=pd.read_csv(r"D:\Data Science Assignments\R-Assignment\Naive_Bayes\SalaryData_Test.csv")

#Data Exploration and manipulation
train.columns
train['workclass'].unique()
train['education'].unique()
train['maritalstatus'].unique()
train['occupation'].unique()
train['relationship'].unique()
train['race'].unique()
train['sex'].unique()
train['native'].unique()
#One hot en-coding the categorical data
train=pd.get_dummies(train,columns=["workclass"],prefix=["WC"])
test=pd.get_dummies(test,columns=["workclass"],prefix=["WC"])
train=pd.get_dummies(train,columns=["education"],prefix=["ED"])
test=pd.get_dummies(test,columns=["education"],prefix=["ED"])
train=pd.get_dummies(train,columns=["maritalstatus"],prefix=["Stat_"])
test=pd.get_dummies(test,columns=["maritalstatus"],prefix=["Stat_"])
train=pd.get_dummies(train,columns=["occupation"],prefix=["occupation"])
test=pd.get_dummies(test,columns=["occupation"],prefix=["occupation"])
train=pd.get_dummies(train,columns=["relationship"],prefix=["Rel_"])
test=pd.get_dummies(test,columns=["relationship"],prefix=["Rel_"])
train=pd.get_dummies(train,columns=["race"],prefix=["R_"])
test=pd.get_dummies(test,columns=["race"],prefix=["R_"])
train=pd.get_dummies(train,columns=["sex"],prefix=["Sex"])
test=pd.get_dummies(test,columns=["sex"],prefix=["Sex"])
train=pd.get_dummies(train,columns=["native"],prefix=["_"])
test=pd.get_dummies(test,columns=["native"],prefix=["_"])

#Defining the training and testing dataset
x_train=train[train.columns.difference(["Salary"])]
y_train=pd.DataFrame(train["Salary"])
x_test=test[test.columns.difference(["Salary"])]
y_test=pd.DataFrame(test["Salary"])

#Making of Naive-Bayes Model
ignb=GaussianNB()
imnb=MultinomialNB()

#Predicting the values based on the models formed
pred_gnb=ignb.fit(x_train,y_train).predict(x_test)
pred_mnb=imnb.fit(x_train,y_train).predict(x_test)

#Visualizing the accuracy with the help of confusion matrix
confusion_matrix(y_test,pred_gnb)
confusion_matrix(y_test,pred_mnb)
pd.crosstab(y_test.values.flatten(),pred_gnb)
pd.crosstab(y_test.values.flatten(),pred_mnb)

#Calculating the accuracy of the Gaussian Naive-Bayes Model
np.mean(y_test.values.flatten()==pred_gnb)
#Calculating the accuracy of the Multinomial Naive-Bayes Model
np.mean(y_test.values.flatten()==pred_mnb)
