# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 14:36:46 2018

@author: CHANDU
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import skew,mode
os.chdir('C:\\Users\\Chandu\\Desktop\\IMAR DATA\\loan prediction')
loan_train= pd.read_csv('train_u6lujuX_CVtuZ9i.csv')
loan_test= pd.read_csv('test_Y3wMUE5_7gLdaTN.csv')

summary= loan_train.describe()
loan_train.info()


loan_train.columns
#['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
#       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
#       'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status']
#-------------------------------
def missing(x):
    return sum(x.isnull())

def counts(x):
    return x.value_counts()

loan_train.apply(missing,axis=0)

#---------gender
counts(loan_train['Gender'])
loan_train['Gender']= loan_train['Gender'].fillna('Male')

#---------------married
counts(loan_train['Married'])
loan_train['Married']= loan_train['Married'].fillna('Yes')

#-----------------dependents------------------
counts(loan_train['Dependents'])
loan_train['new Dependents']= np.where(loan_train['Dependents'].isnull(),
                                    '0', loan_train['Dependents'])

counts(loan_train['new Dependents'])
loan_train['new Dependents'].replace('3+','3',inplace=True)
loan_train['Dependents']= loan_train['new Dependents']

loan_train.drop('new Dependents',axis=1,inplace= True)
counts(loan_train['Dependents'])
#--------------------------self employed-------------------
counts(loan_train['Self_Employed'])
loan_train['Self_Employed']= loan_train['Self_Employed'].fillna('No')


#----------------loan amount------------------
counts(loan_train['LoanAmount'])
loan_train['LoanAmount'].median()
loan_train['LoanAmount']= loan_train['LoanAmount'].fillna(128)

#-------------------------loan amount term----------------------
counts(loan_train['Loan_Amount_Term'])
loan_train['Loan_Amount_Term'].median()
loan_train['Loan_Amount_Term']= loan_train['Loan_Amount_Term'].fillna(360)


#----------------------credit history
counts(loan_train['Credit_History'])
loan_train['Credit_History']= loan_train['Credit_History'].fillna(1)


#---------------------------------------------test-----------------------------
loan_test.apply(missing,axis=0)

counts(loan_test['Gender'])
loan_test['Gender']= loan_test['Gender'].fillna('Male')

#-----------------dependents------------------
counts(loan_test['Dependents'])
loan_test['Dependents']= np.where(loan_test['Dependents'].isnull(),
                                    '0', loan_test['Dependents'])

loan_test['Dependents'].replace('3+','3',inplace=True)

#--------------------------self employed-------------------
counts(loan_test['Self_Employed'])
loan_test['Self_Employed']= loan_test['Self_Employed'].fillna('No')


#----------------loan amount------------------
counts(loan_test['LoanAmount'])
loan_test['LoanAmount'].median()
loan_test['LoanAmount']= loan_test['LoanAmount'].fillna(125)

#-------------------------loan amount term----------------------
counts(loan_test['Loan_Amount_Term'])
loan_test['Loan_Amount_Term'].median()
loan_test['Loan_Amount_Term']= loan_test['Loan_Amount_Term'].fillna(360)


#----------------------credit history
counts(loan_test['Credit_History'])
loan_test['Credit_History']= loan_test['Credit_History'].fillna(1)

#-------------------------LABEL ENCODER--TRAIN-----------------------------------------
loan_train.info()

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()

#-----------gender
loan_train['Gender']= le.fit_transform(loan_train['Gender'])
counts(loan_train['Gender'])

#married
loan_train['Married']= le.fit_transform(loan_train['Married'])
counts(loan_train['Married'])

#Dependents
loan_train['Dependents']= le.fit_transform(loan_train['Dependents'])
counts(loan_train['Dependents'])

#---------eudcation-----------------
#ordina; labe encoder

# using github from sklearn.preprocessing import OrdinalEncoder 
loan_train['Education']= le.fit_transform(loan_train['Education'])
counts(loan_train['Education'])

#-------------self employed
loan_train['Self_Employed']= le.fit_transform(loan_train['Self_Employed'])
counts(loan_train['Self_Employed'])

#--------------property area
loan_train['Property_Area']= le.fit_transform(loan_train['Property_Area'])
counts(loan_train['Property_Area'])

#----------------------------loan status
loan_train['Loan_Status']= le.fit_transform(loan_train['Loan_Status'])
counts(loan_train['Loan_Status'])


#------------label encoder test------------------------------
loan_test.info()

var_mod=['Gender', 'Married', 'Dependents', 'Education','Self_Employed', 
          'Property_Area',]
le= LabelEncoder()

for i in var_mod:
    loan_test[i]=le.fit_transform(loan_test[i])
    
#----------------model 0 no 1 yes------------------------------------------

X=loan_train.iloc[:,2:12]
Y=loan_train['Loan_Status']

#----------------data splitting
from sklearn.model_selection import train_test_split
x_test,x_train,y_test,y_train= train_test_split(X,Y,test_size= 0.2)

#----------------models-------------------------------------------
from sklearn import linear_model
lm= linear_model.LogisticRegression()

model_lm_x_train= lm.fit(x_train,y_train)
predicted_lm_x_train= model_lm_x_train.predict(x_train)
confusion_matrix_x_train= confusion_matrix(y_train,predicted_lm_x_train)
accuracy_matrix_x_train= accuracy_score(y_train,predicted_lm_x_train)
print(accuracy_matrix_x_train)

#-------------------------------------random forest------------------------------
from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier(n_estimators=1000)
model_rf_x_train= rf.fit(x_train,y_train)
predicted_r_x_trainf= model_rf_x_train.predict(x_train)
confusion_rf_x_train= confusion_matrix(y_train,predicted_r_x_trainf)
accuracy_rf_x_train= accuracy_score(y_train,predicted_r_x_trainf)
print(accuracy_rf_x_train)

predicted_rf_x_test= model_rf_x_train.predict(x_test)
confusion_rf_x_test= confusion_matrix(y_test,predicted_rf_x_test)
accuracy_rf_x_test= accuracy_score(y_test,predicted_rf_x_test)
print(accuracy_rf_x_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#-----------------------svm------------------------------------------
from sklearn import svm
sv= svm.SVC(kernel='rbf')

model_sv= sv.fit(x_train,y_train)
predicted_sv_x_train= model_sv.predict(x_train)
confusion_x_train= confusion_matrix(y_train,predicted_sv_x_train)
accurcay_x_train_score= accuracy_score(y_train,predicted_sv_x_train)
print(accurcay_x_train_score)

predicted_sv_x_test= model_sv.predict(x_test)
confusion_x_test= confusion_matrix(y_test,predicted_sv_x_test)
accurcay_x_test_score= accuracy_score(y_test,predicted_sv_x_test)
print(accurcay_x_test_score)




#------------------------------------MLP------------------------------
from sklearn.neural_network import MLPClassifier
MLC= MLPClassifier(activation='relu',solver= 'adam',
                  max_iter=200, 
                  hidden_layer_sizes=(100,100,100))
model_mlc= MLC.fit(x_train,y_train)
predicted_mlc_x_train= MLC.predict(x_train)
confusing_matrix_x_train= confusion_matrix(y_train,predicted_mlc_x_train)
accuracy_x_train= accuracy_score(y_train,predicted_mlc_x_train)
print(accuracy_x_train)


#----------------------test
loan_test.info()
loan_test.drop('Loan_ID',axis=1,inplace=True)
loan_test.drop('index',axis=1,inplace= True)
X_TEST= loan_train.iloc[:,1:11]
predict= model_rf_x_train.predict(X_TEST)
a= pd.DataFrame(predict)
a.to_csv('out.csv')


#------------------------kmeans-------------------------------
X.columns
X.describe()

X_continous= x_train[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term']]

from sklearn.cluster import KMeans
model= KMeans(n_clusters=2, random_state= 123)

model.fit(X_continous)

predicted_model= model.predict(X_continous)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

matrix= confusion_matrix(y_train, predicted_model)
print(matrix)
Accuracy_model= accuracy_score(y_train, predicted_model)
print(Accuracy_model)


# to load lod_session(finename)
import dill
filename = 'LOAN.pkl'
dill.dump_session(filename)









