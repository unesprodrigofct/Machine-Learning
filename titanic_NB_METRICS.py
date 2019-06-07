
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
import sklearn.metrics as mt


# Importing dataset

data = pd.read_csv("train.csv")



#Sex

data["Sex"]=np.where(data["Sex"]=="male",0,1)

#Age

data["Embarked_C"] = pd.get_dummies(data.Embarked).C
data["Embarked_Q"] = pd.get_dummies(data.Embarked).Q
data["Embarked_S"] = pd.get_dummies(data.Embarked).S

print(data)

# View Columns 

data.columns


##input values

data["Pclass"].fillna(data["Pclass"].mode(), inplace=True)
data["Sex"].fillna(data["Sex"].mode(), inplace=True)
data["Age"].fillna(data["Age"].mean(), inplace=True)
data["SibSp"].fillna(data["SibSp"].mean(), inplace=True)


data=data[[
    "Survived",
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked_C",
    "Embarked_Q",
    "Embarked_S"
]].dropna(axis=0, how='any')


X_train, X_test = train_test_split(data, test_size=0.4, random_state=int(time.time()))


gnb = GaussianNB()

used_features =[
     "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked_C",
    "Embarked_Q",
    "Embarked_S"
    ]

gnb.fit(
    X_train[used_features].values,
    X_train["Survived"]
    )

y_predicao = gnb.predict(X_test[used_features])


label = np.asarray(X_test['Survived'])

mt.confusion_matrix(label,y_predicao)

tn, fp, fn, tp = mt.confusion_matrix(label,y_predicao).ravel()

mt.accuracy_score(label,y_predicao)

mt.cohen_kappa_score(label,y_predicao)

mt.average_precision_score(label,y_predicao)
