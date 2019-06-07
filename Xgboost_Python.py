import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import *
from sklearn.grid_search import GridSearchCV
import time
from scipy.stats import ks_2samp
import sklearn.metrics as mt
from matplotlib import pyplot as plt
from xgboost import plot_importance
import seaborn as sns



data = pd.read_csv("train.csv")

data["Sex_male"]=np.where(data["Sex"]=="male",0,1)
data["Sex_female"]=np.where(data["Sex"]=="female",0,1)
data["Embarked_C"] = pd.get_dummies(data.Embarked).C
data["Embarked_Q"] = pd.get_dummies(data.Embarked).Q
data["Embarked_S"] = pd.get_dummies(data.Embarked).S


data["Pclass"].fillna(data["Pclass"].mode(), inplace=True)
data["Sex_male"].fillna(data["Sex_male"].mode(), inplace=True)
data["Sex_female"].fillna(data["Sex_female"].mode(), inplace=True)
data["Age"].fillna(data["Age"].mean(), inplace=True)
data["SibSp"].fillna(data["SibSp"].mean(), inplace=True)


data=data[[
    "Survived",
    "Pclass",
    "Sex_female",
    "Sex_male",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked_C",
    "Embarked_Q",
    "Embarked_S"
]].dropna(axis=0, how='any')


X_train, X_test = train_test_split(data, test_size=0.4, random_state=int(time.time()))


features = list(X_train.columns[1:])

print(features)

parameters = {'nthread':[1,3,6], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': [0.05,0.1,0.3], #so called `eta` value
              'max_depth': [6,9,15],
              'min_child_weight': [11,15,20],
              'silent': [1],
              'subsample': [0.5,0.8],
              'colsample_bytree': [0.7,0.5],
              'n_estimators': [1000], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed': [1337]}


xgb_model = xgb.XGBClassifier()

clf = GridSearchCV(xgb_model, parameters, n_jobs=4, 
                   cv=StratifiedKFold(X_train['Survived'], n_folds=5, shuffle=True), 
                   scoring='roc_auc',
                   verbose=2, refit=True)


clf.fit(X_train[features], X_train["Survived"])

clf.best_estimator_

best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
print('Raw AUC score:', score)
for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))
    
    
 X_test["test_probs"] = clf.predict_proba(X_test[features])[:,1]
 
 
df_label  = X_test.loc[X_test['Survived'] == 1]
df_X  = X_test.loc[X_test['Survived'] == 0]

np.random.seed(12345678)

AUC = mt.roc_auc_score(X_test['Survived'],  X_test["test_probs"])

print("Area roc na base de teste é igual a: ",round(AUC,2))


ks = ks_2samp(df_label.test_probs,df_X.test_probs)

print("KS na base de teste é igual a:",round(ks[0],2))

ax = sns.boxplot(x="Survived", y="test_probs", data=df_label, order = [0,1],color = "red")
ax = sns.boxplot(x="Survived", y="test_probs", data=df_X,order = [0,1],color = "blue")


sns.distplot(df_label.test_probs, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1}, 
                  label = "Survived")

sns.distplot((df_X.test_probs), hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1}, 
                  label = "Dead")

plot_importance(clf.best_estimator_)
