import pandas as pd
import numpy as np
import xgboost as xgb
import itertools
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import *
from sklearn.model_selection import GridSearchCV
import time
from scipy.stats import ks_2samp
import sklearn.metrics as mt
from matplotlib import pyplot as plt
from xgboost import plot_importance
import seaborn as sns
import pyodbc
from sklearn import preprocessing as preprocessing 
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, \
                            roc_curve, precision_recall_curve, auc, average_precision_score
    import pickle


# load data

server   = 'server_name'
database = 'database_name'
username = 'user_name'
password = 'user_password'
driver   = '{ODBC Driver 13 for SQL Server}'
connection_read = pyodbc.connect('DRIVER='+driver+';PORT=1433;SERVER='+server+';PORT=17001;DATABASE='+database+';UID='+username+';DSN_NAME='+ password)

connection_string= 'DSN=DSN_NAME;UID=user_name;DSN_NAME=user_password'
connection= pyodbc.connect(connection_string)

cursor = connection.cursor()
select_string= """ YOUT QUERY FOR IMPORT TABLE""" 

df = pd.read_sql(select_string,connection)

df_QA = df

## LOW VARIANCE

features = list(df_QA.columns[2:])

variable_names = []

Selector = VarianceThreshold(threshold = 0.07)
Selector.fit(df_QA[features])
 
for x in range(0, len(df_QA[features].columns)):
    if(Selector.get_support()[x] == False):
        variable_names.append(df_QA[features].columns[x])
        
        
   
df_QA = df_QA.drop(variable_names,axis = 1)

     
## CORREL FEATURES


features_col = list(df_QA.columns[2:])
    
variaveis = set()

correlation_matrix = df_QA[features_col].corr()

for i in range(correlation_matrix.shape[0]):
        for j in range(i+1):
            if( abs(correlation_matrix.iloc[i,j]) >= 0.9 and abs(correlation_matrix.iloc[i,j]) < 1 ):
                if(correlation_matrix.columns[j] not in variaveis):
                    variaveis.add(correlation_matrix.columns[j])



df_QA = df_QA.drop(variaveis,axis = 1)
            
     
# split train and test data

features_final = list(df_QA.columns[2:])


X_train, X_test = train_test_split(df_QA, test_size=0.3, random_state=int(time.time()))


## Balanceamento Amostal 1:1

df_target = X_train[df.TARGET == 1]
df_x = X_train[X_train.TARGET == 0].sample(len(df_target))
X_train = pd.concat([df_target,df_x],axis = 0)
X_train = X_train.sample(len(X_train))


# tune parameters

parameters = {'nthread':[3], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': [0.05,0.3], #so called `eta` value
              'max_depth': [6,15],
              'min_child_weight': [11,20],
              'silent': [1],
              'subsample': [0.5],
              'colsample_bytree': [0.5],
              'n_estimators': [100], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed': [1337]}


# xgboost

xgb_model = xgb.XGBClassifier()


# GridSearch  and kfolds

clf = GridSearchCV(xgb_model, parameters, n_jobs=1, 
                   cv=5, 
                   scoring='roc_auc',
                   verbose=1, refit=True)




clf.fit(X_train[features_final], X_train["TARGET"])



clf.best_estimator_


# best parameters

   
X_test['SCORE'] = clf.predict_proba(X_test[features_final])[:,1]
 
 
roc_auc_score(X_test['TARGET'],X_test['SCORE'])
# KS AND ROC
 
df_label  = X_test.loc[X_test['TARGET'] == 1]
df_X  = X_test.loc[X_test['TARGET'] == 0]

np.random.seed(12345678)

AUC = mt.roc_auc_score(X_test['TARGET'],  X_test["SCORE"])

print("Area roc na base de teste é igual a: ",round(AUC,2))


ks = ks_2samp(df_label.SCORE,df_X.SCORE)

print("KS na base de teste é igual a:",round(ks[0],2))


# Box Plot

ax = sns.boxplot(x="TARGET", y="SCORE", data=df_label, order = [0,1],color = "red")
ax = sns.boxplot(x="TARGET", y="SCORE", data=df_X,order = [0,1],color = "blue")


# Density 

sns.distplot(df_label.SCORE, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1}, 
                  label = "Readmitio")

sns.distplot((df_X.SCORE), hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1}, 
                  label = "Não Readmitio")


# features importance

1



### gerando os arquivos de scoragem


# Save preprocessing info in dictionary

# save the model to disk

filename = 'model_readmissao.sav'

pickle.dump(clf, open(filename, 'wb'))


