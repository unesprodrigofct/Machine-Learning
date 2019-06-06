# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:45:52 2019

@author: rsilva
"""

## Out of Time



import pandas as pd
import itertools
import numpy as np
import xgboost as xgb
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
username = 'username'
password = 'password'
driver   = '{ODBC Driver 13 for SQL Server}'
connection_read = pyodbc.connect('DRIVER='+driver+';PORT=1433;SERVER='+server+';PORT=17001;DATABASE='+database+';UID='+username+';DSN_NAME='+ password)

connection_string= 'DSN=DSN_NAME;UID=USERNAME;DSN_NAME=password'
connection= pyodbc.connect(connection_string)

cursor = connection.cursor()
select_string= """ YOUR QUERY FOR IMPORT TABLE
				MAKE OUT OF TIME VALIDATION""" 

df_out_time = pd.read_sql(select_string,connection)

## laod

filename = 'YOUR_model.sav'

loaded_model = pickle.load(open(filename, 'rb'))


features_final = ['import your final features(dev in traning model ']



df_out_time['SCORE'] = loaded_model.predict_proba(df_out_time[features_final])[:,1]
 
 
# KS AND ROC
 
df_label  = df_out_time.loc[df_out_time['TARGET'] == 1]
df_X  = df_out_time.loc[df_out_time['TARGET'] == 0]


# calculate precision-recall curve
ks = ks_2samp(df_label.SCORE,df_X.SCORE)

print("KS na base de teste é igual a:",round(ks[0],2))

# Box Plot

ax = sns.boxplot(x="TARGET", y="SCORE", data=df_label, order = [0,1],color = "red")
ax = sns.boxplot(x="TARGET", y="SCORE", data=df_X,order = [0,1],color = "blue")

plt.title('Boxplot dos Scores')
# Density 

sns.distplot(df_label.SCORE, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1}, 
                  label = "YES")

sns.distplot((df_X.SCORE), hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1}, 
                  label = "NO")

plt.title('Analise da Distribuição do Score')


y_test = df_out_time['TARGET']
y_proba_predicted = df_out_time['SCORE']


precision, recall, thresholds = precision_recall_curve(y_test, y_proba_predicted)
avg_precision = average_precision_score(y_test, y_proba_predicted)

step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
plt.step(recall, precision, color='b', alpha=0.2,
             where='post', label = 'Precision-Recall AUC = %0.4f' % avg_precision)
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
plt.legend(loc = 1)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision Recall Curve')
plt.show()
   

 # Plot AUC Curve
fpr, tpr, threshold = roc_curve(y_test, y_proba_predicted)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic (ROC) Curve ')
plt.plot(fpr, tpr, color='darkorange', label = 'ROC AUC = %0.4f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.01, 1])
plt.ylim([0, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


def plot_confusion_matrix(y_test, y_proba_predicted, threshold):
    """
    Plot confusion matrix for a given values of y_test, y_proba_predicted, and threshold
    """
    cm = confusion_matrix(y_test, (y_proba_predicted > threshold).astype(int))
    classes = ['N', 'S']

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()



# Find precision and threshold for 0.95 recall
print('\nFocus on high recall\n')
idx = np.argmax(recall<0.95)
threshold = thresholds[idx]
print('Threshold:', np.round(threshold, 3))
print('Precision:', np.round(precision[idx],4))
print('Recall:', np.round(recall[idx], 4))

plot_confusion_matrix(y_test, y_proba_predicted,threshold)



# Find precision and threshold for 0.95 recall

 # Find recall and threshold for 0.8 precision
 print('\nFocus on high precision\n')
 idx = np.argmax(precision>0.8)
 threshold = thresholds[idx]
 print('Threshold:', np.round(threshold, 3))
 print('Precision:', np.round(precision[idx],4))
 print('Recall:', np.round(recall[idx], 4))
    
 plot_confusion_matrix(y_test, y_proba_predicted, threshold)
