#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os

from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import json



def default_model(path, smelltype, algo: list):
    os.makedirs('default_model', exist_ok=True)
    os.makedirs('default_model\\' + smelltype, exist_ok=True)
    os.makedirs('default_model\\' + smelltype + '\\' +'tuning', exist_ok=True)
    os.makedirs('default_model\\' + smelltype + '\\' +'tuning report', exist_ok=True)

    model_scores=[]

    for a in algo:
        report_fold_all = open('default_model\\' + smelltype + '\\' +'tuning\\' + a[1] + '.jsonl','a')

        predicted_targets = np.array([])
        actual_targets = np.array([])

        for fold in range(1,11):
            
            train = pd.read_csv(path + '\\train_fold_' + str(fold) + ".csv")
            test = pd.read_csv(path + '\\test_fold_' + str(fold) + ".csv")
            
            X_train = train.iloc[:,:-1].to_numpy()
            X_test = test.iloc[:,:-1].to_numpy()
            y_train = train.iloc[:,-1].to_numpy()
            y_test = test.iloc[:,-1].to_numpy()    

            # ***** start model predict *****

            y_pred = ModelPredict(a, fold, X_train, y_train, X_test, y_test, report_fold_all)

            predicted_targets = np.append(predicted_targets, y_pred)
            actual_targets = np.append(actual_targets, y_test)
            
        model_scores.append([actual_targets, predicted_targets, a[1]])

        report_fold_all.close()
    
    return model_scores


def ModelPredict(m, fold, train_X, train_y, test_X, test_y, file):
    model = m[0]
    model.fit(train_X, train_y)
    score = model.score(test_X, test_y)
    pred_y = model.predict(test_X)
    f1score = f1_score(test_y, pred_y)
    
        
    print(f'{m[1]:20} Accuracy: {score:.04f}')        
    print(f'fold {fold}:')
    print(f'Accuracy: {model.score(test_X, test_y)}')
    print(f'f-score: {f1_score(test_y, pred_y)}')
    print(metrics.confusion_matrix(test_y, pred_y))
    print(metrics.classification_report(test_y, pred_y))
    print('-' * 100)
    
    PrintReportFold(file,m,fold,score,f1score)
    
    return pred_y


# In[5]:


def PrintReportFold(report_fold,m,fold,score_fold,f1score_fold):
    save_d = dict(model=m[1], fold=fold, accuracy=score_fold, f1=f1score_fold)
    json.dump(save_d, report_fold, ensure_ascii=False)
    report_fold.write('\n')

# In[6]:


def PrintReport(i,report):
    
    rows = []

    print(f'{i[2]:20}')
    print(metrics.confusion_matrix(i[0], i[1])) 
    print(metrics.classification_report(i[0], i[1]))
    
    report.write(f'{i[2]:20}\n')
    report.write(str(metrics.confusion_matrix(i[0], i[1])) + '\n') 
    report.write(metrics.classification_report(i[0], i[1]))
    
    report_to_csv = pd.DataFrame(metrics.classification_report(i[0], i[1], output_dict=True)).transpose()
    rows =[i[2],report_to_csv['precision'][1],report_to_csv['recall'][1],report_to_csv['f1-score'][1]]
    
    return rows
