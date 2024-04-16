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
from sklearn import tree
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV


# In[3]:


def tuning_model(X, y, smelltype, algo: list):

    os.makedirs('tuning_model', exist_ok=True)
    os.makedirs(f'tuning_model\\{smelltype}', exist_ok=True)
    os.makedirs('tuning_model\\'+smelltype+'_data\\data', exist_ok=True)
    os.makedirs('tuning_model\\'+smelltype + '\\' +'tuning', exist_ok=True)
    os.makedirs('tuning_model\\'+smelltype + '\\' +'tuning report', exist_ok=True)

    model_scores=[]

    for a in algo:

        report_fold_all = open('tuning_model\\' + smelltype + '\\' +'tuning\\' + a[1] + '.jsonl','a')
       
        score = []

        predicted_targets = np.array([])
        actual_targets = np.array([])

        kf = KFold(n_splits=10, shuffle=True, random_state=42)  

        for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
                    
            # spilt train test
            X_train = X[train_index]
            y_train = y[train_index] 
            X_test = X[test_index]
            y_test = y[test_index]

            if not os.path.exists(smelltype+ '_data\\' +'data'):
                # start save train test to csv file
                x_train_data =  pd.DataFrame(X_train)
                y_train_data =  pd.DataFrame(y_train)
                x_test_data = pd.DataFrame(X_test)
                y_test_data = pd.DataFrame(y_test)
                train_data = pd.concat([x_train_data,y_train_data], axis=1)
                train_data.to_csv('tuning_model\\' + smelltype+ '_data\\' +'train_fold_' + str(fold) + '.csv',index=False)
                test_data = pd.concat([x_test_data,y_test_data], axis=1)
                test_data.to_csv('tuning_model\\' + smelltype+ '_data\\' +'test_fold_' + str(fold) + '.csv',index=False)
                # end
            
            # ***** start model predict *****

            y_pred = ModelPredict(a, fold, X_train, y_train, X_test, y_test, report_fold_all)

            predicted_targets = np.append(predicted_targets, y_pred)
            actual_targets = np.append(actual_targets, y_test)
            
        model_scores.append([actual_targets, predicted_targets, a[1]])

        report_fold_all.close()
    
    return model_scores


def ModelPredict(m, fold, train_X, train_y, test_X, test_y, file):
    
    search = HalvingGridSearchCV(estimator=m[0], param_grid=m[2], cv= 10, n_jobs=-1, scoring='accuracy')
    
    model = search.fit(train_X, train_y)
    
    best_model = model.best_estimator_
    score = best_model.score(test_X, test_y)
    pred_y = best_model.predict(test_X)
    f1score = f1_score(test_y, pred_y)


    print(f'{m[1]:20} Accuracy: {score:.04f}')        
    print(f'fold {fold}:')
    print(best_model)
    print(f'Accuracy: {best_model.score(test_X, test_y)}')
    print(f'f-score: {f1_score(test_y, pred_y)}')
    print(metrics.confusion_matrix(test_y, pred_y))
    print(metrics.classification_report(test_y, pred_y))
    print('-' * 100)
    
    PrintReportFold(file,m,fold,score,f1score,test_y,pred_y,model.best_params_)
    
    return pred_y


def PrintReportFold(report_fold,m,fold,score_fold,f1score_fold,y_test_fold,y_pred_fold,best_params):
    save_d = dict(model=m[1], fold=fold, best_params=best_params, accuracy=score_fold, f1=f1score_fold)
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
