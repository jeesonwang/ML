from sklearn import metrics
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV


def ModelPredict(m, train_X, train_y, test_X, test_y):
    model = m[0]
    model.fit(train_X, train_y)
    pred_train = model.predict(train_X)
    pred_test = model.predict(test_X)
    score = model.score(test_X, test_y)
    f1score = f1_score(test_y, pred_test)    
    # print("f1 :", f1score)
    # print(pred_test)
    return pred_train, pred_test, score, f1score

def PrintReportFold(report_fold,m,fold,score_fold,f1score_fold,y_test_fold,y_pred_fold):
    report_fold.write(f'{m[1]:20} \n')
    report_fold.write(f'fold {fold}: \n')
    report_fold.write(f'Accuracy: {score_fold} \n')
    report_fold.write(f'f-score: {f1score_fold} \n')
    report_fold.write(str(metrics.confusion_matrix(y_test_fold, y_pred_fold)) + '\n')
    report_fold.write(metrics.classification_report(y_test_fold, y_pred_fold))
    report_fold.write('-' * 100 + '\n')


def GridSearchModelPredict(m, train_X, train_y, test_X, test_y):
    
    search = HalvingGridSearchCV(estimator=m[0], param_grid=m[2], cv= 10, n_jobs=-1, scoring='accuracy')
    
    model = search.fit(train_X, train_y)
    
    best_model = model.best_estimator_
    score = best_model.score(test_X, test_y)
    pred_train = best_model.predict(train_X)
    pred_test = best_model.predict(test_X)
    f1score = f1_score(test_y, pred_test)
    
    return pred_train, pred_test, score, f1score


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

def write_report_result(model_scores, typesmell):

    select_method=[
        [model_scores, 'all']
    ]

    for sm in select_method:
        rows = []
        for x in sm[0]:
            print(sm[1])
            report = open(typesmell + '\\' + 'tunning report\\' + x[2] + '-report_' + sm[1] + '.txt','w')
            rows_tmp = PrintReport(x,report)
            rows.append(rows_tmp)

        if sm[1] == 'all':
            rows_all = rows
        report.close()
        
    df0 = pd.DataFrame(rows_all, columns=["Model", "precision","recall","f1-score"])
    
    ans=pd.concat([df0], axis=1)
    # ans.to_csv(typesmell + '\\' +typesmell + '-ML-tuning-update.csv',index=False)
    
    return ans

# Find the models' best parameters
import json

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_params_dict(algo, smelltype):
    jsonl_files = []
    for a in algo:
        jsonl_files.append((f'{smelltype}\\tunning\\{a[1]}.jsonl', a[1],))
    result_dict = {}
    for i in range(10):  # 10 entries in each file
        result_dict[i] = {}
        for file_path, al_name in jsonl_files:
            data = load_jsonl(file_path)
            if len(data) > i:
                result_dict[i][al_name] = data[i]['best_params']
            else:
                raise f"{file_path} error"
    return result_dict




