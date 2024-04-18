from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score

from utils import ModelPredict, GridSearchModelPredict


class StackingEnsemble:
    def __init__(self, algo: list, report_file: str='') -> None:
        self.algo = algo
        self.report_file = report_file

    def forward(self, train_X, train_y, test_X, test_y):
        train_pred_dict = {}
        test_pred_dict = {}
        models_acc = []
        models_f1 = []
        for a in self.algo[:-1]:
            pred_train, pred_test, acc, f1 = ModelPredict(a, train_X, train_y, test_X, test_y)
            
            train_pred_dict[f'{a[1]}_predict'] = pred_train
            test_pred_dict[f'{a[1]}_predict'] = pred_test
            models_acc.append(acc)
            models_f1.append(f1)
            
        stacked_train_X = pd.concat([pd.DataFrame(train_X).reset_index(drop=True), pd.DataFrame(train_pred_dict)], axis=1)
        stacked_test_X = pd.concat([pd.DataFrame(test_X).reset_index(drop=True), pd.DataFrame(test_pred_dict)], axis=1)
        # print(stacked_test_X)
        stacked_train_X = stacked_train_X.to_numpy()
        stacked_test_X = stacked_test_X.to_numpy()
        pred_train, pred_test, final_acc, final_f1 = ModelPredict(self.algo[-1], stacked_train_X, train_y, stacked_test_X, test_y)

        return pred_test, final_acc, final_f1
    
    def set_params(self, params: dict=None):
        if params is not None:
            for a in self.algo:
                cur_params = params[a[1]]
                a[0].set_params(**cur_params)



class VoteEnsemble:
    def __init__(self, algo: list, report_file: str='', vote_mode: str='soft') -> None:
        self.algo = algo
        self.report_file = report_file
        self.vote_mode = vote_mode

    def forward(self, train_X, train_y, test_X, test_y):
        self.voting_classifier.fit(train_X, train_y)
        y_pred = self.voting_classifier.predict(test_X)

        vote_accuracy = accuracy_score(test_y, y_pred)
        vote_f1 = f1_score(test_y, y_pred)

        return y_pred, vote_accuracy, vote_f1

    def set_params(self, params: dict=None):
        if params is not None: 
            estimators = []
            for a in self.algo:
                cur_params = params[a[1]]
                classifier = a[0].set_params(**cur_params)
                estimators.append((a[1], classifier))
            self.voting_classifier = VotingClassifier(estimators=estimators, voting=self.vote_mode)
        else:
            estimators = []
            for a in self.algo:
                classifier = a[0]
                estimators.append((a[1], classifier))
            self.voting_classifier = VotingClassifier(estimators=estimators, voting=self.vote_mode)
        