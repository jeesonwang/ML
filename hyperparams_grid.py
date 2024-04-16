from sklearn.gaussian_process.kernels import RBF

grid_knn = { 
    'n_neighbors' : list(range(1,10)),
    'leaf_size' : list(range(1,50)),
    'p':[1,2]
}

grid_dt = { 
    'max_depth' : [None,2,4,6,8,10,12],
    'criterion' : ['gini', 'entropy'],
}

grid_mlp = { 
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

grid_gbt = { 
   'learning_rate': [0.1,0.01],
    'n_estimators' : [100,500,1000],
    'max_depth'    : [3,4,6,8,10,12]
}

grid_svm = { 
    'C':list(range(-1,11)),

}

grid_nb = { 
    'var_smoothing': [1e-11, 1e-10, 1e-9]
}

grid_lr = { 
      'C':list(range(-1,11)), 
      'penalty' : ['l1', 'l2'],
      'max_iter': [20, 50, 100, 200, 500, 1000],                      
      'solver': ['lbfgs', 'liblinear'],   
      'class_weight': [None,'balanced']
}

grid_rf = { 
    'n_estimators': [5, 20, 50, 100],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
}


grid_bagging = {
    'n_estimators': [5, 20, 50, 100],
    'max_samples': [0.5, 0.8, 1.0],
    'max_features': [0.5, 0.8, 1.0],
    'bootstrap': [True, False],
    'bootstrap_features': [True, False]
}

grid_adaboost = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0],
    'algorithm': ['SAMME', 'SAMME.R']
}

grid_extratrees = {
    'n_estimators': [5, 20, 50, 100],
    'max_features': [1, 'sqrt'],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_gaussian = {
    'kernel': [1.0 * RBF(1.0), 2.0 * RBF(1.0)],
    'optimizer': ['fmin_l_bfgs_b', 'fmin_tnc'],
    'n_restarts_optimizer': [0, 5, 10]
}