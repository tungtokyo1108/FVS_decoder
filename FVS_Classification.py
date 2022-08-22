#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 22:22:33 2022

@author: tungdang
"""

import pandas as pd
import numpy as np

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, Sequence, Iterable
from functools import partial, reduce
from itertools import product
import numbers
import operator
import time
import warnings
warnings.simplefilter("ignore")

pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", 60)

import matplotlib.pyplot as plt
import seaborn as sns

import itertools
from scipy import interp
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, RepeatedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV

class AutoML_FVS_Classification():
    
    def __init__(self, random_state = None):
        self.random_state = random_state
    
    def FVS_grid(self, X_train, y_train, X_test, y_test, model, tuned_parameters, my_cv=5, n_selected_features = 100):
        
        F = []
        count = 0
        ddict = {}
        all_F = []
        all_c = []
        all_acc = []
        all_model = []
        start = time.time()
        while count < n_selected_features:
            max_acc = 0
            time_loop = time.time()
            
            for i in X_train.columns:
                    if i not in F:
                        F.append(i)
                        X_train_tmp = X_train[F]
                        acc = 0
                        gsearch_cv = GridSearchCV(estimator = model, param_grid = tuned_parameters, 
                                          scoring = "accuracy", cv = my_cv, n_jobs=-1)
                        gsearch_cv.fit(X_train_tmp, y_train)
                        best_estimator = gsearch_cv.best_estimator_
                        y_pred = best_estimator.predict(X_test[F])
                        acc = metrics.accuracy_score(y_test, y_pred)
                        F.pop()
                        if acc > max_acc:
                            max_acc = acc
                            idx = i
                            best_model = best_estimator
                            
            F.append(idx)
            count += 1
             
            print("The current number of features: {} - Accuracy: {}%".format(count, round(max_acc*100, 2)))
            print("Time for computation: {}".format(time.time() - time_loop))

            all_F.append(np.array(F))
            all_c.append(count)
            all_acc.append(max_acc)
            all_model.append(best_model)
            
        time.time() - start    

        c = pd.DataFrame(all_c)
        a = pd.DataFrame(all_acc)
        f = pd.DataFrame(all_F)    
        f["All"] = f[f.columns[0:]].apply(
                lambda x: ', '.join(x.dropna().astype(str)), axis=1)
            
        all_info = pd.concat([c, a, f["All"]], axis=1)    
        all_info.columns = ['Num_feature', 'Accuracy', 'Feature']    
        all_info = all_info.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
        
        return all_info, all_model, f
    
    def FVS_random(self, X_train, y_train, X_test, y_test, model, hyperparameter, my_cv=5, n_selected_features = 100):
        
        F = []
        count = 0
        ddict = {}
        all_F = []
        all_c = []
        all_acc = []
        all_model = []
        start = time.time()
        while count < n_selected_features:
            max_acc = 0
            time_loop = time.time()
            
            for i in X_train.columns:
                    if i not in F:
                        F.append(i)
                        X_train_tmp = X_train[F]
                        acc = 0
                        rsearch_cv = RandomizedSearchCV(estimator = model, param_distributions=hyperparameter,
                                        scoring="accuracy", cv = my_cv, n_jobs=-1, n_iter=50)
                        rsearch_cv.fit(X_train_tmp, y_train)
                        best_estimator = rsearch_cv.best_estimator_
                        y_pred = best_estimator.predict(X_test[F])
                        acc = metrics.accuracy_score(y_test, y_pred)
                        F.pop()
                        if acc > max_acc:
                            max_acc = acc
                            idx = i
                            best_model = best_estimator
                            
            F.append(idx)
            count += 1
             
            print("The current number of features: {} - Accuracy: {}%".format(count, round(max_acc*100, 2)))
            print("Time for computation: {}".format(time.time() - time_loop))

            all_F.append(np.array(F))
            all_c.append(count)
            all_acc.append(max_acc)
            all_model.append(best_model)
            
        time.time() - start    

        c = pd.DataFrame(all_c)
        a = pd.DataFrame(all_acc)
        f = pd.DataFrame(all_F)    
        f["All"] = f[f.columns[0:]].apply(
                lambda x: ', '.join(x.dropna().astype(str)), axis=1)
            
        all_info = pd.concat([c, a, f["All"]], axis=1)    
        all_info.columns = ['Num_feature', 'Accuracy', 'Feature']    
        all_info = all_info.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
        
        return all_info, all_model, f
    
    def Logistic_FVS(self, X_train, y_train, X_test, y_test, n_selected_features = 100):
        
        c = np.linspace(0.001, 1, 100)

        tuned_parameters = [{"C": c}]
        n_folds = 10 
        #model = LogisticRegression(max_iter=1000)
        model = LogisticRegression(penalty="l1", solver = "liblinear")
        my_cv = TimeSeriesSplit(n_splits = n_folds).split(X_train)
        n_selected_features = n_selected_features
        
        all_info, all_model, f = self.FVS_grid(X_train, y_train, X_test, y_test, model, tuned_parameters, 
                                          my_cv=5, n_selected_features = n_selected_features)
        return all_info, all_model, f
    
    def Naive_Bayes_FVS(self, X_train, y_train, X_test, y_test, n_selected_features = 100):
        
        alphas = np.logspace(0,1,100)
        tuned_parameters = [{"alpha": alphas}]
        n_folds = 10
        model = MultinomialNB()
        my_cv = TimeSeriesSplit(n_splits=n_folds).split(X_train)
        n_selected_features = n_selected_features
        
        all_info, all_model, f = self.FVS_grid(X_train, y_train, X_test, y_test, model, tuned_parameters, 
                                          my_cv=5, n_selected_features = n_selected_features)
        return all_info, all_model, f
    
    def Decision_Tree_FVS(self, X_train, y_train, X_test, y_test, n_selected_features = 100):
        
        max_depth = [5, 10, 25, 50, 75, 100]
        min_samples_leaf = [1, 2, 4, 8, 10]
        min_samples_split = [2, 4, 6, 8, 10]
        max_features = [ "sqrt", "log2", None]
        criterion = ["gini", "entropy"]
        splitter = ["best", "random"]
        
        hyperparameter = {"max_depth": max_depth,
                          "min_samples_leaf": min_samples_leaf,
                          "min_samples_split": min_samples_split,
                          "max_features": max_features,
                          "criterion": criterion,
                          "splitter": splitter}
        
        n_folds = 10
        my_cv = TimeSeriesSplit(n_splits = n_folds).split(X_train)
        model = DecisionTreeClassifier(random_state = 42)
        
        n_selected_features = n_selected_features
        
        all_info, all_model, f = self.FVS_random(X_train, y_train, X_test, y_test, model, 
                                                 hyperparameter, my_cv=5, n_selected_features = n_selected_features)
        
        return all_info, all_model, f
    
    def Random_Forest_FVS(self, X_train, y_train, X_test, y_test, n_selected_features = 100):
        
        # Numer of trees are used
        n_estimators = [5, 10, 50, 100, 150, 200, 250, 300]

        # Maximum depth of each tree
        max_depth = [5, 10, 25, 50, 75, 100]
    
        # Minimum number of samples per leaf 
        min_samples_leaf = [1, 2, 4, 8, 10]
    
        # Minimum number of samples to split a node
        min_samples_split = [2, 4, 6, 8, 10]
    
        # Maximum numeber of features to consider for making splits
        max_features = [ "sqrt", "log2", None]
        
        criterion = ["gini", "entropy"]
    
        hyperparameter = {'n_estimators': n_estimators,
                      'max_depth': max_depth,
                      'min_samples_leaf': min_samples_leaf,
                      'min_samples_split': min_samples_split,
                      'max_features': max_features, 
                      'criterion': criterion}
        n_folds = 10
        my_cv = TimeSeriesSplit(n_splits = n_folds).split(X_train)
        model = RandomForestClassifier(random_state=42)
        
        n_selected_features = n_selected_features
        
        all_info, all_model, f = self.FVS_random(X_train, y_train, X_test, y_test, model, 
                                                 hyperparameter, my_cv=5, n_selected_features = n_selected_features)
        
        return all_info, all_model, f
    
    def Stochastic_Gradient_Descent_FVS(self, X_train, y_train, X_test, y_test, n_selected_features = 100):
        
        # Loss function 
        loss = ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"]
        
        penalty = ["l2", "l1", "elasticnet"]
        
        # The higher the value, the stronger the regularization 
        alpha = np.logspace(-7, -1, 100)
        
        # The Elastic Net mixing parameter 
        l1_ratio = np.linspace(0, 1, 100)
        
        epsilon = np.logspace(-5, -1, 100)
        
        learning_rate = ["constant", "optimal", "invscaling", "adaptive"]
        
        eta0 = np.logspace(-7, -1, 100)
        
        hyperparameter = {"loss": loss,
                          "penalty": penalty,
                          "alpha": alpha,
                          "l1_ratio": l1_ratio,
                          "epsilon": epsilon,
                          "learning_rate": learning_rate,
                          "eta0": eta0}
        
        n_folds = 10
        my_cv = TimeSeriesSplit(n_splits = n_folds).split(X_train)
        model = SGDClassifier(n_jobs = -1)
        
        n_selected_features = n_selected_features
        
        all_info, all_model, f = self.FVS_random(X_train, y_train, X_test, y_test, model, 
                                                 hyperparameter, my_cv=5, n_selected_features = n_selected_features)
        
        return all_info, all_model, f
    
    def Support_Vector_Classify_FVS(self, X_train, y_train, X_test, y_test, n_selected_features = 100):
        
        C = np.logspace(-2, 7, 100)
        kernel = ["linear", "poly", "rbf", "sigmoid"]
        gamma = list(np.logspace(-1, 1, 100))
        gamma.append("scale")
        gamma.append("auto")
        hyperparameter = {"C": C, 
               "kernel": kernel,
               "gamma": gamma}
        n_folds = 10
        model = SVC()
        my_cv = TimeSeriesSplit(n_splits = n_folds).split(X_train)
        
        n_selected_features = n_selected_features
        
        all_info, all_model, f = self.FVS_random(X_train, y_train, X_test, y_test, model, 
                                                 hyperparameter, my_cv=5, n_selected_features = n_selected_features)
        
        return all_info, all_model, f
    
    def Gradient_Boosting_FVS(self, X_train, y_train, X_test, y_test, n_selected_features = 100):
        
        # Numer of trees are used
        n_estimators = [5, 10, 50, 100, 150, 200, 250, 300]

        # Maximum depth of each tree
        max_depth = [5, 10, 25, 50, 75, 100]
    
        # Minimum number of samples per leaf 
        min_samples_leaf = [1, 2, 4, 8, 10]
    
        # Minimum number of samples to split a node
        min_samples_split = [2, 4, 6, 8, 10]
    
        # Maximum numeber of features to consider for making splits
        max_features = ["auto", "sqrt", "log2", None]
        
        criterion = ["friedman_mse", "mse", "mae"]
    
        hyperparameter = {'n_estimators': n_estimators,
                      'max_depth': max_depth,
                      'min_samples_leaf': min_samples_leaf,
                      'min_samples_split': min_samples_split,
                      'max_features': max_features, 
                      'criterion': criterion}
        n_folds = 10
        my_cv = TimeSeriesSplit(n_splits = n_folds).split(X_train)
        model = GradientBoostingClassifier(random_state=42)
        
        all_info, all_model, f = self.FVS_random(X_train, y_train, X_test, y_test, model, 
                                                 hyperparameter, my_cv=5, n_selected_features = n_selected_features)
        
        return all_info, all_model, f
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        