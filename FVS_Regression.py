#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 09:58:05 2021

@author: tungbioinfo
"""

import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", 60)

import warnings 
warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

import itertools
from scipy import interp
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.linear_model import ElasticNet, LarsCV, Lasso, LassoLars
from sklearn.linear_model import MultiTaskElasticNet, MultiTaskLasso
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared, DotProduct, ConstantKernel
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_percentage_error
import math
from scipy.stats import spearmanr
from statsmodels.stats.multitest import fdrcorrection 
from scipy import stats
import time

class AutoML_FVS():
    
    def __init__(self, random_state = None):
        self.random_state = random_state
    
        
    def FVS_grid(self, X_train, y_train, X_test, y_test, model, tuned_parameters, my_cv=5, n_selected_features = 100):
       
        F = []
        count = 0
        ddict = {}
        all_F = []
        all_c = []
        all_acc = []
        all_mse = []
        all_model = []
        start = time.time()
        while count < n_selected_features:
            max_corr = 0
            min_err = np.inf
            time_loop = time.time()
    
            for i in X_train.columns:
                if i not in F:
                    F.append(i)
                    X_train_tmp = X_train[F]
                    acc = 0
                    gsearch_cv = GridSearchCV(estimator = model, param_grid = tuned_parameters, 
                                      scoring = "neg_mean_squared_error", cv = my_cv, n_jobs=-1)
                    gsearch_cv.fit(X_train_tmp, y_train)
                    best_estimator = gsearch_cv.best_estimator_
                    y_pred = best_estimator.predict(X_test[F])
                    mse = mean_squared_error(y_test, y_pred)
                    corr = spearmanr(y_test, y_pred)[0]
                    F.pop()
                    """
                    if mse < min_err:
                        min_err = mse
                        idx = i
                        best_model = best_estimator
                    """
                    if corr > max_corr:
                        max_corr = corr
                        idx = i
                        best_model = best_estimator
                    
            F.append(idx)
            count += 1
            
            print("The current number of features: {} - MSE: {} - Corr: {}".format(count, round(mse, 2), round(corr, 2)))
            print("Time for computation: {}".format(time.time() - time_loop))

            all_F.append(np.array(F))
            all_c.append(count)
            all_acc.append(max_corr)
            all_model.append(best_model)
            all_mse.append(mse)

        c = pd.DataFrame(all_c)
        a = pd.DataFrame(all_acc)
        f = pd.DataFrame(all_F)  
        e = pd.DataFrame(all_mse)  
        f["All"] = f[f.columns[0:]].apply(
            lambda x: ', '.join(x.dropna().astype(str)), axis=1)

        all_info = pd.concat([c, e, a, f["All"]], axis=1)    
        all_info.columns = ['Num_feature', 'Mean_Squared_Error', 'Speaman_correlation', 'Feature']    
        all_info = all_info.sort_values(by='Mean_Squared_Error', ascending=True).reset_index(drop=True)
        
        return all_info, all_model, f
    
    def FVS_random(self, X_train, y_train, X_test, y_test, model, hyperparameter, my_cv=5, n_selected_features = 100):
        
        F = []
        count = 0
        ddict = {}
        all_F = []
        all_c = []
        all_acc = []
        all_mse = []
        all_model = []
        start = time.time()
        while count < n_selected_features:
            max_corr = 0
            min_err = np.inf
            time_loop = time.time()
    
            for i in X_train.columns:
                if i not in F:
                    F.append(i)
                    X_train_tmp = X_train[F]
                    acc = 0
                    rsearch_cv = RandomizedSearchCV(estimator = model, param_distributions = hyperparameter, cv = my_cv,
                                                    scoring = "neg_mean_squared_error", n_iter = 50, n_jobs = -1)
                    rsearch_cv.fit(X_train_tmp, y_train)
                    best_estimator = rsearch_cv.best_estimator_
                    y_pred = best_estimator.predict(X_test[F])
                    mse = mean_squared_error(y_test, y_pred)
                    corr = spearmanr(y_test, y_pred)[0]
                    F.pop()
                    """
                    if mse < min_err:
                        min_err = mse
                        idx = i
                        best_model = best_estimator
                    """
                    if corr > max_corr:
                        max_corr = corr
                        idx = i
                        best_model = best_estimator
                    
            F.append(idx)
            count += 1
            
            print("The current number of features: {} - MSE: {} - Corr: {}".format(count, round(mse, 2), round(corr, 2)))
            print("Time for computation: {}".format(time.time() - time_loop))

            all_F.append(np.array(F))
            all_c.append(count)
            all_acc.append(max_corr)
            all_model.append(best_model)
            all_mse.append(mse)

        c = pd.DataFrame(all_c)
        a = pd.DataFrame(all_acc)
        f = pd.DataFrame(all_F)  
        e = pd.DataFrame(all_mse)  
        f["All"] = f[f.columns[0:]].apply(
            lambda x: ', '.join(x.dropna().astype(str)), axis=1)

        all_info = pd.concat([c, e, a, f["All"]], axis=1)    
        all_info.columns = ['Num_feature', 'Mean_Squared_Error', 'Speaman_correlation', 'Feature']     
        all_info = all_info.sort_values(by='Mean_Squared_Error', ascending=True).reset_index(drop=True)
        
        return all_info, all_model, f
    
    def KernelRidge_FVS(self, X_train, y_train, X_test, y_test, n_selected_features = 100, scoring='neg_mean_squared_error'):
        
        n_samples, n_features = X_train.shape
        
        alphas = np.logspace(-5, 5, 100)
        kernel = ["linear", "poly", "rbf", "sigmoid", "chi2", "laplacian"]
        gamma = list(np.logspace(-2, 2, 100))
        gamma.append("scale")
        gamma.append("auto")
        hyperparameter = {"alpha": alphas, 
               "kernel": kernel,
               "gamma": gamma}
        my_cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
        model = KernelRidge()

        scoring = "neg_mean_squared_error"
        n_selected_features = n_selected_features
        
        all_info, all_model, f = self.FVS_random(X_train, y_train, X_test, y_test, model, 
                                                 hyperparameter, n_selected_features = n_selected_features)
        
        return all_info, all_model, f
    
    
    def RF_FVS(self, X_train, y_train, X_test, y_test, n_selected_features = 100, scoring='neg_mean_squared_error'):
        
        n_samples, n_features = X_train.shape

        n_estimators = [5, 10, 50, 100, 150, 200, 250, 300]
        max_depth = [5, 10, 25, 50, 75, 100]
        min_samples_leaf = [1, 2, 4, 8, 10]
        min_samples_split = [2, 4, 6, 8, 10]
        max_features = ["auto", "sqrt", "log2", None]

        hyperparameter = {'n_estimators': n_estimators,
                  'max_depth': max_depth,
                  'min_samples_leaf': min_samples_leaf,
                  'min_samples_split': min_samples_split,
                  'max_features': max_features,
                  }


        my_cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
        base_model_rf = RandomForestRegressor(criterion="mse", random_state=42)
        n_iter_search = 30

        scoring = "neg_mean_squared_error"
        n_selected_features = n_selected_features

        all_info, all_model, f = self.FVS_random(X_train, y_train, X_test, y_test, base_model_rf, 
                                                 hyperparameter, n_selected_features = n_selected_features)
        
        return all_info, all_model, f
    
    def Stochastic_Gradient_Descent_FVS(self, X_train, y_train, X_test, y_test, n_selected_features = 100, scoring='neg_mean_squared_error'):
        
        n_samples, n_features = X_train.shape

        # Loss function 
        loss = ["squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"]
        
        penalty = ["l2", "l1", "elasticnet"]
        
        # The higher the value, the stronger the regularization 
        alpha = np.logspace(-6, 6, 100)
        
        # The Elastic Net mixing parameter 
        l1_ratio = np.logspace(-6, -1, 100)
        
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
        
        my_cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
        model = SGDRegressor()
        n_iter_search = 30

        scoring = "neg_mean_squared_error"
        n_selected_features = n_selected_features

        all_info, all_model, f = self.FVS_random(X_train, y_train, X_test, y_test, model, 
                                                 hyperparameter, my_cv=5, n_selected_features = n_selected_features)
        
        return all_info, all_model, f
    
    def DecisionTree_FVS(self, X_train, y_train, X_test, y_test, n_selected_features = 100, scoring='neg_mean_squared_error'):
        
        n_samples, n_features = X_train.shape

        max_depth = [5, 10, 25, 50, 75, 100]
        min_samples_leaf = [1, 2, 4, 8, 10]
        min_samples_split = [2, 4, 6, 8, 10]
        max_features = ["auto", "sqrt", "log2", None]
        criterion = ["mse"]
        splitter = ["best", "random"]
        
        hyperparameter = {"max_depth": max_depth,
                          "min_samples_leaf": min_samples_leaf,
                          "min_samples_split": min_samples_split,
                          "max_features": max_features,
                          "criterion": criterion,
                          "splitter": splitter}
        my_cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
        model = DecisionTreeRegressor(random_state = 42)

        scoring = "neg_mean_squared_error"
        n_selected_features = n_selected_features

        all_info, all_model, f = self.FVS_random(X_train, y_train, X_test, y_test, model, 
                                                 hyperparameter, my_cv=5, n_selected_features = n_selected_features)
        
        return all_info, all_model, f
    
    def ElasticNet_FVS(self, X_train, y_train, X_test, y_test, n_selected_features = 100, scoring='neg_mean_squared_error'):
        
        n_samples, n_features = X_train.shape

        alpha = np.logspace(-5, 5, 100)
        
        # The Elastic Net mixing parameter 
        l1_ratio = np.logspace(-10, -1, 100)
        
        hyperparameter = {"alpha": alpha,
                          "l1_ratio": l1_ratio}
        
        my_cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
        model = ElasticNet(max_iter=10000)

        scoring = "neg_mean_squared_error"
        n_selected_features = n_selected_features

        all_info, all_model, f = self.FVS_random(X_train, y_train, X_test, y_test, model, 
                                                 hyperparameter, my_cv=5, n_selected_features = n_selected_features)
        
        return all_info, all_model, f
    
    
    def GaussianProcess_FVS(self, X_train, y_train, X_test, y_test, n_selected_features = 100, scoring='neg_mean_squared_error'):
        
        n_samples, n_features = X_train.shape
        
        kernels = [1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),
           1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1),
           ConstantKernel(0.1, (0.01, 10.0))
               * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2),
           1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),
                        nu=1.5)]
        
        tuned_parameters = [{"kernel": kernels}]
        my_cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
        model = GaussianProcessRegressor()
        
        scoring = "neg_mean_squared_error"
        n_selected_features = n_selected_features

        all_info, all_model, f = self.FVS_grid(X_train, y_train, X_test, y_test, model, tuned_parameters, 
                                          my_cv=5, n_selected_features = n_selected_features)
        return all_info, all_model, f
    
    def LassoLars_FVS(self, X_train, y_train, X_test, y_test, n_selected_features = 100, scoring='neg_mean_squared_error'):
        
        n_samples, n_features = X_train.shape
        
        alphas = np.logspace(-5, 5, 100)
        tuned_parameters = [{"alpha": alphas}]
        my_cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
        model = LassoLars()
        
        scoring = "neg_mean_squared_error"
        n_selected_features = n_selected_features

        all_info, all_model, f = self.FVS_grid(X_train, y_train, X_test, y_test, model, tuned_parameters, 
                                          my_cv=5, n_selected_features = n_selected_features)
        return all_info, all_model, f
    
    def Ridge_FVS(self, X_train, y_train, X_test, y_test, n_selected_features = 100, scoring='neg_mean_squared_error'):
        
        n_samples, n_features = X_train.shape
        
        alphas = np.logspace(-5, 5, 100)
        tuned_parameters = [{"alpha": alphas}]
        my_cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
        model = Ridge()
        
        scoring = "neg_mean_squared_error"
        n_selected_features = n_selected_features

        all_info, all_model, f = self.FVS_grid(X_train, y_train, X_test, y_test, model, tuned_parameters, 
                                          my_cv=5, n_selected_features = n_selected_features)
        return all_info, all_model, f
    
    def Lasso_FVS(self, X_train, y_train, X_test, y_test, n_selected_features = 100, scoring='neg_mean_squared_error'):
        
        n_samples, n_features = X_train.shape
        
        alphas = np.logspace(-5, 5, 100)
        tuned_parameters = [{"alpha": alphas}]
        my_cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
        model = Lasso()
        
        scoring = "neg_mean_squared_error"
        n_selected_features = n_selected_features

        all_info, all_model, f = self.FVS_grid(X_train, y_train, X_test, y_test, model, tuned_parameters, 
                                          my_cv=5, n_selected_features = n_selected_features)
        return all_info, all_model, f
    

        
        
        
    
    



































































