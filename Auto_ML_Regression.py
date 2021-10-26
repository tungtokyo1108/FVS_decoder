#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 13:09:08 2021

@author: tungbioinfo
"""


import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", 60)

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

class AutoML_Regression():
    
    def __init__(self, random_state = None):
        self.random_state = random_state
    
    def Ridge_regression(self, X_train, y_train, X_test, y_test):
        
        alphas = np.logspace(-5, 5, 100)
        tuned_parameters = [{"alpha": alphas}]
        my_cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
        model = Ridge()
        gsearch_cv = GridSearchCV(estimator = model, param_grid = tuned_parameters, 
                                  scoring = "neg_mean_squared_error", cv = my_cv, n_jobs=-1)
        gsearch_cv.fit(X_train, y_train)
        best_model = gsearch_cv.best_estimator_
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return best_model, mse, mae, r2
    
    def Lasso_regression(self, X_train, y_train, X_test, y_test):
        
        alphas = np.logspace(-5, 5, 100)
        tuned_parameters = [{"alpha": alphas}]
        my_cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
        model = Lasso()
        gsearch_cv = GridSearchCV(estimator = model, param_grid = tuned_parameters, 
                                  scoring = "neg_mean_squared_error", cv = my_cv, n_jobs=-1)
        gsearch_cv.fit(X_train, y_train)
        best_model = gsearch_cv.best_estimator_
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return best_model, mse, mae, r2
    
    def Lars_regression(self, X_train, y_train, X_test, y_test):
        
        my_cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
        best_model = LarsCV(cv=my_cv, n_jobs=-1)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return best_model, mse, mae, r2
    
    def ElasticNet_regression(self, X_train, y_train, X_test, y_test):
        
        alpha = np.logspace(-5, 5, 100)
        
        # The Elastic Net mixing parameter 
        l1_ratio = np.logspace(-10, -1, 100)
        
        hyperparameter = {"alpha": alpha,
                          "l1_ratio": l1_ratio}
        
        my_cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
        model = ElasticNet(max_iter=10000)
        rsearch_cv = RandomizedSearchCV(estimator = model, param_distributions = hyperparameter, cv = my_cv,
                                        scoring = "neg_mean_squared_error", n_iter = 100, n_jobs = -1)
        rsearch_cv.fit(X_train, y_train)
        sgb_best = rsearch_cv.best_estimator_
        sgb_best.fit(X_train, y_train)
        y_pred = sgb_best.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return sgb_best, mse, mae, r2
        
    def LassoLars_regression(self, X_train, y_train, X_test, y_test):
        
        alphas = np.logspace(-5, 5, 100)
        tuned_parameters = [{"alpha": alphas}]
        my_cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
        model = LassoLars()
        gsearch_cv = GridSearchCV(estimator = model, param_grid = tuned_parameters, 
                                  scoring = "neg_mean_squared_error", cv = my_cv, n_jobs=-1)
        gsearch_cv.fit(X_train, y_train)
        best_model = gsearch_cv.best_estimator_
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return best_model, mse, mae, r2
    
    def MultiTaskLasso_regression(self, X_train, y_train, X_test, y_test):
        
        alphas = np.logspace(-5, 5, 100)
        tuned_parameters = [{"alpha": alphas}]
        my_cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
        model = MultiTaskLasso()
        gsearch_cv = GridSearchCV(estimator = model, param_grid = tuned_parameters, 
                                  scoring = "neg_mean_squared_error", cv = my_cv, n_jobs=-1)
        gsearch_cv.fit(X_train, y_train)
        best_model = gsearch_cv.best_estimator_
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return best_model, mse, mae, r2
        
    def Support_Vector_regression(self, X_train, y_train, X_test, y_test):
        
        C = np.logspace(-5, 5, 100)
        kernel = ["linear", "poly", "rbf", "sigmoid"]
        gamma = list(np.logspace(-2, 2, 100))
        gamma.append("scale")
        gamma.append("auto")
        hyperparameter = {"C": C, 
               "kernel": kernel,
               "gamma": gamma}
        
        my_cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
        model = SVR()
        rsearch_cv = RandomizedSearchCV(estimator = model, param_distributions = hyperparameter, cv = my_cv,
                                        scoring = "neg_mean_squared_error", n_iter = 100, n_jobs = -1)
        rsearch_cv.fit(X_train, y_train)
        sgb_best = rsearch_cv.best_estimator_
        sgb_best.fit(X_train, y_train)
        y_pred = sgb_best.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return sgb_best, mse, mae, r2
    
    def KernelRidge_regression(self, X_train, y_train, X_test, y_test):
        
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
        rsearch_cv = RandomizedSearchCV(estimator = model, param_distributions = hyperparameter, cv = my_cv,
                                        scoring = "neg_mean_squared_error", n_iter = 100, n_jobs = -1)
        rsearch_cv.fit(X_train, y_train)
        sgb_best = rsearch_cv.best_estimator_
        sgb_best.fit(X_train, y_train)
        y_pred = sgb_best.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return sgb_best, mse, mae, r2
    
    def GaussianProcess_regression(self, X_train, y_train, X_test, y_test):
        
        kernels = [1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),
           1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1),
           ConstantKernel(0.1, (0.01, 10.0))
               * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2),
           1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),
                        nu=1.5)]
        
        tuned_parameters = [{"kernel": kernels}]
        my_cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
        model = GaussianProcessRegressor()
        gsearch_cv = GridSearchCV(estimator = model, param_grid = tuned_parameters, 
                                  scoring = "neg_mean_squared_error", cv = my_cv, n_jobs=-1)
        gsearch_cv.fit(X_train, y_train)
        best_model = gsearch_cv.best_estimator_
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return best_model, mse, mae, r2
    
    def Stochastic_Gradient_Descent(self, X_train, y_train, X_test, y_test):
        
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
        rsearch_cv = RandomizedSearchCV(estimator = model, param_distributions = hyperparameter, cv = my_cv,
                                        scoring = "neg_mean_squared_error", n_iter = 100, n_jobs = -1)
        rsearch_cv.fit(X_train, y_train)
        sgb_best = rsearch_cv.best_estimator_
        sgb_best.fit(X_train, y_train)
        y_pred = sgb_best.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return sgb_best, mse, mae, r2
    
    def DecisionTree_regression(self, X_train, y_train, X_test, y_test):
        
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
        rsearch_cv = RandomizedSearchCV(estimator = model, param_distributions = hyperparameter, cv = my_cv,
                                        scoring = "neg_mean_squared_error", n_iter = 100, n_jobs = -1)
        rsearch_cv.fit(X_train, y_train)
        sgb_best = rsearch_cv.best_estimator_
        sgb_best.fit(X_train, y_train)
        y_pred = sgb_best.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return sgb_best, mse, mae, r2
    
    def Random_Forest(self, X_train, y_train, X_test, y_test):
        
        n_estimators = [5, 10, 50, 100, 150, 200, 250, 300]
        max_depth = [5, 10, 25, 50, 75, 100]
        min_samples_leaf = [1, 2, 4, 8, 10]
        min_samples_split = [2, 4, 6, 8, 10]
        max_features = ["auto", "sqrt", "log2", None]
        criterion = ["mse"]
    
        hyperparameter = {'n_estimators': n_estimators,
                      'max_depth': max_depth,
                      'min_samples_leaf': min_samples_leaf,
                      'min_samples_split': min_samples_split,
                      'max_features': max_features, 
                      'criterion': criterion
                  }
        
        my_cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
        base_model_rf = RandomForestRegressor(random_state=42)
        rsearch_cv = RandomizedSearchCV(estimator=base_model_rf, 
                                   random_state=42,
                                   param_distributions=hyperparameter,
                                   n_iter=50,
                                   cv=my_cv,
                                   scoring="neg_mean_squared_error",
                                   n_jobs=-1)
        rsearch_cv.fit(X_train, y_train)
        rb_best = rsearch_cv.best_estimator_
        rb_best.fit(X_train, y_train)
        y_pred = rb_best.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return rb_best, mse, mae, r2
    
    def fit(self, X_train, y_train, X_test, y_test):
        
        estimators = ["Ridge_regression", "LASSO_regression", "Lars_regression", 
                      "ElasticNet_regression", "LassoLars_regression", "MultiTaskLasso_regression", 
                      #"Support_Vector_regression", 
                      "KernelRidge_regression", "GaussianProcess_regression",
                      "Stochastic_Gradient_Descent", 
                      "DecisionTree_regression", "Random_Forest", 
                      #"Naive_Bayes", "Support_Vector_Classification",
                       #Random_Forest", "Gradient_Boosting", "Extreme_Gradient_Boosting",
                       #"Random_Forest", "Gradient_Boosting",
                       #"Decision_Tree", "Extra_Tree"
                      ]
        
        name_model = []
        all_model = []
        all_mse = []
        all_mae = []
        all_r2_score = []
        
        for est in estimators:
            print(est)
            if est == "Ridge_regression":
                best_model, mse, mae, r2 = self.Ridge_regression(X_train, y_train, X_test, y_test)
            elif est == "LASSO_regression":
                best_model, mse, mae, r2 = self.Lasso_regression(X_train, y_train, X_test, y_test) 
            elif est == "Lars_regression":
                best_model, mse, mae, r2 = self.Lars_regression(X_train, y_train, X_test, y_test) 
            elif est == "ElasticNet_regression":
                best_model, mse, mae, r2 = self.ElasticNet_regression(X_train, y_train, X_test, y_test) 
            elif est == "LassoLars_regression":
                best_model, mse, mae, r2 = self.LassoLars_regression(X_train, y_train, X_test, y_test) 
            elif est == "MultiTaskLasso":
                best_model, mse, mae, r2 = self.MultiTaskLasso_regression(X_train, y_train, X_test, y_test) 
            #elif est == "Support_Vector_regression":
            #    best_model, mse, mae, r2 = self.Support_Vector_regression(X_train, y_train, X_test, y_test)
            elif est == "KernelRidge_regression":
                best_model, mse, mae, r2 = self.KernelRidge_regression(X_train, y_train, X_test, y_test)
            elif est == "GaussianProcess_regression":
                best_model, mse, mae, r2 = self.GaussianProcess_regression(X_train, y_train, X_test, y_test)
            elif est == "Stochastic_Gradient_Descent":
                best_model, mse, mae, r2 = self.Stochastic_Gradient_Descent(X_train, y_train, X_test, y_test)
            elif est == "DecisionTree_regression":
                best_model, mse, mae, r2 = self.DecisionTree_regression(X_train, y_train, X_test, y_test)
            elif est == "Random_Forest":
                best_model, mse, mae, r2 = self.Random_Forest(X_train, y_train, X_test, y_test)
                
            
            name_model.append(est)
            all_model.append(best_model)
            all_mse.append(mse)
            all_mae.append(mae)
            all_r2_score.append(r2)
            
        name = pd.DataFrame(name_model)
        models = pd.DataFrame(all_model)
        mse = pd.DataFrame(all_mse)
        mae = pd.DataFrame(all_mae)
        r2_score = pd.DataFrame(all_r2_score)
        
        all_info = pd.concat([name, mae, mse, r2_score, models], axis = 1)
        all_info.columns = ["Name_Model", "MAE", "MSE", "R2_Score","Best_Model"]
        all_info = all_info.sort_values(by="MAE", ascending=True).reset_index(drop=True)
        
        return all_info

















































