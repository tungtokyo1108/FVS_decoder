#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 22:56:48 2022

@author: tungdang
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pickle 
import os, sys
from joblib import Parallel, delayed
import math
from scipy.stats import spearmanr
from Auto_ML_Multiclass import AutoML_classification
from Auto_ML_Regression import AutoML_Regression
from FVS_Regression import AutoML_FVS_Regression
import warnings 
warnings.simplefilter("ignore")

###############################################################################
############################## Read data set ##################################
###############################################################################

bna = pd.read_csv("ROI_test.csv", index_col="BNAsubjID")
meta = pd.read_csv("Meta_test.csv", index_col="Subject")

# Regression 
y = meta["AgeTag"]

# Classification 
y = meta["Gender"].apply(lambda x: 0 
                             if x == "M" else 1)
class_name = ["Male", "Female"]

X_train, X_test, y_train, y_test = train_test_split(bna, y, test_size=0.3, random_state=42)

###############################################################################
######################## Step 1 - Run Auto_ML #################################
###############################################################################

# Classification
automl = AutoML_classification()
result = automl.fit(X_train, y_train, X_test, y_test)

logistic_best, _, _, _, _ = automl.LogisticRegression(X_train, y_train, X_test, y_test)
evaluate_logistic = automl.evaluate_multiclass(logistic_best, X_train, y_train, X_test, y_test,
                            model = "Losgistic_Regression", num_class=2, class_name = class_name)

rf_best, _, _, _, _ = automl.Random_Forest(X_train, y_train, X_test, y_test)
evaluate_rf = automl.evaluate_multiclass(rf_best, X_train, y_train, X_test, y_test,
                            model = "Random_Forest", num_class=2, class_name = class_name)


# Regression 
automl = AutoML_Regression()
result = automl.fit(X_train, y_train, X_test, y_test)

kr_best, _, _, _ = automl.KernelRidge_regression(X_train, y_train, X_test, y_test)
evaluate_r = automl.evaluate_regression(kr_best, X_train, y_train, X_test, y_test, model="Kernal Ridge regression",
                                        name_target = "AgeTag", feature_evaluate = True)

rd_best, _, _, _ = automl.Ridge_regression(X_train, y_train, X_test, y_test)
evaluate_r = automl.evaluate_regression(rd_best, X_train, y_train, X_test, y_test, model="Ridge regression",
                                        name_target = "AgeTag", feature_evaluate = True)

rf_best, _, _, _ = automl.Random_Forest(X_train, y_train, X_test, y_test)
evaluate_r = automl.evaluate_regression(rf_best, X_train, y_train, X_test, y_test, model="Random Forest",
                                        name_target = "AgeTag", feature_evaluate = False, top_features=10)


####################################################################################################
################### Step 3 - Run forward variable selection (FVS) algorithm ########################
####################################################################################################

fvs = AutoML_FVS_Regression()

all_info, all_model, f = fvs.KernelRidge_FVS(X_train, y_train, X_test, y_test, n_selected_features = 10)

all_info, all_model, f = fvs.RF_FVS(X_train, y_train, X_test, y_test, n_selected_features = 10)

all_info, all_model, f = fvs.Stochastic_Gradient_Descent_FVS(X_train, y_train, X_test, y_test, n_selected_features = 10)

all_info, all_model, f = fvs.DecisionTree_FVS(X_train, y_train, X_test, y_test, n_selected_features = 10)

all_info, all_model, f = fvs.ElasticNet_FVS(X_train, y_train, X_test, y_test, n_selected_features = 10)

all_info, all_model, f = fvs.LassoLars_FVS(X_train, y_train, X_test, y_test, n_selected_features = 10)

all_info, all_model, f = fvs.Ridge_FVS(X_train, y_train, X_test, y_test, n_selected_features = 10)

all_info, all_model, f = fvs.Lasso_FVS(X_train, y_train, X_test, y_test, n_selected_features = 10)

all_info, all_model, f = fvs.GaussianProcess_FVS(X_train, y_train, X_test, y_test, n_selected_features = 10)


###################################################################################

subset = f
subset = subset.drop(columns = "All")
load_grid_model = all_model

best_model_69 = load_grid_model[6]
subset = subset.iloc[6].dropna()
region_subset = bna[subset]

X_train, X_test, y_train, y_test = train_test_split(region_subset, y, test_size=0.3, random_state=42)

best_model_69.fit(X_train, y_train)
evaluate_r = automl.evaluate_regression(best_model_69, X_train, y_train, X_test, y_test, model="Ridge regression",
                                        name_target = "AgeTag", feature_evaluate = True)


