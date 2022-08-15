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
from FVS_algorithm import AutoML_FVS

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
kr_best.fit(X_train, y_train)
y_pred = kr_best.predict(X_test)

dt = {"True AgeTag": y_test, "Predicted AgeTag": y_pred}
df = pd.DataFrame(dt)

g = sns.lmplot(x="True AgeTag", y="Predicted AgeTag", data=df)
g.set(ylim = (min(y_test), max(y_test)))
g.set(xlim = (min(y_test), max(y_test)))
#plt.text(-0.015, 0.006, r'Corr = %.2f' % (spearmanr(y_test, y_pred)[0]))


###############################################################################
################### Step 3 - Run forward algorithm ############################
###############################################################################









































