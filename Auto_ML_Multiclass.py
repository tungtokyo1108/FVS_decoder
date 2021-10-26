#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 22:12:39 2020

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
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier


class AutoML_classification():
    
    def __init__(self, random_state = None):
        self.random_state = random_state
        
    def LogisticRegression(self, X_train, y_train, X_test, y_test):
        
        # Inverse of regularization strength. Smaller values specify stronger regularization. 
        c = np.linspace(0.001, 1, 100)
        
        """
        penalty = ["l2", "l1", "elasticnet"]
        
        # The Elastic Net mixing parameter 
        l1_ratio = np.linspace(0, 1, 100)
        
        solver = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
        
        hyperparameter = {"C": c, 
                          "penalty": penalty,
                          "l1_ratio": l1_ratio, 
                          "solver": solver}
        """
        
        tuned_parameters = [{"C": c}]
        n_folds = 10 
        #model = LogisticRegression(max_iter=1000)
        model = LogisticRegression(penalty="l1", solver = "liblinear")
        my_cv = TimeSeriesSplit(n_splits = n_folds).split(X_train)
        #gsearch_cv = RandomizedSearchCV(estimator = model, param_distributions = hyperparameter, 
        #                          scoring = "f1_macro", cv = my_cv, n_jobs=-1, n_iter = 100)
        
        gsearch_cv = GridSearchCV(estimator = model, param_grid = tuned_parameters, 
                                  scoring = "f1_macro", cv = my_cv, n_jobs=-1)
        gsearch_cv.fit(X_train, y_train)
        best_model = gsearch_cv.best_estimator_
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred, normalize=True) * 100
        precision = np.round(metrics.precision_score(y_test, y_pred, average="macro"), 4)
        recall = np.round(metrics.recall_score(y_test, y_pred, average="macro"), 4)
        f1 = np.round(metrics.f1_score(y_test, y_pred, average="macro"), 4)
        
        return best_model, test_accuracy, precision, recall, f1
    
    def Stochastic_Gradient_Descent(self, X_train, y_train, X_test, y_test):
        
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
        rsearch_cv = RandomizedSearchCV(estimator = model, param_distributions = hyperparameter, cv = my_cv,
                                        scoring = "f1_macro", n_iter = 100, n_jobs = -1)
        rsearch_cv.fit(X_train, y_train)
        sgb_best = rsearch_cv.best_estimator_
        sgb_best.fit(X_train, y_train)
        y_pred = sgb_best.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred, normalize = True) * 100
        precision = np.round(metrics.precision_score(y_test, y_pred, average="macro"), 4)
        recall = np.round(metrics.recall_score(y_test, y_pred, average="macro"), 4)
        f1 = np.round(metrics.f1_score(y_test, y_pred, average="macro"), 4)
        
        return sgb_best, test_accuracy, precision, recall, f1
    
    def Naive_Bayes(self, X_train, y_train, X_test, y_test):
        
        alphas = np.logspace(0,1,100)
        tuned_parameters = [{"alpha": alphas}]
        n_folds = 10
        model = MultinomialNB()
        my_cv = TimeSeriesSplit(n_splits=n_folds).split(X_train)
        gsearch_cv = GridSearchCV(estimator = model, param_grid = tuned_parameters, cv = my_cv, scoring="f1_macro", n_jobs=-1)
        gsearch_cv.fit(X_train, y_train)
        nb_best = gsearch_cv.best_estimator_
        nb_best.fit(X_train, y_train)
        y_pred = nb_best.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred, normalize = True) * 100
        precision = np.round(metrics.precision_score(y_test, y_pred, average="macro"), 4)
        recall = np.round(metrics.recall_score(y_test, y_pred, average="macro"), 4)
        f1 = np.round(metrics.f1_score(y_test, y_pred, average="macro"), 4)
        
        return nb_best, test_accuracy, precision, recall, f1
    
    def LinearDiscriminantAnalysis(self, X_train, y_train, X_test, y_test):
        
        shrinkage = list(np.linspace(0, 1, num = 20))
        shrinkage.append("auto")
        shrinkage.append("None")
        solver = ["lsqr", "eigen"]
        hyper_param = {"shrinkage": shrinkage,
               "solver": solver}
        n_folds = 10
        lda = LinearDiscriminantAnalysis()
        my_cv = TimeSeriesSplit(n_splits = n_folds).split(X_train)
        randomsearch_cv = RandomizedSearchCV(estimator = lda, param_distributions = hyper_param, cv = my_cv,
                                     scoring = "f1_macro", n_iter = 30, n_jobs = -1)
        randomsearch_cv.fit(X_train, y_train)
        lda_best = randomsearch_cv.best_estimator_
        lda_best.fit(X_train, y_train)
        y_pred = lda_best.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred, normalize = True) * 100
        precision = np.round(metrics.precision_score(y_test, y_pred, average="macro"), 4)
        recall = np.round(metrics.recall_score(y_test, y_pred, average="macro"), 4)
        f1 = np.round(metrics.f1_score(y_test, y_pred, average="macro"), 4)
        
        return lda_best, test_accuracy, precision, recall, f1
    
    def Support_Vector_Classify(self, X_train, y_train, X_test, y_test):
        
        C = np.logspace(-2, 7, 100)
        kernel = ["linear", "poly", "rbf", "sigmoid"]
        gamma = list(np.logspace(-1, 1, 100))
        gamma.append("scale")
        gamma.append("auto")
        hyper_param = {"C": C, 
               "kernel": kernel,
               "gamma": gamma}
        n_folds = 10
        svc = SVC()
        my_cv = TimeSeriesSplit(n_splits = n_folds).split(X_train)

        randomsearch_cv = RandomizedSearchCV(estimator = svc, param_distributions = hyper_param, cv = my_cv,
                                     scoring = "f1_macro", n_iter = 50, n_jobs = -1)
        randomsearch_cv.fit(X_train, y_train)
        svc_best = randomsearch_cv.best_estimator_
        svc_best.fit(X_train, y_train)
        y_pred = svc_best.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred, normalize=True) * 100
        precision = np.round(metrics.precision_score(y_test, y_pred, average="macro"), 4)
        recall = np.round(metrics.recall_score(y_test, y_pred, average="macro"), 4)
        f1 = np.round(metrics.f1_score(y_test, y_pred, average="macro"), 4)
        
        return svc_best, test_accuracy, precision, recall, f1
    
    def Random_Forest(self, X_train, y_train, X_test, y_test):
        
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
        
        criterion = ["gini", "entropy"]
    
        hyperparameter = {'n_estimators': n_estimators,
                      'max_depth': max_depth,
                      'min_samples_leaf': min_samples_leaf,
                      'min_samples_split': min_samples_split,
                      'max_features': max_features, 
                      'criterion': criterion}
        n_folds = 10
        my_cv = TimeSeriesSplit(n_splits = n_folds).split(X_train)
        base_model_rf = RandomForestClassifier(random_state=42)
        rsearch_cv = RandomizedSearchCV(estimator=base_model_rf, 
                                   random_state=42,
                                   param_distributions=hyperparameter,
                                   n_iter=50,
                                   cv=my_cv,
                                   scoring="f1_macro",
                                   n_jobs=-1)
        rsearch_cv.fit(X_train, y_train)
        rb_best = rsearch_cv.best_estimator_
        rb_best.fit(X_train, y_train)
        y_pred = rb_best.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred, normalize=True) * 100
        precision = np.round(metrics.precision_score(y_test, y_pred, average="macro"), 4)
        recall = np.round(metrics.recall_score(y_test, y_pred, average="macro"), 4)
        f1 = np.round(metrics.f1_score(y_test, y_pred, average="macro"), 4)
        
        return rb_best, test_accuracy, precision, recall, f1
    
    def Gradient_Boosting(self, X_train, y_train, X_test, y_test):
        
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
        base_model_gb = GradientBoostingClassifier(random_state=42)
        rsearch_cv = RandomizedSearchCV(estimator=base_model_gb, 
                                   random_state=42,
                                   param_distributions=hyperparameter,
                                   n_iter=50,
                                   cv=my_cv,
                                   scoring="f1_macro",
                                   n_jobs=-1)
        rsearch_cv.fit(X_train, y_train)
        gb_best = rsearch_cv.best_estimator_
        gb_best.fit(X_train, y_train)
        y_pred = gb_best.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred, normalize=True) * 100
        precision = np.round(metrics.precision_score(y_test, y_pred, average="macro"), 4)
        recall = np.round(metrics.recall_score(y_test, y_pred, average="macro"), 4)
        f1 = np.round(metrics.f1_score(y_test, y_pred, average="macro"), 4)
        
        return gb_best, test_accuracy, precision, recall, f1
    
    def Extreme_Gradient_Boosting(self, X_train, y_train, X_test, y_test):
        
        n_estimators = [5, 10, 50, 100, 150, 200, 250, 300]
        max_depth = [5, 10, 25, 50, 75, 100]
        min_child_weight = [5, 10, 25, 50, 75, 100]
        gamma = [0.5, 1, 1.5, 2, 5]
        subsample = [0.2, 0.4, 0.6, 0.8, 1]
        colsample_bytree = [0.2, 0.4, 0.6, 0.8, 1]
        
        hyperparameter = {'n_estimators': n_estimators,
                      'max_depth': max_depth,
                      'min_child_weight': min_child_weight,
                      'gamma': gamma,
                      'subsample': subsample,
                      'colsample_bytree': colsample_bytree}
        
        n_folds = 10
        my_cv = TimeSeriesSplit(n_splits = n_folds).split(X_train)
        xgb = XGBClassifier(learning_rate=0.02, objective='multi:softmax', silent=True, nthread=-1)
        rsearch_cv = RandomizedSearchCV(estimator=xgb, param_distributions=hyperparameter, n_iter=50, 
                               scoring='f1_macro', n_jobs=-1, cv=my_cv, verbose=3, random_state=42)
        rsearch_cv.fit(X_train, y_train)
        xgb_best = rsearch_cv.best_estimator_
        xgb_best.fit(X_train, y_train)
        y_pred = xgb_best.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred, normalize=True) * 100
        precision = np.round(metrics.precision_score(y_test, y_pred, average="macro"), 4)
        recall = np.round(metrics.recall_score(y_test, y_pred, average="macro"), 4)
        f1 = np.round(metrics.f1_score(y_test, y_pred, average="macro"), 4)
        
        return xgb_best, test_accuracy, precision, recall, f1
    
    def Decision_Tree(self, X_train, y_train, X_test, y_test):
        
        max_depth = [5, 10, 25, 50, 75, 100]
        min_samples_leaf = [1, 2, 4, 8, 10]
        min_samples_split = [2, 4, 6, 8, 10]
        max_features = ["auto", "sqrt", "log2", None]
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
        dt = DecisionTreeClassifier(random_state = 42)
        rsearch_cv = RandomizedSearchCV(estimator = dt, param_distributions = hyperparameter, n_iter=50,
                                        scoring = "f1_macro", n_jobs = -1, cv = my_cv, random_state = 42)
        rsearch_cv.fit(X_train, y_train)
        dt_best = rsearch_cv.best_estimator_
        dt_best.fit(X_train, y_train)
        y_pred = dt_best.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred, normalize=True) * 100
        precision = np.round(metrics.precision_score(y_test, y_pred, average="macro"), 4)
        recall = np.round(metrics.recall_score(y_test, y_pred, average="macro"), 4)
        f1 = np.round(metrics.f1_score(y_test, y_pred, average="macro"), 4)

        return dt_best, test_accuracy, precision, recall, f1
    
    def Extra_Tree(self, X_train, y_train, X_test, y_test):
        
        max_depth = [5, 10, 25, 50, 75, 100]
        min_samples_leaf = [1, 2, 4, 8, 10]
        min_samples_split = [2, 4, 6, 8, 10]
        max_features = ["auto", "sqrt", "log2", None]
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
        et = ExtraTreeClassifier(random_state = 42)
        rsearch_cv = RandomizedSearchCV(estimator = et, param_distributions = hyperparameter, n_iter = 50,
                                        scoring = "f1_macro", n_jobs = -1, cv = my_cv, random_state = 42)
        rsearch_cv.fit(X_train, y_train)
        et_best = rsearch_cv.best_estimator_
        et_best.fit(X_train, y_train)
        y_pred = et_best.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred, normalize=True) * 100
        precision = np.round(metrics.precision_score(y_test, y_pred, average="macro"), 4)
        recall = np.round(metrics.recall_score(y_test, y_pred, average="macro"), 4)
        f1 = np.round(metrics.f1_score(y_test, y_pred, average="macro"), 4)
        
        return et_best, test_accuracy, precision, recall, f1
        
    def fit(self, X_train, y_train, X_test, y_test):
        
        estimators = ["Losgistic_Regression", "Stochastic_Gradient_Descent", "Naive_Bayes", "Support_Vector_Classification",
                       #Random_Forest", "Gradient_Boosting", "Extreme_Gradient_Boosting",
                       "Random_Forest", "Gradient_Boosting",
                       "Decision_Tree", "Extra_Tree"]
        name_model = []
        all_model = []
        all_acc = []
        all_pre = []
        all_recall = []
        all_f1 = []
        
        for est in estimators:
            print(est)
            if est == "Losgistic_Regression":
                best_model, accuracy, precision, recall, f1 = self.LogisticRegression(X_train, y_train, X_test, y_test)
            elif est == "Stochastic_Gradient_Descent":
                best_model, accuracy, precision, recall, f1 = self.Stochastic_Gradient_Descent(X_train, y_train, X_test, y_test) 
            elif est == "Naive_Bayes":
                best_model, accuracy, precision, recall, f1 = self.Naive_Bayes(X_train, y_train, X_test, y_test)
            elif est == "Support_Vector_Classification":
                best_model, accuracy, precision, recall, f1 = self.Support_Vector_Classify(X_train, y_train, X_test, y_test)
            elif est == "Random_Forest":
                best_model, accuracy, precision, recall, f1 = self.Random_Forest(X_train, y_train, X_test, y_test)
            elif est == "Gradient_Boosting": 
                best_model, accuracy, precision, recall, f1 = self.Gradient_Boosting(X_train, y_train, X_test, y_test)
            #elif est == "Extreme_Gradient_Boosting":
            #    best_model, accuracy, precision, recall, f1 = self.Extreme_Gradient_Boosting(X_train, y_train, X_test, y_test)
            elif est == "Decision_Tree":
                best_model, accuracy, precision, recall, f1 = self.Decision_Tree(X_train, y_train, X_test, y_test)
            elif est == "Extra_Tree":
                best_model, accuracy, precision, recall, f1 = self.Extra_Tree(X_train, y_train, X_test, y_test)
            
            name_model.append(est)
            all_model.append(best_model)
            all_acc.append(accuracy)
            all_pre.append(precision)
            all_recall.append(recall)
            all_f1.append(f1)
        
        name = pd.DataFrame(name_model)
        models = pd.DataFrame(all_model)
        acc = pd.DataFrame(all_acc)
        pr = pd.DataFrame(all_pre)
        re = pd.DataFrame(all_recall)
        f = pd.DataFrame(all_f1)
        
        all_info = pd.concat([name, acc, pr, re, f, models], axis = 1)
        all_info.columns = ["Name_Model", "Accuracy", "Precision", "Recall", "F1_Score","Best_Model"]
        all_info = all_info.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
        
        return all_info
    
    def evaluate_multiclass(self, best_clf, X_train, y_train, X_test, y_test, 
                        model="Random Forest", num_class=3, top_features=2, class_name = ""):
        print("-"*100)
        print("~~~~~~~~~~~~~~~~~~ PERFORMANCE EVALUATION ~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Detailed report for the {} algorithm".format(model))
    
        best_clf.fit(X_train, y_train)
        y_pred = best_clf.predict(X_test)
        #y_pred_prob = best_clf.predict_proba(X_test)
        y_pred_prob = best_clf.predict(X_test)
    
        test_accuracy = accuracy_score(y_test, y_pred, normalize=True) * 100
        points = accuracy_score(y_test, y_pred, normalize=False)
        print("The number of accurate predictions out of {} data points on unseen data is {}".format(
            X_test.shape[0], points))
        print("Accuracy of the {} model on unseen data is {}".format(
            model, np.round(test_accuracy, 2)))
    
        print("Precision of the {} model on unseen data is {}".format(
            model, np.round(metrics.precision_score(y_test, y_pred, average="macro"), 4)))
        print("Recall of the {} model on unseen data is {}".format(
           model, np.round(metrics.recall_score(y_test, y_pred, average="macro"), 4)))
        print("F1 score of the {} model on unseen data is {}".format(
            model, np.round(metrics.f1_score(y_test, y_pred, average="macro"), 4)))
    
        print("\nClassification report for {} model: \n".format(model))
        print(metrics.classification_report(y_test, y_pred))
    
        plt.figure(figsize=(12,12))
        cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
        cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print("\nThe Confusion Matrix: \n")
        print(cnf_matrix)
    
        class_name = class_name
        cmap = plt.cm.Blues
        plt.imshow(cnf_matrix_norm, interpolation="nearest", cmap=cmap)
        plt.colorbar()
        fmt = ".2g"
        thresh = cnf_matrix_norm.max()/2
        for i, j in itertools.product(range(cnf_matrix_norm.shape[0]), range(cnf_matrix_norm.shape[1])):
            plt.text(j,i,format(cnf_matrix_norm[i,j], fmt), ha="center", va="center", 
                 color="white" if cnf_matrix_norm[i,j] > thresh else "black", fontsize=35)
    
        plt.xticks(np.arange(num_class), labels = class_name, fontsize=30, rotation=45, 
                   horizontalalignment='right')
        plt.yticks(np.arange(num_class), labels = class_name, fontsize=30)
    
    
        plt.ylabel("True label", fontsize=30)
        plt.xlabel("Predicted label", fontsize=30)
        plt.ylim((num_class - 0.5, -0.5))
        plt.show()
    
        print("\nROC curve and AUC")
        y_pred = best_clf.predict(X_test)
        y_pred_prob = best_clf.predict_proba(X_test)
        y_test_cat = np.array(pd.get_dummies(y_test))

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(num_class):
            fpr[i], tpr[i], _ = metrics.roc_curve(y_test_cat[:,i], y_pred_prob[:,i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_class)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_class):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        
        mean_tpr /= num_class
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

        plt.figure(figsize=(12,12))
        plt.plot(fpr["macro"], tpr["macro"], 
                 label = "macro-average ROC curve with AUC = {} - Accuracy = {}%".format(
                 round(roc_auc["macro"], 2), round(test_accuracy, 2)),
                 color = "navy", linestyle=":", linewidth=4)
        colors = sns.color_palette()
        for i, color in zip(range(num_class), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label = "ROC curve of class {0} (AUC = {1:0.2f})".format(i, roc_auc[i]))   
        plt.plot([0,1], [0,1], "k--", lw=3, color='red')
        plt.title("ROC-AUC for {}".format(model), fontsize=20)
        plt.xlabel("False Positive Rate", fontsize=15)
        plt.ylabel("True Positive Rate", fontsize=15)
        plt.legend(loc="lower right")
        plt.show()
    
    
        if model == "Random Forest" or model == "XGBoost":
            importances = best_clf.feature_importances_
            indices = np.argsort(importances)[::-1]
        
            feature_tab = pd.DataFrame({"Features": list(X_train.columns),
                                    "Importance": importances})
            feature_tab = feature_tab.sort_values("Importance", ascending = False).reset_index(drop=True)
        
            index = feature_tab["Features"].iloc[:top_features]
            importance_desc = feature_tab["Importance"].iloc[:top_features]  
            feature_space = []
            for i in range(indices.shape[0]-1, -1, -1):
                feature_space.append(X_train.columns[indices[i]])
        
            fig, ax = plt.subplots(figsize=(20,20))
            ax = plt.gca()
            plt.title("Feature importances", fontsize=30)
            plt.barh(index, importance_desc, align="center", color="blue", alpha=0.6)
            plt.grid(axis="x", color="white", linestyle="-")
            plt.xlabel("The Average of Decrease in Impurity", fontsize=20)
            plt.ylabel("Features", fontsize=30)
            plt.yticks(fontsize=30)
            plt.xticks(fontsize=20)
            ax.tick_params(axis="both", which="both", length=0)
            plt.show()
        
            return {"importance": feature_tab, 
                    "y_pred": y_pred,
                    "y_pred_prob": y_pred_prob}
    
        return {"y_pred": y_pred,
                "y_pred_prob": y_pred_prob}









































