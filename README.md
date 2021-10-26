# AutoML_FVS: A new approach to identify a important group of brain region from fMRI databases 

## Main commands and options

### 1. Automatic machine learning approaches 

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from Auto_ML_Multiclass import AutoML_classification
from Auto_ML_Regression import AutoML_Regression
```

Import data and run automatic machine learning algorithm 

```
ress_BPD_brain = pd.read_csv("BPD_brain.csv", header=None)
ress_BPD_meta = pd.read_csv("BPD_rrs.csv", header=None)
y = ress_BPD_meta["RRS_Brooding"]

X_train, X_test, y_train, y_test = train_test_split(ress_BPD_brain, y, test_size=0.3, random_state=42)

automl = AutoML_Regression()
result = automl.fit(X_train, y_train, X_test, y_test)

```
Outputs are shown in table 

| Name_Model                  | MSE       | MAE      | R2_Score |
| --------------------------- |:---------:|:--------:|:--------:|
| KernelRidge_regression      | 15.396256 | 3.192229 | 0.064702 |
| LassoLars_regression        | 15.589468 | 3.193472 | 0.052964 |
| MultiTaskLasso_regression   | 15.589468 | 3.193472 | 0.052964 |
| Ridge_regression            | 16.224041 | 3.228338 | 0.014415 |
| ElasticNet_regression       | 16.345417 | 3.239610 | 0.007041 |
| Lars_regression             | 16.729033 | 3.252711 | -0.01626 |
| Stochastic_Gradient_Descent | 17.032267 | 3.353871 | -0.03468 |
| LASSO_regression            | 17.079694 | 3.252711 | -0.03756 |
| DecisionTree_regression     | 18.015736 | 3.422243 | -0.09442 |
| Random_Forest               | 18.892558 | 3.570010 | -0.14769 |

### 2. Forward variable selection algorithm

After selecting the best algorithm for analyzing our database, we go to the next step that run forward variable selection to identify a important group of brain regions. For example, in our database, the kernel ridge regression is the best model with the smallest value of MSE. Thus, we start with combination of the kernel ridge regression and forward variable selection. 

```
from FVS_algorithm import AutoML_FVS
fvs = AutoML_FVS()
all_info, all_model, f = fvs.KernelRidge_FVS(X_train, y_train, X_test, y_test, n_selected_features = 200)

[Parallel(n_jobs=-1)]: Using backend LokyBackend with 80 concurrent workers.
[Parallel(n_jobs=-1)]: Done  40 tasks      | elapsed:   38.9s
[Parallel(n_jobs=-1)]: Done 246 out of 246 | elapsed:  2.4min finished
The current number of features: 1 - MSE: 12.51

[Parallel(n_jobs=-1)]: Using backend LokyBackend with 80 concurrent workers.
[Parallel(n_jobs=-1)]: Done  40 tasks      | elapsed:   36.9s
[Parallel(n_jobs=-1)]: Done 246 out of 246 | elapsed:  2.4min finished
The current number of features: 2 - MSE: 10.48

[Parallel(n_jobs=-1)]: Using backend LokyBackend with 80 concurrent workers.
[Parallel(n_jobs=-1)]: Done  40 tasks      | elapsed:   41.4s
[Parallel(n_jobs=-1)]: Done 246 out of 246 | elapsed:  2.5min finished
The current number of features: 3 - MSE: 9.67

.....

```

### 3. Evaluate the performances
