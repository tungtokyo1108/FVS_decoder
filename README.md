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

### 3. Evaluate the performances
