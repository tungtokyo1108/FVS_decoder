# AutoML_FVS: A new approach to identify a small group of brain region from fMRI databases 

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

### 2. Forward variable selection algorithm

### 3. Evaluate the performances
