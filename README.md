# MLhousePricing

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
from datetime import datetime
from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.impute import SimpleImputer


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


![image](https://github.com/user-attachments/assets/9858d83f-21ce-4929-995c-0885ca5a87f2)

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
test2 = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print ("House prices dataset is loaded!")

print(train.describe())

![image](https://github.com/user-attachments/assets/ae3f0ad2-48ad-4ea4-b386-e47d39ae4538)


print("train set : \n",train.head())
print("test set : \n",test.head())


#Check if salePrice exists in train set
if 'SalePrice' in train.columns:
    y_train = train['SalePrice']
    print("y_train contain the salePrice")
else:
    print("Error: 'SalePrice' column is missing in the training data.")
    raise ValueError("'SalePrice' column is missing.")

![image](https://github.com/user-attachments/assets/9a0ba28f-e154-49b0-bf71-6fe056f4fa69)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.boxplot(x=train['SalePrice'], ax=axes[0, 0])
axes[0, 0].set_title('Distribution of house prices')
axes[0, 0].set_xlabel('Sale Price')

sns.scatterplot(x="LotArea", y="SalePrice", data=train, ax=axes[0, 1])
axes[0, 1].set_title("Relation between LotArea and SalePrice")
axes[0, 1].set_xlabel("LotArea")
axes[0, 1].set_ylabel("SalePrice")

avg_price_by_year = train.groupby('YearBuilt')['SalePrice'].mean()
sns.lineplot(x=avg_price_by_year.index, y=avg_price_by_year.values, ax=axes[1, 0])
axes[1, 0].set_title("Average SalePrice by YearBuilt")
axes[1, 0].set_xlabel("Year Built")
axes[1, 0].set_ylabel("Average SalePrice")

sns.boxplot(x="OverallQual", y="SalePrice", data=train, ax=axes[1, 1])
axes[1, 1].set_title("SalePrice by OverallQual")
axes[1, 1].set_xlabel("OverallQual")
axes[1, 1].set_ylabel("SalePrice")

##Adjust the space between the graphs
plt.tight_layout()

##display allign graphs
plt.show()

![image](https://github.com/user-attachments/assets/7542288f-b853-4d7f-9e99-0764f3655546)


# saving the target before removing
y_train = train['SalePrice'] if 'SalePrice' in train.columns else None
y_test = test['SalePrice'] if 'SalePrice' in test.columns else None

# Convert all columns to numeric categories if necessary
for col in train.select_dtypes(include=['object']).columns:
    if train[col].nunique() < 50:
        train[col] = train[col].astype('category').cat.codes 
    else:
        train = pd.get_dummies(train, columns=[col], drop_first=True)

for col in test.select_dtypes(include=['object']).columns:
    if test[col].nunique() < 50:
        test[col] = test[col].astype('category').cat.codes
    else:
        test = pd.get_dummies(test, columns=[col], drop_first=True)

# column types
print("train types:", train.dtypes)
print("test types:", test.dtypes)

# Saving numeric columns
numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns

# Defining columns to remove
columns_to_drop = ['SalePrice', 'Id']

# Check if the columns exists in DF
columns_to_drop_existing = [col for col in columns_to_drop if col in train.columns]

# Remove unnecessary columns (salePrice, id)
train = train.drop(columns=columns_to_drop_existing, errors='ignore')
test = test.drop(columns=['Id'], errors='ignore')

# Handle missing values with the mean strategy
numeric_imputer = SimpleImputer(strategy='mean')

# Replace infinity values with Nan
train[numeric_cols] = train[numeric_cols].replace([np.inf, -np.inf], np.nan)
test[numeric_cols] = test[numeric_cols].replace([np.inf, -np.inf], np.nan)

train[numeric_cols] = numeric_imputer.fit_transform(train[numeric_cols])
test[numeric_cols] = numeric_imputer.transform(test[numeric_cols])

plt.figure(figsize=(12, 6))

# Graph before handling exceptions
plt.subplot(1, 2, 1)
for col in ['GrLivArea', 'TotRmsAbvGrd', 'OverallQual', 'OverallCond']:
    plt.hist(train[col], bins=30, alpha=0.5, label=col)
plt.title('Before Outlier Treatment')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend()

# Handle exception values
for col in ['GrLivArea', 'TotRmsAbvGrd', 'OverallQual', 'OverallCond']:
    lower_limit = train[col].quantile(0.01)
    upper_limit = train[col].quantile(0.99)
    train[col] = np.clip(train[col], lower_limit, upper_limit)

# After handle the "special" values
plt.subplot(1, 2, 2)
for col in ['GrLivArea', 'TotRmsAbvGrd', 'OverallQual', 'OverallCond']:
    plt.hist(train[col], bins=30, alpha=0.5, label=col)
plt.title('After Outlier Treatment')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()

# New Features (tran & test)
train['HouseAge'] = train['YrSold'] - train['YearBuilt']
train['TotalArea'] = train['GrLivArea'] + train['BsmtFinSF1'] + train['BsmtFinSF2']
train['TotalRooms'] = train['TotRmsAbvGrd'] + train['FullBath'] + train['HalfBath']

test['HouseAge'] = test['YrSold'] - test['YearBuilt']
test['TotalArea'] = test['GrLivArea'] + test['BsmtFinSF1'] + test['BsmtFinSF2']
test['TotalRooms'] = test['TotRmsAbvGrd'] + test['FullBath'] + test['HalfBath']

# Validating missing values ​​after feature engineering
missing_data_train = train.isnull().sum()
missing_data_test = test.isnull().sum()

# Handle missing values after FE
train[numeric_cols] = numeric_imputer.fit_transform(train[numeric_cols])
test[numeric_cols] = numeric_imputer.transform(test[numeric_cols])

print("missing data after feature engineering (train):", missing_data_train[missing_data_train > 0])
print("missing data after feature engineering (test):", missing_data_test[missing_data_test > 0])

# Align the sets
train, test = train.align(test, join='left', axis=1, fill_value=0)

# Define variables
X_train = train
X_test = test

print(f"Shape of X_train: {X_train.shape}, Shape of X_test: {X_test.shape}")
if y_train is not None:
    print(f"Shape of y_train: {y_train.shape}")
else:
    print("Column 'SalePrice' is missing in the train dataset.")


![image](https://github.com/user-attachments/assets/6fc0c7e6-6886-43fc-998d-14db71c7f214)


# search parameters (tree number, tree depth, sample splits, minimum leaf)
# Model parameters
models_params = [
    ('Random Forest', RandomForestRegressor(random_state=42), {
        'model__n_estimators': [20, 30,60],
        'model__max_depth': [2, 3, 5],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2]
    }),
    ('Linear Regression', LinearRegression(), {
        'model__fit_intercept': [True, False]
    }),
    ('Gradient Boosting', GradientBoostingRegressor(random_state=42), {
        'model__n_estimators': [30, 60],
        'model__learning_rate': [0.01, 0.1],
        'model__max_depth': [3, 5, 7]
    }),
    ('Support Vector Regression', SVR(), {
        'model__kernel': ['linear', 'rbf'],
        'model__C': [1, 10],
        'model__gamma': ['scale', 'auto']
    })
]

summary_results = []

# Go over the models
for model_name, model, param_grid in models_params:
    if model_name == 'Linear Regression':
        #  Linear Regression add the normalization
        pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('model', model)])
    else:
        pipeline = Pipeline(steps=[('model', model)])

    #  grid search 5-fold cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)

    # Align set with training
    grid_search.fit(X_train, y_train)

    # Best results
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_


    if 'SalePrice' in test.columns:
        y_test = test['SalePrice']
        test_score = best_model.score(X_test, y_test)
    else:
        test_score = None


    
    summary_results.append({
        'Model': model_name,
        'Best CV R^2': best_score,
        'Test R^2': test_score,
        'Best Parameters': best_params
    })

# Get the best model, include handle situations with None values
best_overall_model = max(summary_results, key=lambda x: x['Test R^2'] if x['Test R^2'] is not None else -float('inf'))


print("\nBest Model:")
print(f"Model: {best_overall_model['Model']}")
print(f"Best hyperparameters: {best_overall_model['Best Parameters']}")

test_r2 = best_overall_model['Test R^2']
if test_r2 is not None:
    print(f"R^2 on test: {test_r2:.4f}")
else:
    print("R^2 on test: Not Available")


summary_df = pd.DataFrame(summary_results)


print("\nModels Summary:")
print(summary_df)

print("\nBest Model:")
print(f"Model: {best_overall_model['Model']}")
print(f"Best hyperparameters: {best_overall_model['Best Parameters']}")

The picture include 3 - fold instead of 5, I will add 5 as well in the same page.
![image](https://github.com/user-attachments/assets/00c1ad17-83d3-4c85-bb27-4e9bdde6beb7)



# Pipeline with the best model
best_model = Pipeline(steps=[
    ('scaler', StandardScaler()), 
    ('model', RandomForestRegressor(
        max_depth=5,
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=20,
        random_state=42
    ))
])

# Retrain the model with the set
best_model.fit(X_train, y_train)

# Test set Prediction
y_pred = best_model.predict(X_test)

print("First 5 predictions on test set:")
print(y_pred[:5])

if 'SalePrice' in test.columns:
    y_test = test['SalePrice']
    from sklearn.metrics import r2_score
    test_r2 = r2_score(y_test, y_pred)
    print(f"R^2 on test set: {test_r2:.4f}")
else:
    print("No SalePrice column in test set for evaluation.")

![image](https://github.com/user-attachments/assets/a5c70f44-bd0b-4011-9780-9ce55f491e1f)

