#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve
import xgboost as xgb
import lightgbm as lgbm

ccdata = pd.read_csv(r'C:\Users\rihab\Downloads\DataSets\creditcard_2023.csv')

# DATA EXPLORATION
print('Columns:\n', ccdata.columns)
print('First Few Rows:\n', ccdata.head())
print('Description:\n', ccdata.describe)
print('Correlation:\n', ccdata.corr()['Class'])
print('Shape:\n', ccdata.shape)
print('Count of null values in each column:\n', ccdata.isnull().sum())

# DATA VISUALIZATION
ccdata.hist(figsize = (20, 20)) # plot histograms of each parameter 
plt.show()

corr_matrix = ccdata.corr() # plot correlation heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# DATA MODELING
X = ccdata.drop(['id', 'Class'], axis=1) #features/inputs
y = ccdata['Class'] #target/outputs

print(y.value_counts()) # show number of frauds and not frauds

# Split data into test and train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the input data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest model
clfmodel = RandomForestClassifier(verbose=2)
clfmodel.fit(X_train_scaled, y_train)

# Tune the model's hyperparameters
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 6]
}
# Prioriize for the best f1 score
grid_search = GridSearchCV(clfmodel, param_grid, scoring='f1', cv=5)
grid_search.fit(X_train_scaled, y_train)

# MODEL EVALUATION

# Predict and evaluate the Random Forest model
y_pred = clfmodel.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
# Predictions for true positives, false positives, false negatives and true negatives
confusionmatrix = confusion_matrix(y_test, y_pred)

print("Accuracy:\n", accuracy)
print('Random Forest Model:\n', classification_report(y_test, y_pred, output_dict=True))
print('AUC:\n', auc)
print('Confusion Matrix:\n',confusionmatrix)
print("Best Parameters:\n", grid_search.best_params_)
print("Best Model:\n", grid_search.best_estimator_)