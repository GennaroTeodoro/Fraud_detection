from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import GridSearchCV
from ProgettoDSML.util import *

fraud_Dataset = pd.read_csv("C:\\Users\\Gennaro Teodoro\\Desktop\\PS_20174392719_1491204439457_log.csv", sep=',')

fraud_Dataset = resample(fraud_Dataset, 3)

# Andiamo a creare l'attributo balance_rapp
fraud_Dataset["balance_rapp"] = (fraud_Dataset["oldbalanceOrg"]+0.1)/(fraud_Dataset["oldbalanceDest"]+0.1)

fraud_Dataset["balance_rapp"] = fraud_Dataset["balance_rapp"].astype(float)

# Andiamo ad utilizzare ora la tecnica del one hot encoding sull'attributo categorico type

fraud_Dataset_dummy = pd.get_dummies(fraud_Dataset["type"])
fraud_Dataset = fraud_Dataset.join(fraud_Dataset_dummy)

fraud_Dataset = fraud_Dataset.drop(columns=["type", "nameOrig", "nameDest", "newbalanceDest", "oldbalanceDest"])


print(fraud_Dataset.info())

fraud_Dataset_without_label = fraud_Dataset[["step", "amount", "oldbalanceOrg", "newbalanceOrig", "isFlaggedFraud",
                                             "balance_rapp", "CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]]

fraud_Dataset_label = fraud_Dataset["isFraud"]

scaler = StandardScaler()
scaled = scaler.fit_transform(fraud_Dataset_without_label)

columns = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "isFlaggedFraud", "balance_rapp",
           "CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


fraud_dataset_scaled = pd.DataFrame(scaled, columns=columns)


fraud_dataset_scaled.insert(11, "isFraud", fraud_Dataset_label.values)

print(fraud_dataset_scaled.info())

# fine fase preparazione inizio fase di training

dataset_without_label = fraud_dataset_scaled[["step", "amount", "oldbalanceOrg", "newbalanceOrig", "isFlaggedFraud",
                                              "balance_rapp", "CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]]

dataset_label = fraud_dataset_scaled["isFraud"]

param_grid = [
    {'random_state': [42], 'max_iter': [100, 200, 300, 400, 500, 600, 700],
     'class_weight': ["balanced"]},

    {'random_state': [42], 'max_iter': [100, 200, 300, 400, 500, 600, 700]},

    {'random_state': [42], 'max_iter': [100, 200, 300, 400, 500, 600, 700],
     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']},

    {'random_state': [42], 'max_iter': [100, 200, 300, 400, 500, 600, 700],
     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 'class_weight': ["balanced"]},

    {'random_state': [42], 'max_iter': [100, 200, 300, 400, 500, 600, 700],
     'solver': ['liblinear', 'saga'], 'penalty':['l1', 'l2', 'elasticnet']},

    {'random_state': [42], 'max_iter': [100, 200, 300, 400, 500, 600, 700],
     'solver': ['liblinear', 'saga'], 'penalty': ['l1', 'l2', 'elasticnet'], 'class_weight': ["balanced"]}
]

lr_clf = LogisticRegression()

grid_search = GridSearchCV(lr_clf, param_grid, cv=5, scoring="recall")
grid_search.fit(dataset_without_label, dataset_label)
print(grid_search.best_params_)
