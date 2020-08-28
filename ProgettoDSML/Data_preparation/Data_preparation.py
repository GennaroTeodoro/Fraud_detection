import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from ProgettoDSML.util import *

fraud_Dataset = pd.read_csv("C:\\Users\\Gennaro Teodoro\\Desktop\\PS_20174392719_1491204439457_log.csv", sep=',')

fraud_Dataset = resample(fraud_Dataset, 3)

# Andiamo a creare ora dei nuovi attributi
fraud_Dataset["OldBalance_amount_percentage"] = (fraud_Dataset["oldbalanceOrg"]+0.1)/fraud_Dataset["amount"]
fraud_Dataset["balance_rapp"] = (fraud_Dataset["oldbalanceOrg"]+0.1)/(fraud_Dataset["oldbalanceDest"]+0.1)

fraud_Dataset["OldBalance_amount_percentage"] = fraud_Dataset["OldBalance_amount_percentage"].astype(float)
fraud_Dataset["balance_rapp"] = fraud_Dataset["balance_rapp"].astype(float)

# rivediamo le matrici di correlazioni in virtù dei nuovi attributi

print("Andiamo ora a valutare le correlazioni di tutti gli attributi con l'attributo isFraud")
corr_matrix = fraud_Dataset.corr()
print(corr_matrix["isFraud"].sort_values(ascending=False))


attributi_correlati_maggiormente = ["isFraud", "balance_rapp", "OldBalance_amount_percentage"]
pd.plotting.scatter_matrix(fraud_Dataset[attributi_correlati_maggiormente], figsize=(12, 8))
plt.show()


fraud_Dataset.plot(kind="scatter", y="isFraud", x="balance_rapp", alpha=0.1)
plt.show()


fraud_Dataset = fraud_Dataset.drop(columns=["OldBalance_amount_percentage"])

# Andiamo ad utilizzare ora la tecnica del one hot encoding sull'attributo categorico type

fraud_Dataset_dummy = pd.get_dummies(fraud_Dataset["type"])
fraud_Dataset = fraud_Dataset.join(fraud_Dataset_dummy)

# rimuovo type perchè non mi è più utile e nameOrig e name Dest perchè non li userò dopo
fraud_Dataset = fraud_Dataset.drop(columns=["type", "nameOrig", "nameDest"])

print(fraud_Dataset.head())

corr_matrix = fraud_Dataset.corr()
print(corr_matrix["isFraud"].sort_values(ascending=False))


scaler = StandardScaler()
scaled = scaler.fit_transform(fraud_Dataset)

columns = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
           "isFraud", "isFlaggedFraud", "amount_step", "balance_diff", "balance_rapp", "balance_diff_abs",
           "CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

fraud_dataset_scaled = pd.DataFrame(scaled, columns=columns)
