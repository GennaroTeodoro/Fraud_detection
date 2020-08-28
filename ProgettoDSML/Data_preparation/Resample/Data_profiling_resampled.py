import pandas as pd
from matplotlib import pyplot as plt
from ProgettoDSML.util import *

fraud_Dataset = pd.read_csv("C:\\Users\\Gennaro Teodoro\\Desktop\\PS_20174392719_1491204439457_log.csv", sep=',')

fraud_Dataset = resample(fraud_Dataset, 1)

print(fraud_Dataset.info())
print(fraud_Dataset.describe())

print("1. la cardinalità del dataset è:", len(fraud_Dataset))
print("2. Vediamo ora i valori per ogni colonna di min, max, median, avarege, count: \n")
numeric_column_names = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
                        "isFraud", "isFlaggedFraud"]

for numeric_column_name in numeric_column_names:
    print("\n" + numeric_column_name)
    print("min:", fraud_Dataset[numeric_column_name].min(),
          "\tmax:", fraud_Dataset[numeric_column_name].max(),
          "\tmedian:", fraud_Dataset[numeric_column_name].median(),
          "\tavarege:", fraud_Dataset[numeric_column_name].mean(),
          "\tcount:", fraud_Dataset[numeric_column_name].count())

print("\n3. Vediamo ora il numero di valori null per ogni colonna:\n")
print(fraud_Dataset.isnull().sum())

print("\n4. Vediamo ora il numero di valori distinti per ogni colonna:\n", fraud_Dataset.nunique())

column_names = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest", "oldbalanceDest",
                "newbalanceDest", "isFraud", "isFlaggedFraud"]

print("5. Calcoliamo ora la percentuale di unicità per ogni attributo")
for column_name in column_names:
    # calcolo la percentuale facendo numero di valori distinti diviso il numero totale di valori
    print(column_name+":", fraud_Dataset[column_name].nunique()/fraud_Dataset[column_name].count())

print("6. Vediamo ora l'istogramma dei valori\n")

fraud_Dataset.hist(bins=50)

for numeric_column_name in numeric_column_names:
    plt.hist(fraud_Dataset[numeric_column_name], bins=50, edgecolor="black")
    plt.ticklabel_format(style="plain")
    plt.legend(title=numeric_column_name)
    plt.show()


print("7. Vediamo ora la frequenza dei valori  più ricorrenti in ogni colonna\n")
for column_name in column_names:
    print(column_name+":", fraud_Dataset[column_name].value_counts(normalize="true"))

print("8. Calcoliamo ora i quartili\n")
print("25%", fraud_Dataset.quantile(0.25))
print("50%", fraud_Dataset.quantile(0.50))
print("75%", fraud_Dataset.quantile(0.75))
print("100%", fraud_Dataset.quantile(1))


# Creiamo ora i dataset di training e di test
# Andiamo ad aggiungere ora la colonna Id al dataset su cui andare a eseguire lo split del dataset
fraud_Dataset_with_id = fraud_Dataset.reset_index()
train_set, test_set = split_train_test_by_id(fraud_Dataset_with_id, 0.3, "index")
print(len(test_set))
print(len(train_set))

print("Andiamo ora a valutare le correlazioni di tutti gli attributi con l'attributo isFraud")
corr_matrix = fraud_Dataset.corr()
print(corr_matrix["isFraud"].sort_values(ascending=False))


attributi_correlati_maggiormente = ["isFraud", "amount", "isFlaggedFraud", "step", "oldbalanceOrg"]
pd.plotting.scatter_matrix(fraud_Dataset[attributi_correlati_maggiormente], figsize=(12, 8))
plt.show()


fraud_Dataset.plot(kind="scatter", y="isFraud", x="amount", alpha=0.1)
plt.show()
