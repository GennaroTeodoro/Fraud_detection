from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, plot_precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
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

# Andiamo ad aggiungere ora la colonna Id al dataset su cui andare a eseguire lo split del dataset
fraud_Dataset_with_id = fraud_dataset_scaled.reset_index()
train_set, test_set = split_train_test_by_id(fraud_Dataset_with_id, 0.3, "index")

# andiamo a tenere soltanto gli attributi numerici

train_set_numeric = train_set[["step", "amount", "oldbalanceOrg", "newbalanceOrig", "isFlaggedFraud", "balance_rapp",
                              "CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]]

train_set_numeric_labels = train_set["isFraud"]

test_set_numeric = test_set[["step", "amount", "oldbalanceOrg", "newbalanceOrig", "isFlaggedFraud", "balance_rapp",
                            "CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]]

test_set_numeric_labels = test_set["isFraud"]


print(len(train_set_numeric), len(test_set_numeric))


rnd_clf = RandomForestClassifier(criterion='entropy', n_estimators=125, random_state=42, max_features='auto')
rnd_clf.fit(train_set_numeric, train_set_numeric_labels)

print("accurcy on train set: ")
predict_label = rnd_clf.predict(train_set_numeric)
print(rnd_clf.__class__.__name__, accuracy_score(train_set_numeric_labels, predict_label))

print("accurcy on test set: ")
predict_label = rnd_clf.predict(test_set_numeric)
print(rnd_clf.__class__.__name__, accuracy_score(test_set_numeric_labels, predict_label))

print("confusion matrix random forest output: ", confusion_matrix(test_set_numeric_labels, predict_label))


print("la precision del classificatore è:", precision_score(test_set_numeric_labels, predict_label))
print("Il recall del classificatore è:",  recall_score(test_set_numeric_labels, predict_label))

average_precision = average_precision_score(test_set_numeric_labels, predict_label)

disp = plot_precision_recall_curve(rnd_clf, test_set_numeric, test_set_numeric_labels)
disp.ax_.set_title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
plt.show()

fpr, tpr, tresholds = roc_curve(test_set_numeric_labels, predict_label)
roc_auc = auc(fpr, tpr)


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Roc curve')
plt.legend(loc="lower right")
plt.show()
