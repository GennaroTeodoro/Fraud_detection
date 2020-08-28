import hashlib
import numpy as np

def resample(fraud_dataset, percentage, random_state=42):
    fraud_dataset_fraud = fraud_dataset[fraud_dataset["isFraud"] == 1]
    fraud_dataset_no_fraud = fraud_dataset[fraud_dataset["isFraud"] == 0].sample(int(len(fraud_dataset_fraud) * percentage),
                                                                                 random_state=random_state)
    dataframe_resampled = fraud_dataset_fraud.append(fraud_dataset_no_fraud)

    return dataframe_resampled


def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.sha512):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]