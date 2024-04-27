import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from MyMLP import MyMLP

from imblearn.over_sampling import SMOTE

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")

# set seed
random.seed(72)

##### DATA #####
# import data
mrna_data = pd.read_csv('data/mRNA_RSEM_UQ_log2_Tumor.cct', delimiter="\t", index_col=0)
clinical_data = pd.read_csv('data/clinical_table_140.tsv', delimiter="\t")

# arrange mrna_data as rows of features
mrna_data = mrna_data.sort_index(axis=1) # sort by patient name
mrna_data = mrna_data.T # transpose
features = mrna_data.values # convert to np array

# get cancer type for each patient
patient_labels = clinical_data[['case_id', 'histology_diagnosis']].values
patient_labels = patient_labels[patient_labels[:, 0].argsort()] # sort by patient name
labels = patient_labels[:, 1] # just labels

encode = LabelEncoder()
encoded_labels = encode.fit_transform(labels) # label encoding

##### EVALUATE #####
model = MyMLP(5, 100)

# leave one out cross validation
loo = LeaveOneOut()

accuracy_scores = []

true_positives = 0
false_positives = 0
false_negatives = 0
true_negatives = 0

for train_index, test_index in loo.split(features):
    train_features, test_features = features[train_index], features[test_index]
    train_labels, test_labels = encoded_labels[train_index], encoded_labels[test_index]

    # Under sample
    del_indices = np.where(train_labels == 1)[0]
    del_indices = np.random.choice(del_indices, 130, replace=False)
    undersampled_features = np.delete(train_features, del_indices, axis=0)
    undersampled_labels = np.delete(train_labels, del_indices, axis=0)

    # print(undersampled_features.shape)

    oversample = SMOTE(k_neighbors=3, sampling_strategy='auto', random_state=72)
    oversampled_features, oversampled_labels = oversample.fit_resample(undersampled_features, undersampled_labels)

    shuffle_indices = np.random.permutation(oversampled_features.shape[0])
    shuffled_oversampled_features = oversampled_features[shuffle_indices]
    shuffled_oversampled_labels = oversampled_labels[shuffle_indices]

    # train model
    train_loss = model.train(shuffled_oversampled_features, shuffled_oversampled_labels)

    # test model
    pred, accuracy = model.test(test_features, test_labels)
    accuracy_scores.append(accuracy)

    print(shuffled_oversampled_features.shape)
    print(shuffled_oversampled_labels.shape)
    print(test_labels)
    print(pred)
    
    # Accumulate true positives, false positives, true negatives, and false negatives
    if test_labels[0] == 1 and pred[0][0] == 1:
        true_positives += 1
        print("true pos")
        print()
    elif test_labels[0] == 0 and pred[0][0] == 1:
        false_positives += 1
        print("false pos")
        print()
    elif test_labels[0] == 1 and pred[0][0] == 0:
        false_negatives += 1
        print("false neg")
        print()
    elif test_labels[0] == 0 and pred[0][0] == 0:
        true_negatives += 1
        print("true neg")
        print()

# evaluate model
avg_accuracy = (true_positives + true_negatives) / len(features)
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)

print("TP: ", true_positives)
print("FP: ", false_positives)
print("FN: ", false_negatives)
print("TN: ", true_negatives)
print()

print("Accuracy:", avg_accuracy)
print("Precision:", precision)
print("Recall:", recall)

# confusion matrix
conf_matrix = np.array([[true_negatives, false_positives], [false_negatives, true_positives]])
plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

