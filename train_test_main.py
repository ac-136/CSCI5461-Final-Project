import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

from MyMLP import MyMLP

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
model = MyMLP(20, 100)

# Split data into train and test sets
train_features, test_features, train_labels, test_labels = train_test_split(features, encoded_labels, test_size=0.4, random_state=72)

# Train model
train_loss = model.train(train_features, train_labels)

# Test model
pred, accuracy = model.test(test_features, test_labels)

# Precision, recall
precision = precision_score(test_labels, pred)
recall = recall_score(test_labels, pred)

# Confusion matrix
conf_matrix = confusion_matrix(test_labels, pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# confusion matrix
plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

