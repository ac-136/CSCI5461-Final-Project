import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

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
# TODO: define model

# leave one out cross validation
loo = LeaveOneOut()

# evaluate model with mean score
scores = cross_val_score(model, features, encoded_labels, cv = loo)
mean_score = scores.mean()
print("Mean Score:", mean_score)

# other evaluation metrics
pred = cross_val_predict(model, features, encoded_labels, cv=loo)
accuracy = accuracy_score(encoded_labels, pred)
precision = precision_score(encoded_labels, pred)
recall = recall_score(encoded_labels, pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# confusion matrix
conf_matrix = confusion_matrix(encoded_labels, pred)
plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

