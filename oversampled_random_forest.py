import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

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

dup_indices = np.where(encoded_labels == 0)[0]
oversampled_features = np.concatenate((features, features[dup_indices], features[dup_indices], features[dup_indices], features[dup_indices], features[dup_indices], features[dup_indices], features[dup_indices], features[dup_indices], features[dup_indices], features[dup_indices], features[dup_indices], features[dup_indices], features[dup_indices], features[dup_indices], features[dup_indices], features[dup_indices], features[dup_indices], features[dup_indices], features[dup_indices], features[dup_indices]), axis=0)
oversampled_labels = np.concatenate((encoded_labels, encoded_labels[dup_indices], encoded_labels[dup_indices], encoded_labels[dup_indices], encoded_labels[dup_indices], encoded_labels[dup_indices], encoded_labels[dup_indices], encoded_labels[dup_indices], encoded_labels[dup_indices], encoded_labels[dup_indices], encoded_labels[dup_indices], encoded_labels[dup_indices], encoded_labels[dup_indices], encoded_labels[dup_indices], encoded_labels[dup_indices], encoded_labels[dup_indices], encoded_labels[dup_indices], encoded_labels[dup_indices], encoded_labels[dup_indices], encoded_labels[dup_indices], encoded_labels[dup_indices]), axis=0)
shuffle_indices = np.random.permutation(oversampled_features.shape[0])
shuffled_oversampled_features = oversampled_features[shuffle_indices]
shuffled_oversampled_labels = oversampled_labels[shuffle_indices]

# # CORRELATION COEFFICIENTS
# np.seterr(divide='ignore')
# # Calculate correlation coefficients
# correlation_coefficients = np.abs(np.corrcoef(features, encoded_labels, rowvar=False)[:-1, -1])
# # Sort features based on correlation coefficients
# sorted_indices = np.argsort(correlation_coefficients)[::-1]  # Sort in descending order
# selected_features = features[:, sorted_indices[:100]]  # Select top 100 features (adjust this as needed)

##### EVALUATE #####
# TODO: define model
model = RandomForestClassifier(n_estimators=100)

# leave one out cross validation
loo = LeaveOneOut()

# evaluate model with mean score
scores = cross_val_score(model, shuffled_oversampled_features, shuffled_oversampled_labels, cv = loo)
mean_score = scores.mean()
print("Mean Score:", mean_score)

# other evaluation metrics
pred = cross_val_predict(model, shuffled_oversampled_features, shuffled_oversampled_labels, cv=loo)
accuracy = accuracy_score(shuffled_oversampled_labels, pred)
precision = precision_score(shuffled_oversampled_labels, pred)
recall = recall_score(shuffled_oversampled_labels, pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# confusion matrix
conf_matrix = confusion_matrix(shuffled_oversampled_labels, pred)
plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('oversampled_random_forest_confusion_matrix')

# output
# Mean Score: 0.9642857142857143
# Accuracy: 0.9642857142857143
# Precision: 0.9642857142857143
# Recall: 1.0