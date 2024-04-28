import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.neighbors import KNeighborsClassifier

# Load the data
mrna_data = pd.read_csv('data/mRNA_RSEM_UQ_log2_Tumor.cct', sep='\t', index_col=0)
clinical_data = pd.read_csv('data/clinical_table_140.tsv', sep='\t') 

# arrange mrna_data as rows of features
mrna_data = mrna_data.sort_index(axis=1) # sort by patient name
mrna_data = mrna_data.T # transpose
features = mrna_data.values # convert to np array

patient_labels = clinical_data[['case_id', 'histology_diagnosis']].values
patient_labels = patient_labels[patient_labels[:, 0].argsort()] # sort by patient name
labels = patient_labels[:, 1] # just labels

encode = LabelEncoder()
encoded_labels = encode.fit_transform(labels) # label encoding

model = KNeighborsClassifier(n_neighbors=3)

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

    # if test_labels[0] == 0:
    #     oversample = SMOTE(k_neighbors=3, sampling_strategy=1, random_state=72)
    #     oversampled_features, oversampled_labels = oversample.fit_resample(trained_features, trained_labels)
    # else:
    #     oversample = SMOTE(k_neighbors=4, sampling_strategy=1, random_state=72)
    #     oversampled_features, oversampled_labels = oversample.fit_resample(trained_features, trained_labels)
    
    shuffle_indices = np.random.permutation(oversampled_features.shape[0])
    shuffled_oversampled_features = oversampled_features[shuffle_indices]
    shuffled_oversampled_labels = oversampled_labels[shuffle_indices]


    # Train model
    model.fit(shuffled_oversampled_features, shuffled_oversampled_labels)

    # Test model
    predictions = model.predict(test_features)

    # print(shuffled_oversampled_features.shape)
    # print(shuffled_oversampled_labels.shape)
    # print(test_labels)
    # print(predictions)
    # print()

    if test_labels[0] == 1 and predictions[0] == 1:
        true_positives += 1
    elif test_labels[0] == 0 and predictions[0] == 1:
        false_positives += 1
    elif test_labels[0] == 1 and predictions[0] == 0:
        false_negatives += 1
    elif test_labels[0] == 0 and predictions[0] == 0:
        true_negatives += 1

# Evaluate model
precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
accuracy = (true_positives + true_negatives) / len(features)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Confusion matrix
conf_matrix = np.array([[true_negatives, false_positives], [false_negatives, true_positives]])
plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix_from_knn_smote.png')
"""
Accuracy: 0.81
Precision: 0.99
Recall: 0.81
"""