import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the data
mrna_data = pd.read_csv('mRNA_RSEM_UQ_log2_Tumor.cct', sep='\t', index_col=0)
clinical_data = pd.read_csv('clinical_table_140.tsv', sep='\t') 

# arrange mrna_data as rows of features
mrna_data = mrna_data.sort_index(axis=1) # sort by patient name
mrna_data = mrna_data.T # transpose
features = mrna_data.values # convert to np array

patient_labels = clinical_data[['case_id', 'histology_diagnosis']].values
patient_labels = patient_labels[patient_labels[:, 0].argsort()] # sort by patient name
labels = patient_labels[:, 1] # just labels

encode = LabelEncoder()
encoded_labels = encode.fit_transform(labels) # label encoding

model = GaussianNB(priors=None, var_smoothing=1e-09)

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

    dup_indices = np.where(train_labels == 0)[0]
    oversampled_features = np.concatenate([train_features] + [train_features[dup_indices]]*20, axis=0)
    oversampled_labels = np.concatenate([train_labels] + [train_labels[dup_indices]]*20, axis=0)

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
accuracy = (true_positives + true_negatives) / len(shuffled_oversampled_features)

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
plt.show()
