import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression  # Logistic Regression as an example classifier
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
mrna_data = pd.read_csv('mRNA_RSEM_UQ_log2_Tumor.cct', sep='\t', index_col=0)
clinical_data = pd.read_csv('clinical_table_140.tsv', sep='\t')
# Arrange mrna_data as rows of features
mrna_data = mrna_data.sort_index(axis=1)  # sort by patient name
mrna_data = mrna_data.T  # transpose
features = mrna_data.values  # convert to np array

patient_labels = clinical_data[['case_id', 'histology_diagnosis']].values
patient_labels = patient_labels[patient_labels[:, 0].argsort()]  # sort by patient name
labels = patient_labels[:, 1]  # just labels

encode = LabelEncoder()
encoded_labels = encode.fit_transform(labels)  # label encoding

# Apply PCA
pca = PCA(n_components=10)  # you can adjust the number of components
pca_features = pca.fit_transform(features)

# Classifier
model = LogisticRegression()

# Leave one out cross validation
loo = LeaveOneOut()

# Evaluate model with mean score
scores = cross_val_score(model, pca_features, encoded_labels, cv=loo)
mean_score = scores.mean()
print("Mean Score:", mean_score)

# Other evaluation metrics
pred = cross_val_predict(model, pca_features, encoded_labels, cv=loo)
accuracy = accuracy_score(encoded_labels, pred)
precision = precision_score(encoded_labels, pred, average='macro')
recall = recall_score(encoded_labels, pred, average='macro')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Confusion matrix
conf_matrix = confusion_matrix(encoded_labels, pred)
plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Output
# Mean Score: 0.9285714285714286
# Accuracy: 0.9285714285714286
# Precision: 0.556390977443609
# Recall: 0.5777777777777778
