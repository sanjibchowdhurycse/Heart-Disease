import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA

# -----------------------------
# 1️⃣ Load dataset
# -----------------------------
file_path = "/home/sanjib/LAB_PC/Assignment/archive/cleaned_merged_heart_dataset.csv"
df = pd.read_csv(file_path)

# Quick check
print(df.head())
print(df.info())

# -----------------------------
# 2️⃣ Exploratory Data Visualization
# -----------------------------
# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig("correlation_heatmap.png")  # Save image
plt.close()

# Pairplot for a few features
sns.pairplot(df[['age','trestbps','chol','thalachh','oldpeak','target']], hue='target')
plt.savefig("pairplot_features.png")  # Save image
plt.close()

# -----------------------------
# 3️⃣ Preprocessing
# -----------------------------
categorical_cols = ['sex','cp','fbs','restecg','exang','slope','ca','thal']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 4️⃣ PCA for Visualization
# -----------------------------
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_train_pca[y_train==0,0], X_train_pca[y_train==0,1], label='No Disease', alpha=0.7)
plt.scatter(X_train_pca[y_train==1,0], X_train_pca[y_train==1,1], label='Disease', alpha=0.7)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA Visualization of Training Data')
plt.legend()
plt.savefig("pca_visualization.png")
plt.close()

# -----------------------------
# 5️⃣ Naive Bayes
# -----------------------------
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

# -----------------------------
# 6️⃣ Logistic Regression
# -----------------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

# -----------------------------
# 7️⃣ Metrics
# -----------------------------
metrics = {
    "Model": ["Naive Bayes", "Logistic Regression"],
    "Accuracy": [accuracy_score(y_test, y_pred_nb), accuracy_score(y_test, y_pred_lr)],
    "Precision": [precision_score(y_test, y_pred_nb), precision_score(y_test, y_pred_lr)],
    "Recall": [recall_score(y_test, y_pred_nb), recall_score(y_test, y_pred_lr)],
    "F1 Score": [f1_score(y_test, y_pred_nb), f1_score(y_test, y_pred_lr)]
}

metrics_df = pd.DataFrame(metrics)
print(metrics_df)

# -----------------------------
# 8️⃣ Metrics Comparison Plot
# -----------------------------
labels = metrics_df['Model']
accuracy = metrics_df['Accuracy']
precision = metrics_df['Precision']
recall = metrics_df['Recall']
f1 = metrics_df['F1 Score']

x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots(figsize=(10,6))
ax.bar(x - 1.5*width, accuracy, width, label='Accuracy')
ax.bar(x - 0.5*width, precision, width, label='Precision')
ax.bar(x + 0.5*width, recall, width, label='Recall')
ax.bar(x + 1.5*width, f1, width, label='F1 Score')

ax.set_ylabel('Scores')
ax.set_title('Comparison of Naive Bayes and Logistic Regression')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0,1)
ax.legend()
plt.savefig("model_metrics_comparison.png")
plt.close()
