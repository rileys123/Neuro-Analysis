import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\rsril\OneDrive\Desktop\neuro_analysis\Subjects\features.csv")

# Encode emotion labels
label_encoder = LabelEncoder()
df["emotion_encoded"] = label_encoder.fit_transform(df["emotion"])

# Prepare features and labels
X = df.drop(columns=["subject", "emotion", "emotion_encoded"])
y = df["emotion_encoded"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Classification Report:\n")
from sklearn.utils.multiclass import unique_labels
labels_in_test = unique_labels(y_test, y_pred)
print(classification_report(y_test, y_pred, labels=labels_in_test, target_names=label_encoder.inverse_transform(labels_in_test)))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Feature importance
importances = clf.feature_importances_
features = X.columns
feature_df = pd.DataFrame({"Feature": features, "Importance": importances})
feature_df = feature_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_df, x="Importance", y="Feature", palette="viridis")
plt.title("Feature Importances")
plt.tight_layout()
plt.show()
