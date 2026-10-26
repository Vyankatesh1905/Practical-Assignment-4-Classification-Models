
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {
    "SVM": make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", probability=True, random_state=42)
    ),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN (k=5)": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    ),
    "Gaussian Naive Bayes": GaussianNB()
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    results.append({
        "Algorithm": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(
            y_test, y_pred, average="weighted"
        ), 4),
        "Recall": round(recall_score(
            y_test, y_pred, average="weighted"
        ), 4),
        "F1-Score": round(f1_score(
            y_test, y_pred, average="weighted"
        ), 4),
        "Log Loss": round(log_loss(y_test, y_prob), 4)
    })

df = pd.DataFrame(results)

print("\nCLASSIFICATION MODEL COMPARISON")
print("-" * 90)
print(df.to_string(index=False))

print("\nPERFORMANCE SUMMARY")
print("Highest Accuracy:",
      df.loc[df["Accuracy"].idxmax(), "Algorithm"])
print("Highest F1-Score:",
      df.loc[df["F1-Score"].idxmax(), "Algorithm"])
print("Lowest Log Loss:",
      df.loc[df["Log Loss"].idxmin(), "Algorithm"])

print("\nProgram executed successfully!")
