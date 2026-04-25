import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# create models folder
if not os.path.exists("models"):
    os.makedirs("models")

# load dataset
df = pd.read_csv("Iris.csv")

if "Id" in df.columns:
    df = df.drop("Id", axis=1)

# encode target
le = LabelEncoder()
df["Species"] = le.fit_transform(df["Species"])

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# models
models = {
    "Logistic": LogisticRegression(max_iter=500),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),  # important for confidence
    "DecisionTree": DecisionTreeClassifier(),
    "NeuralNet": MLPClassifier(max_iter=500)
}

best_model = None
best_acc = 0

# train + find best
for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(name, "Accuracy:", acc)

    if acc > best_acc:
        best_acc = acc
        best_model = model

# save best model
pickle.dump(best_model, open("models/best_model.pkl", "wb"))

# save scaler + encoder
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
pickle.dump(le, open("models/label_encoder.pkl", "wb"))

print("✅ Best model saved with accuracy:", best_acc)