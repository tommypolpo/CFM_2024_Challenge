import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# Features
# pandas takes care of the header (titles of the colums): they are not loaded in X
X = pd.read_csv("X_train_features.csv")

# Labels
y = pd.read_csv("Y_train.csv")["eqt_code_cat"]


# Remove obs_id (it is just an identifier, not feature)
X = X.drop(columns=["obs_id"])


# Split into training and validation sets
# the option stratify=y divides X_train, X_val with the same proportion of each label
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Create the model
model = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_val)

# Evaluate
accuracy = accuracy_score(y_val, y_pred)

print("Validation accuracy:", accuracy)

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(importance.head(20))



cm = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(10,10))
plt.imshow(cm)
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()