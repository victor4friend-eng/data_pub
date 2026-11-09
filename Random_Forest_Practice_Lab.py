# 1. Sample delivery data
import numpy as np
rng = np.random.default_rng(42)
km = rng.integers(2, 41, 400)
stops = rng.integers(1, 16, 400)
rain = rng.integers(0, 2, 400)
X = np.column_stack([km, stops, rain])
print(X[:4])


# 2. Mark late deliveries
minutes = 12 + 1.2 * km + 2.5 * stops
minutes = minutes + 14 * rain
minutes = minutes + rng.normal(0, 10, len(X))
y = (minutes > 65).astype(int)
print(np.bincount(y))


# 3. Split the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
print(len(y_train), len(y_test))


# 4. Train the Random Forest
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(
    n_estimators=100, max_depth=5, random_state=42
)
forest.fit(X_train, y_train)
pred = forest.predict(X_test)


# 5. Check the test results
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print(f"Accuracy: {accuracy_score(y_test, pred):.2%}")
print(confusion_matrix(y_test, pred, labels=[0, 1]))


# 6. Try the same route in dry weather and rain
routes = [[22, 8, 0], [22, 8, 1]]
names = np.array(["On time", "Late"])
print(names[forest.predict(routes)])
print(forest.predict_proba(routes)[:, 1].round(2))


# 7. Compare the forest with a single tree
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(
    max_depth=5, random_state=42)
tree.fit(X_train, y_train)
print(f"One tree: {tree.score(X_test, y_test):.2%}")
print(f"Forest:   {forest.score(X_test, y_test):.2%}")


# 8. Practice with a different route
my_route = [[28, 6, 0]]
print(names[forest.predict(my_route)])
print(forest.predict_proba(my_route)[0, 1].round(2))
