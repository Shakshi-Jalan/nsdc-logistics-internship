import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv("Superstore_cleaned_2.csv", encoding="latin1")
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)
df["Shipping Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
df["Ship Mode"] = df["Ship Mode"].replace("s", "Standard Class")

# -----------------------------------------
# PROBLEM 1: REGRESSION - predict Shipping Days
# -----------------------------------------
features = ["Ship Mode", "Region", "Category", "Segment"]
target = "Shipping Days"

X = pd.get_dummies(df[features], drop_first=True)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
}

print("=== REGRESSION: Predicting Shipping Days ===\n")
results = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    results[name] = (mae, rmse, r2)
    print(f"{name}: MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")

# Cross-validation on best model (Random Forest)
rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=5, scoring="neg_mean_absolute_error")
print("\n5-fold CV MAE (Random Forest):", (-cv_scores).round(3), "Mean:", round(-cv_scores.mean(),3))

# Hyperparameter tuning
param_grid = {"n_estimators":[100,200], "max_depth":[5,8,12]}
grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring="neg_mean_absolute_error")
grid.fit(X_train, y_train)
print("\nBest params:", grid.best_params_)
best_rf = grid.best_estimator_
pred_best = best_rf.predict(X_test)
print("Tuned RF -> MAE:", round(mean_absolute_error(y_test,pred_best),3),
      "RMSE:", round(np.sqrt(mean_squared_error(y_test,pred_best)),3),
      "R2:", round(r2_score(y_test,pred_best),3))

# Feature importance
importances = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 feature importances:")
print(importances.head(10).round(3))

# -----------------------------------------
# PROBLEM 2: CLASSIFICATION - predict Slow Shipment (>=5 days)
# -----------------------------------------
df["Slow Shipment"] = (df["Shipping Days"] >= 5).astype(int)
y2 = df["Slow Shipment"]

X2_train, X2_test, y2_train, y2_test = train_test_split(X, y2, test_size=0.2, random_state=42, stratify=y2)

clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
clf.fit(X2_train, y2_train)
pred2 = clf.predict(X2_test)

print("\n=== CLASSIFICATION: Predicting Slow Shipment (>=5 days) ===")
print("Accuracy:", round(accuracy_score(y2_test, pred2),3))
print("Precision:", round(precision_score(y2_test, pred2),3))
print("Recall:", round(recall_score(y2_test, pred2),3))
print("F1:", round(f1_score(y2_test, pred2),3))
print("Confusion matrix:\n", confusion_matrix(y2_test, pred2))

# -----------------------------------------
# OPTIMIZATION SIMULATION
# -----------------------------------------
print("\n=== OPTIMIZATION SIMULATION ===")
current_avg = df["Shipping Days"].mean()
print("Current average shipping days (all orders):", round(current_avg,3))

# Identify worst Region x Category combos on Standard Class
std = df[df["Ship Mode"]=="Standard Class"]
worst_combo = std.groupby(["Region","Category"])["Shipping Days"].mean().sort_values(ascending=False)
print("\nWorst 5 Region x Category combos (Standard Class avg shipping days):")
print(worst_combo.head(5).round(2))

# Simulate: upgrade Standard Class orders in the single worst combo to Second Class average duration
worst_region, worst_category = worst_combo.index[0]
mask = (df["Ship Mode"]=="Standard Class") & (df["Region"]==worst_region) & (df["Category"]==worst_category)
n_affected = mask.sum()
second_class_avg = df[df["Ship Mode"]=="Second Class"]["Shipping Days"].mean()

df_sim = df.copy()
df_sim["Shipping Days"] = df_sim["Shipping Days"].astype(float)
df_sim.loc[mask, "Shipping Days"] = second_class_avg
new_avg = df_sim["Shipping Days"].mean()

print(f"\nWorst combo: {worst_region} / {worst_category}  ({n_affected} orders)")
print("If upgraded to Second Class average duration:")
print("New overall average shipping days:", round(new_avg,3))
print("Improvement:", round(current_avg-new_avg,4), "days")