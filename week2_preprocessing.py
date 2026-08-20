import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------
# 1. LOAD THE DATASET
# --------------------------------------------------

df = pd.read_csv("Superstore_cleaned_2.csv", encoding="latin1")

print("Dataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# --------------------------------------------------
# 2. BASIC DATA EXPLORATION
# --------------------------------------------------

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nColumn names:")
print(df.columns.tolist())


# --------------------------------------------------
# 3. CHECK MISSING VALUES
# --------------------------------------------------

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nTotal missing values:")
print(df.isnull().sum().sum())


# --------------------------------------------------
# 4. CHECK DUPLICATE RECORDS
# --------------------------------------------------

duplicate_count = df.duplicated().sum()

print("\nNumber of duplicate records:", duplicate_count)


# --------------------------------------------------
# 5. CHECK DATA TYPES
# --------------------------------------------------

print("\nData types:")
print(df.dtypes)


# --------------------------------------------------
# 6. CONVERT DATE COLUMNS
# --------------------------------------------------

df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    dayfirst=True,
    errors="coerce"
)

df["Ship Date"] = pd.to_datetime(
    df["Ship Date"],
    dayfirst=True,
    errors="coerce"
)

print("\nDate columns converted successfully.")


# --------------------------------------------------
# 7. CREATE SHIPPING DAYS
# --------------------------------------------------

df["Shipping Days"] = (
    df["Ship Date"] - df["Order Date"]
).dt.days

print("\nShipping Days summary:")
print(df["Shipping Days"].describe())


# --------------------------------------------------
# 8. CHECK INVALID SHIPPING DAYS
# --------------------------------------------------

negative_shipping_days = (
    df["Shipping Days"] < 0
).sum()

print(
    "\nNegative shipping days:",
    negative_shipping_days
)


# --------------------------------------------------
# 9. FIX INCONSISTENT CATEGORICAL VALUES
# --------------------------------------------------

print("\nShip Mode values before cleaning:")
print(df["Ship Mode"].value_counts())

# A single record contained "s" instead of a valid ship mode.
# Its shipping duration (4 days) matches Standard Class, so it is
# standardised accordingly rather than dropped.
df["Ship Mode"] = df["Ship Mode"].replace("s", "Standard Class")

print("\nShip Mode values after cleaning:")
print(df["Ship Mode"].value_counts())


# --------------------------------------------------
# 10. CHECK OUTLIERS USING IQR (across key numeric fields)
# --------------------------------------------------

print("\nIQR Outlier Detection")

outlier_columns = ["Sales", "Profit", "Discount", "Quantity", "Shipping Days"]

for col in outlier_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower_limit) |
        (df[col] > upper_limit)
    ]

    print(f"\n{col}:")
    print("  Q1:", Q1)
    print("  Q3:", Q3)
    print("  IQR:", IQR)
    print("  Lower Limit:", lower_limit)
    print("  Upper Limit:", upper_limit)
    print("  Number of outliers:", len(outliers))
    print(f"  Percentage of outliers: {len(outliers)/len(df)*100:.2f}%")


# --------------------------------------------------
# 11. CHECK NUMERICAL COLUMNS
# --------------------------------------------------

numeric_columns = df.select_dtypes(
    include=np.number
).columns

print("\nNumerical columns:")
print(numeric_columns.tolist())


# --------------------------------------------------
# 12. NORMALIZATION
# --------------------------------------------------

# Select numerical features for normalization
features_to_scale = [
    "Sales",
    "Quantity",
    "Discount",
    "Profit",
    "Shipping Days"
]

scaler = MinMaxScaler()

df_scaled = df.copy()

df_scaled[features_to_scale] = scaler.fit_transform(
    df_scaled[features_to_scale]
)

print("\nNormalized data:")
print(df_scaled[features_to_scale].head())


# --------------------------------------------------
# 13. FINAL DATA CHECK
# --------------------------------------------------

print("\nFinal dataset shape:")
print(df.shape)

print("\nFinal missing values:")
print(df.isnull().sum().sum())

print("\nFinal duplicate count:")
print(df.duplicated().sum())


# --------------------------------------------------
# 14. SAVE CLEANED DATASET
# --------------------------------------------------

df.to_csv(
    "Superstore_week2_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")