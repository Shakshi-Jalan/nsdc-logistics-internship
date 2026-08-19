import pandas as pd

# 1. Load the dataset
df = pd.read_csv("Superstore_cleaned_2.csv", encoding="latin1")

# 2. View the first five records
print(df.head())

# 3. Check dataset information
print(df.info())

# 4. Generate descriptive statistics
print(df.describe())

# 5. Convert date columns into datetime format
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)

# 6. Calculate shipping duration in days
df["Shipping Days"] = (
    df["Ship Date"] - df["Order Date"]
).dt.days

# 7. Calculate average shipping days
average_shipping_days = df["Shipping Days"].mean()
print("Average Shipping Days:", average_shipping_days)

# 8. Analyse shipping performance by shipping mode
shipping_mode_analysis = df.groupby("Ship Mode")["Shipping Days"].mean()
print("\nAverage Shipping Days by Ship Mode:")
print(shipping_mode_analysis)

# 9. Analyse shipping performance by region
regional_analysis = df.groupby("Region")["Shipping Days"].mean()
print("\nAverage Shipping Days by Region:")
print(regional_analysis)