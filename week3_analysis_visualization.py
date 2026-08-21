import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1. LOAD THE CLEANED DATASET
# --------------------------------------------------

df = pd.read_csv("Superstore_week2_cleaned.csv", encoding="latin1")

print("Dataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# --------------------------------------------------
# 2. BASIC DATA EXPLORATION
# --------------------------------------------------

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate records:")
print(df.duplicated().sum())


# --------------------------------------------------
# 3. CONVERT DATE COLUMNS
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


# --------------------------------------------------
# 4. CREATE YEAR COLUMN
# --------------------------------------------------

df["Year"] = df["Order Date"].dt.year

print("\nYears in dataset:")
print(df["Year"].unique())


# --------------------------------------------------
# 5. BASIC STATISTICS
# --------------------------------------------------

print("\nBasic Statistics:")

print(
    df[
        [
            "Sales",
            "Quantity",
            "Discount",
            "Profit",
            "Shipping Days"
        ]
    ].describe()
)


# --------------------------------------------------
# 6. AVERAGE SHIPPING DAYS
# --------------------------------------------------

average_shipping = df["Shipping Days"].mean()

print("\nOverall Average Shipping Days:")
print(round(average_shipping, 2))


# --------------------------------------------------
# 7. SHIPPING DAYS BY YEAR
# --------------------------------------------------

shipping_by_year = (
    df.groupby("Year")["Shipping Days"]
    .mean()
)

print("\nAverage Shipping Days by Year:")
print(shipping_by_year)


# --------------------------------------------------
# 8. SHIPPING DAYS BY SHIP MODE
# --------------------------------------------------

shipping_by_mode = (
    df.groupby("Ship Mode")["Shipping Days"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Shipping Days by Ship Mode:")
print(shipping_by_mode)


# --------------------------------------------------
# 9. SHIPPING DAYS BY REGION
# --------------------------------------------------

shipping_by_region = (
    df.groupby("Region")["Shipping Days"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Shipping Days by Region:")
print(shipping_by_region)


# --------------------------------------------------
# 10. DISCOUNT VS PROFIT
# --------------------------------------------------

print("\nDiscount and Profit Summary:")
print(
    df[
        ["Discount", "Profit"]
    ].describe()
)


# --------------------------------------------------
# 11. CORRELATION ANALYSIS
# --------------------------------------------------

numeric_columns = [
    "Sales",
    "Quantity",
    "Discount",
    "Profit",
    "Shipping Days"
]

correlation = df[numeric_columns].corr()

print("\nCorrelation Matrix:")
print(correlation)


# ==================================================
# VISUALIZATIONS
# ==================================================


# --------------------------------------------------
# GRAPH 1: SHIPPING DAYS DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["Shipping Days"],
    bins=8,
    edgecolor="black"
)

plt.title("Distribution of Shipping Days")
plt.xlabel("Shipping Days")
plt.ylabel("Number of Orders")

plt.tight_layout()

plt.savefig(
    "01_shipping_days_distribution.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# GRAPH 2: AVERAGE SHIPPING DAYS BY YEAR
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    shipping_by_year.index,
    shipping_by_year.values,
    marker="o"
)

plt.title("Average Shipping Days by Year")
plt.xlabel("Year")
plt.ylabel("Average Shipping Days")

plt.xticks(shipping_by_year.index)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "02_average_shipping_by_year.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# GRAPH 3: AVERAGE SHIPPING DAYS BY SHIP MODE
# --------------------------------------------------

plt.figure(figsize=(8, 5))

shipping_by_mode.plot(kind="bar")

plt.title("Average Shipping Days by Ship Mode")
plt.xlabel("Ship Mode")
plt.ylabel("Average Shipping Days")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    "03_shipping_by_ship_mode.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# GRAPH 4: AVERAGE SHIPPING DAYS BY REGION
# --------------------------------------------------

plt.figure(figsize=(8, 5))

shipping_by_region.plot(kind="bar")

plt.title("Average Shipping Days by Region")
plt.xlabel("Region")
plt.ylabel("Average Shipping Days")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "04_shipping_by_region.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# GRAPH 5: DISCOUNT VS PROFIT
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Discount"],
    df["Profit"],
    alpha=0.5
)

plt.title("Discount vs Profit")
plt.xlabel("Discount")
plt.ylabel("Profit")

plt.axhline(
    y=0,
    linestyle="--"
)

plt.tight_layout()

plt.savefig(
    "05_discount_vs_profit.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# GRAPH 6: CORRELATION HEATMAP
# --------------------------------------------------

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f"
)

plt.title(
    "Correlation Between Logistics and Business Variables"
)

plt.tight_layout()

plt.savefig(
    "06_correlation_heatmap.png",
    dpi=300
)

plt.close()


# ==================================================
# FINAL RESULTS
# ==================================================

print("\n" + "=" * 50)
print("WEEK 3 ANALYSIS COMPLETED")
print("=" * 50)

print(
    "\nOverall Average Shipping Days:",
    round(average_shipping, 2)
)

print("\nAverage Shipping Days by Year:")
print(shipping_by_year.round(2))

print("\nSlowest Ship Mode:")
print(shipping_by_mode.idxmax())

print(
    "Average:",
    round(shipping_by_mode.max(), 2),
    "days"
)

print("\nFastest Ship Mode:")
print(shipping_by_mode.idxmin())

print(
    "Average:",
    round(shipping_by_mode.min(), 2),
    "days"
)

print("\nRegion with Highest Average Shipping Days:")
print(shipping_by_region.idxmax())

print(
    "Average:",
    round(shipping_by_region.max(), 2),
    "days"
)

print("\nRegion with Lowest Average Shipping Days:")
print(shipping_by_region.idxmin())

print(
    "Average:",
    round(shipping_by_region.min(), 2),
    "days"
)

print("\nSix visualization files created successfully!")

print("\nFiles created:")
print("1. 01_shipping_days_distribution.png")
print("2. 02_average_shipping_by_year.png")
print("3. 03_shipping_by_ship_mode.png")
print("4. 04_shipping_by_region.png")
print("5. 05_discount_vs_profit.png")
print("6. 06_correlation_heatmap.png")

print("\nWeek 3 completed successfully!")