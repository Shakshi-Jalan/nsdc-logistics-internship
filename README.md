# Logistics Delivery and Shipping Performance Analysis

Week 1 task for NSDC–YuvaIntern Logistics Data Analyst internship. Analyses shipping performance (average shipping days, ship mode, region) using the Superstore dataset (9,994 records).

**Run:** `pip install pandas` then `python week1_logistics.py`

**Key finding:** Average shipping time is 3.96 days; 39.95% of orders are "slow" (5+ days), with Standard Class shipping being the slowest mode (5.01 days).
# Week 2: Data Collection, Cleaning and Preprocessing

Week 2 task Focuses on cleaning and preparing the Superstore dataset for further logistics analysis using Python.

### Data Cleaning Steps

- Checked missing values
- Checked duplicate records
- Checked data types
- Converted Order Date and Ship Date into date format
- Created a Shipping Days column
- Checked for negative shipping days
- Corrected an inconsistent Ship Mode value
- Detected outliers using the IQR method
- Normalized numerical columns using Min-Max Scaling
- Performed final data validation
- Saved the cleaned dataset

**Key findings:** The dataset contains 9,994 records and 21 original columns. There are 0 missing values, 0 duplicate records, and 0 negative shipping days. One inconsistent Ship Mode value (`s`) was corrected to `Standard Class`. The average shipping time is 3.96 days. The final cleaned dataset contains 9,994 records and 22 columns.

**Run:** `pip install pandas numpy scikit-learn` then `python week2_preprocessing.py`

### Tools Used

- Python
- Pandas
- NumPy
- Scikit-learn
