# Logistics Delivery and Shipping Performance Analysis
A personal data analysis project exploring shipping and delivery performance 
using the Superstore retail dataset — covering data cleaning, exploratory 
analysis, visualization, and predictive modelling in Python.

## Week 1: Analysis
Analyses shipping performance (average shipping days, ship mode, region) using the Superstore dataset (9,994 records).

**Run:** `pip install pandas` then `python week1_logistics.py`

**Key finding:** Average shipping time is 3.96 days; 39.95% of orders are "slow" (5+ days), with Standard Class shipping being the slowest mode (5.01 days).
## Week 2: Data Collection, Cleaning and Preprocessing

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

## Week 3 – Advanced Data Analysis and Visualization

The Week 3 task focused on Exploratory Data Analysis (EDA), advanced analysis and visualization using Python.

### Analysis Performed

- Basic statistical analysis
- Shipping performance by year
- Shipping performance by Ship Mode
- Shipping performance by Region
- Discount vs Profit analysis
- Correlation analysis
- Data visualization

### Key Findings

- Overall average shipping time: **3.96 days**
- Average shipping time was **4.20 days in 2014**.
- Average shipping time decreased to **3.67 days in 2016**.
- Average shipping time increased slightly to **3.84 days in 2017**.
- Standard Class was the slowest shipping mode at **5.01 days**.
- Same Day was the fastest shipping mode at approximately **0.04 days**.
- Central had the highest average shipping time at **4.06 days**.
- East had the lowest average shipping time at **3.91 days**.
- Discount and Profit had a negative correlation of **-0.219**.
- Sales and Profit had a positive correlation of **0.479**.
- Shipping Days had very weak correlation with Sales and Profit.

### Visualizations

The following visualizations were created:

1. Shipping Days Distribution
2. Average Shipping Days by Year
3. Shipping Performance by Ship Mode
4. Shipping Performance by Region
5. Discount vs Profit
6. Correlation Heatmap

### Python File

`week3_analysis_visualization.py`

## Tools Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- VS Code
- GitHub

## Week 4 – Predictive Modeling and Optimization in Logistics Systems

The Week 4 task focused on building predictive models to forecast shipping performance and using model insights to propose logistics optimization strategies.

### Analysis Performed

- Regression modeling to predict Shipping Days
- Model comparison (Linear Regression, Decision Tree, Random Forest)
- 5-fold cross-validation
- Hyperparameter tuning using GridSearchCV
- Feature importance analysis
- Classification modeling to predict Slow Shipments (≥5 days)
- Optimization simulation based on model insights

### Key Findings

- The tuned Random Forest model achieved an **R² of 0.672** and an **MAE of 0.819 days**.
- **Ship Mode** was by far the most important feature for predicting shipping time, with Standard Class alone accounting for the majority of the model's predictive power.
- The classification model for Slow Shipments achieved **71.7% accuracy** and **85% recall**.
- The worst-performing combination was **West Region + Technology Category** under Standard Class, averaging **5.09 days**.
- An optimization simulation showed that upgrading this worst-performing group could reduce the overall average shipping time from **3.958 to 3.893 days**.

### Optimization Recommendations

1. Monitor Standard Class shipments closely, since they have the strongest effect on shipping time.
2. Use the classification model to flag potentially slow shipments early.
3. Focus on poor-performing combinations such as West + Technology.
4. Consider selective shipping upgrades for orders predicted to be slow.
5. Retrain the model periodically as new logistics data becomes available.

### Python File

`week4_predictive_modeling.py`

**Run:** `pip install pandas numpy scikit-learn` then `python week4_predictive_modeling.py`
