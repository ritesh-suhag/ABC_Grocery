
# Importing required packages -
import pandas as pd
from sklearn.utils import shuffle
import pickle

# Loading data -
loyalty_scores = pd.read_excel("grocery_database.xlsx", sheet_name = "loyalty_scores")
customer_details = pd.read_excel("grocery_database.xlsx", sheet_name = "customer_details")
transactions = pd.read_excel("grocery_database.xlsx", sheet_name = "transactions")

# Merging the tables -
data_for_regression = pd.merge(customer_details, loyalty_scores, how = "left", on = "customer_id")

# Feature engineering - 
sales_summary = transactions.groupby(by = "customer_id").agg({"sales_cost" : "sum",
                                                              "num_items" : "sum",
                                                              "transaction_id" : "count",
                                                              "product_area_id" : "nunique"}).reset_index()
sales_summary.columns = ["customer_id", "total_sales", "total_items", "transaction_count", "product_area_count"]
sales_summary["average_basket_value"] = sales_summary["total_sales"]/sales_summary["transaction_count"]

# Merging the new table with the customer details - 
data_for_regression = pd.merge(data_for_regression, sales_summary, how = "inner", on = "customer_id")

# Splitting into 2 df - one which has loyalty score and another which doesn't.
regression_modelling = data_for_regression.loc[data_for_regression["customer_loyalty_score"].notna()]
regression_scoring = data_for_regression.loc[data_for_regression["customer_loyalty_score"].isna()]

# Removing customer loyalty as these are 0.
regression_scoring.drop(["customer_loyalty_score"], axis = 1, inplace = True)

# Dropping the unnecessary customer idd column for training -
regression_modelling.drop(["customer_id"], axis = 1, inplace = True)
regression_modelling = shuffle(regression_modelling, random_state = 42)

# Checking the numbre of NAs
regression_modelling.isna().sum()
regression_modelling.dropna(how = "any", inplace = True)

# Checking potential columns which might contain outliers -
outlier_investigation = regression_modelling.describe()

# Selecting columns to remov outliers from - 
outlier_columns = ["distance_from_store", "total_sales", "total_items"]

# Removing outliers -
for column in outlier_columns :
    lower_quartile = regression_modelling[column].quantile(0.25)
    upper_quartile = regression_modelling[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr * 2      # Widening the factor trying not to remove too many outliers.
    min_border = lower_quartile - iqr_extended
    max_border = upper_quartile + iqr_extended
    
    outliers = regression_modelling[( regression_modelling[column] < min_border ) | ( regression_modelling[column] > max_border )].index
    print(f"{len(outliers)} outliers detected in column {column}")
    
    regression_modelling.drop(outliers, inplace = True)

# Saving the file -
pickle.dump(regression_modelling, open("abc_regression_modelling.p", "wb"))
pickle.dump(regression_scoring, open("abc_regression_scoring.p", "wb"))
