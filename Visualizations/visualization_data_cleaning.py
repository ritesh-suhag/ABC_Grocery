

# Importing required packages
import pandas as pd
import pickle


# Loading in the data -
transactions = pd.read_excel("grocery_database.xlsx", sheet_name = "transactions")
product_areas = pd.read_excel("grocery_database.xlsx", sheet_name = "product_areas")

# Merging the tables -
df = pd.merge(transactions, product_areas, how = "left")

# checking for NAs -
df.isna().sum()
# we have 0 NAs.

# Getting the month, week - 
df['transaction_month'] = pd.to_datetime(df['transaction_date']).dt.month
df['transaction_date'] = pd.to_datetime(df['transaction_date'])
df['transaction_week'] = df['transaction_date'].dt.week


# Getting summary stats - 
daily_department_summary_stats = df.groupby(['product_area_name', 'transaction_date']).agg({'sales_cost' : ["sum"],
                                                                                 'num_items' : ['sum'],
                                                                                 'customer_id' : "nunique"}).reset_index()
weekly_department_summary_stats = df.groupby(['product_area_name', 'transaction_week']).agg({'sales_cost' : ["sum"],
                                                                                 'num_items' : ['sum'],
                                                                                 'customer_id' : "nunique"}).reset_index()
monthly_department_summary_stats = df.groupby(['product_area_name', 'transaction_month']).agg({'sales_cost' : ["sum"],
                                                                                 'num_items' : ['sum'],
                                                                                 'customer_id' : "nunique"}).reset_index()

daily_overall_summary_stats = df.groupby(['transaction_date']).agg({'sales_cost' : ["sum"],
                                                                                 'num_items' : ['sum'],
                                                                                 'customer_id' : "nunique"}).reset_index()
weekly_overall_summary_stats = df.groupby(['transaction_week']).agg({'sales_cost' : ["sum"],
                                                                                 'num_items' : ['sum'],
                                                                                 'customer_id' : "nunique"}).reset_index()
monthly_overall_summary_stats = df.groupby(['transaction_month']).agg({'sales_cost' : ["sum"],
                                                                                 'num_items' : ['sum'],
                                                                                 'customer_id' : "nunique"}).reset_index()

# Saving the data files and function separately - 
pickle.dump([daily_department_summary_stats, weekly_department_summary_stats, monthly_department_summary_stats, 
             daily_overall_summary_stats, weekly_overall_summary_stats, monthly_overall_summary_stats], 
            open('summary_files.p', 'wb'))



