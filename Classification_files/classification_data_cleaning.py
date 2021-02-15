
# Loading the required packages - 
import pandas as pd
import pickle
from sklearn.utils import shuffle


# Reading in the give initial data - 
data_for_model = pickle.load(open("data/initial_data.p", "rb"))

# Dropiing customer id as we don't really need it.
data_for_model.drop(["customer_id"], axis = 1, inplace = True)

# Shuffling the data -
data_for_model = shuffle(data_for_model, random_state = 42)

# Checking class balance - 
data_for_model["signup_flag"].value_counts()

# To get the percent of classes - 
data_for_model["signup_flag"].value_counts(normalize = True)

# Checking the NA values -
data_for_model.isna().sum()
# Since the number of NA are very low we can directly drop them.

# Dropping missing values - 
data_for_model.dropna(how = "any", inplace = True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DEAL WITH OUTLIERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Investigating if any columns have potential outliers -
outlier_investigation = data_for_model.describe()

# Choosing columns to remove outliers -
outlier_columns = ["distance_from_store", "total_sales", "total_items"]

for column in outlier_columns :
    lower_quartile = data_for_model[column].quantile(0.25)
    upper_quartile = data_for_model[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr * 2      # Widening the factor trying not to remove too many outliers.
    min_border = lower_quartile - iqr_extended
    max_border = upper_quartile + iqr_extended
    
    outliers = data_for_model[( data_for_model[column] < min_border ) | ( data_for_model[column] > max_border )].index
    print(f"{len(outliers)} outliers detected in column {column}")
    
    data_for_model.drop(outliers, inplace = True)

pickle.dump(data_for_model, open('abc_classification_modelling.p', 'wb'))



