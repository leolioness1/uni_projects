# Import packages
import sqlite3
import pandas as pd
import numpy as np
from datetime import date
import scipy.cluster.hierarchy as clust_hier
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import matplotlib.font_manager
from pandas.util.testing import assert_frame_equal
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from collections import defaultdict
from matplotlib.colors import rgb2hex, colorConverter
from itertools import combinations
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import SpectralClustering
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.cluster.vq import kmeans2
import os
from kmodes.kmodes import KModes
from sklearn import mixture
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
from sompy.sompy import SOMFactory
from sompy.visualization.bmuhits import BmuHitsView
from sompy.visualization.mapview import View2D
from sompy.visualization.hitmap import HitMapView
from sompy.visualization.plot_tools import plot_hex_map
import logging
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from IPython.display import Image           # Decision Tree Visualization
from sklearn.externals.six import StringIO  # Decision Tree Visualization
from sklearn.tree import export_graphviz    # Decision Tree Visualization
import pydotplus   # Must be installed manually in anaconda prompt with: conda install pydotplus
from sklearn import tree
import collections

#defining functions to be used further down

# Define the lower and upper quartiles boundaries for plotting the boxplots
# and for dropping values. Numbers between (0,1) and qtl1 < qtl2

def boxplot_all_columns(df_in, qtl_1, qtl_2):
    """
    qtl_1 is the lower quantile use to plot the boxplots. Number between (0,1)
    qtl_2 is the upper quantile use to plot the boxplots. Number between (0,1)
    """
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(15, 15))
    ax = sns.boxplot(data=df_in, orient="h", palette="Set2", whis=[qtl_1, qtl_2])
    plt.show()


# Define quartiles for plotting the boxplots and dropping rows
def IQR_drop_outliers(df_in, qtl_1, qtl_2):
    '''
    qtl_1 is the lower quantile use to drop the rows. Number between (0, 1)
    qtl_2 is the upper quantile use to drop the rows. Number between (0, 1)
    '''
    lower_range = df_in.quantile(qtl_1)
    upper_range = df_in.quantile(qtl_2)
    # df_out is filtered with values within the quartiles boundaries
    df_out = df_in[~((df_in < lower_range) | (df_in > upper_range)).any(axis=1)]
    df_outliers = df_in[((df_in < lower_range) | (df_in > upper_range)).any(axis=1)]
    return df_out, df_outliers

# Reference:
# https://medium.com/@prashant.nair2050/hands-on-outlier-detection-and-treatment-in-python-using-1-5-iqr-rule-f9ff1961a414
# We are now going to scale the data so we can do effective clustering of our variables
# Standardize the data to have a mean of ~0 and a variance of 1


def corr_plot(data):
    corr = data.corr()

    # Set Up Mask To Hide Upper Triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    np.triu_indices_from(mask)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(10, 13))
    heatmap = sns.heatmap(corr,
                          mask=mask,
                          square=True,
                          linewidths=0.5,
                          cmap='coolwarm',
                          cbar_kws={'shrink': 0.4,
                                    'ticks': [-1, -.5, 0, 0.5, 1]},
                          vmin=-1,
                          vmax=1)
    # add the column names as labels
    ax.set_yticklabels(corr.columns)
    ax.set_xticklabels(corr.columns,rotation=45)
    sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
    del corr, mask, heatmap

# -------------- Querying the database file
# set db path
my_path = 'insurance.db'

# Open connection to DB
conn = sqlite3.connect(my_path)

# Create a cursor
cursor = conn.cursor()

# Execute the query to check what tables exist in the database
cursor.execute('SELECT name from sqlite_master where type= "table"')

# Print the results
print(cursor.fetchall())

# Showing the tables names and the columns names inside each table
# To show the tables and their columns names before importing the db is a good exploratory method
db_filename = 'insurance.db'
newline_indent = '\n   '

conn = sqlite3.connect(db_filename)
conn.text_factory = str
cur = conn.cursor()

result = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
table_names = sorted(list(zip(*result))[0])
print("\ntables are:" + newline_indent + newline_indent.join(table_names))

for table_name in table_names:
    result = cur.execute("PRAGMA table_info('%s')" % table_name).fetchall()
    column_names = list(zip(*result))[1]
    print(("\ncolumn names for %s:" % table_name) + newline_indent + (newline_indent.join(column_names)))

# Set the queries to import all the data from each table
query_lob = """
SELECT * 
FROM
LOB
"""

query_engage = """
SELECT * 
FROM
Engage
"""

# -------------- Exploring and cleaning the data

# Load the data into dfs
lob_df = pd.read_sql_query(query_lob, conn)
engage_df = pd.read_sql_query(query_engage, conn)

# Check the shape of the tables to check number of rows and columns
print(lob_df.shape)
print(engage_df.shape)

# Take a look a both data frames
lob_df.head()
engage_df.head()

# We import the excel data to check if it is the same as that we receive from the DB
excel_df = pd.read_csv(os.path.join(os.getcwd(), "A2Z Insurance.csv"))

# Left join the 2 tables on Customer Identity and reset the index
combined_df = pd.merge(engage_df, lob_df, on='Customer Identity', how='left')

# Load original columns in a variable for later use
original_columns = combined_df.columns

# Show a description of the data frame
orig_desc=combined_df.describe(include='all')
print(orig_desc)

# Drop 'index_x' and 'index_y' since they are not useful anymore
combined_df.drop(['index_y', 'index_x'], axis=1, inplace=True)

# Check if the data from the database data source is identical to that of the static csv file provided
try:
    if assert_frame_equal(excel_df, combined_df) is None:
        print("The Dataframes are equal")
    else:
        print("Ups!")

except AssertionError as error:
    outcome = 'There are some differences in the Dataframes: {}'.format(error)

# Make customer Identity the index
combined_df.set_index('Customer Identity', inplace=True)
combined_df.columns

# clear original dfs to clean the environment
del lob_df, engage_df, excel_df, table_names

# The data is the same so we proceed using the data coming from the database

# # Set simpler columns names to facilitate analysis
combined_df.set_axis(['policy_creation_year',
                      'birth_year',
                      'education_lvl',
                      'gross_monthly_salary',
                      'geographic_area',
                      'has_children',
                      'customer_monetary_value',
                      'claims_rate',
                      'motor_premiums',
                      'household_premiums',
                      'health_premiums',
                      'life_premiums',
                      'work_premiums'],
                     axis=1, inplace=True)

# Check the shape of the table to check if no lost data
print(combined_df.shape)

# Show the type of each column
print(combined_df.dtypes)

#Data Transformation and Missing value analysis

# Since education level is the only column with string values
# It is possible to split the education columns into two columns, one with numeric part and the other one with the education description
combined_df[['edu_code', 'edu_desc']] = combined_df['education_lvl'].str.split(" - ", expand=True)

# Create a one-hot encoded set of the type values for 'education_lvl'
# (ended up not using this due to it's increase in dimensionality of our input space)
#edu_values = combined_df.edu_desc.unique()

# Delete education_lvl and the edu_code columns, since its information is captured in the edu_desc cat column
combined_df = combined_df.drop(['education_lvl', 'edu_code'], axis=1)



# Checking for missing data using isnull() function & calculating the % of null values per column
# Show the distribution of missing data per column
print('This is the missing data distribution per column (%):\n',
      round((combined_df.isnull().sum() / len(combined_df)) * 100, 2))

print('This is the data distribution per column equal to 0(%):\n',
      round(((combined_df ==0).sum() / len(combined_df)) * 100, 2))

#We decided to fill null values in the premium types columns with 0, as these didn't have any zeros before
#therefore we interpreted this as meaning that the customer did not have this type of policy
combined_df[['motor_premiums', 'household_premiums', 'health_premiums', 'life_premiums', 'work_premiums']]=combined_df[['motor_premiums', 'household_premiums', 'health_premiums', 'life_premiums', 'work_premiums']].fillna(0)

# Show the percentage of all rows with missing data
print('The sum of percentage of missing data for all rows is: ',
      round((combined_df.isnull().sum() / len(combined_df)).sum() * 100, 2), '% \n',
      'which are ', combined_df.isnull().sum().sum(), 'rows of the total ', len(combined_df), 'rows')

# Assuming there are no more than 1 missing value per row,
# The number of rows with null values is below 0.89%,
# which is a reasonable amount of rows that can be dropped
# and continue with the 10204 customers
# comparing sizes of data frames if we dropped rows with at least one null value
print("Original data frame length:",
      len(combined_df),
      "\nNew data frame length:",
      len(combined_df.dropna(axis=0, how='any')),
      "\nNumber of rows with at least 1 NA value: ",
      (len(combined_df) - len(combined_df.dropna(axis=0, how='any'))),
      "\nWhich is ",
      round(((len(combined_df) - len(combined_df.dropna(axis=0, how='any'))) / len(combined_df)) * 100, 2),
      "% of the orignal data.")

# making new data frame 'null_values' with dropped NA values
null_df = combined_df[combined_df.isna().any(axis=1)]
# Drop rows with NA values
df = combined_df.dropna(axis=0, how='any')
df.isnull().sum()
# Defining each column type value with a  dictionary
type_dict = {
    'policy_creation_year': int,
    'birth_year': int,
    'gross_monthly_salary': float,
    'geographic_area': str,
    'has_children': int,
    'customer_monetary_value': float,
    'claims_rate': float,
    'motor_premiums': float,
    'household_premiums': float,
    'health_premiums': float,
    'life_premiums': float,
    'work_premiums': float,
    'edu_desc': str
}

df = df.astype(type_dict)
df.dtypes

#########################################################
# --------- Logic Validation of each variable -----------
#########################################################

# The current year of the database is 2016.
# Check for wrong values on each column, based on the following premises:

# customer_id: there should not be repeated values, only positive values
print("1. Amount of duplicate rows on 'customer_id' column:",
      len(df[df.index.duplicated(keep=False)]),
      '\n')

# policy_creation_year: should not exist values larger than 2016, only positive values
print("2. Are there values greater than 2016 on the 'policy_creation_year'?: ",
      (df['policy_creation_year'] > 2016).any(),
      '\n')

# Count of values by year
df.groupby('policy_creation_year')['geographic_area'].count()

# It's better to remove the year with a not valid value
df = df[df.policy_creation_year != 53784]

print("3. Before data validation, are there values greater than 2016 on the 'policy_creation_year'?: ",
      (df['policy_creation_year'] > 2016).any(),
      '\n')

# birth_year: should not exist values larger than 2016, only positive values
print("4. Are there values greater than 2016 on the 'birth_year'?: ",
      (df['birth_year'] > 2016).any(),
      '\n')

# Check for the older birth year:
# Count of values by year
df.groupby('birth_year')['geographic_area'].count()

# It's better to remove the year with a not valid value
# There's only one customer with a too old birth date, let's drop this row:
df = df[df.birth_year != 1028]  # Goodbye Wolverine

# gross_monthly_salary: only positive values
# Show the lowest salaries by sorting the data frame by this column
df.gross_monthly_salary.sort_values(ascending=False).tail(3)

# geographic_area: nothing to check because no further info is provided for these codes

# has_children: the values should be 0 or 1
if df['has_children'].isin([0, 1]).sum() == len(df):
    print('5. All values from has_children column are binary values.', '\n')
else:
    print('5. Not all values from has_children column are binary values.',
          ' Additional check is necessary.', '\n')

# we decide to map the binary variable to a string variable to treatit as a categorical variable
df['has_children'] = df['has_children'].map({1: 'Yes', 0: 'No'})

# birth_year: should not exist values larger than policy year creation
print("6. Are there values greater than policy_creation _year on the 'birth_year' + 18 years?: ",
      sum(np.where(df['policy_creation_year'] < (df['birth_year'] + 18), 1, 0)))

# Due to the high levels of inconsistent data in the birth year column we decide to drop this column as the data in it cannot be trusted
df.drop('birth_year', axis=1, inplace=True)

# customer_monetary_value (CMV), nothing to verify
# claims_rate, nothing to verify
# all premiums, nothing to verify
# all the other columns, (nothing to verify)

#########################################################
# --------- Calculating additional variables ------------
#########################################################

# With the additional information given,
# it's possible to obtain extra valuable information.

# #create feature for number of active premiums per customer
df['active_premiums'] = df[['motor_premiums', 'household_premiums', 'health_premiums', 'life_premiums', 'work_premiums']].gt(0).sum(axis=1)

#create feature for number of premiums cancelled this year but were active the previous year per customer
#a negative number for the premium indicates a reversal i.e. that a policy was active the previous year but canceled this year
#after correlation analysis we decide to make this a pct of total active premiums last year (5)
df['cancelled_premiums_pct']=df[['motor_premiums', 'household_premiums', 'health_premiums',
       'life_premiums', 'work_premiums']].lt(0).sum(axis=1)/df[['motor_premiums', 'household_premiums', 'health_premiums',
       'life_premiums', 'work_premiums']].ne(0).sum(axis=1)

# Calculate how old is each customer's first policy
# and create a new column named 'Cust_pol_age'
today_year = int(date.today().year)
df['cust_tenure'] = today_year - df['policy_creation_year']

# we decided to no longer generate customer age after discovering a big chunk of the data is unreliable

# For 'claims_rate' (CR) it's possible to clear the 'Amount paid by the insurance company'
# claims_rate = (Amount paid by the insurance company)/(Total Premiums)
# Therefore:(Amount paid by the insurance company) = (claims_rate)*(Total Premiums)
# where: (Total Premiums) can be calculated prior, as the sum of:
# 'motor_premiums', 'household_premiums', 'health_premiums',
# 'life_premiums', 'work_premium

# Calculating and adding 'total_premiums' to the data frame
df['total_premiums'] = df['motor_premiums'] + \
                       df['household_premiums'] + \
                       df['health_premiums'] + \
                       df['life_premiums'] + \
                       df['work_premiums']

# # Calculate 'Amount paid by the insurance company' assuming claims_rate is the same as that over the last 2 yrs
df['amt_paidby_comp'] = df['claims_rate'] * df['total_premiums']

# For 'customer_monetary_value' (CMV), it's possible to clear the given formula:
# CMV = (Customer annual profit)(number of years as customer) - (acquisition cost)
# Therefore:
# (acquisition cost) = (Customer annual profit)(number of years as customer) - CMV
#     where: (Customer annual profit) and (number of years as customer)
# can be calculated prior, as a requirement to get (acquisition cost)
#  we assumed Customer annual profit = total premiums - amt_paidby_comp and number of years as customer was the cust_tenure

# Calculating the acquisition cost assuming claims_rate is the same as that over the last 2 yrs
df['cust_acq_cost'] = (df['total_premiums']-df['amt_paidby_comp']) * df['cust_tenure'] - df['customer_monetary_value']

# Calculate the premium/wage proportion
df['premium_wage_ratio'] = df['total_premiums'] / (df['gross_monthly_salary'] * 12)

#we select the categorical columns so we can analyse the numerical values visually more easily
# Categorical boolean mask
categorical_feature_mask = df.dtypes == object
# filter categorical columns using mask and turn it into a list
categorical_cols = df.columns[categorical_feature_mask].tolist()
df.columns
# -------------- Detecting outliers
# After logical validation, we check for outliers using different methods:
# 1) Histograms
def hist_df(df_in):
    fig, axes = plt.subplots(len(df_in.columns) // 3, 3, figsize=(20, 48))

    i = 0
    for triaxis in axes:
        for axis in triaxis:
            df_in.hist(column=df_in.columns[i], bins=100, ax=axis)
            i = i + 1
    fig.savefig("outliers_hist.png")

# # Apply histogram function to the entire data frame
hist_df(df.drop(categorical_cols, axis=1))

#########################################################
# ------------------- Excluding outliers ----------------
#########################################################

# Standardize the numerical variables to plot them all with the same scale
scaler = StandardScaler()
X_std = scaler.fit_transform(df.drop(categorical_cols, axis=1))
X_std_df = pd.DataFrame(X_std, columns=df.drop(categorical_cols, axis=1).columns)

qtl_1 = 0.01  # lower boundary
qtl_2 = 0.99  # upper boundary

# Apply box-plot function to the selected columns
boxplot_all_columns(X_std_df, qtl_1, qtl_2)
print(df.shape)

# There are outliers, so let's remove them with the 'IQR_drop_outliers' function
df, df_outliers = IQR_drop_outliers(df, qtl_1, qtl_2)
print(df.shape)

# Standardize the data after dropping the outliers to plot them with the same scale
scaler = StandardScaler()
X_std = scaler.fit_transform(df.drop(categorical_cols, axis=1))
X_std_df = pd.DataFrame(X_std, columns=df.drop(categorical_cols, axis=1).columns)

# Plot without outliers
boxplot_all_columns(X_std_df, qtl_1, qtl_2)

#Allocate the categorical columns to a new Dataframe
df_cat = df.loc[:, categorical_cols]
df.drop(categorical_cols, axis=1, inplace=True)


#-------- Re-Scale the data before plotting -------------
# Re-scale df
scaler.inverse_transform(X=X_std_df)
df = pd.DataFrame(scaler.inverse_transform(X=X_std_df),
                  columns=df.columns, index=df.index)

#########################################################
# ------------------- Plotting variables ----------------
#########################################################
df.columns

# Plot variables of df to get more insights about the customers and the business
# ----------- 1. Pie chart for number of customers by region (geographic area):
df_region = df_cat.groupby('geographic_area')['geographic_area'].count()
color_palette_list = ["#4878D0", "#6ACC64", "#D65F5F", "#956CB4", "#D5BB67", "#82C6E2"]

fig, ax = plt.subplots()
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#909090'
plt.rcParams['axes.labelcolor'] = '#909090'
plt.rcParams['xtick.color'] = '#909090'
plt.rcParams['ytick.color'] = '#909090'
plt.rcParams['font.size'] = 12
labels = [df_region.index[0], df_region.index[1], df_region.index[2], df_region.index[3]]
counts = [df_region[0], df_region[1], df_region[2], df_region[3]]
explode = (0.1, 0.1, 0.1, 0.1)


def func(pct, allvals):
    absolute = int(pct / 100. * np.sum(allvals))
    return "{:.1f}%\n({:,})".format(pct, absolute)

ax.pie(counts, explode=explode,
       colors=color_palette_list[0:4], autopct=lambda pct: func(pct, df_region),
       shadow=False, startangle=0,
       pctdistance=1.2, labeldistance=1.1)
ax.axis('equal')
ax.legend(labels, loc='best', title="Region")
ax.set_title("Number of customers by geographic area")
plt.show()

del df_region, color_palette_list, labels, counts, explode

# ----------- 2. Pie chart for number of customers by education level:
df_educ = df_cat.groupby('edu_desc')['edu_desc'].count()
color_palette_list = ["#4878D0", "#6ACC64", "#D65F5F", "#956CB4", "#D5BB67", "#82C6E2"]

fig, ax = plt.subplots()
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#909090'
plt.rcParams['axes.labelcolor'] = '#909090'
plt.rcParams['xtick.color'] = '#909090'
plt.rcParams['ytick.color'] = '#909090'
plt.rcParams['font.size'] = 12
labels = [df_educ.index[0], df_educ.index[1], df_educ.index[2], df_educ.index[3]]
counts = [df_educ[0], df_educ[1], df_educ[2], df_educ[3]]
explode = (0.1, 0.1, 0.1, 0.1)


def func(pct, allvals):
    absolute = int(pct / 100. * np.sum(allvals))
    return "{:.1f}%\n({:,})".format(pct, absolute)


ax.pie(counts, explode=explode,
       colors=color_palette_list[0:4], autopct=lambda pct: func(pct, df_educ),
       shadow=False, startangle=45,
       pctdistance=1.2, labeldistance=1.1)
ax.axis('equal')
ax.legend(labels, loc='best', title="Education")
ax.set_title("Number of customers by education level")
plt.show()

del df_educ, color_palette_list, labels, counts, explode

# ----------- 4. Histogram for salary:
sns.distplot(df['gross_monthly_salary'], kde=False, color='green', bins=20)
plt.title('Gross Monthly Salary', fontsize=16)
plt.xlabel('Customers salaries (EUR)', fontsize=14)
plt.ylabel('Frequency', fontsize=16)
plt.show()

# ----------- 4. Density curves for salary by each education level:
# Typically, high salaries are related with high levels of education.
# Let's plot a Density curve of salary for each education level to check the distributions

# Auxiliary df for education levels and salaries:
df_educ = pd.merge(df['gross_monthly_salary'], df_cat['edu_desc'],
                   left_on='Customer Identity',
                   right_on='Customer Identity',
                   how='left')
# df_educ.shape
B_df = df_educ[df_educ.edu_desc == 'Basic']
H_df = df_educ[df_educ.edu_desc == 'High School']
D_df = df_educ[df_educ.edu_desc == 'BSc/MSc']
P_df = df_educ[df_educ.edu_desc == 'PhD']

sns.distplot(B_df['gross_monthly_salary'], hist=False, kde=True, label='Basic')
sns.distplot(H_df['gross_monthly_salary'], hist=False, kde=True, label='High School')
sns.distplot(D_df['gross_monthly_salary'], hist=False, kde=True, label='BSc/MSc')
sns.distplot(P_df['gross_monthly_salary'], hist=False, kde=True, label='PhD')

# Plot formatting
plt.legend(prop={'size': 14})
plt.title('Salary distribution for each education level', size=16)
plt.xlabel('Gross Monthly Salary (EUR)', size=14)
plt.ylabel('Density', size=14)
plt.show()

# Just the clients with Basic education has a low, skewed salary density
# The other three are homogeneously dense distributed between 0.9 k and 4.5 k (EUR)
del df_educ, B_df, H_df, D_df, P_df

# ----------- 5. Histogram for tenure:
# Check the distribution of this variable individually:
sns.distplot(df['cust_tenure'], kde=False)
plt.title('Customer tenure', fontsize=18)
plt.xlabel('Customer tenure (years)', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.show()

df['cust_tenure'].describe()

# This Histogram shows the behavior of customer acquisition by
# the insurance company over time. It explicitly shows a
# constant distribution on the policy age for customers.
# Which means that the company acquired the same quantity of customers
# each year: 420. Except, for the year 1983 (2016 - 33 years),
# when the company gained 870 customers.
# Therefore, in absolute terms, the company always keep the same quantity of customers.

# ----------- 5. Sum of Premiums by LOB by Claim Rate ()
# Distribution of Claim Rate variable
sns.distplot(df['claims_rate'])

# Auxiliary df for plotting:
color_palette_list = ["#4878D0", "#6ACC64", "#D65F5F", "#956CB4", "#D5BB67", "#82C6E2"]
color_palette_list = ["#956CB4", "#6ACC64", "#4878D0", "#D65F5F", "#D5BB67"]
claim_prem_df = df[['motor_premiums',
                    'household_premiums',
                    'health_premiums',
                    'life_premiums',
                    'work_premiums',
                    'claims_rate']]

# Since the rates can variate too much on the decimals, let's create segments
claim_prem_df['claims_bin'] = pd.cut(claim_prem_df['claims_rate'],
                                     [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
                                     labels=['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%',
                                             '80-90%', '90-100%', '100-110%', '110-120%', '120-130%', '130-140%', '140-150%'])
claim_prem_df = claim_prem_df.drop('claims_rate', axis=1)
# Group by claim rate  bin and sum the premiums
claim_prem_df.groupby('claims_bin').sum().plot.bar(stacked=True,
                                                   width=0.92,
                                                   colors=color_palette_list)
# Plot formatting
plt.legend(prop={'size': 12}, loc='best')
plt.title('Sum of Premiums by claims rate and type of Premium')
plt.xlabel('Claim Rate')
plt.ylabel('Premiums (EUR)')
plt.show()


del claim_prem_df, color_palette_list

# ----------- 6. Pie chart for sum of premiums by LOB:
df_premium = df[['motor_premiums',
                 'household_premiums',
                 'health_premiums',
                 'life_premiums',
                 'work_premiums']]

df_premium = pd.melt(df_premium, var_name='Premium', value_name='Value')

df_premium = df_premium.groupby('Premium')['Value'].sum()
color_palette_list = ["#4878D0", "#6ACC64", "#D65F5F", "#956CB4", "#D5BB67", "#82C6E2"]

fig, ax = plt.subplots()
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#909090'
plt.rcParams['axes.labelcolor'] = '#909090'
plt.rcParams['xtick.color'] = '#909090'
plt.rcParams['ytick.color'] = '#909090'
plt.rcParams['font.size'] = 12
labels = [df_premium.index[0], df_premium.index[1], df_premium.index[2], df_premium.index[3], df_premium.index[4]]
counts = [df_premium[0], df_premium[1], df_premium[2], df_premium[3], df_premium[4]]
explode = (0.1, 0.1, 0.1, 0.1, 0.1)


def func(pct, allvals):
    absolute = int(pct / 100. * np.sum(allvals))
    return "{:.1f}%\n({:,})".format(pct, absolute)


ax.pie(counts, explode=explode,
       colors=color_palette_list[0:5], autopct=lambda pct: func(pct, df_premium),
       shadow=False, startangle=0,
       pctdistance=1.2, labeldistance=1.3)
ax.axis('equal')
ax.legend(labels, loc='best', title="Premium")
ax.set_title("Sum of Premiums by LOB")
plt.show()


del df_premium, color_palette_list, labels, counts, explode

# ----------- 8. Count of customers by LOB:
df_premium = df[['motor_premiums',
                 'household_premiums',
                 'health_premiums',
                 'life_premiums',
                 'work_premiums']]

df_premium = pd.melt(df_premium, var_name='Premium', value_name='Value')
df_premium.head()

df_premium = df_premium[df_premium['Value'] > 0].groupby('Premium')['Value'].count()

color_palette_list = ["#4878D0", "#6ACC64", "#D65F5F", "#956CB4", "#D5BB67", "#82C6E2"]

bar_plot = sns.barplot(x=df_premium.index,
                       y=df_premium.values,
                       color=color_palette_list[2])
len(df)
# Create labels
label = [df_premium.iloc[0],
         df_premium.iloc[1],
         df_premium.iloc[2],
         df_premium.iloc[3],
         df_premium.iloc[4]]

# Text on the top of each barplot
for i in range(len(df_premium)):
    plt.text(x=df_premium[i] - 0.5, y=df_premium.iloc[i] + 0.1, s=label[i], size=6)

plt.title('Number of customers with positive Premiums')
plt.show()

del df_premium, color_palette_list, bar_plot, label

# ----------- 9. Check the numbers of reversals:
# Add a column for customers that are leaving the company
# Negative values in any of the premiums.

health_reversals_df = df[df['health_premiums'] < 0]
household_reversals_df = df[df['household_premiums'] < 0]
life_reversals_df = df[df['life_premiums'] < 0]
motor_reversals_df = df[df['motor_premiums'] < 0]
work_reversals_df = df[df['work_premiums'] < 0]

# Show the number of reversal by premium
reversals_df = pd.DataFrame(list(zip(['Health', 'Household', 'Life', 'Motor', 'Work'],
                                     [len(health_reversals_df),
                                      len(household_reversals_df),
                                      len(life_reversals_df),
                                      len(motor_reversals_df),
                                      len(work_reversals_df)])),
                            columns=['Premium', 'Reversals'])
reversals_df.set_index('Premium', inplace=True)
reversals_df.sum()

color_palette_list = ["#4878D0", "#6ACC64", "#D65F5F", "#956CB4", "#D5BB67", "#82C6E2"]

fig, ax = plt.subplots()
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#909090'
plt.rcParams['axes.labelcolor'] = '#909090'
plt.rcParams['xtick.color'] = '#909090'
plt.rcParams['ytick.color'] = '#909090'
plt.rcParams['font.size'] = 12
labels = [reversals_df.index[1], reversals_df.index[0], reversals_df.index[2], reversals_df.index[3]]
counts = [reversals_df.iloc[0], reversals_df.iloc[1], reversals_df.iloc[2], reversals_df.iloc[3]]
explode = (0.1, 0.1, 0.1, 0.1)


def func(pct, allvals):
    absolute = int(pct / 100. * np.sum(allvals))
    return "{:.1f}%\n({:,})".format(pct, absolute)

ax.pie(counts, explode=explode,
       colors=color_palette_list[0:4], autopct=lambda pct: func(pct, reversals_df),
       shadow=False, startangle=30,
       pctdistance=1.2, labeldistance=1.2)
ax.axis('equal')
ax.legend(labels, loc='best', title="Premium")
ax.set_title("Number of customers with reversal condition")
plt.show()
# reversals_df.plot(kind='pie', subplots=True, autopct='%1.1f%%')
del color_palette_list, labels, counts, explode

# Most of the reversals are from: Household, Life and Work premiums.
# Append all reversals
reversals_df = health_reversals_df.append(household_reversals_df)
reversals_df.append(life_reversals_df)
reversals_df.append(motor_reversals_df)
reversals_df.append(work_reversals_df)
# 2153 reversal condition

del health_reversals_df, household_reversals_df, life_reversals_df, motor_reversals_df, work_reversals_df

# ----------- 10. Check the CMV and CAC variables:
# CMV: Customer Monetary Value. CAC: Customer Acquisition Cost
# Plotting CMV
sns.distplot(df['customer_monetary_value'], hist=True, kde=True, label='CMV')
plt.legend(prop={'size': 12})
plt.title('Customer Monetary Value distribution')
plt.xlabel('CMV (EUR)')
plt.ylabel('Density')
plt.show()

# Plotting CAC
sns.distplot(df['cust_acq_cost'], hist=True, kde=True, label='CAC')
plt.legend(prop={'size': 12})
plt.title('Customer Acquisition Cost distribution')
plt.xlabel('CAC (EUR)')
plt.ylabel('Density')
plt.show()

# Now, let's plot CAC vs CMV for each customer to see if there's some trend:
sns.jointplot(x=df['cust_acq_cost'],
              y=df['customer_monetary_value'],
              data=df)

ax.set(xlabel='Customer Monetary Value (EUR)', ylabel='Customer Acquisition Cost (EUR)')
plt.show()

# Just looking at these 2 variables, there's no a straight trend, but a positive trend.
# Which implies, that the company invest more on gaining or maintaining customers with high salaries

# Clustering can be performed using those two variables

# --------------APbC vs PWR-----
# APbC: Amount paid by the company, last 2 years. PWR: Premium Wage Ratio
sns.distplot(df['premium_wage_ratio'])
plt.show
# Now, let's plot CR vs PWR for each customer to see if there's some trend:
sns.jointplot(x=df['premium_wage_ratio'],
              y=df['amt_paidby_comp'], data=df,
              color='green')
plt.show

# Total amount paid by the insurance companyÑ
df['amt_paidby_comp'].sum()

df['gross_monthly_salary'].describe()

# ------ Count-plot for customers by salary_bin and education level
# Define a new data frame with only categorical values for this part
df_Engage = df_cat.join(df['gross_monthly_salary'])
# Converting gross_monthly_salary into categorical variable, using bins
df_Engage['salary_segment'] = pd.cut(df_Engage['gross_monthly_salary'],
                                 [0, 1000, 2000, 3000, 4000, 5000, 6000],
                                 labels=['0-1k', '1k-2k', '2k-3k', '3k-4k', '4k-5k', '5k-6k'])

# Drop 'gross_monthly_salary'
df_Engage = df_Engage.drop('gross_monthly_salary', axis=1)

df_Engage.dtypes
plt.figure()
ax = sns.countplot(x=df_Engage['salary_segment'], hue='edu_desc', data=df_Engage,
                   order=df_Engage['salary_segment'].value_counts().index,
                   hue_order=['Basic', 'High School', 'BSc/MSc', 'PhD'])
plt.legend(loc='upper right')

for p in ax.patches:
    ax.annotate(format(p.get_height(), '1.0f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.show()

# Most of the customers belong to the '1k-2k', '2k-3k', '3k-4k' bins
# Most of the customers education level is High School and BSc/MSc
# There is no correlation between education level and salary
# So, education level should not be a valuable variable for the analysis

# ------ Count-plot for customers by salary_bin and geographic_area
plt.figure()

df_plot = df_Engage.groupby(['salary_segment', 'geographic_area']).size().reset_index().pivot(columns='salary_segment',
                                                                                          index='geographic_area',
                                                                                          values=0)
df_plot.plot(kind='bar', stacked=True)
plt.title('Count of customers by salary segment and by region', fontsize=14)
# With this plot we can determine 3 things:
# 1. There are more customers in region 4
# 2. Region 2 is the region with less customers
# 3. The proportion of salary ranges is almost homogeneous over all the regions

plt.show()
del df_plot

# ------ Count-plot for customers by salary_bin and has_children
plt.figure()

df_plot = df_Engage.groupby(['salary_segment', 'has_children']).size().reset_index().pivot(columns='salary_segment',
                                                                                    index='has_children', values=0)
ax = df_plot.plot(kind='bar', stacked=True)

# With this plot we can determine 2 main things:
# 1. There are more customers with children: 7000 has children and 3000 does not.
# 2. The proportion of higher salaries is higher over the customers with no children.

plt.show()
del df_plot, df_Engage

# Looking at the plots of categorical from the Engage data frame,
# 1. The main variable in this data frame is salary.
# 2. Age can be an important variable as well, but since a lot of the data is wrong
# 	then is better not to trust in this variable.

# -------------- Plotting correlation matrix
# # Compute the correlation matrix
corr = df.corr()

# Set Up Mask To Hide Upper Triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
np.triu_indices_from(mask)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(10, 13))
heatmap = sns.heatmap(corr,
                      mask=mask,
                      square=True,
                      linewidths=0.5,
                      cmap='coolwarm',
                      cbar_kws={'shrink': 0.4,
                                'ticks': [-1, -.5, 0, 0.5, 1]},
                      vmin=-1,
                      vmax=1,
                      annot=True,
                      annot_kws={'size': 10})
# add the column names as labels
ax.set_yticklabels(corr.columns, rotation=0)
ax.set_xticklabels(corr.columns)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
plt.show()
del corr, mask, heatmap

#########################################################
# ------------------- Standardization ----------------
#########################################################

# We are now going to scale the data so we can do effective clustering of our variables
# Standardize the data to have a mean of ~0 and a variance of 1
scaler = StandardScaler()
X_std = scaler.fit_transform(df)
X_std_df = pd.DataFrame(X_std, columns=df.columns)


# --------------------- Re-Scale the data ----------------
# # Re-scale df
# scaler.inverse_transform(X=X_std_df)
# df = pd.DataFrame(scaler.inverse_transform(X=X_std_df),
#                   columns=df.columns)
