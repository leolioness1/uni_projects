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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.cluster.vq import kmeans2
import os
from kmodes.kmodes import KModes
from sklearn import mixture
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score


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
print(combined_df.describe(include='all'))

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
combined_df.dtypes

# Since education level is the only column with string values
# it is convenient to transform it from categorical to numerical.
# Before doing that, it is possible to split the education columns into two
# columns, one with numeric part and the other one with the education description
combined_df[['edu_code', 'edu_desc']] = combined_df['education_lvl'].str.split(" - ", expand=True)

# Create a one-hot encoded set of the type values for 'education_lvl'
edu_values = combined_df.edu_desc.unique()

# Delete education_lvl columns, since its information is into the two new dummy columns
combined_df = combined_df.drop(['education_lvl', 'edu_code'], axis=1)

# Checking for missing data using isnull() function & calculating the % of null values per column
# Show the distribution of missing data per column
print('This is the missing data distribution per column (%):\n',
      round((combined_df.isnull().sum() / len(combined_df)) * 100, 2))

# Show the percentage of all rows with missing data, no matter which column
print('The sum of percentage of missing data for all rows is: ',
      round((combined_df.isnull().sum() / len(combined_df)).sum() * 100, 2), '% \n',
      'which are ', combined_df.isnull().sum().sum(), 'rows of the total ', len(combined_df), 'rows')

# Assuming there are no more than 1 missing value per row,
# The number of rows with null values is below 3%,
# which is a reasonable amount of rows that can be dropped
# and continue with the 96.6% of the dataframe
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
df.columns
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

# we decide to map the binary variable to a string variable to treat is as a categorical variable
df['has_children'] = df['has_children'].map({1: 'Yes', 0: 'No'})

# birth_year: should not exist values larger than policy year creation
print("6. Are there values greater than policy_creation _year on the 'birth_year'?: ",
      sum(np.where(df['policy_creation_year'] < (df['birth_year'] + 18), 1, 0)))

# Due to the high levels of inconsistent data in the birth year column we decide to drop this column as the data in it cannot be trusted
df.drop('birth_year', axis=1, inplace=True)

# customer_monetary_value (CMV), nothing to verify
# claims_rate, nothing to verify
# all premiums, nothing to verify
# #create feature for number of active premiums per customer
# df['number_active_premiums']=df[['motor_premiums', 'household_premiums', 'health_premiums',
#        'life_premiums', 'work_premiums']].gt(0).sum(axis=1)

#create feature for number of premiums cancelled this year but were active the previous year per customer
#a negative number for the premium indicates a reversal i.e. that a policy was active the previous year but canceled this year
df['cancelled_premiums_pct']=df[['motor_premiums', 'household_premiums', 'health_premiums',
       'life_premiums', 'work_premiums']].lt(0).sum(axis=1)/5

# all the other columns, (nothing to verify)

#########################################################
# --------- Calculating additional variables ------------
#########################################################

# With the additional information given,
# it's possible to obtain extra valuable information.

# For 'customer_monetary_value' (CMV), it's possible to clear the given formula:
# CMV = (Customer annual profit)(number of years as customer) - (acquisition cost)
# Therefore:

# (acquisition cost) = (Customer annual profit)(number of years as customer) - CMV

#     where: (Customer annual profit) and (number of years as customer)
# can be calculated prior, as a requirement to get (acquisition cost)

# Calculate how old is each customer's first policy
# and create a new column named 'Cust_pol_age'
today_year = int(date.today().year)
df['cust_pol_age'] = today_year - df['policy_creation_year']

# we decided to no longer generate customer age after discovering a big chunk of the data is unreliable

# dropping the year column as this information has now been captured in the age variable created
df.drop(['policy_creation_year'], axis=1, inplace=True)

# Calculating and adding 'total_premiums' to the data frame
df['total_premiums'] = df['motor_premiums'] + \
                       df['household_premiums'] + \
                       df['health_premiums'] + \
                       df['life_premiums'] + \
                       df['work_premiums']

df['motor_premiums'] = df['motor_premiums'] / df['total_premiums']
df['household_premiums'] = df['household_premiums'] / df['total_premiums']
df['health_premiums'] = df['health_premiums'] / df['total_premiums']
df['life_premiums'] = df['life_premiums'] / df['total_premiums']
df['work_premiums'] = df['work_premiums'] / df['total_premiums']

# # # Calculating the acquisition cost assuming claims_rate constant last 2 yrs
# df['cust_acq_cost'] = (df['total_premiums']-df['claims_rate'] * df['total_premiums']) * df['cust_pol_age'] - df['customer_monetary_value']

# For 'claims_rate' (CR) it's possible to clear the 'Amount paid by the insurance company'
# claims_rate = (Amount paid by the insurance company)/(Total Premiums)
# Therefore:(Amount paid by the insurance company) = (claims_rate)*(Total Premiums)
# where: (Total Premiums) can be calculated prior, as the sum of:
# 'motor_premiums', 'household_premiums', 'health_premiums',
# 'life_premiums', 'work_premiums

# # Calculate 'Amount paid by the insurance company' (
#df['amt_paidby_comp_2yrs'] = df['claims_rate'] * df['total_premiums']*2
#df.drop(['claims_rate'], axis=1, inplace=True)

# Calculate the premium/wage proportion
df['premium_wage_ratio'] = df['total_premiums'] / (df['gross_monthly_salary'] * 12)
df.drop(['total_premiums'], axis=1, inplace=True)

# Categorical boolean mask
categorical_feature_mask = df.dtypes == object
# filter categorical columns using mask and turn it into a list
categorical_cols = df.columns[categorical_feature_mask].tolist()

#########################################################
# ------------------- Excluding outliers ----------------
#########################################################

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
    Q1 = df_in.quantile(qtl_1)
    Q3 = df_in.quantile(qtl_2)
    # IQR = Q3 - Q1
    # lower_range = Q1 - (1.5 * IQR)
    # upper_range = Q3 + (1.5 * IQR)

    # df_out is filtered with values within the quartiles boundaries
    df_out = df_in[~((df_in < Q1) | (df_in > Q3)).any(axis=1)]
    df_outliers = df_in[((df_in < Q1) | (df_in > Q3)).any(axis=1)]
    return df_out, df_outliers


# Reference:
# https://medium.com/@prashant.nair2050/hands-on-outlier-detection-and-treatment-in-python-using-1-5-iqr-rule-f9ff1961a414
# We are now going to scale the data so we can do effective clustering of our variables
# Standardize the data to have a mean of ~0 and a variance of 1
scaler = StandardScaler()
X_std = scaler.fit_transform(df.drop(categorical_cols, axis=1))
X_std_df = pd.DataFrame(X_std, columns=df.drop(categorical_cols, axis=1).columns)

qtl_1 = 0.01  # lower boundary
qtl_2 = 0.99  # upper boundary

# Apply box-plot function to the selected columns
#boxplot_all_columns(X_std_df, qtl_1, qtl_2)

# There are outliers, so let's remove them with the 'IQR_drop_outliers' function
df, df_outliers = IQR_drop_outliers(df, qtl_1, qtl_2)

# Standardize the data after dropping the outliers
scaler = StandardScaler()
X_std = scaler.fit_transform(df.drop(categorical_cols, axis=1))
X_std_df = pd.DataFrame(X_std, columns=df.drop(categorical_cols, axis=1).columns)

# Plot without outliers
# boxplot_all_columns(X_std_df, qtl_1, qtl_2)

#Allocate the categorical columns to a new Dataframe

df_cat = df.loc[:, categorical_cols]
df.drop(categorical_cols, axis=1, inplace=True)

#########################################################
# ------------------- Standardization ----------------
#########################################################

# We are now going to scale the data so we can do effective clustering of our variables
# Standardize the data to have a mean of ~0 and a variance of 1
scaler = StandardScaler()
X_std = scaler.fit_transform(df)
X_std_df = pd.DataFrame(X_std, columns=df.columns, index=df.index)
X_std_df.columns

#########################################################
# --------------- Gaussian Mixture Models --------------s-
#########################################################
# 0) This should be standardized df:
to_GMM = X_std_df

# 1) Set the GMM parameters
gmm = mixture.GaussianMixture(n_components=6,
                              init_params='kmeans', # {‘kmeans’, ‘random’}, defaults to ‘kmeans’.
                              max_iter=1000,
                              n_init=10,
                              verbose=1)
# 2) Fit the model
gmm.fit(to_GMM)

# 3) Get labels from clusters
labels = gmm.predict(X_std)

# 3.1) Scores
EM_score_samp = gmm.score_samples(X_std)
# 3.2) Prediction probability
EM_pred_prob = gmm.predict_proba(X_std)

# 3.3) Count the number of unique labels (clusters)
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

# 5) Count what?
unique, counts = np.unique(labels, return_counts=True)

print(np.asarray((unique, counts)).T)

# Let's check how are they distributed
# 6) Apply PCA to reduce dimensionality:

# 6.1) PCA with 2 principal components:
from sklearn.decomposition import PCA

pca = PCA(n_components=2).fit(to_GMM)  # fit to 2 principal components
pca_2d = pca.transform(to_GMM)

# Set a marker and a color to each label (cluster)
for i in range(0, pca_2d.shape[0]):
    if labels[i] == 0:
        c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
    elif labels[i] == 1:
        c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
    elif labels[i] == 2:
        c3 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='*')
    elif labels[i] == 3:
        c4 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='y', marker='s')
    elif labels[i] == 4:
        c5 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='m', marker='p')
    elif labels[i] == 5:
        c6 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='c', marker='H')
    elif labels[i] == 6:
        c7 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='k', marker='o')

plt.legend([c1, c2, c3, c4, c5, c6],
           ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7'])
plt.title('Mean Shift found  %i' % n_clusters_ + ' clusters')
plt.show()

# 6.2) PCA with 3 principal components:
from sklearn.decomposition import PCA

pca = PCA(n_components=3).fit(to_GMM)  # fit to 3 principal components
pca_3d = pca.transform(to_GMM)
# Add my visuals
my_color = []
my_marker = []
# Load my visuals
# Set a marker and a color to each label (cluster)
for i in range(pca_3d.shape[0]):
    if labels[i] == 0:
        my_color.append('r')
        my_marker.append('+')
    elif labels[i] == 1:
        my_color.append('g')
        my_marker.append('o')
    elif labels[i] == 2:
        my_color.append('b')
        my_marker.append('*')
    elif labels[i] == 3:
        my_color.append('y')
        my_marker.append('s')
    elif labels[i] == 4:
        my_color.append('m')
        my_marker.append('p')
    elif labels[i] == 5:
        my_color.append('c')
        my_marker.append('H')
    elif labels[i] == 6:
        my_color.append('k')
        my_marker.append('o')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# for i in range(pca_3d.shape[0]):
for i in range(250):
    ax.scatter(pca_3d[i, 0],
               pca_3d[i, 1],
               pca_3d[i, 2], c=my_color[i], marker=my_marker[i])

ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')

# --------------- PCA analysis ---------------------------
# Performing complete PCA analysis, with all components
pca_init = PCA()
pca = pca_init.fit(X_std_df)
n_components = X_std_df.shape[1]

# Showing the percentage explained by each component and
# and the cumulative sum of this percentage on a table
pca_board = pd.DataFrame({"Explained var. (%)": np.round(pca.explained_variance_ratio_ * 100, decimals=1),
                          "Cumulative var. (%)": np.round(np.cumsum(pca.explained_variance_ratio_ * 100), decimals=2)})
pca_board.index.name = 'PC'
pca_board.index += 1

print("{}\n".format(pca_board))
pca_index = []

for i in range(1, len(X_std_df.columns) + 1):
    pca_index.append('PC' + str(i))

print(pd.DataFrame(pca.components_,
                   columns=X_std_df.columns,
                   index=pca_index))

# Plotting the Cumulative Sum of the Explained Variance
plt.figure()
pc_nr = len(pca.explained_variance_ratio_)
x_pos = np.arange(1, pc_nr + 1)
plt.plot(x_pos, np.cumsum(pca.explained_variance_ratio_),
         color='orange',
         label='Cumulative exp. var.')

plt.bar(x_pos, pca.explained_variance_ratio_,
        label='Individual exp. var.')

plt.xlabel('Number of component')
plt.ylabel('Variance (%)')  # for each component
plt.title('Individual and cumulative explained variance')
plt.legend(loc='center right', frameon=True)
plt.show()

# Plotting the contribution of each variable to the PCA
plt.matshow(pca.components_, cmap='Spectral_r')

pca_index = []
for i in range(1, len(X_std_df.columns) + 1):
    pca_index.append('PC' + str(i))

tick_list = []
for i in range(0, len(X_std_df.columns)):
    tick_list.append(i)

plt.yticks(tick_list, pca_index, fontsize=10)
plt.colorbar()
plt.xticks(range(len(X_std_df.columns)), X_std_df.columns, rotation=65, ha='left')
plt.tight_layout()
plt.show()

# --------------- end of GMM -------------------
