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


# This is a summary of the labs sessions topics we’ve covered
# Just to put checkmarks on the techniques we are using in this project:

# Session 1: query a database (sqlite3)
# Session 2: query and join tables (sqlite3)
# Session 3: explore data (describe, shape, info, unique, dtypes, sum, mean, head, tail, groupby, columns, iloc)
# Session 4: continuation of session 3

# Session 5 (Impute):
#	from sklearn.impute import SimpleImputer
# 	from sklearn.neighbors import KNeighborsClassifier
#	from sklearn.neighbors import KNeighborsRegressor

# Session 5 (Programming): Clustering
#	minMax	(scale, normalization)
#	from scipy.spatial.distance import Euclidean

# Session 6 (Normalizing and PCA):
#	from sklearn.preprocessing import MinMaxScaler
#	from sklearn.preprocessing import StandardScaler
#	from sklearn.decomposition import PCA

# Session 7 (): correlation matrix, encoding:
#	from sklearn.preprocessing import OneHotEncoder
#	from sklearn import preprocessing
#	le_status = preprocessing.LabelEncoder()

# Session 8 (Encoding):
# (transforming categorical to numerical) the status values of each customer
#	from sklearn import preprocessing
#	le_status = preprocessing.LabelEncoder()

# Session 9 (Kmeans, Silhouttes):
#	elbow_plot function
#	silhouette
#	Kmeans

# Session 10 (Hier. Clustering, K-modes):
#	from scipy.cluster.hierarchy import dendrogram, linkage
#	from scipy.cluster import hierarchy
#	from pylab import rcParams
#	from sklearn.cluster import AgglomerativeClustering
#	import sklearn.metrics as sm
# 	from kmodes.kmodes import KModes

# Session 11 (Classification tree):
#	from sklearn.model_selection import cross_val_score
#	from sklearn import tree
#	from sklearn.tree import DecisionTreeClassifier, plot_tree
#	from sklearn.model_selection import train_test_split # Import train_test_split function
#	from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
#	from dtreeplt import dtreeplt
#	import graphviz

# Session 12 (Self Organizing Map):
#	from sompy.visualization.mapview import View2DPacked
#	from sompy.visualization.mapview import View2D
#	from sompy.visualization.bmuhits import BmuHitsView
#	from sompy.visualization.hitmap import HitMapView

# Session 13 (DB Scan & Mean Shift):
#	from sklearn.cluster import DBSCAN
#	from sklearn import metrics
#	from sklearn.cluster import MeanShift, estimate_bandwidth

# Session 14 (GaussianMixture):
#	from sklearn import mixture


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

# -------------- Detecting outliers
# After logical validation, we check for outliers using different methods:
# 1) Histograms
def outliers_hist(df_in):
    fig, axes = plt.subplots(len(df_in.columns) // 3, 3, figsize=(20, 48))

    i = 0
    for triaxis in axes:
        for axis in triaxis:
            df_in.hist(column=df_in.columns[i], bins=100, ax=axis)
            i = i + 1
    fig.savefig("outliers_hist.png")

# Apply histogram function to the entire data frame
outliers_hist(df.drop(categorical_cols, axis=1))

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

qtl_1 = 0.05  # lower boundary
qtl_2 = 0.95  # upper boundary

# Apply box-plot function to the selected columns
boxplot_all_columns(X_std_df, qtl_1, qtl_2)

# There are outliers, so let's remove them with the 'IQR_drop_outliers' function
df, df_outliers = IQR_drop_outliers(df, qtl_1, qtl_2)

# Standardize the data after dropping the outliers
scaler = StandardScaler()
X_std = scaler.fit_transform(df.drop(categorical_cols, axis=1))
X_std_df = pd.DataFrame(X_std, columns=df.drop(categorical_cols, axis=1).columns)

# Plot without outliers
boxplot_all_columns(X_std_df, qtl_1, qtl_2)

#Allocate the categorical columns to a new Dataframe

df_cat = df.loc[:, categorical_cols]
df.drop(categorical_cols, axis=1, inplace=True)

# Clustering can be performed using those two variables
scaler = StandardScaler()
X_std = scaler.fit_transform(df)
X_std_df = pd.DataFrame(X_std, columns=df.columns)

# ----------------------K-modes-------------------------
# separate df into engage and consume
df_Engage = df_cat.join(df['gross_monthly_salary'])
df_Engage.dtypes

# Converting gross_monthly_salary into categorical variable, using bins
df_Engage['salary_bin'] = pd.cut(df_Engage['gross_monthly_salary'],
                                 [0, 1000, 2000, 3000, 4000, 5000, 6000],
                                 labels=['0-1k', '1k-2k', '2k-3k', '3k-4k', '4k-5k', '5k-6k'])
# Drop 'gross_monthly_salary', since the goal is to perform K-modes
df_Engage = df_Engage.drop('gross_monthly_salary', axis=1)
# drop gross_monthly_salary as this information has been captured in total_premiums
df.drop('gross_monthly_salary', axis=1, inplace=True)

# Take a look at the new df_Engage full categorical
df_Engage.head()
df_Engage['salary_bin'] = df_Engage['salary_bin'].astype(str)
df_Engage.dtypes

# Choosing K for kmodes by comparing Cost against each K. Copied from:
# https://www.kaggle.com/ashydv/bank-customer-clustering-k-modes-clustering
cost = []
for num_clusters in list(range(1, 5)):
    kmode = KModes(n_clusters=num_clusters, init="Cao", n_init=1, verbose=1)
    kmode.fit_predict(df_Engage)
    cost.append(kmode.cost_)

y = np.array([i for i in range(1, 5, 1)])
plt.figure()
plt.plot(y, cost)
plt.savefig('K-mode_elbow.png')

## ------  K-modes with k of 2
kmodes_clustering = KModes(n_clusters=2, init='Cao', n_init=50, verbose=1)
clusters_cat = kmodes_clustering.fit_predict(df_Engage)

pca = PCA(2)

# Turn the dummified df into two columns with PCA
plot_columns = pca.fit_transform(X_std_df.iloc[:,0:13])
X_std_df.shape

LABEL_COLOR_MAP = {0 : 'r',
                   1 : 'k',
                   2 : 'b'}

fig, ax = plt.subplots()
for c in np.unique(clusters_cat):
    ix = np.where(clusters_cat == c)
    ax.scatter(plot_columns[:,1][ix], plot_columns[:,0][ix], c = LABEL_COLOR_MAP[c], label = kmodes_clustering.cluster_centroids_[c], s = 50, marker='.')
ax.legend()
fig.savefig("Kdo")
plt.show()

# Print the cluster centroids
print("The mode of each variable for each cluster:\n{}".format(kmodes_clustering.cluster_centroids_)) #This gives the mode of each variable for each cluster.
df_cat_centroids = pd.DataFrame(kmodes_clustering.cluster_centroids_,
                                   columns=df_Engage.columns)

unique, counts = np.unique(kmodes_clustering.labels_, return_counts=True)
cat_counts = pd.DataFrame(np.asarray((unique, counts)).T, columns=['Label', 'Number'])
cat_centroids = pd.concat([df_Engage, cat_counts], axis=1)
del cat_counts



# ------ Count-plot for customers by salary_bin and education level
plt.figure()
ax = sns.countplot(x=df_Engage['salary_bin'], hue='edu_desc', data=df_Engage,
                   order=df_Engage['salary_bin'].value_counts().index,
                   hue_order=['Basic', 'High School', 'BSc/MSc', 'PhD'])
plt.legend(loc='upper right')

for p in ax.patches:
    ax.annotate(format(p.get_height(), '1.0f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.savefig('salary_bin_educ.png')
plt.show()
# Most of the customers belong to the '1k-2k', '2k-3k', '3k-4k' bins
# Most of the customers education level is High School and BSc/MSc
# There is no correlation between education level and salary
# So, education level should not be a valuable variable for the analysis

# ------ Count-plot for customers by salary_bin and geographic_area
plt.figure()

df_plot = df_Engage.groupby(['salary_bin', 'geographic_area']).size().reset_index().pivot(columns='salary_bin',
                                                                                          index='geographic_area',
                                                                                          values=0)
df_plot.plot(kind='bar', stacked=True)

# With this plot we can determine 3 things:
# 1. There are more customers in region 4
# 2. Region 2 is the region with less customers
# 3. The proportion of salary ranges is almost homogeneous over all the regions

plt.savefig('salary_bin_geo.png')
del df_plot

# ------ Count-plot for customers by salary_bin and has_children
plt.figure()

df_plot = df_Engage.groupby(['salary_bin', 'has_children']).size().reset_index().pivot(columns='salary_bin',
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
                      annot=True)
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

#########################################################
# ---------------------- Clustering --------------------
#########################################################

# Selecting the number of clusters with silhouette analysis
# Reference:
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
def silhouette_analysis(df_in, n, m):
    '''
    Selecting the number of clusters with
    silhouette analysis.
    df_in = numerical data frame (should be normalized)
    n = lowest number of cluster desired to analyse (n >=2 )
    m = highest number of cluster desired to analyse (m is included)
    m >= n
    '''

    range_n_clusters = list(range(n, m + 1))

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        # One column for the silhouette plot and
        # other column for the clustering plot
        fig, (ax1) = plt.subplots(1, 1)
        fig.set_size_inches(10, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range within [-1, 1]
        ax1.set_xlim([-1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(df_in) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(df_in)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(df_in, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(df_in, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

    plt.show()
# ------------------------------------------------------
# Applying Silhouette Analysis function over normalized df
silhouette_analysis(X_std, 2, 6)


# ## Experiment with alternative clustering techniques
# variance zero cols must go
corr = X_std_df.corr()  # Calculate the correlation of the above variables
sns.set_style("whitegrid")
# sns.heatmap(corr) #Plot the correlation as heat map

sns.set(font_scale=1.0)
cmap = sns.diverging_palette(h_neg=210, h_pos=350, s=90, l=30, as_cmap=True)
# clustermap
clustermap = sns.clustermap(data=corr, cmap="coolwarm")
clustermap.savefig('corr_clustermap.png')

descr=df.describe()

#sns.pairplot(data=corr, size=2)
# ## Perform PCA
# "First of all Principal Component Analysis is a good name. It does what it says on the tin. PCA finds the principal components of data. ...
# They are the directions where there is the most variance, the directions where the data is most spread out."
# guide: https://medium.com/@dmitriy.kavyazin/principal-component-analysis-and-k-means-clustering-to-visualize-a-high-dimensional-dataset-577b2a7a5fe2
# Create a PCA instance: pca
pca = PCA(n_components=len(X_std_df.columns))
principalComponents = pca.fit_transform(X_std)
# Plot the explained variances
features = range(1, pca.n_components_+1)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('Variance %')
plt.show()
# plt.clf()

# Save components to a DataFrame
PCA_components = pd.DataFrame(principalComponents)

# # plot scatter plot of pca
# plt.scatter(PCA_components[0], PCA_components[1], alpha=.1, color='black')
# plt.xlabel('PCA 1')
# plt.ylabel('PCA 2')
# plt.show()
n_cols = 6

ks = range(1, 10)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)

    # Fit model to samples
    model.fit(PCA_components.iloc[:, :n_cols])

    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)

plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.show()
# plt.clf()

# ### we show a steep dropoff of inertia at k=6 so we can take k=6
# ## Perform cluster analysis on PCA variables

n_clusters = 6

X = PCA_components.values
labels = KMeans(n_clusters, random_state=0).fit_predict(X)

fig, ax = plt.subplots(figsize=(15, 15))
ax.scatter(X[:, 0], X[:, 1], c=labels,
           s=30, cmap='viridis', marker='.')

# add legend
# txts = df.index.values
# for i, txt in enumerate(txts):
#     ax.annotate(txt, (X[i, 0], X[i, 1]), fontsize=7)

# # Script to attempt to find good clusters for Data Objects in a datalake arrangement
# Uses both hierarchical dendrograms and K means with PCA to find good clusters

# Hierarchical clustering
# Try different methods for clustering, check documentation:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html?highlight=linkage#scipy.cluster.hierarchy.linkage
# https://www.analyticsvidhya.com/blog/2019/05/beginners-guide-hierarchical-clustering/
# https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering
# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html
# Hierarchical clustering, does not require the user to specify the number of clusters.
# Initially, each point is considered as a separate cluster, then it recursively clusters the points together depending upon the distance between them.
# The points are clustered in such a way that the distance between points within a cluster is minimum and distance between the cluster is maximum.
# Commonly used distance measures are Euclidean distance, Manhattan distance or Mahalanobis distance. Unlike k-means clustering, it is "bottom-up" approach.

# ## Try several different clustering heuristics
methods = ['ward', 'single', 'complete', 'average', 'weighted', 'centroid', 'median']
fig = plt.figure(figsize=(30, 100))
for n, method in enumerate(methods):
    try:
        Z = linkage(X_std, method)
        ax = fig.add_subplot(len(methods), 1, n + 1)
        dendrogram(Z, ax=ax, labels=df.index, truncate_mode='lastp', color_threshold=0.62 * max(Z[:, 2]))
        ax.tick_params(axis='x', which='major', labelsize=20)
        ax.tick_params(axis='y', which='major', labelsize=20)
        ax.set_title('{} Method Dendrogram'.format(method))

    except Exception as e:
        print('Error caught:'.format(e))

fig.savefig('all_methods_dendrogram.png'.format(method))
plt.show()

Z = linkage(X_std, 'ward')
Z2 = linkage(X_std, 'centroid',optimal_ordering=True)

# Ward variance minimization algorithm

method = "ward"
fig = plt.figure(figsize=(30, 20))
ax = fig.add_subplot(1, 1, 1)
dendrogram(Z, ax=ax, labels=df.index, truncate_mode='lastp', color_threshold=0.62 * max(Z[:, 2]))
ax.tick_params(axis='y', which='major', labelsize=20)
ax.set_title('{} Method Dendrogram'.format(method))
fig.savefig('{}_method_dendrogram.png'.format(method))


# Centroid Algorithm

method = 'centroid'
fig = plt.figure(figsize=(30, 20))
ax = fig.add_subplot(1, 1, 1)
dendrogram(Z2, ax=ax, labels=df.index, truncate_mode='lastp', color_threshold=0.62 * max(Z2[:, 2]))
ax.tick_params(axis='x', which='major', labelsize=20)
ax.tick_params(axis='y', which='major', labelsize=20)
ax.set_xlabel('Data Object')
ax.set_title('{} Method Dendrogram'.format(method))
fig.savefig('{}_method_dendrogram.png'.format(method))

# DBSCAN

db = DBSCAN(eps=1, min_samples=10).fit(X_std)

labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

unique_clusters, count_clusters = np.unique(db.labels_, return_counts=True)

# -1 is the noise
print(np.asarray((unique_clusters, count_clusters)))

# Visualising the clusters
fig = plt.figure(figsize=(30, 20))
pca = PCA(n_components=2).fit(X_std)
pca_2d = pca.transform(X_std)

for i in range(0, pca_2d.shape[0]):
    if db.labels_[i] == 0:
        c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
    elif db.labels_[i] == 1:
        c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
    elif db.labels_[i] == 2:
        c4 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='k', marker='v')
    elif db.labels_[i] == 3:
        c5 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='y', marker='s')
    elif db.labels_[i] == 4:
        c6 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='m', marker='p')
    elif db.labels_[i] == 5:
        c7 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='c', marker='H')
    elif db.labels_[i] == -1:
        c3 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='*')

plt.legend([c1, c2, c4, c5, c6, c7, c3],
           ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6', 'Noise'])
plt.title('DBSCAN finds 6 clusters and noise')
fig.savefig('DBSCAN finds 6 clusters and noise.png')
plt.show()

# plt.clf()


pca = PCA(n_components=3).fit(X_std)
pca_3d = pca.transform(X_std)
# Add my visuals
my_color = []
my_marker = []
# Load my visuals
for i in range(pca_3d.shape[0]):
    if labels[i] == 0:
        my_color.append('r')
        my_marker.append('+')
    elif labels[i] == 1:
        my_color.append('b')
        my_marker.append('o')
    elif labels[i] == 2:
        my_color.append('g')
        my_marker.append('*')
    elif labels[i] == -1:
        my_color.append('k')
        my_marker.append('<')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(500):
    # for i in range(pca_3d.shape[0]):
    ax.scatter(pca_3d[i, 0], pca_3d[i, 1], pca_3d[i, 2], c=my_color[i], marker=my_marker[i])

ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')


# # The following bandwidth can be automatically detected using
# my_bandwidth = estimate_bandwidth(X_std,
#                                   quantile=0.1,
#                                   n_samples=1000)
#
# ms = MeanShift(bandwidth=my_bandwidth,
#                # bandwidth=0.15,
#                bin_seeding=True)
#
# ms.fit(X_std)
# labels = ms.labels_
# cluster_centers = ms.cluster_centers_
#
# labels_unique = np.unique(labels)
# n_clusters_ = len(labels_unique)
#
# # Values
# scaler.inverse_transform(X=cluster_centers)
#
# # Count
# unique, counts = np.unique(labels, return_counts=True)
#
# print(np.asarray((unique, counts)).T)
#
# # lets check our are they distributed
# pca = PCA(n_components=2).fit(X_std_df)
# pca_2d = pca.transform(X_std_df)
# for i in range(0, pca_2d.shape[0]):
#     if labels[i] == 0:
#         c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1],s=None, c='r')
#     elif labels[i] == 1:
#         c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1],s=None, c='g')
#     elif labels[i] == 2:
#         c3 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='*')
#
# # plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2', 'Cluster 3 '])
# # plt.title('Mean Shift found 3 clusters')
# plt.legend([c1, c2], ['Cluster 1', 'Cluster 2'])
# plt.title('Mean Shift found 2 clusters')
# plt.show()
# #plt.clf()
#
# # 3D
# pca = PCA(n_components=3).fit(X_std)
# pca_3d = pca.transform(X_std)
# # Add my visuals
# my_color = []
# my_marker = []
# # Load my visuals
# for i in range(pca_3d.shape[0]):
#     if labels[i] == 0:
#         my_color.append('r')
#         my_marker.append('+')
#     elif labels[i] == 1:
#         my_color.append('b')
#         my_marker.append('o')
#     elif labels[i] == 2:
#         my_color.append('g')
#         my_marker.append('*')
#     elif labels[i] == 3:
#         my_color.append('k')
#         my_marker.append('<')
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # for i in range(pca_3d.shape[0]):
# for i in range(250):
#     ax.scatter(pca_3d[i, 0],
#                pca_3d[i, 1],
#                pca_3d[i, 2], c=my_color[i], marker=my_marker[i])
#
# ax.set_xlabel('PCA 1')
# ax.set_ylabel('PCA 2')
# ax.set_zlabel('PCA 3')
#


gmm = mixture.GaussianMixture(n_components= 5,
                              init_params='kmeans', # {‘kmeans’, ‘random’}, defaults to ‘kmeans’.
                              max_iter=1000,
                              n_init=10,
                              verbose = 1)

gmm.fit(X_std)


EM_labels_ = gmm.predict(X_std)

#Individual
EM_score_samp = gmm.score_samples(X_std)
#Individual
EM_pred_prob = gmm.predict_proba(X_std)


print(scaler.inverse_transform(gmm.means_))

# finding best initialisation method
def compare_init_methods(data, list_init_methods, K_n):
    """
    This function returns a plot comparing the range of initialisation methods cluster plots with 4 runs.
    data: original data DataFrame
    list_init_methods: list of initialisation methods for the Scipy learn kmeans2 method.
    K_n: integer representing the number of clusters i.e. value of k in the kmeans funtion
    """
    keys = []
    centroids_list = []
    labels_list = []
    # for i in range(1, 5):
    fig, axs = plt.subplots(len(list_init_methods), 2, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
    #     fig.suptitle('Initialization Method Comparision nr {}'.format(i))

    for index, init_method in enumerate(list_init_methods):
        centroids, labels = kmeans2(data, k=K_n, minit=init_method,iter=50)
        keys.append(init_method)
        centroids_list.append(centroids)
        print(centroids)
        labels_list.append(labels)
        print(labels)
        axs[index, 0].plot(data[labels == 0, 0], data[labels == 0, 1], 'ob',
                           data[labels == 1, 0], data[labels == 1, 1], 'or',
                           data[labels == 2, 0], data[labels == 2, 1], 'oy',
                           data[labels == 3, 0], data[labels == 3, 1], 'og')
        axs[index, 0].plot(centroids[:, 0], centroids[:, 1], 'sk', markersize=5)

        axs[index, 1].plot(data[labels == 0, 2], data[labels == 0, 3], 'ob',
                           data[labels == 1, 2], data[labels == 1, 3], 'or',
                           data[labels == 2, 2], data[labels == 2, 3], 'oy',
                           data[labels == 3, 2], data[labels == 3, 3], 'og')
        axs[index, 1].plot(centroids[:, 2], centroids[:, 3], 'sk', markersize=5)
        axs[index, 0].set_title("Method:{}".format(init_method), y=0.7)
    return keys, centroids_list, labels_list


# run different initialisation methods and optimal k value(elbow)

def elbow_plot(data,max_k):
    """
    This function returns a plot and prints a dataframe of plot values.
    data: original data DataFrame
    max_k: integer representing the max of the range of values of k from [1,k]
    """

    # elbow
    cluster_range = range(1,max_k)
    sse = {}
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k,
                    random_state=0,
                    n_init = 50,
                    max_iter = 300).fit(data)
        data["Clusters"] = kmeans.labels_
        sse[k] = kmeans.inertia_
        # Inertia: Sum of distances of samples to their closest cluster center
    plt.figure(figsize=(8,5))
    plt.plot(list(sse.keys()), list(sse.values()),
             linewidth=1.5,
             linestyle="-",
             marker = "X",
             markeredgecolor="salmon",
             color = "black")
    plt.title ("K-Means elbow graph", loc = "left",fontweight = "bold")
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.axvline(x = 4, alpha = 0.4, color = "salmon", linestyle = "--")
    plt.show()
    clusters_df = pd.DataFrame.from_dict(sse,orient='index',columns=['Inertia'])
    print (clusters_df)


k_max = 8
elbow_plot(X_std_df,k_max)

init_methods = ['points', '++']
number_K = 4

keys, centroids_list, labels_list = compare_init_methods(X_std, init_methods,number_K)

# pick best kmeans iteration and initialisation method from plots above (please chnage accordingly)
#best_init = 3
best_method = "points"

centroids_dict = dict(zip(keys, centroids_list))
labels_dict = dict(zip(keys, labels_list))
print(" Labels: \n {} \n Centroids: \n {}".format(list(labels_dict[best_method]),
                                                  centroids_dict[best_method]))
