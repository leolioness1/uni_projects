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


# Session 12 (Self Organizing Map):
#	from sompy.visualization.mapview import View2DPacked
#	from sompy.visualization.mapview import View2D
#	from sompy.visualization.bmuhits import BmuHitsView
#	from sompy.visualization.hitmap import HitMapView


# Session 10 (Hier. Clustering, K-modes):
#	from scipy.cluster.hierarchy import dendrogram, linkage
#	from scipy.cluster import hierarchy
#	from pylab import rcParams
#	from sklearn.cluster import AgglomerativeClustering
#	import sklearn.metrics as sm
# 	from kmodes.kmodes import KModes


# Session 9 (Kmeans, Silhouttes):
#	elbow_plot function
#	silhouette
#	Kmeans

# Session 13 (DB Scan & Mean Shift & Spectral):
#	from sklearn.cluster import DBSCAN
#	from sklearn import metrics
#	from sklearn.cluster import MeanShift, estimate_bandwidth

# Session 14 (GaussianMixture):
#	from sklearn import mixture


# Session 11 (Classification tree & KNN):
#	from sklearn.model_selection import cross_val_score
#	from sklearn import tree
#	from sklearn.tree import DecisionTreeClassifier, plot_tree
#	from sklearn.model_selection import train_test_split # Import train_test_split function
#	from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
#	from dtreeplt import dtreeplt
#	import graphviz

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
import sompy
from sompy.sompy import SOMFactory
from sompy.visualization.bmuhits import BmuHitsView
from sompy.visualization.mapview import View2D
from sompy.visualization.hitmap import HitMapView
from sompy.visualization.plot_tools import plot_hex_map
import logging
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import SpectralClustering
from IPython.display import Image           # Decision Tree Visualization
from sklearn.externals.six import StringIO  # Decision Tree Visualization
from sklearn.tree import export_graphviz    # Decision Tree Visualization
import pydotplus   # Must be installed manually in anaconda prompt with: conda install pydotplus
from sklearn import tree
import collections
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

#defining functions to be used further down

# Selecting the number of clusters with silhouette analysis
# Reference:
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
logging.getLogger('matplotlib.font_manager').disabled = True
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
        clusterer = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0,n_init=30)
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

# run different initialisation methods and optimal k value(elbow)

def elbow_plot(data,max_k, optimal_k):
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
                        init='k-means++',
                        random_state=0,
                        n_init = 30)
        kmeans.fit(data)
        #data["Clusters"] = kmeans.labels_
        sse[k] =kmeans.inertia_
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
    plt.axvline(x = optimal_k, alpha = 0.4, color = "salmon", linestyle = "--")
    plt.show()
    kmeans_clusters_df = pd.DataFrame.from_dict(sse,orient='index',columns=['Inertia'])
    print (kmeans_clusters_df)


# finding best initialisation method
def compare_init_methods(data, list_init_methods, K_n):
    """
    This function returns a plot comparing the range of initialisation methods cluster plots
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
        centroids, labels = kmeans2(data, k=K_n, minit=init_method,iter=100)
        keys.append(init_method)
        centroids_list.append(centroids)
        labels_list.append(labels)
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
df_outliers.drop(categorical_cols, axis=1, inplace=True)

#########################################################
# ------------------- Input Space Reduction -------------
#########################################################
# Standardize the data after dropping the outliers
scaler = StandardScaler()
X_std = scaler.fit_transform(df)
X_std_df = pd.DataFrame(X_std, columns=df.columns)

# variance zero cols must go
corr = X_std_df.corr()  # Calculate the correlation of the above variables
sns.set_style("whitegrid")
# sns.heatmap(corr) #Plot the correlation as heat map

sns.set(font_scale=1.0)
cmap = sns.diverging_palette(h_neg=210, h_pos=350, s=90, l=30, as_cmap=True)
# clustermap
clustermap = sns.clustermap(data=corr, cmap="coolwarm")
clustermap.savefig('corr_clustermap.png')
#--------------------------------------------------------

# Create a copy of the df in order to perform clustering on
# This clust_df data frame IS NOT Standardized
clust_df = df.copy(deep=True)

# As an attempt to reduce the input space we will be dropping some variables from the DataFrame before performing clustering
# dropping the year column as this information has now been captured in the age variable created
clust_df.drop(['policy_creation_year'], axis=1, inplace=True)
#drop the tenure because it is detected as a redundant attributes removed
clust_df.drop('cust_tenure',axis=1,inplace=True)


#after correlation analysis we decide to drop this variable as it is correlated with cancelled_pct
clust_df.drop('active_premiums',axis=1,inplace=True)

#since claims_rate, customer_monetary_value,amt_paidby_comp and cust_acq_cost are highly correlated with each other
#mainly because they are mainly linear combinations of each other, we decide to keep customer aquisition cost since it is the most informative for the insurance industry
# clust_df.drop('claims_rate', axis=1, inplace=True)
# clust_df.drop('customer_monetary_value',axis=1,inplace=True)
clust_df.drop('cust_acq_cost',axis=1,inplace=True)
clust_df.drop('amt_paidby_comp',axis=1,inplace=True)

#we decide to make the other premium values as a pct of total premiums to capture that information here
#this was also after correlation analysis and to reduce size of the input space
clust_df['motor_premiums_pct'] = df['motor_premiums'] / df['total_premiums']
clust_df['household_premiums_pct'] = df['household_premiums'] / df['total_premiums']
clust_df['health_premiums_pct'] = df['health_premiums'] / df['total_premiums']
clust_df['life_premiums_pct'] = df['life_premiums'] / df['total_premiums']
clust_df['work_premiums_pct'] = df['work_premiums'] / df['total_premiums']
clust_df.drop(['motor_premiums', 'household_premiums', 'health_premiums', 'life_premiums', 'work_premiums'],axis=1, inplace=True)

#We drop the total_premiums since it's value has been captured by each of the individual types of premiums as a pct of total premiums
#and in the above new variable
clust_df.drop(['total_premiums'], axis=1, inplace=True)
# drop gross_monthly_salary as this information has been captured in total_premiums
clust_df.drop('gross_monthly_salary', axis=1, inplace=True)

# Apply the same transformations to outliers df:
df_outliers.drop(['policy_creation_year', 'cust_tenure', 'active_premiums', 'cust_acq_cost', 'amt_paidby_comp'], axis=1, inplace=True)
df_outliers['motor_premiums_pct'] = df_outliers['motor_premiums'] / df_outliers['total_premiums']
df_outliers['household_premiums_pct'] = df_outliers['household_premiums'] / df_outliers['total_premiums']
df_outliers['health_premiums_pct'] = df_outliers['health_premiums'] / df_outliers['total_premiums']
df_outliers['life_premiums_pct'] = df_outliers['life_premiums'] / df_outliers['total_premiums']
df_outliers['work_premiums_pct'] = df_outliers['work_premiums'] / df_outliers['total_premiums']
df_outliers.drop(['motor_premiums', 'household_premiums', 'health_premiums', 'life_premiums', 'work_premiums', 'total_premiums', 'gross_monthly_salary'], axis=1, inplace=True)

#########################################################
# ------ Standardization & sub-setting data frame -------
#########################################################
# We are now going to scale the data so we can do effective clustering of our variables
# This data frame name is already written on the clustering codes
# Except for K-modes, which uses categorical variables...
# Standardize the data to have a mean of ~0 and a variance of 1
scaler = StandardScaler()
X_std = scaler.fit_transform(clust_df)
X_std_df = pd.DataFrame(X_std, columns=clust_df.columns)


# plotting correlation plot after droping vars before clustering
corr = X_std_df.corr()  # Calculate the correlation of the above variables
sns.set_style("whitegrid")
sns.set(font_scale=1.0)
cmap = sns.diverging_palette(h_neg=210, h_pos=350, s=90, l=30, as_cmap=True)
# clustermap
clustermap = sns.clustermap(data=corr, cmap="coolwarm")
clustermap.savefig('corr_clust_clustermap.png')

# split main clustering data frame into products and value data frames

X_prod_df = clust_df[['motor_premiums_pct',
                          'household_premiums_pct',
                          'health_premiums_pct',
                          'life_premiums_pct',
                          'work_premiums_pct']]

X_value_df = clust_df[['cancelled_premiums_pct',
                           'claims_rate',
                           'customer_monetary_value',
                           'premium_wage_ratio']]

# Define the unstandardized data frame to work with from now:
to_use_clust_df = clust_df.copy(deep=True)

# We are now going to scale the data so we can do effective clustering of our variables
# This data frame name is already written on the clustering codes
# Except for K-modes, which uses categorical variables...
# Standardize the data to have a mean of ~0 and a variance of 1
scaler = StandardScaler()
Std_clust = scaler.fit_transform(to_use_clust_df)
Std_clust_df = pd.DataFrame(Std_clust, columns=to_use_clust_df.columns)

#########################################################
# --------------------- PCA + KMeans --------------------
#########################################################

# "First of all Principal Component Analysis is a good name. It does what it says on the tin. PCA finds the principal components of data. ...
# They are the directions where there is the most variance, the directions where the data is most spread out."
# We try to perform Factor Analysis1 in order to extract the underlying factors of all vars, and then perform k-means on the obtained factors.
# A disadvantage of this procedure might be the difficult interpretability of the factor
# guide: https://medium.com/@dmitriy.kavyazin/principal-component-analysis-and-k-means-clustering-to-visualize-a-high-dimensional-dataset-577b2a7a5fe2

#1) Create a PCA instance: pca variance analysis
pca = PCA(n_components=len(X_std_df.columns))
principalComponents = pca.fit_transform(X_std)

# 2) Showing the percentage explained by each component and
# and the cumulative sum of this percentage on a table
pca_board = pd.DataFrame({"Explained var. (%)": np.round(pca.explained_variance_ratio_ * 100, decimals=1),
                          "Cumulative var. (%)": np.round(np.cumsum(pca.explained_variance_ratio_ * 100), decimals=2)})
pca_board.index.name = 'PC'
pca_board.index += 1
print("{}\n".format(pca_board))
pca_index = []
for i in range(1, len(X_std_df.columns) + 1):
    pca_index.append('PC' + str(i))
pca_inter=pd.DataFrame(pca.components_,
                   columns=X_std_df.columns,
                   index=pca_index)
pca_inter.head()

# Save components to a DataFrame
PCA_components = pd.DataFrame(principalComponents)

# -----------------------------------------------------
# 3) Plotting the Cumulative Sum of the Explained Variance
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

# Most of the variance is captured by the first 4 Principal Components therefore we perform kmeans using the first 4 Components
n_cols = 4

max_k=10
n_clusters = 5
#elbow plot
elbow_plot(PCA_components.iloc[:, :n_cols],max_k,n_clusters)

# ### we show a steep dropoff of inertia at k=4 so we can take k=4
# ## Perform cluster analysis on PC combonents
cluster_names=['Cluster 1', 'Cluster 2', 'Cluster 3','Cluster 4','Cluster 5']
X = PCA_components.iloc[:, :n_cols].values

pca_km = KMeans(n_clusters,init='k-means++',random_state=0, n_init=30)
pca_km.fit(X)

fig, ax = plt.subplots()
# ax.set_title("Kmean on the main {} PCs".format(n__cols))
scatter=ax.scatter(X[:, 0], X[:, 1], c=pca_km.labels_,
           s=30, cmap='viridis', marker='.')
# produce a legend with the unique colors from the scatter
plt.legend(handles=scatter.legend_elements()[0], labels=list(np.unique(pca_km.labels_)),bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.1)
plt.scatter(pca_km.cluster_centers_[0][0], pca_km.cluster_centers_[0][1], s=40, c='r', marker='s')
plt.scatter(pca_km.cluster_centers_[1][0], pca_km.cluster_centers_[1][1], s=40, c='r', marker='s')
plt.scatter(pca_km.cluster_centers_[2][0], pca_km.cluster_centers_[2][1], s=40, c='r', marker='s')
plt.scatter(pca_km.cluster_centers_[3][0], pca_km.cluster_centers_[3][1], s=40, c='r', marker='s')
if n_clusters==5:
    plt.scatter(pca_km.cluster_centers_[4][0], pca_km.cluster_centers_[4][1], s=40, c='r', marker='s')

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("KMeans using PCs and {} clusters".format(n_clusters))
plt.show()
# Print the cluster centroids
print("The centroids of each variable for each cluster:\n{}".format(pca_km.cluster_centers_)) # This gives the centroids for each cluster.
pca_km_centroids = pd.DataFrame(pca_km.cluster_centers_, columns=PCA_components.iloc[:, :n_cols].columns)
pca_km_unique, pca_km_counts = np.unique(pca_km.labels_, return_counts=True)
pca_km_counts = pd.DataFrame(np.asarray((pca_km_unique, pca_km_counts)).T, columns=['Label', 'Number'])
pca_km_centroids_df = pd.concat([pca_km_centroids, pca_km_counts], axis=1)
print(pca_km_centroids_df)
pca_km_labels_df = Std_clust_df.copy(deep=True)
pca_km_labels_df['pca_km_labels'] = pca_km.labels_
fig = plt.figure(figsize=(30, 100))
tidy = pca_km_labels_df.melt(id_vars='pca_km_labels')
sns.barplot(x='pca_km_labels', y='value', hue='variable', data=tidy)
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.1)
plt.show()
# Print results
print('pca_kmeans_{}: {}'.format(n_clusters,silhouette_score(Std_clust_df, pca_km.labels_,
                                           metric='euclidean')))

# -----------------------------------------------------
# 4) Plotting the contribution of each variable to the PCA in order to try and interpret the cluster centroids
plt.matshow(pca.components_, cmap='Spectral_r')

pca_index = []
for i in range(1, len(X_std_df.columns) + 1):
    pca_index.append('PC' + str(i))

tick_list = []
for i in range(0, len(X_std_df.columns)):
    tick_list.append(i)

plt.yticks(tick_list, pca_index, fontsize=10)
plt.xticks(range(len(X_std_df.columns)), X_std_df.columns, rotation=65, ha='left')
plt.show()

#########################################################
# -------------- Self-Organizing Maps -------------------
#########################################################
sm = SOMFactory().build(data=X_std_df.values,
               mapsize=(20, 20),
               normalization='var',
               initialization='random',     #'random', 'pca'
               component_names=X_std_df.columns,
               lattice='hexa',              #'rect','hexa'
               training='batch')            #'seq','batch'

sm.train(n_job=4,
         verbose='info',
         train_rough_len=30,
         train_finetune_len=50)
# Comments about SOM:
"""
        :param data: data to be clustered, represented as a matrix of n rows,
            as inputs and m cols as input features
        :param neighborhood: neighborhood object calculator.  Options are:
            - gaussian
            - bubble
            - manhattan (not implemented yet)
            - cut_gaussian (not implemented yet)
            - epanechicov (not implemented yet)
        :param normalization: normalizer object calculator. Options are:
            - var
        :param mapsize: tuple/list defining the dimensions of the som.
            If single number is provided is considered as the number of nodes.
        :param mask: mask
        :param mapshape: shape of the som. Options are:
            - planar
            - toroid (not implemented yet)
            - cylinder (not implemented yet)
        :param lattice: type of lattice. Options are:
            - rect
            - hexa
        :param initialization: method to be used for initialization of the som.
            Options are:
            - pca
            - random
        :param name: name used to identify the som
        :param training: Training mode (seq, batch)
"""
final_clusters = pd.DataFrame(sm._data, columns=X_std_df.columns)
som_labels = pd.DataFrame(sm._bmu[0])
final_clusters = pd.concat([final_clusters, som_labels], axis=1)
final_clusters.rename(columns={0 : 'Labels'}, inplace=True)
view2D = View2D(20, 20, "", text_size=7)
view2D.show(sm, col_sz=5, which_dim="all", denormalize=True)
plt.show()

vhts = BmuHitsView(10, 10, "Hits Map", text_size=7)
vhts.show(sm, anotate=True, onlyzeros=False, labelsize=10, cmap="Blues", logaritmic=False)

u = sompy.umatrix.UMatrixView(50, 50, 'umat', show_axis=True, text_size=8, show_text=True)
u.build_u_matrix(sm, distance=1000, row_normalized=False)
u.show(sm, distance2=8000, row_normalized=False, show_data=True, contooor=False, blob=False)

# K-Means Clustering
sm.cluster(5)   # According to median method of Hierarchical Clustering
hits = HitMapView(10, 10, "Clustering", text_size=7)
a = hits.show(sm, labelsize=12)

#As these labels get assigned to the BMUs and not each data sample the same analyse is not performed
som_kmeans_label=sm.cluster(5)


########################################################
# ----------------------K-means-------------------------
########################################################
# Define the number of K
k_max = 8
# Plotting elbow graph with function
elbow_plot(Std_clust_df, k_max,5)

init_methods = ['points', '++']
number_K = 5
keys, centroids_list, labels_list = compare_init_methods(Std_clust_df.values, init_methods,number_K)

# pick best initialisation method from plots above (please change accordingly)

best_method = "++"

centroids_dict = dict(zip(keys, centroids_list))
labels_dict = dict(zip(keys, labels_list))
kmeans_labels =list(labels_dict[best_method])
kmeans_centroids = scaler.inverse_transform(X=centroids_dict[best_method])
print(" Labels: \n {} \n Centroids: \n {}".format(kmeans_labels,kmeans_centroids ))
# Print results
print('kmeans_{}: {}'.format(number_K,silhouette_score(Std_clust_df, kmeans_labels,metric='euclidean')))
km_centroids = pd.DataFrame(kmeans_centroids, columns=Std_clust_df.columns)
km_unique, km_counts = np.unique(kmeans_labels, return_counts=True)
km_counts = pd.DataFrame(np.asarray((km_unique, km_counts)).T, columns=['Label', 'Number'])
km_centroids_df = pd.concat([km_centroids, km_counts], axis=1)
print(km_centroids_df)
km_centroids_df.to_csv("kmeans_centroids.csv")
km_labels_df = Std_clust_df.copy(deep=True)
km_labels_df['km_labels'] = kmeans_labels
fig = plt.figure(figsize=(30, 100))
tidy = km_labels_df.melt(id_vars='km_labels')
sns.barplot(x='km_labels', y='value', hue='variable', data=tidy)
plt.show()

########################################################
# ----------------------K-modes-------------------------
########################################################
# Selecting variables to use in K-modes and make Engage_df
Engage_df = df_cat.join(df[['gross_monthly_salary', 'cust_tenure']])
Engage_df.columns

# Converting gross_monthly_salary into categorical variable, using bins
Engage_df['salary_bin'] = pd.cut(Engage_df['gross_monthly_salary'],
                                 [0, 1000, 2000, 3000, 4000, 5000, 6000],
                                 labels=['0-1k', '1k-2k', '2k-3k', '3k-4k', '4k-5k', '5k-6k'])
# Converting cust_pol_age into categorical variable, using bins
Engage_df['tenure_bin'] = pd.cut(Engage_df['cust_tenure'],
                                 [20, 25, 30, 35, 40, 45, 50],
                                 labels=['20-25', '25-30', '30-35', '35-40', '40-45', '45-50'])

# Drop 'gross_monthly_salary' and 'cust_pol_age', since the goal is to perform K-modes
Engage_df = Engage_df.drop(['gross_monthly_salary', 'cust_tenure'], axis=1)
# Take a look at the new Engage_df full categorical
Engage_df.head()
Engage_df['salary_bin'] = Engage_df['salary_bin'].astype(str)
Engage_df['tenure_bin'] = Engage_df['tenure_bin'].astype(str)
Engage_df.columns

for col in Engage_df:
    print(Engage_df[col].unique())

# Choosing K for kmodes by comparing Cost against each K. Copied from:
# https://www.kaggle.com/ashydv/bank-customer-clustering-k-modes-clustering
cost = []
for num_clusters in list(range(1, 10)):
    kmode = KModes(n_clusters=num_clusters, init="Cao", n_init=30, verbose=1)
    kmode.fit_predict(Engage_df)
    cost.append(kmode.cost_)

y = np.array([i for i in range(1, 10, 1)])
plt.figure(figsize=(8, 5))
plt.plot(y, cost,
         linewidth=1.5,
         linestyle="-",
         marker="X",
         markeredgecolor="salmon",
         color="black")
plt.title("K-Modes elbow graph", loc="left", fontweight="bold")
plt.xlabel("Number of cluster")
plt.ylabel("Cost")
plt.axvline(x = 5, alpha = 0.4, color = "salmon", linestyle = "--")
plt.show()

## ------  K-modes with k of 3
k = 3
kmodes_clustering = KModes(n_clusters=k, init='Cao', verbose=1)
kmode_labels = kmodes_clustering.fit_predict(Engage_df)

# Turn the dummified df into two columns with PCA
pca = PCA(2)
plot_columns = pca.fit_transform(X_std_df)
X_std_df.shape

LABEL_COLOR_MAP = {0: 'b',
                   1: 'g',
                   2: 'r',
                   3: 'c',
                   4: 'm'}

fig, ax = plt.subplots()
for c in np.unique(kmode_labels):
    ix = np.where(kmode_labels == c)
    ax.scatter(plot_columns[:, 1][ix],
               plot_columns[:, 0][ix],
               c=LABEL_COLOR_MAP[c],
               label=kmodes_clustering.cluster_centroids_[c],
               s=30, marker='.')
ax.legend()
plt.title('K-modes over first 2 Principal Components, with %i' % k + ' clusters')
fig.savefig("Kdo")
plt.show()

# Print the cluster centroids
print("The mode of each variable for each cluster:\n{}".format(kmodes_clustering.cluster_centroids_)) # This gives the mode of each variable for each cluster.
kmodes_centroids = pd.DataFrame(kmodes_clustering.cluster_centroids_,columns=Engage_df.columns)

kmodes_unique, kmodes_counts = np.unique(kmodes_clustering.labels_, return_counts=True)
kmodes_counts = pd.DataFrame(np.asarray((kmodes_unique, kmodes_counts)).T, columns=['Label', 'Number'])
kmodes_centroids_df = pd.concat([kmodes_centroids, kmodes_counts], axis=1)
print(kmodes_centroids_df)
#########################################################
# --------------- Gaussian Mixture Models ---------------
#########################################################
# 0) This should be standardized df:
to_GMM = Std_clust_df

# 1) Set the GMM parameters
gmm = mixture.GaussianMixture(n_components=5,
                              init_params='kmeans', # {‘kmeans’, ‘random’}, defaults to ‘kmeans’.
                              max_iter=1000,
                              n_init=30,
                              verbose=1)
# 2) Fit the model
gmm.fit(to_GMM)

# 3) Get labels from clusters
gmm_labels = gmm.predict(to_GMM)

# 3.1) Scores
EM_score_samp = gmm.score_samples(to_GMM)
# 3.2) Prediction probability
EM_pred_prob = gmm.predict_proba(to_GMM)

gmm_n_clusters_ = len(np.unique(gmm_labels))

# 5) Count what?
gmm_unique, gmm_counts = np.unique(gmm_labels, return_counts=True)
gmm_counts = pd.DataFrame(np.asarray((gmm_unique, gmm_counts )).T, columns=['Label', 'Number'])
gmm_labels_df = Std_clust_df.copy(deep=True)
gmm_labels_df['gmm_labels'] = gmm_labels
gmm_centroids = gmm_labels_df.groupby('gmm_labels').mean()
gmm_centroids = scaler.inverse_transform(X=gmm_centroids)
gmm_centroids = pd.DataFrame(gmm_centroids, columns=Std_clust_df.columns)
gmm_centroids_df = pd.concat([gmm_centroids, gmm_counts], axis=1)
print(" Labels: \n {} \n Centroids: \n {}".format(gmm_labels,gmm_centroids ))
# Print results
print('Gaussian-Mixture: {}'.format(number_K,silhouette_score(Std_clust_df,gmm_labels,metric='euclidean')))
print(gmm_centroids_df)
gmm_centroids_df.to_csv("gaussian_centroids.csv")
fig = plt.figure(figsize=(30, 100))
tidy = gmm_labels_df.melt(id_vars='gmm_labels')
sns.barplot(x='gmm_labels', y='value', hue='variable', data=tidy)
plt.show()


#########################################################
# ---------------------- DBSCAN-------------------------
#########################################################
#https://towardsdatascience.com/cluster-analysis-create-visualize-and-interpret-customer-segments-474e55d00ebb

#db = DBSCAN(eps=0.7, min_samples=10).fit(Std_clust_df.values) #params for value_df
db = DBSCAN(eps=0.9, min_samples=50).fit(Std_clust_df.values)

dbs_n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

dbs_unique, dbs_counts = np.unique(db.labels_, return_counts=True)
dbs_counts = pd.DataFrame(np.asarray((dbs_unique, dbs_counts)).T, columns=['Label', 'Number'])
dbs_labels_df = Std_clust_df.copy(deep=True)
dbs_labels_df['dbscan_labels'] = db.labels_
dbs_centroids = (dbs_labels_df.loc[dbs_labels_df.dbscan_labels!=-1, :].groupby('dbscan_labels').mean())
dbs_centroids = scaler.inverse_transform(X=dbs_centroids)
dbs_centroids = pd.DataFrame(dbs_centroids, columns=Std_clust_df.columns)
print(" Labels: \n {} \n Centroids: \n {}".format(db.labels_,dbs_centroids ))
# Print results
print('DBSCAN: {}'.format(number_K,silhouette_score(Std_clust_df,db.labels_,metric='euclidean')))
dbs_centroids_df = pd.concat([dbs_centroids, dbs_counts.loc[dbs_counts.Label!=-1, :].reset_index()], axis=1)
print(dbs_centroids_df)
dbs_centroids_df.to_csv("dbs_centroids.csv")
fig = plt.figure(figsize=(30, 100))
tidy = dbs_labels_df.melt(id_vars='dbscan_labels')
sns.barplot(x='dbscan_labels', y='value', hue='variable', data=tidy)
plt.show()

###################################################
#------------------- Mean Shift -------------------
###################################################
# 0) This should be standardized df:
to_MS = Std_clust_df
# 1) Estimate bandwith:
# The following bandwidth can be automatically estimated using
my_bandwidth = estimate_bandwidth(to_MS,
                                  quantile=0.1,    # Quantile of all the distances/ 0.3 is the default
                                  n_samples=3000)

# 2) Create an object for Mean Shift:
ms = MeanShift(bandwidth=my_bandwidth,
               # bandwidth=0.15,
               bin_seeding=True)

# 3) Apply Mean Shift to data frame:
ms.fit(to_MS)
ms_cluster_centers = ms.cluster_centers_
# Count the number of unique labels (clusters)
ms_n_clusters_ = len(np.unique(ms.labels_))
# # 4) Re-scale the cluster_centers
ms_cluster_centers2=scaler.inverse_transform(X=ms_cluster_centers)
# Print the cluster centroids
print(" Labels: \n {} \n Centroids: \n {}".format(ms.labels_,ms_cluster_centers2 ))
print('Mean-Shift: {}'.format(number_K,silhouette_score(Std_clust_df,ms.labels_,metric='euclidean')))
ms_centroids = pd.DataFrame(ms_cluster_centers2,columns=to_MS.columns)
ms_unique, ms_counts = np.unique(ms.labels_, return_counts=True)
ms_counts = pd.DataFrame(np.asarray((ms_unique,ms_counts)).T, columns=['Label', 'Number'])
ms_centroids_df = pd.concat([ms_centroids, ms_counts], axis=1)
print(ms_centroids_df)
ms_centroids_df.to_csv("meanshift_centroids.csv")
ms_labels_df = Std_clust_df.copy(deep=True)
ms_labels_df['ms_labels'] = ms.labels_
fig = plt.figure(figsize=(30, 100))
tidy = ms_labels_df.melt(id_vars='ms_labels')
sns.barplot(x='ms_labels', y='value', hue='variable', data=tidy)
plt.show()


#########################################################
# ---------------- Hierarchical clustering--------------
#########################################################
# ## Try several clustering heuristics using the standardised dataframe
methods = ['ward', 'complete', 'average', 'weighted', 'centroid', 'median']  #single (removed as not good)
fig = plt.figure(figsize=(30, 100))
for n, method in enumerate(methods):
    #can try with diff data combinations: final_clusters.drop('Labels', axis=1).values , X_std_df , X_prod_df, X_value_df
    try:
        Z = linkage(Std_clust_df, method)
        ax = fig.add_subplot(len(methods), 1, n + 1)
        dendrogram(Z, ax=ax, labels=df.index, truncate_mode='level',p=3, color_threshold=0.62 * max(Z[:, 2]))
        #ax.tick_params(axis='x', which='major', labelsize=20)
        #ax.tick_params(axis='y', which='major', labelsize=20)
        ax.set_xlabel("Number of points in node (or index of point if no parenthesis).")
        ax.set_title('{} Method Dendrogram'.format(method))
        plt.axis('off')
    except Exception as e:
        print('Error caught:'.format(e))

fig.savefig('all_methods_dendrogram.png'.format(method))
plt.show()
# Ward variance minimization algorithm provides the most clear clusters, it is based on the Eucleadian distance
# perform hierarchical clustering on the output of the SOM # can change linkage distance calculation method e.g centroid
Z = linkage(final_clusters.drop('Labels', axis=1).values, 'ward')
method = "ward"
fig = plt.figure(figsize=(30, 20))
ax = fig.add_subplot(1, 1, 1)
dendrogram(Z, ax=ax, labels=df.index, truncate_mode='lastp', color_threshold=0.62 * max(Z[:, 2]))
ax.tick_params(axis='y', which='major', labelsize=20)
ax.set_title('{} Method Dendrogram'.format(method))
fig.savefig('{}_method_dendrogram.png'.format(method))

#https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_clustering.html#sphx-glr-auto-examples-cluster-plot-agglomerative-clustering-py
fig = plt.figure(figsize=(30, 100))
k_list=[4,5]
methods_list =['average','complete','ward']
n=0
for index,n_clusters in enumerate(k_list):
    for k_index, linkage in enumerate(methods_list):
        n+=1
        plt.subplot(len(k_list), len(methods_list), n)
        print(linkage)
        model = AgglomerativeClustering(linkage=linkage,n_clusters=n_clusters)
        model.fit(final_clusters.drop('Labels', axis=1).values)
        plt.scatter(X[:, 0], X[:, 1], c=model.labels_,
                    cmap=plt.cm.nipy_spectral)
        plt.title('linkage=%s , n_cluster=%i' % (linkage,n_clusters),
                  fontdict=dict(verticalalignment='top'))
        plt.axis('equal')
        plt.axis('off')
        plt.suptitle('Hierarchical clustering comparision', size=17)

model_unique, model_counts = np.unique(model.labels_, return_counts=True)
model_counts = pd.DataFrame(np.asarray((model_unique, model_counts)).T, columns=['Label', 'Number'])
model_labels_df = Std_clust_df.copy(deep=True)
model_labels_df['agg_labels'] = model.labels_
model_centroids = scaler.inverse_transform(X=model_labels_df.groupby('agg_labels').mean())
model_centroids = pd.DataFrame(model_centroids, columns=Std_clust_df.columns)
print(" Labels: \n {} \n Centroids: \n {}".format(model.labels_,model_centroids ))
# Print results
print('agg_cluster: {}'.format(number_K,silhouette_score(Std_clust_df, model.labels_,metric='euclidean')))
model_centroids_df = pd.concat([model_centroids, model_counts], axis=1)
print(model_centroids_df)
model_centroids_df.to_csv("agg_centroids.csv")
fig = plt.figure(figsize=(30, 100))
tidy = model_labels_df.melt(id_vars='agg_labels')
sns.barplot(x='agg_labels', y='value', hue='variable', data=tidy)
plt.show()

# Try different methods for clustering, check documentation:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html?highlight=linkage#scipy.cluster.hierarchy.linkage
# https://www.analyticsvidhya.com/blog/2019/05/beginners-guide-hierarchical-clustering/
# https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering
# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html
# Hierarchical clustering, does not require the user to specify the number of clusters.
# Initially, each point is considered as a separate cluster, then it recursively clusters the points together depending upon the distance between them.
# The points are clustered in such a way that the distance between points within a cluster is minimum and distance between the cluster is maximum.
# Commonly used distance measures are Euclidean distance, Manhattan distance or Mahalanobis distance. Unlike k-means clustering, it is "bottom-up" approach.

###################################################
# ---------------- Cluster Visualisation ------------------
###################################################
# 0.1) Make sure that the 'labels' values correspond to the
# clustering method you pretend to plot
# #ms.labels_,db.labels_
names_methods_list =["Kmeans","Gaussian-Mixture", "Agglomerative (Ward distance"] # "Mean-Shift", "DBSCAN",
for i, name in enumerate(names_methods_list):
    # name="Mean-Shift"
    # labels_for_pca =ms.labels_
    label_methods_list = [kmeans_labels, gmm_labels, model.labels_]
    labels_for_pca = label_methods_list[i]
    print(len(np.unique(labels_for_pca)))
    # 0.2) Set the data frame to analyse
    # It should be standardized!
    df_for_pca = Std_clust_df

    ###################################################
    #------------ Plotting 2D PCA results ------------
    ###################################################

    fig = plt.figure(figsize=(30, 100))
    pca = PCA(n_components=2).fit(df_for_pca)  # fit to 2 principal components
    pca_2d = pca.transform(df_for_pca)
    cluster_names=['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4','Cluster 5', 'Cluster 6', 'Cluster 7', 'Cluster 8', 'Cluster 9']
    # Set a marker and a color to each label (cluster)
    for i in range(0, pca_2d.shape[0]):
        if labels_for_pca[i] == 0:
            c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
        elif labels_for_pca[i] == 1:
            c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
        elif labels_for_pca[i] == 2:
            c3 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='*')
        elif labels_for_pca[i] == 3:
            c4 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='y', marker='s')
        elif labels_for_pca[i] == 4:
            c5 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='m', marker='p')
        elif labels_for_pca[i] == 5:
            c6 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='c', marker='H')
        elif labels_for_pca[i] == 6:
            c7 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='k', marker='o')
        elif labels_for_pca[i] == 7:
            c8 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='w', marker='.')
        elif labels_for_pca[i] == 8:
            c9 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='p')

    plt.legend([c1,c2,c3,c4,],cluster_names[:len(np.unique(labels_for_pca))])
    plt.title('%i' % len(np.unique(labels_for_pca) )+ ' clusters found from {} method'.format(name))
    plt.show()

###################################################
#------------ Plotting 3D PCA results ------------
###################################################
pca = PCA(n_components=3).fit(df_for_pca)  # fit to 3 principal components
pca_3d = pca.transform(df_for_pca)
# Add my visuals
my_color = []
my_marker = []
# Load my visuals
# Set a marker and a color to each label (cluster)
for i in range(pca_3d.shape[0]):
    if labels_for_pca[i] == 0:
        my_color.append('r')
        my_marker.append('+')
    elif labels_for_pca[i] == 1:
        my_color.append('g')
        my_marker.append('o')
    elif labels_for_pca[i] == 2:
        my_color.append('b')
        my_marker.append('*')
    elif labels_for_pca[i] == 3:
        my_color.append('y')
        my_marker.append('s')
    elif labels_for_pca[i] == 4:
        my_color.append('m')
        my_marker.append('p')
    elif labels_for_pca[i] == 5:
        my_color.append('c')
        my_marker.append('H')
    elif labels_for_pca[i] == 6:
        my_color.append('k')
        my_marker.append('o')


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

del df_for_pca, labels_for_pca

###################################################
# ------------- Silhouette Analysis ---------------
###################################################

# ------------------------------------------------------
# Applying Silhouette Analysis function over normalized df
silhouette_analysis(Std_clust_df, 4, 5)

###################################################
# ------------- Evaluate Clustering methods ---------------
###################################################

print('kmeans: {}'.format(silhouette_score(Std_clust_df, kmeans_labels,
                                           metric='euclidean')))
print('PCA kmeans:{}'.format(silhouette_score(Std_clust_df,
                                          pca_km.labels_,
                                          metric='euclidean')))
print('DBSCAN: {}'.format(silhouette_score(Std_clust_df, db.labels_,
                                           metric='euclidean')))
print('Mean-Shit: {}'.format(silhouette_score(Std_clust_df, ms.labels_,
                                           metric='euclidean')))
print('Gaussian-Mixture: {}'.format(silhouette_score(Std_clust_df, gmm_labels,
                                           metric='euclidean')))
print('Agglomerative: {}'.format(silhouette_score(Std_clust_df, model.labels_,
                                           metric='euclidean')))

#########################################################
# ------------- Decision Tree Classifier ----------------
#########################################################
# Define the data frame with labels column
DT_df = clust_df
dt_labels = kmeans_labels

# Define the target variable 'y'
X = DT_df
y = dt_labels  # The target is the cluster label

# Split up the data into a training set and a test set
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Define the parameters for the Decision Tree
clf = DecisionTreeClassifier(criterion='gini',
                             splitter='best',
                             max_depth=None,
                             random_state=10,
                             min_samples_leaf=3)

# Train the model
clf = clf.fit(X_train, y_train)

# Evaluation of the decision tree results
predict_test = clf.predict(X_test)

conf_matrix = confusion_matrix(y_test, predict_test)
accuracy = accuracy_score(y_test, predict_test)

# Show confusion matrix
conf_matrix

# Show accuracy
accuracy

# Print a classification tree report for the test
print(classification_report(y_test, predict_test))

features = list(DT_df.columns)
features

# -------- Predicting clusters for outliers df ----------
# Evaluation of the decision tree results
to_predict=df_outliers.dropna(axis=0, how='any')
to_predict['km_labels']=clf.predict(to_predict)
predict_labels_df = clust_df.copy(deep=True)

predict_labels_df['km_labels'] = kmeans_labels
clust_df_with_outliers = pd.concat([predict_labels_df,to_predict],axis=0, sort=True)
clust_df_with_outliers_std = scaler.fit_transform(clust_df_with_outliers.iloc[:,:-1])
clust_df_with_outliers_std_df = pd.DataFrame(clust_df_with_outliers_std, columns=clust_df.columns)
clust_df_with_outliers_std_df['km_labels'] = clust_df_with_outliers['km_labels']
with_outliers_unique, with_outliers_counts = np.unique(predict_labels_df['km_labels'].tolist(), return_counts=True)
with_outliers_counts = pd.DataFrame(np.asarray((with_outliers_unique, with_outliers_counts)).T, columns=['Label', 'Number'])
with_outliers_centroids = clust_df_with_outliers.groupby('km_labels').mean()
with_outliers_centroids_df = pd.concat([with_outliers_centroids, with_outliers_counts], axis=1)
print(with_outliers_centroids_df)
model_centroids_df.to_csv("with_outliers_centroids.csv")
fig = plt.figure(figsize=(30, 100))
tidy = clust_df_with_outliers_std_df.melt(id_vars='km_labels')
sns.barplot(x='km_labels', y='value', hue='variable', data=tidy)
plt.show()



# Plotting Decision Tree
# Create DOT data
dot_data = tree.export_graphviz(clf,
                                out_file=None,
                                feature_names=DT_df.columns,
                                class_names=None,
                                filled=True,
                                rounded=True,
                                special_characters=True)

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)
# Show graph
Image(graph.create_png())
# Create PNG
graph.write_png("clf_tree.png")

# References:
# https://towardsdatascience.com/decision-tree-algorithm-explained-83beb6e78ef4
# https://www.youtube.com/watch?v=z-AGmGmR6Z8
# https://www.geeksforgeeks.org/python-decision-tree-regression-using-sklearn/
