# Import packages
import sqlite3
import pandas as pd
import numpy as np
from datetime import date
import scipy.cluster.hierarchy as clust_hier
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
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
path1 = "C:\\Users\\Leonor.furtado\\OneDrive - Accenture\\Uni\\Data Mining\\project\\"
path2 = ""
excel_df = pd.read_csv("A2Z Insurance.csv")

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
combined_df = combined_df.drop(['education_lvl', 'edu_desc'], axis=1)

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
null_values = combined_df[combined_df.isna().any(axis=1)]

# Drop rows with NA values
df = combined_df.dropna(axis=0, how='any')

# Defining each column type value with a  dictionary
type_dict = {
    'policy_creation_year': int,
    'birth_year': int,
    'gross_monthly_salary': float,
    'geographic_area': int,
    'has_children': int,
    'customer_monetary_value': float,
    'claims_rate': float,
    'motor_premiums': float,
    'household_premiums': float,
    'health_premiums': float,
    'life_premiums': float,
    'work_premiums': float,
    'edu_code': int
}
df.columns
df = df.astype(type_dict)
df.dtypes

# -------------- Business logic validation of the data
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
          ' Additional check is neccesary.', '\n')

# birth_year: should not exist values larger than policy year creation
df["customer younger than policy"] = np.where(df['policy_creation_year'] < (df['birth_year'] + 18), 1, 0)
print("6. Are there values greater than policy_creation _year on the 'birth_year'?: ",
      sum(df["customer younger than policy"]))

df = df[df["customer younger than policy"] == 0]

# customer_monetary_value (CMV), nothing to verify
# claims_rate, nothing to verify
# all premiums, nothing to verify
# all the other columns, (nothing to verify)


# --------------Outliers-----
outlier = df.iloc[[172, 9150, 8867], :]
df.drop([172, 9150, 8867], inplace=True)


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


# 2) Boxplots
def outliers_boxplot(df_in):
    fig, axes = plt.subplots(len(df_in.columns), 1, figsize=(20, 30))

    i = 0
    for my_box in axes:
        sns.boxplot(data=df_in.iloc[:, i],
                    orient='h',
                    ax=my_box).set_title(df_in.columns[i])
        i = i + 1
    fig.savefig("outliers_boxplot.png")


# Apply histogram function to the entire data frame
outliers_hist(df)

# After looking at the histograms, we can check further for outliers
# just on the following attributes:
check_list = ['gross_monthly_salary',
              'customer_monetary_value',
              'claims_rate',
              'motor_premiums',
              'household_premiums',
              'health_premiums',
              'life_premiums',
              'work_premiums']

df_check = df[df.columns.intersection(check_list)]

# Now, let's identify outliers with Box Plot over normalized df_check
# Standardize the data to have a mean of ~0 and a variance of 1

df_norm = pd.DataFrame(StandardScaler().fit_transform(df_check))
# Rename columns of normalized data frame
df_norm.columns = check_list

# Apply boc-plot function to the selected columns
outliers_boxplot(df_norm)

# References for detecting outliers:
# https://www.dataquest.io/blog/tutorial-advanced-for-loops-python-pandas/
# https://wellsr.com/python/python-create-pandas-boxplots-with-dataframes/
# https://notebooks.ai/rmotr-curriculum/outlier-detection-using-boxplots-a89cd3ee
# https://seaborn.pydata.org/examples/horizontal_boxplot.html

# -------------- Plotting correlation matrix and correlogram
# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Correlogram with regression
sns.pairplot(df, kind="reg")
plt.show()

# Correlogram without regression
sns.pairplot(df, kind="scatter")
plt.show()

# References for plotting correlation matrix on seaborn:
# https://seaborn.pydata.org/examples/many_pairwise_correlations.html
# https://python-graph-gallery.com/111-custom-correlogram/
# -------------- Caculating additional columns

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

# Do the same with 'birth_year column' to get the customers' ages
df['cust_age'] = today_year - df['birth_year']
df.head()

# dropping the year columns as this information has now been captured in the age variables created
df.drop(['policy_creation_year', 'birth_year'], axis=1, inplace=True)

# # Calculating and adding 'Customer annual profit' to the data frame
# df['cust_annual_prof'] = df['gross_monthly_salary']*12  # Please, let me know if 12 is OK, in Panama is 13
#
# # Calculating the acquisition cost:
# df['cust_acq_cost'] = df['cust_annual_prof']*df['cust_pol_age'] - df['customer_monetary_value']

# For 'claims_rate' (CR) it's possible to clear the 'Amount paid by the insurance company'
# claims_rate = (Amount paid by the insurance company)/(Total Premiums)
# Therefore:
# (Amount paid by the insurance company) = (claims_rate)*(Total Premiums)

# where: (Total Premiums) can be calculated prior, as the sum of:
# 'motor_premiums', 'household_premiums', 'health_premiums',
# 'life_premiums', 'work_premiums'

# Calculating and adding 'total_premiums' to the data frame
df['total_premiums'] = df['motor_premiums'] + \
                       df['household_premiums'] + \
                       df['health_premiums'] + \
                       df['life_premiums'] + \
                       df['work_premiums']

# Calculate 'Amount paid by the insurance company'
df['amt_paidby_comp'] = df['claims_rate'] * df['total_premiums']

# We are now going to scale the data so we can do effective clustering of our variables
# Standardize the data to have a mean of ~0 and a variance of 1
scaler = StandardScaler()
X_std = scaler.fit_transform(df)
X_std_df = pd.DataFrame(X_std, columns=df.columns)

# ## Experiment with alternative clustering techniques

# variance zero cols must go
corr = df.corr()  # Calculate the correlation of the above variables
sns.set_style("whitegrid")
# sns.heatmap(corr) #Plot the correlation as heat map

sns.set(font_scale=1.0)
cmap = sns.diverging_palette(h_neg=210, h_pos=350, s=90, l=30, as_cmap=True)
# clustermap
clustermap = sns.clustermap(data=corr, cmap="coolwarm", annot_kws={"size": 12})
clustermap.savefig('corr_clustermap.png')
# ## Perform PCA
# "First of all Principal Component Analysis is a good name. It does what it says on the tin. PCA finds the principal components of data. ...
# They are the directions where there is the most variance, the directions where the data is most spread out."
# guide: https://medium.com/@dmitriy.kavyazin/principal-component-analysis-and-k-means-clustering-to-visualize-a-high-dimensional-dataset-577b2a7a5fe2

# Create a PCA instance: pca
pca = PCA(n_components=6)
principalComponents = pca.fit_transform(X_std)
# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.show()
plt.clf()

# Save components to a DataFrame
PCA_components = pd.DataFrame(principalComponents)
#
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
plt.clf()

# # ### we show a steep dropoff of inertia at k=6 so we can take k=6
# # ## Perform cluster analysis on PCA variables
#
# n_clusters = 6
#
# X = PCA_components.values
# labels = KMeans(n_clusters, random_state=0).fit_predict(X)
#
# fig, ax = plt.subplots(figsize=(15, 15))
# ax.scatter(X[:, 0], X[:, 1], c=labels,
#            s=50, cmap='viridis');
#
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

Z = linkage(X_std, 'ward')
Z2 = linkage(X_std, 'single', optimal_ordering=True)

# Ward variance minimization algorithm

method = "ward"
fig = plt.figure(figsize=(30, 20))
ax = fig.add_subplot(1, 1, 1)
dendrogram(Z, ax=ax, labels=df.index, truncate_mode='lastp', color_threshold=4)
ax.tick_params(axis='y', which='major', labelsize=20)
ax.set_xlabel('Data Object')
fig.savefig('{}_method_dendrogram.png'.format(method))

# ## Nearest Point Algorithm

method = 'single'
fig = plt.figure(figsize=(30, 20))
ax = fig.add_subplot(1, 1, 1)
dendrogram(Z2, ax=ax, labels=df.index, truncate_mode='lastp', color_threshold=1.25)
ax.tick_params(axis='x', which='major', labelsize=20)
ax.tick_params(axis='y', which='major', labelsize=20)
ax.set_xlabel('Data Object')
fig.savefig('{}_method_dendrogram.png'.format(method))
fig.clf()

# # ## Try several different clustering heuristics
# methods = ['ward', 'single', 'complete', 'average', 'weighted', 'centroid', 'median', ]
# fig = plt.figure(figsize=(30, 100))
# for n, method in enumerate(methods):
#     try:
#         Z = linkage(X_std, method)
#         ax = fig.add_subplot(len(methods), 1, n + 1)
#         dendrogram(Z, ax=ax, labels=df.index, truncate_mode='lastp', color_threshold=0.62 * max(Z[:, 2]))
#         ax.tick_params(axis='x', which='major', labelsize=20)
#         ax.tick_params(axis='y', which='major', labelsize=20)
#         ax.set_xlabel(method)
#         fig.savefig('{}_method_dendrogram.png'.format(method))
#
#     except Exception as e:
#         print('Error caught:'.format(e))
# plt.show()

# DBSCAN

db = DBSCAN(eps=1, min_samples=10).fit(X_std)

labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

unique_clusters, count_clusters = np.unique(db.labels_, return_counts=True)

# -1 is the noise
print(np.asarray((unique_clusters, count_clusters)))

# Visualising the clusters

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
           ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 5', 'Noise'])
plt.title('DBSCAN finds 6 clusters and noise')
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

# mean shift
to_MS = X_std
# The following bandwidth can be automatically detected using
my_bandwidth = estimate_bandwidth(to_MS,
                                  quantile=0.2,
                                  n_samples=1000)

ms = MeanShift(bandwidth=my_bandwidth,
               # bandwidth=0.15,
               bin_seeding=True)

ms.fit(to_MS)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

# Values
scaler.inverse_transform(X=cluster_centers)

# Count
unique, counts = np.unique(labels, return_counts=True)

print(np.asarray((unique, counts)).T)

# lets check our are they distributed
pca = PCA(n_components=3).fit(to_MS)
pca_2d = pca.transform(to_MS)
for i in range(0, pca_2d.shape[0]):
    if labels[i] == 0:
        c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
    elif labels[i] == 1:
        c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
    elif labels[i] == 2:
        c3 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='*')

plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2', 'Cluster 3 '])
plt.title('Mean Shift found 3 clusters')
plt.show()
plt.clf()

# 3D
pca = PCA(n_components=3).fit(to_MS)
pca_3d = pca.transform(to_MS)
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
    elif labels[i] == 3:
        my_color.append('k')
        my_marker.append('<')

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
