# Import packages
import os
import sqlite3
from datetime import date

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.util.testing import assert_frame_equal
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from kmodes.kmodes import KModes

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

#We import the excel data to check if it is the same as that we receive from the DB
excel_df=pd.read_csv(os.path.join(os.getcwd(),"A2Z Insurance.csv"))

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
null_df= combined_df[combined_df.isna().any(axis=1)]

# Drop rows with NA values
df = combined_df.dropna(axis=0, how='any')

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

#we decide to map the binary variable to a string variable to treat is as a categorical variable
df['has_children'] = df['has_children'].map({1:'Yes', 0: 'No'})


# birth_year: should not exist values larger than policy year creation
print("6. Are there values greater than policy_creation _year on the 'birth_year'?: ",
      sum(np.where(df['policy_creation_year'] < (df['birth_year'] + 18), 1, 0)))

# -------------- Plotting correlation matrix to check if birthday_year can be
# associated to another variable due to their high correlation
# # Compute the correlation matrix
corr = df.corr()

# Set Up Mask To Hide Upper Triangle

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
np.triu_indices_from(mask)
mask[np.triu_indices_from(mask)] = True

fig, ax = plt.subplots(figsize=(11, 15))
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
                      annot_kws={'size': 12})
#add the column names as labels
ax.set_yticklabels(corr.columns, rotation=0)
ax.set_xticklabels(corr.columns)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
fig.savefig("corr_matrix.png")

# The correlation matrix shows a correlation between salary and birth year
# So let's drop birth_year column, but keep in mind that people with high salary might be older do to the correlation

# Due to the high levels of inconsistent data in the birth year column we decide to drop this column as the data in it cannot be trusted
df.drop('birth_year', axis=1, inplace=True)

# customer_monetary_value (CMV), nothing to verify
# claims_rate, nothing to verify
# all premiums, nothing to verify
# all the other columns, (nothing to verify)

# Categorical boolean mask
categorical_feature_mask = df.dtypes == object
# filter categorical columns using mask and turn it into a list
categorical_cols = df.columns[categorical_feature_mask].tolist()

df_cat = df.loc[:, categorical_cols]

df.drop(categorical_cols, axis=1, inplace=True)

# -------------- Detecting outliers
# After logical validation, we check for outliers using different methods:
# 1) Histograms
def hist_all_columns(df_in):
    fig, axes = plt.subplots(len(df_in.columns) // 3, 3, figsize=(20, 48))

    i = 0
    for triaxis in axes:
        for axis in triaxis:
            df_in.hist(column=df_in.columns[i], bins=100, ax=axis)
            i = i + 1
    fig.savefig("outliers_hist.png")


# Apply histogram function to the entire data frame
# hist_all_columns(df)

# -------------- Calculating additional columns

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

# Calculating the acquisition cost:
df['cust_acq_cost'] = df['total_premiums']*df['cust_pol_age'] - df['customer_monetary_value']
#
# For 'claims_rate' (CR) it's possible to clear the 'Amount paid by the insurance company'
# claims_rate = (Amount paid by the insurance company)/(Total Premiums)
# Therefore:
# (Amount paid by the insurance company) = (claims_rate)*(Total Premiums)
#
# where: (Total Premiums) can be calculated prior, as the sum of:
# 'motor_premiums', 'household_premiums', 'health_premiums',
# 'life_premiums', 'work_premiums'

# Calculate 'Amount paid by the insurance company'
df['amt_paidby_comp_2yrs'] = df['claims_rate'] * df['total_premiums']

#Calculate the premium/wage proportion

df['premium_wage_ratio'] = df['total_premiums']/(df['gross_monthly_salary']*12)


# --------------K-modes-----
# separate df into engage and consume
df_Engage = df_cat.join(df['gross_monthly_salary'], how='inner')
df_Engage.dtypes

# Plotting 'gross_monthly_salary' before converting to categorical
# just to see the range and determine the bins
plt.figure()
sns.distplot(df_Engage['gross_monthly_salary'])
plt.savefig('salary_histogram.png')

# Oh! There are a few people with a really high salary
# The bin will be 1000 (1k), and the salaries above 6k will be considered outliers.
# The rows that contains those individuals will be store into this data frame:
salary_outliers = df_Engage[df_Engage['gross_monthly_salary'] > 6000]
# Check the shape of this data frame
salary_outliers.head()   # Ja! just 2 guys above $6k. Messi and CR7

df_Engage = df_Engage[df_Engage['gross_monthly_salary'] <= 6000]
df_Engage.shape         # 9985 individuals below $6k

# Converting gross_monthly_salary into categorical variable, using bins
df_Engage['salary_bin'] = pd.cut(df_Engage['gross_monthly_salary'],
                                 [0, 1000, 2000, 3000, 4000, 5000, 6000],
                                 labels=['0-1k', '1k-2k', '2k-3k', '3k-4k', '4k-5k', '5k-6k'])

# Drop 'gross_monthly_salary', since the goal is to perform K-modes
df_Engage = df_Engage.drop('gross_monthly_salary', axis=1)

# Take a look at the new df_Engage full categorical
df_Engage.head()
df_Engage.columns
df_Engage.dtypes
df_Engage['salary_bin'] = df_Engage['salary_bin'].astype(str)
df_Engage.dtypes

## ------ DM lab code for K-modes
kmodes_clustering = KModes(n_clusters=8, init='random', n_init=50, verbose=1)
clusters_cat = kmodes_clustering.fit_predict(df_Engage)

# Print the cluster centroids
print(kmodes_clustering.cluster_centroids_)
df_Engage_centroids = pd.DataFrame(kmodes_clustering.cluster_centroids_,
                                   columns=['geographic_area',
                                            'has_children',
                                            'edu_desc',
                                            'salary_bin'])

unique, counts = np.unique(kmodes_clustering.labels_, return_counts=True)
cat_counts = pd.DataFrame(np.asarray((unique, counts)).T, columns=['Label', 'Number'])
cat_centroids = pd.concat([df_Engage_centroids, cat_counts], axis=1)

# This was my result for k =8 :
#   geo     child   educ            salary      my_label
# [['4.0'   'Yes'   'BSc/MSc'       '2k-3k']
#  ['4.0'   'Yes'   'High School'   '1k-2k']
#  ['4.0'   'Yes'   'High School'   '2k-3k']
#  ['1.0'   'No'    'High School'   '3k-4k']
#  ['1.0'   'Yes'   'BSc/MSc'       '3k-4k']
#  ['4.0'   'Yes'   'BSc/MSc'       '1k-2k']
#  ['3.0'   'No'    'BSc/MSc'       '4k-5k']
#  ['4.0'   'No'    'BSc/MSc'       '3k-4k']]

# I think, this K-modes should be performed by region
# and reduce the number of clusters. To see, if hopefully
# we can have a cluster for each educ level. At least, educ level and salary are correlated

# # --------------- This part is under construction !!!!!!!!!!!!!!!!!
# # Plotting 'cust_pol_age' before converting
# sns.distplot(df['cust_pol_age'], bins=10)
#
# # Converting cust_policy_age into categorical variable
# df['pol_age_bin'] = pd.cut(df['cust_pol_age'],
#                            [0, 10, 20, 30, 40, 50],
#                            labels=['0-10', '10-20', '20-30', '30-40', '40-50'])
# # Drop 'cust_pol_age'
# df = df.drop('cust_pol_age', axis=1)
#
# # Doing a copy of of the data frame before
# df_copy = df.copy()
# # --------------- This part is under construction !!!!!!!!!!!!!!!!!





# We are now going to scale the data so we can do effective clustering of our variables
# Standardize the data to have a mean of ~0 and a variance of 1
scaler = StandardScaler()
X_std = scaler.fit_transform(df)
X_std_df = pd.DataFrame(X_std, columns=df.columns)

# --------------Outliers-----

# Define the lower and upper quartiles boundaries for plotting the boxplots
# and for dropping values. Numbers between (0,1) and qtl1 < qtl2
qtl_1 = 0.05  # lower boundary
qtl_2 = 0.95  # upper boundary

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
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)

    # df_out is filtered with values within the quartiles boundaries
    df_out = df_in[~((df_in < lower_range) | (df_in > upper_range)).any(axis=1)]
    df_outliers = df_in[((df_in < lower_range) | (df_in > upper_range)).any(axis=1)]
    return df_out, df_outliers


# Reference:
# https://medium.com/@prashant.nair2050/hands-on-outlier-detection-and-treatment-in-python-using-1-5-iqr-rule-f9ff1961a414

# Apply box-plot function to the selected columns
boxplot_all_columns(X_std_df, qtl_1, qtl_2)

# There are outliers, so let's remove them with the 'IQR_drop_outliers' function
df, df_outliers = IQR_drop_outliers(df, qtl_1, qtl_2)

# We are now going to scale the data so we can do effective clustering of our variables
# Standardize the data to have a mean of ~0 and a variance of 1
scaler = StandardScaler()
X_std = scaler.fit_transform(df)
X_std_df = pd.DataFrame(X_std, columns=df.columns)

# Plot without outliers
boxplot_all_columns(X_std_df, qtl_1, qtl_2)

# -------------- Plotting correlation matrix
# # Compute the correlation matrix
corr = df.corr()

# Set Up Mask To Hide Upper Triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
np.triu_indices_from(mask)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 15))
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
                      annot_kws={'size': 12})
#add the column names as labels
ax.set_yticklabels(corr.columns, rotation=0)
ax.set_xticklabels(corr.columns)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
plt.show()

# Doing individual scatter plots to find trends in customers and classify them

# Get columns names
X_std_df.columns

# Set seaborn palette colors
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

# Pair-plot just for products and customers salaries
salary_premiums = X_std_df[['gross_monthly_salary',
                            'total_premiums',
                            'motor_premiums',
                            'work_premiums',
                            'health_premiums',
                            'life_premiums',
                            'household_premiums']]
with sns.color_palette('Paired'):
    sns.pairplot(data=salary_premiums,
                 y_vars=['total_premiums',
                         'motor_premiums',
                         'work_premiums',
                         'health_premiums',
                         'life_premiums',
                         'household_premiums'],
                 x_vars=['gross_monthly_salary'],
                 markers='x')
    plt.show()

# Pair-plot just for products and claims rate
claims_premiums = X_std_df[['claims_rate',
                            'total_premiums',
                            'motor_premiums',
                            'work_premiums',
                            'health_premiums',
                            'life_premiums',
                            'household_premiums']]
with sns.color_palette(flatui):
    sns.pairplot(data=claims_premiums,
                 y_vars=['total_premiums',
                         'motor_premiums',
                         'work_premiums',
                         'health_premiums',
                         'life_premiums',
                         'household_premiums'],
                 x_vars=['claims_rate'],
                 markers='x')
    plt.show()

# Perform clustering over education and salary

# Perform clustering over geography and salary

# Perform clustering over life_premiums and has_children

# Perform clustering over life_premiums and has_children

# Perform clustering over policy_age and salary
pol_age_salary = X_std_df[['cust_pol_age', 'gross_monthly_salary']]

with sns.color_palette(flatui):
    sns.pairplot(data=pol_age_salary,
                 y_vars=['cust_pol_age'],
                 x_vars=['gross_monthly_salary'],
                 markers='1')
    plt.show()

# ## Experiment with alternative clustering techniques
# variance zero cols must go
corr = X_std_df.corr()  # Calculate the correlation of the above variables
sns.set_style("whitegrid")
# sns.heatmap(corr) #Plot the correlation as heat map

sns.set(font_scale=1.0)
cmap = sns.diverging_palette(h_neg=210, h_pos=350, s=90, l=30, as_cmap=True)
# clustermap
clustermap = sns.clustermap(data=corr, cmap="coolwarm", annot_kws={"size": 12})
clustermap.savefig('corr_clustermap.png')

sns.pairplot(data=corr, size = 2)
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

Z = linkage(salary_premiums[['gross_monthly_salary', 'total_premiums']], 'ward')
Z2 = linkage(salary_premiums[['gross_monthly_salary', 'total_premiums']], 'single', optimal_ordering=True)

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

#finding best initialisation method
def compare_init_methods(data,list_init_methods,K_n):
    """
    This function returns a plot comparing the range of initialisation methods cluster plots with 4 runs.
    data: original data DataFrame
    list_init_methods: list of initialisation methods for the Scipy learn kmeans2 method.
    K_n: integer representing the number of clusters i.e. value of k in the kmeans funtion
    """
    keys=[]
    centroids_list=[]
    labels_list=[]
    for i in range(1,5):
        fig, axs = plt.subplots(len(list_init_methods), 2, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
        fig.suptitle('Initialization Method Comparision nr {}'.format(i))

        for index, init_method in enumerate(list_init_methods):
            centroids, labels = kmeans2(data, k=K_n,minit=init_method)
            keys.append("{}:{}".format(i,init_method))
            centroids_list.append(centroids)
            labels_list.append(labels)
            axs[index, 0].plot(data[labels == 0, 0], data[labels == 0, 1], 'ob',
                 data[labels == 1, 0], data[labels == 1, 1], 'or',
                 data[labels == 2, 0], data[labels == 2, 1], 'oy')
            axs[index, 0].plot(centroids[:, 0], centroids[:, 1], 'sg', markersize=5)


            axs[index, 1].plot(data[labels == 0, 2], data[labels == 0, 3], 'ob',
                 data[labels == 1, 2], data[labels == 1, 3], 'or',
                 data[labels == 2, 2], data[labels == 2, 3], 'oy')
            axs[index, 1].plot(centroids[:, 2], centroids[:, 3], 'sg', markersize=5)
            axs[index, 0].set_title("Method:{}".format(init_method), y=0.7)
    return keys, centroids_list, labels_list


def kneigh_fillna(df, X, y):
    """
    This function returns a dataframe with filled na values using KNeighbours.
    df: original data DataFrame
    X: list of the static columns names to be used to predict na values.
    y: string corresponding to the name of the value to fill na values
    """
    # select the other variables to be used in the regressor for prediction and fill in their missing values with respective averages
    # (avg fillna applied because kmeans regressor doesn't work with variables that have missing values)
    tempavg = df.drop(columns=y).fillna(df.drop(columns=y).mean())
    # filter on columns of interest
    reg = pd.concat([tempavg, df[y]], axis=1)

    # splitting the reg dataframe to incomplete table
    # contains the rows of each Nan value of variable to be predicted in reg df
    incomplete = df[reg[y].isna()]

    # splitting the reg dataframe to complete table
    # contains the rows of each valid value of the variable to be predicted in reg df
    complete = df[~reg.index.isin(incomplete.index)]

    # setting the kNeighbours regressor using euclidean distances
    regressor = KNeighborsRegressor(10, weights='distance', metric='euclidean')

    # fitting the regressor to the dataframe using the complete df filling empty values in the dependant variable with the mean
    fitting = regressor.fit(reg[~reg.index.isin(incomplete.index)].loc[:, X], complete.loc[:, [y]])

    # predicting the values using the incomplete df filling empty values in the dependant variable with the mean
    incomplete[y] = fitting.predict(reg[reg[y].isna()].loc[:, X])

    # applying concatenate to the original df
    df = pd.concat([complete, incomplete], axis=0)

    # reset the index
    df.reset_index(drop=True, inplace=True)

    # return the original df with filled values
    return df