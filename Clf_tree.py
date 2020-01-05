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

# --------------------- Re-Scale the data ----------------
# # Re-scale df
# scaler.inverse_transform(X=X_std_df)
# df = pd.DataFrame(scaler.inverse_transform(X=X_std_df),
#                   columns=df.columns)
# -------------------------------------------------------


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


#########################################################
# -------------------- K-means --------------------------
#########################################################

#k_max = 8
#elbow_plot(X_std_df,k_max)
#
#init_methods = ['points', '++']
#number_K = 4
#
#keys, centroids_list, labels_list = compare_init_methods(X_std, init_methods,number_K)
#
## pick best kmeans iteration and initialisation method from plots above (please chnage accordingly)
##best_init = 3
#best_method = "points"
#
#centroids_dict = dict(zip(keys, centroids_list))
#labels_dict = dict(zip(keys, labels_list))
#print(" Labels: \n {} \n Centroids: \n {}".format(list(labels_dict[best_method]),
#                                                  centroids_dict[best_method]))

#########################################################
# ---------------- Spectral Clustering ------------------
#########################################################
from sklearn.cluster import SpectralClustering
from mpl_toolkits.mplot3d import Axes3D
# 1) Choosing variables to perform clustering
SC_df = df[['customer_monetary_value', 'claims_rate', 'premium_wage_ratio']]

# 2) Standardize the data frame before performing Clustering
scaler = StandardScaler()
X_std = scaler.fit_transform(SC_df)
X_stdSC_df = pd.DataFrame(X_std, columns=SC_df.columns, index=SC_df.index)

# 3.a) Determine the correct number of clusters using the elbow plot
elbow_plot(X_stdSC_df, 9)   # Let's try 3 clusters

# The Spectral Clustering Method can be developed with 2 types of affinity.
# We will use both types and compare the results to choose the best
# 4.a) Using affinity = ‘rbf’ ( Kernel of the euclidean distanced )
# Building the clustering model using affinity = ‘rbf’
spectral_model_rbf = SpectralClustering(n_clusters=3, affinity='rbf')

# Training the model and Storing the predicted cluster labels
labels_rbf = spectral_model_rbf.fit_predict(X_stdSC_df)

# 5.a) Plotting the results for rbf:
# Building the label to colour mapping
colours = {}
colours[0] = 'b'
colours[1] = 'g'
colours[2] = 'r'
colours[3] = 'c'
colours[4] = 'm'
colours[5] = 'y'
colours[6] = 'b'

# Building the colour vector for each data point
cvec = [colours[label] for label in labels_rbf]

# Plotting the clustered scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xs = [X_stdSC_df['customer_monetary_value']]
ys = [X_stdSC_df['claims_rate']]
zs = [X_stdSC_df['premium_wage_ratio']]

ax.scatter(xs, ys, zs, c=cvec, marker='o')

ax.set_xlabel('CMV')
ax.set_ylabel('Claims Rate')
ax.set_zlabel('PWR')

plt.show()

#-------- Re-Scale the data before plotting -------------
# Re-scale df
scaler.inverse_transform(X=X_stdSC_df)
SC_df = pd.DataFrame(scaler.inverse_transform(X=X_stdSC_df),
                     columns=X_stdSC_df.columns, index=X_stdSC_df.index)

# Add the clusters labels to SC_df
SC_df = pd.DataFrame(pd.concat([SC_df, pd.DataFrame(labels_rbf)], axis=1))
SC_df.columns = ['CMV', 'Claims Rate', 'PWR', 'Labels']
SC_df.tail()        # I think I'm loosing the Customer Identity code around here
SC_df.dropna(inplace=True)

del scaler, X_stdSC_df, X_std


#########################################################
# ------------- Decision Tree Classifier ----------------
#########################################################
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from IPython.display import Image           # Decision Tree Visualization
from sklearn.externals.six import StringIO  # Decision Tree Visualization
from sklearn.tree import export_graphviz    # Decision Tree Visualization
import pydotplus   # Must be installed manually in anaconda prompt with: conda install pydotplus
import pydot
from sklearn import tree
import collections
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


# Define the target variable 'y'
X = SC_df.drop('Labels', axis=1)
y = SC_df[['Labels']]  # The target is the cluster label

# Split up the data into a training set and a test set
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create Decision Tree classifier object
clf = DecisionTreeClassifier()

# Define the parameters conditions for the Decision Tree
# dtree = DecisionTreeClassifier(random_state=0, max_depth=None)
# dtree = DecisionTreeClassifier(criterion='entropy')

# Train the model
clf = clf.fit(X_train, y_train)

# Evaluation of the decision tree results
predictions = clf.predict(X_test)

conf_matrix = confusion_matrix(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)

# Show confusion matrix
conf_matrix

# Show accuracy
accuracy

# Print a classification tree report
print(classification_report(y_test, predictions))

features = list(SC_df.columns[0:3])
features

# Plotting Decision Tree
# Create DOT data
dot_data = tree.export_graphviz(clf, 
                                out_file=None, 
                                feature_names=['class_'+ str(x) for x in np.unique(y)],  
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
# ------------------------------------------------------
