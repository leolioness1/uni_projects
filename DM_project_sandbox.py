# Import packages
import sqlite3
import pandas as pd
from datetime import date
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from pandas.util.testing import assert_frame_equal
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns



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
path = "C:\\Users\\Leonor.furtado\\OneDrive - Accenture\\Uni\\Data Mining\\project\\"
excel_df=pd.read_csv(path + "A2Z Insurance.csv")

# Left join the 2 tables on Customer Identity and reset the index
combined_df = pd.merge(engage_df, lob_df, on='Customer Identity', how='left')

# Load original columns in a variable for later use
original_columns = combined_df.columns

# Show a description of the data frame
print(combined_df.describe(include='all'))

# Drop 'index_x' and 'index_y' since they are not useful anymore
combined_df.drop(['index_y', 'index_x'], axis=1, inplace=True)

#Check if the data from the database data source is identical to that of the static csv file provided
try:
    if assert_frame_equal (excel_df,combined_df) is None:
        print("The Dataframes are equal")
    else:
        print("Ups!")

except AssertionError as error:
    outcome = 'There are some differences in the Dataframes: {}'.format(error)

#Make customer Identity the index
combined_df.set_index('Customer Identity', inplace=True)


#clear original dfs to clean the environment
del lob_df, engage_df, excel_df, table_names

#The data is the same so we proceed using the data coming from the database

# # Set simpler columns names to facilitate analysis
combined_df.set_axis( ['policy_creation_year',
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
edu_enc = pd.get_dummies(combined_df['edu_desc'])
edu_enc.head()
edu_values = combined_df.edu_desc.unique()

# Concatenate back to the DataFrame
combined_df = pd.concat([combined_df, edu_enc], axis=1)

# Delete education_lvl columns, since its information is into the two new dummy columns
combined_df = combined_df.drop(['education_lvl','edu_code','edu_desc'], axis=1)

# Checking for missing data using isnull() function & calculating the % of null values per column
# Show the distribution of missing data per column
print('This is the missing data distribution per column (%):\n', round((combined_df.isnull().sum() / len(combined_df))*100, 2))


# Show the percentage of all rows with missing data, no matter which column
print('The sum of percentage of missing data for all rows is: ',
      round((combined_df.isnull().sum() / len(combined_df)).sum()*100, 2), '% \n',
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
      "\nWhich is ", round(((len(combined_df) - len(combined_df.dropna(axis=0, how='any'))) / len(combined_df)) * 100, 2),
      "% of the orignal data.")

# Drop rows with NA values
# making new data frame 'df' with dropped NA values
df = combined_df.dropna(axis=0, how='any')


# Defining each column type value with a  dictionary
type_dict = {
            'policy_creation_year' : int,
            'birth_year' : int,
            'gross_monthly_salary' : float,
            'geographic_area' : int,
            'has_children' : int,
            'customer_monetary_value' : float,
            'claims_rate' : float,
            'motor_premiums' : float,
            'household_premiums' : float,
            'health_premiums' : float,
            'life_premiums' : float,
            'work_premiums' : float,
            'BSc/MSc' : int,
            'Basic' : int,
            'High School' : int,
            'PhD' : int}
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
df = df[df.birth_year != 1028]   # Goodbye Wolverine


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

# customer_monetary_value (CMV), nothing to verify
# claims_rate, nothing to verify
# all premiums, nothing to verify
# all the other columns, (nothing to verify)


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
df.drop(['policy_creation_year','birth_year'], axis=1, inplace=True)

# Calculating and adding 'Customer annual profit' to the data frame
df['cust_annual_prof'] = df['gross_monthly_salary']*12  # Please, let me know if 12 is OK, in Panama is 13

# Calculating the acquisition cost:
df['cust_acq_cost'] = df['cust_annual_prof']*df['cust_pol_age'] - df['customer_monetary_value']

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
df['amt_paidby_comp'] = df['claims_rate']*df['total_premiums']

df.dtypes
df.shape
df.columns
#We are now going to scale the data so we can do effective clusterign of our variables

# Standardize the data to have a mean of ~0 and a variance of 1
X_std = StandardScaler().fit_transform(df)


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
plt.xticks(features)
# Save components to a DataFrame
PCA_components = pd.DataFrame(principalComponents)


# plot scatter plot of pca
plt.scatter(PCA_components[0], PCA_components[1], alpha=.1, color='black')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

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
plt.xticks(ks)
plt.show()

# ### we show a steep dropoff of inertia at k=6 so we can take k=6
# ## Perform cluster analysis on PCA variables

n_clusters = 6

X = PCA_components.values
labels = KMeans(n_clusters, random_state=0).fit_predict(X)

fig, ax = plt.subplots(figsize=(15, 15))
ax.scatter(X[:, 0], X[:, 1], c=labels,
           s=50, cmap='viridis');

txts = df.index.values
for i, txt in enumerate(txts):
    ax.annotate(txt, (X[i, 0], X[i, 1]), fontsize=7)

# # Script to attempt to find good clusters for Data Objects in a datalake arrangement by the number of uses
# Uses both hierarchical dendrograms and K means with PCA to find good clusters

#Hierarchical clustering
# Try different methods for clustering, check documentation:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html?highlight=linkage#scipy.cluster.hierarchy.linkage
# https://www.analyticsvidhya.com/blog/2019/05/beginners-guide-hierarchical-clustering/
# https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering
# Hierarchical clustering, does not require the user to specify the number of clusters.
# Initially, each point is considered as a separate cluster, then it recursively clusters the points together depending upon the distance between them.
# The points are clustered in such a way that the distance between points within a cluster is minimum and distance between the cluster is maximum.
# Commonly used distance measures are Euclidean distance, Manhattan distance or Mahalanobis distance. Unlike k-means clustering, it is "bottom-up" approach.

Z = linkage(PCA_components, 'ward')
Z2 = linkage(PCA_components, 'single')

# Ward variance minimization algorithm

method = "ward"
fig = plt.figure(figsize=(30, 20))
ax = fig.add_subplot(1, 1, 1)
dendrogram(Z,labels=df.index)
ax.tick_params(axis='x', which='major', labelsize=20)
ax.tick_params(axis='y', which='major', labelsize=20)
ax.set_xlabel('Data Object')
fig.savefig('{}_method_dendrogram.png'.format(method))


# ## Nearest Point Algorithm

method = 'single'
fig = plt.figure(figsize=(30, 20))
ax = fig.add_subplot(1, 1, 1)
dendrogram(Z2, ax=ax, labels=df.index, p=0.5, truncate_mode=None, color_threshold=1.25)
ax.tick_params(axis='x', which='major', labelsize=20)
ax.tick_params(axis='y', which='major', labelsize=20)
ax.set_xlabel('Data Object')
fig.savefig('{}_method_dendrogram.png'.format(method))


# ## Try several different clustering heuristics
methods = ['ward', 'single', 'complete', 'average', 'weighted', 'centroid', 'median', ]
fig = plt.figure(figsize=(30, 100))
for n, method in enumerate(methods):
    try:
        Z = linkage(X_std, method, optimal_ordering=True)

        ax = fig.add_subplot(len(methods), 1, n + 1)
        dendrogram(Z, ax=ax, labels=df.index, p=4, truncate_mode=None, color_threshold=0.62 * max(Z[:, 2]))
        ax.tick_params(axis='x', which='major', labelsize=20)
        ax.tick_params(axis='y', which='major', labelsize=20)
        ax.set_xlabel(method)
        fig.savefig('{}_method_dendrogram.png'.format(method))

    except Exception as e:
        print('Error caught:'.format(e))
plt.show()


# ## Experiment with alternative clustering techniques

# variance zero cols must go
corr = df.T.corr()  # Calculate the correlation of the above variables
# sns.heatmap(corr) #Plot the correlation as heat map
sns.set(font_scale=1.0)
cmap = sns.diverging_palette(h_neg=210, h_pos=350, s=90, l=30, as_cmap=True)
sns.clustermap(data=corr, cmap="Blues", annot_kws={"size": 12})

