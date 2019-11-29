# Import packages
import sqlite3
import pandas as pd
import os
from datetime import date
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

# Left join the 2 tables on Customer Identity and reset the index
combined_df = pd.merge(engage_df, lob_df, on='Customer Identity', how='left')

# Load original columns in a variable for later use
original_columns = combined_df.columns

# Show a description of the data frame
print(combined_df.describe(include='all'))

# Drop 'index_x' and 'index_y' since they are not useful anymore
combined_df.drop(['index_y', 'index_x'], axis=1, inplace=True)

# Set simpler columns names to facilitate analysis
combined_df.set_axis(['customer_id',
                      'policy_creation_year',
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

# Delete education_lvl column, since its information is into the two new columns: 'edu_numb', 'edu_desc'
combined_df = combined_df.drop(['education_lvl','edu_code'], axis=1)

# Create a one-hot encoded set of the type values for 'education_lvl'
edu_enc = pd.get_dummies(combined_df['edu_desc'])
edu_enc.head()
edu_values = combined_df.edu_desc.unique()

# Concatenate back to the DataFrame
combined_df = pd.concat([combined_df, edu_enc], axis=1)


# Calculate how old is each customer's first policy
# and create a new column named 'Cust_pol_age'
today_year = int(date.today().year)
combined_df['cust_pol_age'] = today_year - combined_df['policy_creation_year']

# Do the same with 'birth_year column' to get the customers' ages
combined_df['cust_age'] = today_year - combined_df['birth_year']

combined_df.head()

# Checking for missing data using isnull() function & calculating the % of null values per column
null_values_prc = (combined_df.isnull().sum() / len(combined_df))*100

# Show the distribution of missing data per column
print('This is the missing data distribution per column (%):\n', round(null_values_prc, 2))
print('')

# Show the percentage of all rows with missing data, no matter which column
null_values_prc_sum = (combined_df.isnull().sum() / len(combined_df)).sum()

print('The sum of percentage of missing data for all rows is: ',
      round(null_values_prc_sum*100, 2), '% \n',
     'which are ', combined_df.isnull().sum().sum(), 'rows of the total ', len(combined_df), 'rows')

# Assuming there are no more than 1 missing value per row,
# The number of rows with null values is below 5%,
# which is a reasonable amount of rows that can be dropped
# and continue with the 96.6% of the dataframe

# Drop rows with NA values
# making new data frame 'df' with dropped NA values
df = combined_df.dropna(axis=0, how='any')

# comparing sizes of data frames
print("Original data frame length:",
      len(combined_df),
      "\nNew data frame length:",
      len(df),
      "\nNumber of rows with at least 1 NA value: ",
      (len(combined_df) - len(df)),
      "\nWhich is ", round(((len(combined_df) - len(df)) / len(combined_df)) * 100, 2),
      "% of the orignal data.")

# Defining each column type value with a  dictionary
type_dict = {'customer_id': int,
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
            'edu_desc' : str,
            'cust_pol_age' : int,
            'cust_age' : int,
            'BSc/MSc' : int,
            'Basic' : int,
            'High School' : int,
            'PhD' : int}
df.columns
df = df.astype(type_dict)
df.dtypes

# -------------- Caculating additional columns

# With the additional information given,
# it's possible to obtain extra valuable information.

# For 'customer_monetary_value' (CMV), it's possible to clear the given formula:
# CMV = (Customer annual profit)(number of years as customer) - (acquisition cost)
# Therefore:

# (acquisition cost) = (Customer annual profit)(number of years as customer) - CMV

#     where: (Customer annual profit) and (number of years as customer)
# can be calculated prior, as a requirement to get (acquisition cost)

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

# -------------- Business logic validation of the data
# The current year of the database is 2016.
# Check for wrong values on each column, based on the following premises:

# customer_id: there should not be repeated values, only positive values
print("1. Amount of duplicate rows on 'customer_id' column:",
      len(df[df.duplicated(['customer_id'])]),
      '\n')

# policy_creation_year: should not exist values larger than 2016, only positive values
print("2. Are there values greater than 2016 on the 'policy_creation_year'?: ",
      (df['policy_creation_year'] > 2016).any(),
      '\n')

# Count of values by year
df.groupby('policy_creation_year')['customer_id'].count()

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
df.groupby('birth_year')['customer_id'].count()

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
