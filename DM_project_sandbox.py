# Import packages
import sqlite3
import pandas as pd

#set db path
my_path = 'C:\\Users\Leonor.furtado\OneDrive - Accenture\\Uni\Data Mining\project\insurance.db'

# Open connection to DB
conn = sqlite3.connect(my_path)

# Create a cursor
cursor = conn.cursor()

# Execute the query to check what tables exist in the database
cursor.execute('SELECT name from sqlite_master where type= "table"')

# Print the results
print(cursor.fetchall())

#set the queries to import all the data from each table
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

#load the data into dfs
lob_db = pd.read_sql_query(query_lob,conn)
engage_db = pd.read_sql_query(query_engage,conn)

#check the shape of the tables to check number of rows
print(lob_db.shape)
print(engage_db.shape)

#left join the 2 tables on Customer Identity and reset the index
combined_df= pd.merge(engage_db, lob_db, on='Customer Identity', how='left')
combined_df.drop(['index_y','index_x'],axis=1,inplace=True)
combined_df.set_axis(['customer_id','policy_creation_year','birth_year','education_lvl','growth_monthly_salary','geographic_area','has_children','customer_monetary_value','claims_rate','motor_premiums','household_premiums','health_premiums','life_premiums','work_premiums'], axis=1,inplace=True)

#check shape of the table to check if no lost data
print(combined_df.shape)

combined_df.dtypes

# Create a one-hot encoded set of the type values
edu_enc=pd.get_dummies(combined_df['education_lvl'])
edu_enc.head()
edu_values=combined_df.education_lvl.unique()

# Concatenate back to the DataFrame
combined_df = pd.concat([combined_df, edu_enc], axis=1)
combined_df.dtypes
