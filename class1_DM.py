# Import packages
import sqlite3
import pandas as pd

#set db path
my_path = 'C:\\Users\Leonor.furtado\Downloads\datamining.db'

# Open connection to DB
conn = sqlite3.connect(my_path)

# Create a cursor
cursor = conn.cursor()

# Execute the query
cursor.execute('SELECT name from sqlite_master where type= "table"')

cursor.execute('SELECT count(distinct(id)) from customers')
cursor.execute('SELECT count(distinct(a.id)),gender from customers a LEFT JOIN genders d ON a.gender_id = d.id group by gender')
query1 = """
SELECT sum(mnt),gender,education FROM customers a 
LEFT JOIN education_levels b 
ON a.education_id = b.id
LEFT JOIN marital_status c 
ON a.marital_status_id = c.id
LEFT JOIN genders d
ON a.gender_id = d.id
LEFT JOIN recommendations e
ON a.recommendation_id = e.id group by gender, education;
"""

query2 = """
SELECT avg(mnt),status FROM customers a 
LEFT JOIN education_levels b 
ON a.education_id = b.id
LEFT JOIN marital_status c 
ON a.marital_status_id = c.id
LEFT JOIN genders d
ON a.gender_id = d.id
LEFT JOIN recommendations e
ON a.recommendation_id = e.id group by status;
"""
df1 = pd.read_sql_query(query1,conn)
df2 = pd.read_sql_query(query2,conn)
# Print the results
print(cursor.fetchall())

#query to join all tables

query = """
SELECT * FROM customers a 
LEFT JOIN education_levels b 
ON a.education_id = b.id
LEFT JOIN marital_status c 
ON a.marital_status_id = c.id
LEFT JOIN genders d
ON a.gender_id = d.id
LEFT JOIN recommendations e
ON a.recommendation_id = e.id ;
"""


#save data to df
df = pd.read_sql_query(query,conn)
df.shape

#get columns
list(df)
df.columns.values.tolist()

#get # rows
df.shape[0]

#get # cols
df.shape[1]