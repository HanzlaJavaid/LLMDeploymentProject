
import sqlite3

# Connect to or create a new database file
conn = sqlite3.connect('feedbackDB.db')

# Create a cursor object to execute SQL commands
cursor = conn.cursor()

# Define a SQL command to create a table
create_table_query = '''
CREATE TABLE IF NOT EXISTS feedback_table1 (
    unix_time DATETIME,
    user_message TEXT NOT NULL,
    ai_message TEXT NOT NULL
);
'''

# Execute the create table command
cursor.execute(create_table_query)

# Commit the changes and close the connection
conn.commit()
conn.close()
