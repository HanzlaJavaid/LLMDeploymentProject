import sqlite3
import datetime
class DatabaseManager:
    def __init__(self, db_name='feedbackDB.db'):
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()

    def insert_data(self, user_message, ai_message):
        current_datetime = datetime.datetime.now()
        print("Current datetime:", current_datetime)
        insert_query = "INSERT INTO feedback_table1 (unix_time,user_message, ai_message) VALUES (?, ?,?)"
        self.cursor.execute(insert_query, (current_datetime,user_message, ai_message))
        self.conn.commit()

    def fetch_data(self):
        select_query = "SELECT * FROM feedback_table"
        self.cursor.execute(select_query)
        rows = self.cursor.fetchall()
        return rows

    def close_connection(self):
        self.conn.close()


