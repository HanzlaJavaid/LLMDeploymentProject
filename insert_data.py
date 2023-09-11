import sqlite3
from datetime import datetime
import pytz  # This is the timezone module

def insert(HumanMessage, AIMessage):
	# Create a UTC timezone object
	utc_timezone = pytz.timezone('UTC')

	# Get the current time in UTC
	current_utc_datetime = datetime.now(utc_timezone)
	datetime_utc = current_utc_datetime.strftime('%Y-%m-%d %H:%M:%S')


	conn = sqlite3.connect('feedbackDB.db', check_same_thread=False)
	cursor = conn.cursor()


	#def insert(HumanMessage, AIMessage):
	# Insert data
	cursor.execute("INSERT INTO feedback  (timestamp, HumanMessage, AIMessage) VALUES (?,?,?)", (datetime_utc, HumanMessage, AIMessage))


	# Commit the changes and close the connection	
	conn.commit()
	conn.close()
