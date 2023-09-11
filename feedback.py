import csv
import os

def feedback_message(question, response):
	data_to_write = [['question','response'], [question,response]]	
	# File path
	file_path = 'feedback.csv'

	# Check if the file exists
	file_exists = os.path.exists(file_path)

	# Open the CSV file in write or append mode
	with open(file_path, 'a', newline='') as file:
		writer = csv.writer(file)
		# If the file doesn't exist, write the header row
		if not file_exists:
			writer.writerow(data_to_write[0])
		# Write or append data rows
		for row in data_to_write[1:]:
			writer.writerow(row)
