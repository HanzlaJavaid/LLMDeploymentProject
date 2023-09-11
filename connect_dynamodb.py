import boto3

from datetime import datetime

# Get the current date and time
current_datetime = datetime.now()

# Format the datetime object as a string in the desired format with milliseconds
formatted_timestamp = current_datetime.strftime("%Y%m%d %H:%M:%S:%f")[:-3]
current_date = current_datetime.strftime("%Y%m%d")

# Print the formatted timestamp
#print(current_date)



class DynamoDBManager:
    def __init__(self, table_name='feedback-test', aws_region='eu-central-1'):
        self.table_name = table_name
        self.aws_region = aws_region
        self.dynamodb = boto3.resource('dynamodb', region_name=self.aws_region)
        self.table = self.dynamodb.Table(self.table_name)

    def create_document(self, document):
        try:
            document['current_ds'] = current_date
            document['current_timestamp'] = formatted_timestamp
            response = self.table.put_item(Item=document)
            print("Document created successfully:", response)
            return response
        except Exception as e:
            print("Error creating document:", str(e))

    def fetch_document(self, current_ds):
        try:
            filter_expression = 'current_ds > :target_date'
            expression_attribute_values = {':target_date':current_ds}
            response = self.table.scan(FilterExpression=filter_expression,ExpressionAttributeValues=expression_attribute_values)
            item=response['Items']
            while 'LastEvaluatedKey' in response:
                response=self.table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
                item.extend(response['Items'])
            print(f'The length of data: {len(item)}')
            if item:
                return item
            else:
                print("Document not found")
                return None
        except Exception as e:
            print("Error fetching document:", str(e))


