import mysql.connector
from mysql.connector import Error

HOST = 'audiogeniousdb.cfc53pvygwhf.us-east-2.rds.amazonaws.com'
ID = 'admin'
PASSWD = 'password1!'
DB = 'audiogenious'

# Function to create a new connection
def create_server_connection():
    connection = None
    try:
        connection = mysql.connector.connect(
            host=HOST,
            user=ID,
            passwd=PASSWD,
            database=DB
        )
        print("MySQL Database connection successful")
    except Error as err:
        print(f"Error: '{err}'")
    return connection

def execute_query(query, data):
    connection = create_server_connection()
    cursor = connection.cursor()
    try:
        print(query % (data))
        cursor.execute(query, data)
        connection.commit()
        print("Query successful")
    except Error as err:
        print(f"Error: '{err}'")
    finally:
        cursor.close()
        connection.close()

