import mysql.connector
from mysql.connector import Error

HOST = 'audiogeniousdb.cfc53pvygwhf.us-east-2.rds.amazonaws.com'
ID = 'admin'
PASSWD = 'password1!'
DB = 'audiogenious'

class DBConnection:
    def __init__(self):
        self.connection = None
        self.connect()

    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=HOST,
                user=ID,
                passwd=PASSWD,
                database=DB
            )
            print("MySQL Database connection successful")
        except Error as err:
            print(f"Error: '{err}'")
            self.connection = None  # Ensure connection is set to None if error

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self.connection.close()
            print("Database connection closed.")

def execute_query(query, data):
    with DBConnection() as connection:
        if connection is None:
            print("Failed to connect to the database.")
            return False
        cursor = connection.cursor()
        try:
            cursor.execute(query, data)
            connection.commit()
            print("Query successful")
            return True
        except Error as err:
            print(f"Error: '{err}'")
            return False
        finally:
            cursor.close()
