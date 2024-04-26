# import sqlite3

# # Connect to the SQLite database
# try:
#     connection = sqlite3.connect("cars.db")
# except sqlite3.Error as e:
#     print("Error connecting to database:", e)
#     exit()

# # Create a cursor object to execute SQL queries
# cursor = connection.cursor()

# # Execute a query to fetch the column names of the CleanDataset table
# try:
#     cursor.execute("PRAGMA table_info(CleanDataset)")
# except sqlite3.Error as e:
#     print("Error executing query:", e)
#     cursor.close()
#     connection.close()
#     exit()

# # Fetch all the rows (column information)
# column_info = cursor.fetchall()

# # Extract column names from the column information
# column_names = [info[1] for info in column_info]

# if column_names:
#     # Print the column names
#     print("Column names of CleanDataset table:")
#     for name in column_names:
#         print(name)
# else:
#     print("No columns found in CleanDataset table.")

# # Close the cursor and connection
# cursor.close()
# connection.close()


import sqlite3

def get_column_names(database_path, table_name):
    # Connect to the SQLite database
    connection = sqlite3.connect(database_path)

    # Create a cursor object to execute SQL queries
    cursor = connection.cursor()

    # Execute a query to fetch the column names of the specified table
    cursor.execute(f"PRAGMA table_info({table_name})")

    # Fetch all rows from the cursor
    rows = cursor.fetchall()

    # Extract column names from the rows
    column_names = [row[1] for row in rows]

    # Close the cursor and connection
    cursor.close()
    connection.close()

    # Return column names separated by commas
    return ', '.join(column_names)

# Call the function to get column names separated by commas
column_names = get_column_names("cars.db", "first_run_2017_CleanDataset")
print("Column names separated by comma:", column_names)
