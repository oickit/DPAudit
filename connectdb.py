import pymysql
import time

start = time.time()
# database connection
conn = pymysql.connect(
    host="localhost",
    port=3306,
    user="root",
    password="1234",
    charset="utf8mb4"
)

# test the connection
print(conn.get_server_info)

# create a cursor object
cursor = conn.cursor()

# choose database
conn.select_db("test")

# example query
cursor.execute("SELECT COUNT(*) FROM test.person WHERE address=1")
results1 = cursor.fetchall()
for result in results1:
    print(result)

# close cursor and connection
cursor.close()
conn.close()

end = time.time()
print(end-start)