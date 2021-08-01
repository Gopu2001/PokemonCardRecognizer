# Import required modules
import requests                   # Retrieve the HTML document for bs4
import sys, os                    # For system operations
import threading                  # Complete 2+ Tasks using the same CPU
import time                       # Display time span of each threaded task
import sqlite3 as sql             # Connect to and populate SQLite databases
import traceback                  # error viewing after catching

# Static variables
START_TIME = time.time()

# Helper for Creating a SQLite Connection (Later)
def create_connection(db_file):
	conn = None
	try:
		conn = sql.connect(db_file)
		curs = conn.cursor()
	except sql.Error as e:
		print(e)
		if os.path.exists(db_file):
			curs.close()
			conn.close()
		sys.exit(1)

	return conn, curs

def main(connection, cursor, databanks):
	# format and execute sqlite3 command
	banks = f"""('{"', '".join(databanks)}')"""
	for link_num, (link, name, bank) in enumerate(cursor.execute(f"SELECT link, name, bank FROM Pokemon WHERE bank in {banks}")):
		# download file
		response = requests.get(link)
		start_length = len(f"Downloading {name+str(link_num)}.jpg ... ")
		print(f"Downloading {name+str(link_num)}.jpg ... ", end="", flush=True)
		with open(f"images/{name+str(link_num)}.jpg", "wb") as f:
			f.write(response.content)
		print(" "*(60-start_length) + "Done")

# ready ... set ... GO!
if __name__ == "__main__":
	try:
		databanks = ["Excrystalguardians", "Teamrocket"]
		connection, cursor = create_connection("card_images.db")
		main(connection, cursor, databanks)
	except:
		print()
		traceback.print_exc()
	finally:
		print()
		cursor.close()
		connection.close()
		end_time = time.time()
		print(f"Completed in {(end_time - START_TIME) /    1} seconds")
		print(f"Completed in {(end_time - START_TIME) /   60} minutes")
		print(f"Completed in {(end_time - START_TIME) / 3600} hours")
