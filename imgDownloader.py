# Import required modules
import requests                   # Retrieve the HTML document for bs4
import sys, os                    # For system operations
import threading                  # Complete 2+ Tasks using the same CPU
import time                       # Display time span of each threaded task
import sqlite3 as sql             # Connect to and populate SQLite databases
import traceback                  # Error viewing after catching
from io import BytesIO            # Convert Bytes object into a TMP file
from PIL import Image             # Resize images before ssaving them

# Static variables
START_TIME = time.time()
WIDTH, HEIGHT = 300, 420
SIZE = (WIDTH, HEIGHT)
BAD_FILENAME_CHARS = [
	'#',	'%',	'&',
	'{',	'}',	'\\',
	'$',	'!',	':',
	';',	'@',	'<',
	'>',	'*',	'?',
	'+',	'=',	'`',
	'|',	"'",	'"',
]
FILES = os.listdir("images/")

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

def main_all(connection, cursor):
	error_list = []
	errors = []
	# execute sqlite3 command
	for link_num, (link, name, bank) in enumerate(cursor.execute(f"SELECT link, name, bank FROM Pokemon")):
		# download file
		filename = f"{link_num}.{name}.{bank}.jpg"
		filename = ''.join(char for char in filename if not char in BAD_FILENAME_CHARS) # remove unwanted chars
		if filename in FILES:
			print(':=:\t', str(link_num).ljust(15)[:15], name.ljust(15)[:15], bank.ljust(15)[:15])
			continue
		response = requests.get(link)
		start_length = len(f"Downloading {filename} ... ")
		print(f"Downloading {filename} ... ", end="", flush=True)
		try:
			Image.open(BytesIO(response.content)).resize(SIZE).convert('RGB').save(f"images/{filename}")
			# follow the robots.txt file for tcgplayer.com // none of the others have a crawl-delay
			if 'tcgplayer.com' in link:
				time.sleep(10)
		except:
			errors.append(traceback.format_exc())
			error_list.append((link, name, bank))
		print(" "*(112-start_length) + "Done") # 112 is constant for "Done" to print neatly
	return errors, error_list

# ready ... set ... GO!
if __name__ == "__main__":
	try:
		#databanks = ["Excrystalguardians", "Teamrocket"]
		connection, cursor = create_connection("card_images.db")
		#main(connection, cursor, databanks)
		errors, error_list = main_all(connection, cursor)
		print("------------------------------------------------------------Error List START")
		print(error_list)
		print("------------------------------------------------------------Error List END")
		print("------------------------------------------------------------Error Traceback START")
		print(errors)
		print("------------------------------------------------------------Error Traceback END")
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
