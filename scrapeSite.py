# Import required modules
from selenium import webdriver    # Enable Searching after Javascript is enabled
from bs4 import BeautifulSoup     # Helper module for reading data from websites
from bs4 import NavigableString   # Checking for type 'NavigableString'
import requests                   # Retrieve the HTML document for bs4
import sys, os                    # For system operations
import threading                  # Complete 2+ Tasks using the same CPU
import time                       # Display time span of each threaded task
import unicodedata                # Normalize whatever strings are read in
import logging                    # Perform Logging when Threading tasks
import sqlite3 as sql             # Connect to and populate SQLite databases
import urllib.request             # Downloader of Images
import traceback                  # error viewing after catching

# Static variables
START_TIME = time.time()
OPTIONS = webdriver.ChromeOptions()
OPTIONS.add_argument('headless')
OPTIONS.add_argument('log-level=3') # fatal errors only
OPTIONS.add_argument('no-sandbox') # potentially solve tab crash problems
OPTIONS.add_argument('mute-audio') # to resolve issues with MediaEvents
OPTIONS.add_argument('disable-gpu') # potentially solve GpuChannelMsg_CreateCommandBuffer
BROWSER = webdriver.Chrome(options=OPTIONS)
BASE_URL = "https://www.serebii.net"

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

sqlite3_commands = []

def populate_table(serebii_links, operation_name):
    global sqlite3_commands, OPTIONS, BASE_URL
    BROWSER = webdriver.Chrome(options=OPTIONS)
    # get the links of all images and put it in sqlite database
    for bank in range(len(serebii_links)):
        image_bank = serebii_links[bank]
        print(f"{operation_name}: Starting Link ({image_bank.split('/')[-1].title()})")
        BROWSER.get(BASE_URL + image_bank)
        document = BeautifulSoup(BROWSER.page_source, 'html.parser')
        table = document.find("table", class_="dextable")
        if table != None:
            rows = list(list(table.children)[1].children)[1:]
            lastImgNum = 0
            extra_pack = False
            for r in range(len(rows)):
                row = list(list(table.children)[1].children)[1:][r]
                if type(row) != NavigableString:
                    box = row.find_all("td")[2]
                    link = box.contents[0].get("href").split(".")[0].split("/")[-1]
                    if link.isdigit():
                        link = BASE_URL + image_bank + "/" + str(int(link)) + ".jpg"
                    else:
                        link = BASE_URL + image_bank + "/" + link.lower() + ".jpg"
                    pokemon_name = box.get_text()
                    # print(f"Pokemon {r} of {len(rows)}: {pokemon_name.ljust(30)}", end="\r")
                    bank_name = image_bank.split("/")[-1].title()
                    insertion_command = f'insert into Pokemon (link, name, bank) values ("{link}", "{pokemon_name.strip()}", "{bank_name}")'
                    sqlite3_commands.append(insertion_command)
                    # try:
                    #     cursor.execute(insertion_command)
                    #     connection.commit()
                    # except:
                    #     print()
                    #     print(insertion_command)
                    #     traceback.print_exc()
                    #     raise Error("Error")
        bank_name = image_bank.split("/")[-1].title()
        print(f"{operation_name}: Finished Link ({bank_name})")

def main(connection, cursor):
    # need to scrape from both their english and english promo websites
    BROWSER.get(BASE_URL+"/card/engpromo.shtml")
    promo_doc = BeautifulSoup(BROWSER.page_source, 'html.parser')
    BROWSER.get(BASE_URL+"/card/english.shtml")
    english_doc = BeautifulSoup(BROWSER.page_source, 'html.parser')

    # serebii_links will contain all the generations of cards listed in a table
    serebii_links = []

    # first scrape the links from the promo document
    table = promo_doc.find('tbody')
    for row in list(table.children)[1:]:
        if type(row) != NavigableString:
            serebii_links.append(row.contents[1].contents[0].get('href'))

    # scrape the links from the english document
    table = english_doc.find('tbody')
    for row in list(table.children)[1:]:
        if type(row) != NavigableString:
            serebii_links.append(row.contents[1].contents[0].get('href'))

    # serebii_links = serebii_links[-1:]  # for testing purposes
    # serebii_links = ["/card/swshpromos"]
    # serebii_links = serebii_links[86:88]

    # (re)create the Pokemon card database
    cursor.execute("drop table if exists Pokemon")
    command = "create table Pokemon (link text not null primary key, name text not null, bank text not null)"
    cursor.execute(command)

    # split up the data processing with
    t1 = threading.Thread(target=populate_table, args=(serebii_links[:len(serebii_links)//3],"Thread 1",))
    t2 = threading.Thread(target=populate_table, args=(serebii_links[len(serebii_links)//3:2*len(serebii_links)//3],"Thread 2",))
    t3 = threading.Thread(target=populate_table, args=(serebii_links[2*len(serebii_links)//3:],"Thread 3",))

    # start each thread
    t1.start()
    t2.start()
    t3.start()

    # wait for the threads to finish
    t1.join()
    t2.join()
    t3.join()

    print("Completed Scraping. Now Appending Scraped Results to Database")
    for insertion_command in sqlite3_commands:
        try:
            cursor.execute(insertion_command)
            connection.commit()
        except:
            print()
            print(insertion_command)
            traceback.print_exc()
            raise Error("Error")

# ready ... set ... GO!
if __name__ == "__main__":
    try:
        connection, cursor = create_connection("card_images.db")
        main(connection, cursor)
    except:
        print()
        traceback.print_exc()
    finally:
        cursor.close()
        connection.close()
        BROWSER.close()
        BROWSER.quit()
        end_time = time.time()
        print(f"Completed in {(end_time - START_TIME) /    1} seconds")
        print(f"Completed in {(end_time - START_TIME) /   60} minutes")
        print(f"Completed in {(end_time - START_TIME) / 3600} hours")

        print("Please check the links if they are formatted correctly")
