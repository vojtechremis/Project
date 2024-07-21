"""
Script: database.py

Desc: Functions and Classes for keeping SQL requests simple during project implementation.

"""

root_directory = '/Users/vojtechremis/Desktop/VŠ/BP/bachelorproject_git/bachelorproject/'

import sqlite3
from sqlite3 import Error
import pandas as pd

#Importing shared modules from /VojtaWork/Inc
import sys
sys.path.append(root_directory+'Inc')
import log
Log = log.log("log_database.txt")


# db connection class
class Connection:
    def __init__(self, dbPath):
        try:
            self.con = sqlite3.connect(dbPath)
            Log.info('SQL Database sucessfuly connected w/ dbPath: '+dbPath)
            self.cursor = self.con.cursor()
        except sqlite3.Error:
            Log.error('SQL Database connection error w/ dbPath: '+dbPath)
            self.cursor = None
    
    def getCursor(self):
        if  self.cursor:
            return self.cursor

    def getAllTables(self):
        if self.cursor:
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            response = self.cursor.fetchall()
            return response

    def getTableColumns(self, tableName):
        if self.cursor:
            self.cursor.execute("PRAGMA table_info('"+tableName+"');")
            response = self.cursor.fetchall()
            return response

    def tableToDF(self, SQL):
        if self.cursor:
            try:
                result = pd.read_sql_query(SQL, self.con)
                return result
            except ValueError:
                Log.error('Function tableToDF Error!')
                Log.error(ValueError)
    
    def dataframeToTable(self, tableName, dataframe, append=False):
        if self.cursor:
            if append == True:
                dataframe.to_sql(tableName, self.con, if_exists='append', index = False)
            else:
                dataframe.to_sql(tableName, self.con, if_exists='fail', index = False)

def createDatabase(pathToDB, columns=None):
    con = sqlite3.connect(pathToDB)
    con.commit()