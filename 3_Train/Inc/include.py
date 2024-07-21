"""
Script: include.py
Desc: Function loadRectangles() loads rectangles from file.
"""

root_directory = '/Users/vojtechremis/Desktop/VŠ/BP/bachelorproject_git/bachelorproject/'

import numpy as np
import ast

# Custom class
import sys
sys.path.append(root_directory+'Inc')
import log

Log = log.log()

def loadRectangles(FilePath):
    try:
        with open(FilePath, 'r') as file:
            lines = file.readlines()
            rectanglesList = [ast.literal_eval(line) for line in lines]
            X_return = np.array(rectanglesList)[:, :4]
            Y_return = np.array(rectanglesList)[:, 4]
            return X_return, Y_return
    except Exception as e:
        Log.error(f'Error when loading rectangles [{FilePath}], error: {e}.')