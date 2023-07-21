"""
Functions to manipulate asc files from MGB forecast model
@author: Vinícius A. Siqueira (19/07/2023)
IPH-UFRGS
"""

import pandas as pd

def read_mini_gtp(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(line.strip().split())
    headers = data.pop(0)  # Remove the header line from the data and store it as headers
    df = pd.DataFrame(data, columns=headers)
    
    return df
