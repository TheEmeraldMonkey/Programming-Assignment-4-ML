import numpy as np
import math
import matplotlib.pyplot as plt
import pickle 
from sklearn import tree
import pandas as pd

def convert(value):
    try:
        # Try to convert to float
        return int(value)
    except ValueError:
        # If it fails, encode as hex
        return value.encode('utf-8').hex()
def decode_hex(value):
    try:
        # Try to decode assuming it's hex
        return bytes.fromhex(value).decode('utf-8')
    except ValueError:
        # If it fails (it's just a number), leave it as-is
        return value

if __name__ == '__main__':
    # Load the training data
    filename = f'./wcmatches.csv'
    #M = np.genfromtxt('./wcmatches.csv', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    M = np.genfromtxt('./wcmatches.csv', skip_header=0, delimiter=',', dtype=str)


    # Step 3: Apply it across the entire array
    vectorized_convert = np.vectorize(convert)  # Applies function to every element
    converted_data = vectorized_convert(M)

    # Step 2: Vectorize it to apply over the array
    vectorized_decode = np.vectorize(decode_hex, otypes=[str])
    decoded_data = vectorized_decode(converted_data)
    print(decoded_data)
    
    #create filter
    country_list = ['Brazil', 'Germany', 'Italy', 'Argentina', 'France', 'Spain', 'England', 'Uruguay']
    hex_vals = vectorized_convert(country_list)
    print(f'hex vals: {hex_vals}')
    #target_value = 'Uruguay'.encode('utf-8').hex()
    #print(f'target value: {target_value}')

    # Find which rows match
    mask = np.isin(converted_data[:, 1], hex_vals)

    # Apply the mask to get the filtered rows
    filtered_rows = converted_data[mask]
    decoded_data = vectorized_decode(filtered_rows)

    print(decoded_data)

    #M = np.genfromtxt('./FIFA - 2014.csv', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = converted_data[:, 3]
    Xtrn = converted_data[:, 4:]
    
    #print(f'ytest: {ytrn}')
    #print(f'Xtst: {Xtrn}')
        
        