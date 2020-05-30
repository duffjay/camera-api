import json
import numpy as np

def read_app_config(filename):
    with open(filename) as json_file:
        config = json.load(json_file)
    return config


def format_numpy_float(np_array):
    '''
    you can use np.set_printoptions - if you don't care about linefeeds
    if you want the array on one line, then you need a list and this works well

    works with 1 dimensional list only
    '''
    formatted_list = []
    for item in np_array.tolist():
        formatted_list.append("%.2f"%item)
    return formatted_list