from typing import Dict, List
import json
import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
    i= 0
    while i<len(lst):
        #take out group of n element by iteration from input lst
        group =[]
        for j in range(n):
            if i+j <len(lst):
                group.append(lst[i+j])
        # reverse the group
        reversed_group= []
        for k in range(len(group)):
            reversed_group.insert(0,group[k])   
     # appending reversed group back into the lst
        for item in reversed_group:
            if i < len(lst):
                lst[i]=item
            else: 
                lst.append(item)
            i += 1        
    return lst

#-------------------------Use of the Function 1.-------------------------------------------------------------------#
# print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], n=4))
# print(reverse_by_n_elements([10, 20, 30, 40, 50, 60, 70], n=4))
#-------------------------------------------------------------------------------------------------------#

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
#Your code here
    dict={}
    #grouping the strings by their lengths
    for word_string in lst:
        length = len(word_string)
        if length not in dict:
            dict[length]=[]
        dict[length].append(word_string)
    # sorting dict by keys which is lengths  using dictionary comprehension
    dict = {key:dict[key] for key in sorted(dict)}
    return dict

#------------------------Use of the funtion group_by_length()-----------------------------------------------------#
# print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))
# print(group_by_length(["one", "two", "three", "four"]))
#-------------------------------------------------------------------------------------------------------#

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here
    flattened_dict = {}
    stack = [(nested_dict, '')]  # Using Stack for keeping track of (child, parent_key)

    while stack:
        current_dict, parent_key = stack.pop()

        for key, value in current_dict.items():
            # Creating new key
            new_key = f"{parent_key}{sep}{key}" if parent_key else key

            if isinstance(value, dict):
                # Push dict to stack for and process it later and do current processing of dict
                stack.append((value, new_key))
            elif isinstance(value, list):
                # Handle lists: index each element
                for index, item in enumerate(value):
                    list_key = f"{new_key}[{index}]"
                    if isinstance(item, dict):
                        stack.append((item, list_key))
                    else:
                        flattened_dict[list_key] = item
            else:
                # Assigning the value in the flattened_dict as new_key
                flattened_dict[new_key] = value

    # Converting the flattened dictionary to a  JSON string with double-quoted keys
    return f"json\n{json.dumps(flattened_dict, indent=4)}"

#---------------------------Use of the function: flatten_dict()----------------------------------------------------#
'''
# Example Input:
nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}
# Call the function and print
print(flatten_dict(nested_dict))
'''
#-------------------------------------------------------------------------------------------------------#

from itertools import permutations
def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    permus = set(permutations(nums))

    #converting set back to list
    return [list(p) for p in permus]

#-------------------Use of the function of permutation-------------------------------------------------#
# nums=[1,1,2]
# print(unique_permutations(nums))
#-------------------------------------------------------------------------------------------------------#

import re
def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    # using the regex to find the date pattern : 
    #for dd_mm_yyyy = r'\b\d{2}-\d{2}-\d{4}\b' , for  mm_dd_yyyy = r'\b\d{2}/\d{2}/\d{4}\b', for yyyy_mm_dd= r'\b\d{4}\.\d{2}\.\d{2}\b
    # and combining all of them by using the '|' OR operator
    dates_pattern = r"\b(\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}\.\d{2}\.\d{2})\b"
    
    date_matches= re.findall(dates_pattern,text)

    return date_matches

#---------------------Use of function find_all_dates()--------------------------------------------------------#
# text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
# print(find_all_dates(text))
#-------------------------------------------------------------------------------------------------------#

import pandas as pd
from math import radians, sin, cos, sqrt, atan2
import polyline

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """

    # Decoding the polyline string into a list of (latitude, longitude) tuples
    coordinates = polyline.decode(polyline_str)
    
    #Converting the list of coordinates to a Pandas DataFrame
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    # Initializing the 'distance' column with 0 for the first point
    df['distance'] = 0.0
    
    #Calculatting the Haversine distance  inside the function

    R = 6371000  # Radius of the Earth in meters
    for i in range(1, len(df)):

        #  current and previous coordinates
        lat1, lon1 = df.loc[i - 1, 'latitude'], df.loc[i - 1, 'longitude']
        lat2, lon2 = df.loc[i, 'latitude'], df.loc[i, 'longitude']
        
        # Converting degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Calculate the differences between latitudes and longitudes
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        # Haversine formula
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        
        # Distance in meters
        distance = R * c
        
        # Updates the distance in the DataFrame
        df.loc[i, 'distance'] = distance
    
    return pd.DataFrame({
        'latitude': df['latitude'],
        'longitude': df['longitude'],
        'distance': df['distance']
    })

#-------------------------Use of the function ployline_to_dataframe------------------------------------#
# # Example:
# polyline_str = '_p~iF~ps|U_ulLnnqC_mqNvxq`@'
# df = polyline_to_dataframe(polyline_str)
# print(df)
#-------------------------------------------------------------------------------------------------------#

def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here
    n = len(matrix)

    #Rotate  matrix by 90 degrees clockwise
    rotated_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - i - 1] = matrix[i][j]

# Transforming each element by replacing it with the sum of all elements in its row and column, excluding itself
    transformed_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):            
            # Calculating the sum of elements in the same row and column
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            transformed_matrix[i][j] = row_sum + col_sum

    return transformed_matrix

#---------------------Example use of the function-------------------------------------------------------#
# #Example:
# matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# transformed_matrix = rotate_and_multiply_matrix(matrix)
# print(transformed_matrix)
#-------------------------------------------------------------------------------------------------------#


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here

    # Maping weekday names to numbers (Monday=0, Sunday=6)
    weekday_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    
    # Converting startDay and endDay to numeric day of the week
    df['startDayNum'] = df['startDay'].map(weekday_map)
    df['endDayNum'] = df['endDay'].map(weekday_map)
    
    # Converting times into hours
    df['startHour'] = pd.to_datetime(df['startTime'], format='%H:%M:%S').dt.hour
    df['endHour'] = pd.to_datetime(df['endTime'], format='%H:%M:%S').dt.hour
    
    # to complete Full day and week  requirement
    full_week = set(range(7))   # Monday=0, Sunday=6
    full_day = set(range(24))   # 0=12 AM, 23=11 PM

    def completeness_time_check(group):
        # Create a set of all days covered
        days_covered = set()
        hours_covered = set()

        # For each row in the group, determining the range of days and hours covered
        for _, row in group.iterrows():
            start_day = row['startDayNum']
            end_day = row['endDayNum']
            start_hour = row['startHour']
            end_hour = row['endHour']

            # Adding the range of days to days_covered  in set
            if start_day <= end_day:
                days_covered.update(range(start_day, end_day + 1))
            else:
                days_covered.update(range(start_day, 7))  # Wrap around the week
                days_covered.update(range(0, end_day + 1))

            # Adding the range of hours to hours_covered set for each day
            if start_hour <= end_hour:
                hours_covered.update(range(start_hour, end_hour + 1))
            else:
                hours_covered.update(range(start_hour, 24))
                hours_covered.update(range(0, end_hour + 1))

        # Checking if all 7 days and 24 hours are covered
        return not (days_covered == full_week and hours_covered == full_day)
    
    # so Grouping by id and id_2
    grouped = df.groupby(['id', 'id_2'])

    # Apply the time completeness function 
    completeness_result = grouped.apply(completeness_time_check)

    # Return the result 
    return pd.Series(completeness_result.values, index=completeness_result.index, name="incorrect_timestamps")

#--------------------Use of the function time_check----------------------------------------------------#
# #Example:
# file_path = 'datasets/dataset-1.csv'
# df = pd.read_csv(file_path)
# # Call the function and print the result
# print(time_check(df))
#-------------------------------------------------------------------------------------------------------#