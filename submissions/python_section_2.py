import pandas as pd
import numpy as np
from datetime import time

def calculate_distance_matrix(df)->pd.DataFrame():
#def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:    
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here

    # Extracting all unique IDs
    unique_locations = pd.Index(sorted(set(df['id_start']).union(set(df['id_end']))))
    
    # Initializing distance matrix with infinity for unknown distances
    distance_matrix = pd.DataFrame(np.inf, index=unique_locations, columns=unique_locations)

    #Setting diagonal to 0 ( for distance to itself)
    np.fill_diagonal(distance_matrix.values, 0)

    #Filling the matrix with the known distances from original DataFrame
    for _, row in df.iterrows():
        id_start, id_end, distance = row['id_start'], row['id_end'], row['distance']
        distance_matrix.loc[id_start, id_end] = distance
        distance_matrix.loc[id_end, id_start] = distance  # Ensure symmetry

    # Using Floyd-Warshall algorithm to finding the shortest paths between all pairs
    n = len(unique_locations)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                distance_matrix.iloc[i, j] = min(distance_matrix.iloc[i, j], 
                                                 distance_matrix.iloc[i, k] + distance_matrix.iloc[k, j])

    # Assigning the distance matrix back to df and return it
    df = distance_matrix
    return df

file_path = 'datasets/dataset-2.csv'
df = pd.read_csv(file_path)
result = calculate_distance_matrix(df)

#--------------------------------Use of calculate_distance_matrix() function-------------------------------#

#print(result)
#----------------------------------------------------------------------------------------------------------#


def unroll_distance_matrix(df)->pd.DataFrame():
#def unroll_distance_matrix(df: pd.DataFrame)->pd.DataFrame:

    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    data = []
    for id_start in df.index:
        for id_end in df.columns:
            if id_start != id_end:
                data.append({
                    'id_start': id_start, 
                    'id_end': id_end, 
                    'distance': df.loc[id_start, id_end]
                })
    df = pd.DataFrame(data)
    return df

unrolled_result = unroll_distance_matrix(result) # Data Frame from Q9

#---------------------------------------Use/Example of Function unroll_distance_matrix()--------------------------#
#print(unrolled_result.head())

#save the unrolled _distance_matrix csv result file
#unrolled_result.to_csv('unrolled_distance_matrix.csv', index=False)
#----------------------------------------------------------------------------------------------------------#


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
#def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> pd.DataFrame:

    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    # average distance calculation for ref_id
    reference_avg = df[df['id_start'] == reference_id]['distance'].mean()

# 10% Threshold Calculation:by lower and upper bounds  calculating as ±10% 
    lower_bound = reference_avg * 0.9
    upper_bound = reference_avg * 1.1

# Filtering the ID's within distance thresold
    avg_distances = df.groupby('id_start')['distance'].mean()
    ids_within_threshold = avg_distances[(avg_distances.between(lower_bound, upper_bound))].index

    df = df[df['id_start'].isin(ids_within_threshold)] # sorted list for the ids within thresold

    return df
#----------------------------------- Use/Example of the funtion 10---------------------------------------------#

# as unrolled_result  is the  dataframe result from Question 10
reference_id = 1001400  # Example reference ID
result_df = find_ids_within_ten_percentage_threshold(unrolled_result, reference_id)

#print(result_df)
#----------------------------------------------------------------------------------------------------------#


def calculate_toll_rate(df)-> pd.DataFrame():
#def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:

    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    # Coefficients for toll rate calculation
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    # Calculate toll rates by multiplying the distance with the their rates
    df['moto'] = df['distance'] * rates['moto']
    df['car'] = df['distance'] * rates['car']
    df['rv'] = df['distance'] * rates['rv']
    df['bus'] = df['distance'] * rates['bus']
    df['truck'] = df['distance'] * rates['truck']
    
    #df = df.drop(columns=['distance'])
    # updated DataFrame
    return df

# the  unrolled_result  is the dataframe from Question 10
toll_rate_df = calculate_toll_rate(unrolled_result)

#---------------------------------------------Use/Example of the Funtion culate_toll_rate()#------------------#
# Print or save the updated DataFrame
# print(toll_rate_df)
# print(toll_rate_df.head())
#toll_rate_df.to_csv('toll_rates.csv', index=False)  # Save to CSV if needed

#-------------------------------------------------------------------------------------------------------#

def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    discount_weekdays = {
        '00:00:00-10:00:00': 0.8,
        '10:00:00-18:00:00': 1.2,
        '18:00:00-23:59:59': 0.8
    }
    discount_weekends = 0.7
    
    # time intervals and days

    time_intervals = [('00:00:00', '10:00:00'), ('10:00:00', '18:00:00'), ('18:00:00', '23:59:59')]
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekends = ['Saturday', 'Sunday']
    
    # list to collect new rows
    new_rows = []
    
    # iterating the  each row in the original DataFrame
    for _, row in df.iterrows():
        base_rates = row[['moto', 'car', 'rv', 'bus', 'truck']].to_dict()

        # Process weekdays
        for day in weekdays:
            for start_time, end_time in time_intervals:
                factor = discount_weekdays[f'{start_time}-{end_time}']
                new_rates = {k: v * factor for k, v in base_rates.items()}
                new_rows.append({
                    'id_start': row['id_start'],
                    'id_end': row['id_end'],
                    'distance': row['distance'],
                    'start_day': day,
                    'start_time': time.fromisoformat(start_time),
                    'end_day': day,
                    'end_time': time.fromisoformat(end_time),
                    **new_rates
                })

        # Iteratinng the  weekends
        for day in weekends:
            for start_time, end_time in time_intervals:
                new_rates = {k: v * discount_weekends for k, v in base_rates.items()}
                new_rows.append({
                    'id_start': row['id_start'],
                    'id_end': row['id_end'],
                    'distance': row['distance'],
                    'start_day': day,
                    'start_time': time.fromisoformat(start_time),
                    'end_day': day,
                    'end_time': time.fromisoformat(end_time),
                    **new_rates
                })

    # Convert list of new rows to a DataFrame
    df = pd.DataFrame(new_rows)
    
    return df

# the toll_rate_df is the dataframe from Question 12
time_based_df = calculate_time_based_toll_rates(toll_rate_df)

#------------------------------------------ Use/exmaple/print of the above function ------------------------#
# Print the first few rows of the new DataFrame
print(time_based_df.head())
#print(time_based_df)
#-------------------------------------------------------------------------------------------------------#