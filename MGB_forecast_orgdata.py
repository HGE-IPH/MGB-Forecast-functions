"""
Functions to manipulate and organize MGB ensemble forecasts
@author: Vinícius A. Siqueira (21/07/2023)
IPH-UFRGS
"""

import numpy as np
import pandas as pd   

#=====================================================================================
#Codes to organize forecast data

def create_ensemble_dataframe(ensemble_data, forecast_dates):
    # Ensure the ensemble_data is a 2D or 3D numpy array
    if not isinstance(ensemble_data, np.ndarray) or ensemble_data.ndim not in (2, 3):
        raise ValueError("ensemble_data must be a 2D or 3D numpy array")

    # If ensemble_data is 2D, add a new axis to simulate a single forecast date
    if ensemble_data.ndim == 2:
        ensemble_data = ensemble_data[:, :, np.newaxis]

    num_lead_times = ensemble_data.shape[0]
    num_members = ensemble_data.shape[1]
    num_forecast_dates = ensemble_data.shape[2]

    # Check if the length of forecast_dates matches num_forecast_dates
    if len(forecast_dates) != num_forecast_dates:
        raise ValueError("Length of forecast_dates must match the number of forecast dates in ensemble_data")

    # Reshape the 3D array to (num_forecast_dates * num_lead_times, num_members)
    reshaped_values = ensemble_data.transpose(2, 0, 1).reshape(-1, num_members)

    # Create a new MultiIndex for the DataFrame 
    forecast_index = pd.MultiIndex.from_product([forecast_dates, range(1, num_lead_times + 1)],
                                                names=['Forecast Date', 'LT'])

    # Create the DataFrame with column names as 'M1', 'M2', 'M3', etc.
    column_names = [f"M{i+1}" for i in range(num_members)]
    forecast_df = pd.DataFrame(reshaped_values, index=forecast_index, columns=column_names)

    return forecast_df



# Function that aggregates ensemble forecasts (of a pandas dataframe) into different time scales
# Observations are only averaged if the "max_nan_percentage_obs" criterion is met
# if max_nan_percentage_obs is set to a value different from None (0 to 1), it assumes that observations are in the first column (before members) 
def average_lead_times_dataframe(ensemble_df, lead_times_to_aggregate, max_nan_percentage_obs=None):
    # Get the unique forecast dates in the DataFrame
    forecast_dates = ensemble_df.index.get_level_values('Forecast Date').unique()

    # Perform averaging for each forecast date separately
    averaged_dfs = []
    updated_lead_times = []
    
    for date in forecast_dates:
        # Use boolean indexing to select all rows with the current forecast date
        df_slice = ensemble_df[ensemble_df.index.get_level_values('Forecast Date') == date]
        num_lead_times = df_slice.index.get_level_values('LT').nunique()

        # Calculate the number of slices based on lead_times_to_aggregate for each forecast date
        num_slices = num_lead_times // lead_times_to_aggregate

        # Create slices for the DataFrame based on lead_times_to_aggregate for each forecast date
        slices = [slice(i * lead_times_to_aggregate, (i + 1) * lead_times_to_aggregate) for i in range(num_slices)]

        # Perform averaging for each slice and append to the list
        averaged_dfs.extend([df_slice.iloc[slice_, :] for slice_ in slices])
        
        # Record the updated lead times for each slice
        updated_lead_times.extend(range(1, len(slices) + 1))
           
    # Check option for maximum nan percentage
    if max_nan_percentage_obs is not None:
        
        for i, obsfor_group in enumerate(averaged_dfs):
        
            # Calculate the number of NaN values in the slice
            sum_nan_values = obsfor_group['Obs'].isnull().sum()
                           
            # Calculate the percentage of NaN values
            percentage_nan_values = sum_nan_values / lead_times_to_aggregate
            
            # Check if the percentage of NaN values satisfies the minimum requirements
            if percentage_nan_values > max_nan_percentage_obs:                          
               # Set Observed in all lead times as nan values. Observed values are set in first row
               averaged_dfs[i].iloc[:,0] = np.nan           
               

    # Concatenate the results for all forecast dates
    averaged_df = pd.concat([df.groupby(['Forecast Date']).mean() for df in averaged_dfs], axis=0)

    # Insert the updated LT column
    averaged_df.insert(0, 'LT', updated_lead_times)

    # Set 'Forecast Date' and 'LT' columns as the MultiIndex
    averaged_df.set_index('LT', append=True, inplace=True)

    return averaged_df


# Pair observed data with forecast dates and lead times
def pair_obs_forecast_LTs(observed_df, ensemble_df): 
    indexes_fcst = ensemble_df.index.get_level_values('Forecast Date')
    indexes_LT = ensemble_df.index.get_level_values('LT')
    observed_values = np.empty((len(indexes_fcst)))
    
    i=0
    # Iterate through each forecast date and lead time and pair the observed value
    for (forecast_date, lead_time), forecast_values in ensemble_df.iterrows():
        # Calculate the corresponding date based on the lead time
        target_date = forecast_date + pd.Timedelta(days=lead_time - 1)
    
        # Extract the observed value for the target date
        observed_values[i] = observed_df.loc[target_date].values
        i+=1
    
    # Create a new DataFrame from the list of paired data
    paired_df = pd.DataFrame({'Forecast Date': indexes_fcst, 'LT': indexes_LT, 'Obs': observed_values})
    
    # Set the forecast dates and lead times as the index to the DataFrame
    paired_df.set_index(['Forecast Date', 'LT'], inplace=True)
    
    return paired_df


# Function to aggregate forecasts across lead times directly in the original matrix
# More efficient and useful for large domains
def aggregate_lead_times_matrix(ensemble_data, n_aggregated_LTs, axis):

   # Validate the axis parameter
    if axis >= ensemble_data.ndim:
        raise ValueError("Invalid axis. The specified axis is out of bounds for the ensemble_forecast array.")

    # Validate the num_days_for_averaging
    if n_aggregated_LTs <= 0 or n_aggregated_LTs > ensemble_data.shape[axis]:
        raise ValueError("Invalid num_days_for_averaging. The value must be positive and not exceed the dimension size.")

    # Calculate the number of segments
    num_LT_segments = ensemble_data.shape[axis] // n_aggregated_LTs

    # List to store the averages of each segment
    LT_segment_averages = []

    # Calculate the average for each segment
    for i in range(num_LT_segments):
        start = i * n_aggregated_LTs
        end = (i + 1) * n_aggregated_LTs
        LT_segment_slice = [slice(None)] * ensemble_data.ndim
        LT_segment_slice[axis] = slice(start, end)
        LT_segment_data = ensemble_data[tuple(LT_segment_slice)]
        LT_segment_average = np.mean(LT_segment_data, axis=axis)
        LT_segment_averages.append(LT_segment_average)

    # Combine the segment averages along the specified axis
    averaged_data = np.stack(LT_segment_averages, axis=axis)

    return averaged_data




