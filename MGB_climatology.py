"""
Functions to produce climatologies
@authors: Erik Quedi, Vinícius A. Siqueira 
IPH-UFRGS
"""

import numpy as np
import pandas as pd
import datetime

# Function that returns the climatology, considering the average of values across target date_times
def get_climatological_values(target_datetimes, values, vector_dates, left_right_windows=0):
	
    '''
    # values can be a 1D or 2D numpy matrix
    # left_right_windows is the number of periods that should be added at both sides of target_dates, to increase samples
    # important: the size of each period will be equal to target_datetimes size
    '''
    
    #Set option to perform average in values as default
    perform_average=True
    
    # If there is a single date as target, do not compute average 
    if target_datetimes.size == 1: perform_average=False
    
    # Check the number of dimensions. If only one, adds a virtual axis to avoid problems 
    if values.ndim == 1:
       values = values[:, np.newaxis]
    
    # Get the number of points
    nPoints = values.shape[1]
    
    idate = target_datetimes[0]  # Initial date in the window of interest
    if idate.day == 29 and idate.month == 2: idate = idate + pd.DateOffset(days=1) #Avoid problems with leap years
    
    tmpDays = np.asarray([i.day for i in vector_dates])
    tmpMonth = np.asarray([i.month for i in vector_dates])
    tmpYears = np.asarray([i.year for i in vector_dates])
    tmpUniqueYears = np.unique(tmpYears)
    
    # Compute the size of windows 
    tdate_len = len(target_datetimes)
    side_offset = tdate_len * left_right_windows 
    n_windows = (left_right_windows * 2) + 1
    
    climAvgData = np.empty((tmpUniqueYears.size * n_windows, nPoints))
    
    #Initialize vector as nan to handle potential missing values due to incomplete windows
    climAvgData[:] = np.nan
    
    counter=0 #counter initialization
    for i in range(tmpUniqueYears.size):
		
        iyear = tmpUniqueYears[i]
        leftIdx = np.where(np.logical_and(np.logical_and(tmpDays == idate.day, tmpMonth == idate.month), tmpYears == iyear))[0][0]

        # Compute limits for the full window
        leftWindowIdx = leftIdx - side_offset
        rightWindowIdx = leftIdx + (tdate_len - 1) + side_offset
        
        # Check limits, skip the current year if it has lesser data than requested
        if np.logical_or(leftWindowIdx < 0, rightWindowIdx >= len(vector_dates)): continue
    
        # If perform_average option is enabled, loop through window slices to perform moving average
        if perform_average==True:
            for wd in range(n_windows):
            
                # Get slices representing each subset window
                slice_window = np.arange(leftWindowIdx + tdate_len * wd, leftWindowIdx + tdate_len * (wd + 1))
                
                # Perform averaging
                climAvgData[counter] = np.nanmean(values[slice_window,:], axis=0)                
                counter += 1 # Increase counter
        
        else:  # If perform_average option is disabled (single date), return the full window of values
       
            # get start and end positions to store values
            start_pos = counter
            end_pos = counter + n_windows - 1
        
            # Get climatological values from input data and increase counter
            climAvgData[start_pos:end_pos + 1] = values[leftWindowIdx:rightWindowIdx + 1,:]            
            counter += n_windows
  
    return climAvgData



# Function that returns a dataframe with climatology values (for unique calendar days of the year) from a set of forecast dates 
def get_df_moving_climatology_forecast(forecast_df, aggregated_LTs, source_values, source_dates, left_right_windows):
    
    # Get indexes from hindcast dataframe
    fcst_dates = forecast_df.index.get_level_values('Forecast Date').unique()
    fcst_LTs = forecast_df.index.get_level_values('LT').unique()
    
    # Set arbitrary year for calendar dates
    temp_year = fcst_dates.year[-1]
    
    # Create unique pairs of month and day
    unique_pairs = set()
    for dt in fcst_dates:
            day_month_pair = (dt.day, dt.month)
            unique_pairs.add(day_month_pair)
    
    # Create datetimes with the previous calendar dates
    datetime_set = []
    for day, month in unique_pairs:
            dt = datetime.datetime(temp_year, month, day)
            datetime_set.append(dt)
    
    # Sort datetimes
    datetime_set.sort()
    datetime_set = np.array(datetime_set)
    
    # Obtain climatologies for each calendar date and lead time
    climatology_list = []
    LT_list = []
    new_date_list = []
    for dt in datetime_set:
        for lt in fcst_LTs:
            dt2 = dt + pd.DateOffset(days = aggregated_LTs * (lt-1))
            clim_val = get_climatological_values(pd.date_range(dt2, periods = aggregated_LTs), source_values, source_dates, left_right_windows=left_right_windows)
            climatology_list.append(clim_val.T)
            new_date_list.append(dt)
            LT_list.append(lt)
      
    # Join climatologies into a single dataframe
    clim_df = pd.concat([pd.DataFrame(clim) for clim in climatology_list], axis = 0)       
    # Reset the index to the default sequential integer index, to allow blending to the forecast dates and LTs
    clim_df = clim_df.reset_index(drop=True)
    # concatenate dataframes of forecast dates-LT with climatology values
    clim_df = pd.concat([pd.DataFrame({'Calendar Date': np.array(new_date_list), 'LT': np.array(LT_list).astype('int32')}), clim_df], axis = 1)
    # Set index of dataframe as multi index
    clim_df.set_index(['Calendar Date', 'LT'], inplace=True)
    
    return clim_df
    



# Function to get percentiles from a dataframe of climatologies (with unique calendar dates)
def compute_percentiles_from_climatology_df(climatology_df, percentiles, method='linear'):

    calendar_dts = climatology_df.index.get_level_values('Calendar Date')
    calendar_LTs = climatology_df.index.get_level_values('LT')

    # Initialize percentile array
    prct = np.zeros((len(climatology_df.index),len(percentiles)))    
    
    # Loop through rows and compute percentiles
    for i, (dts, row) in enumerate(climatology_df.iterrows()):
        row_val = row.values
        # remove nan values from series
        row_val = row_val[np.isnan(row_val)==0]
        # compute percentiles
        prct[i] = np.percentile(row_val, percentiles, method='linear')
        
    # Create a new dataframe with values
    prct_clim_df = pd.concat([pd.DataFrame({'Calendar Date': calendar_dts, 'LT': calendar_LTs}), pd.DataFrame(prct, columns=percentiles)], axis=1)

    # Set index
    prct_clim_df.set_index(['Calendar Date', 'LT'], inplace=True)

    return prct_clim_df




# Compute the corresponding percentile of a given value (or values) retrieved from time series
def get_percentile_from_values(value, time_series):
    
    # Remove NaN values from the time series
    time_series = time_series[~np.isnan(time_series)]
    
    # Sort the time series data
    sorted_series = np.sort(time_series)

    # Find the rank of the given value in the sorted time series.  
    rank = np.searchsorted(sorted_series, value) + 1

    # Compute percentil according to weibull distribution
    percentile = 100 * (rank / (len(time_series) + 1))

    return percentile    
    