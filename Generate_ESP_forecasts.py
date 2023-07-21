"""
Code to generate ESP precipitation forecasts for MGB model
@author: Vinícius A. Siqueira
IPH-UFRGS
"""
import numpy as np
from datetime import datetime

 
def generate_esp_forecast(precipitation_data, date_range, target_date, lead_times, resampling_years, cross_validation=True):
    
    # cross_validation = True: will not include the same year of the target date as an ensemble member
    # cross_validation = False: will include the same year of the target date as an ensemble member
    
    # Step 1: Check if the target_date is February 29th and adjust it to February 28th to handle leap years
    target_date_obj = datetime.strptime(target_date, '%Y-%m-%d')
    if target_date_obj.month == 2 and target_date_obj.day == 29:
        target_date_obj = datetime(target_date_obj.year, 2, 28)

    # Step 2: Filter precipitation data for the adjusted target_date and desired years (excluding target year if cross-validation is True)
    filtered_indices = [idx for idx, date_rg in enumerate(date_range) if
                        date_rg.strftime('%m-%d') == target_date_obj.strftime('%m-%d') and (not cross_validation or date_rg.year != target_date_obj.year) and date_rg.year in resampling_years]

    # Step 3: Generate the resampled precipitation sequences starting from the adjusted target_date
    num_time_series = precipitation_data.shape[1]

    # Determine the number of ensemble members based on cross-validation option and target year inclusion in resampling_years
    num_ensemble_members = len(resampling_years) - (1 if cross_validation and target_date_obj.year in resampling_years else 0)

    # Precipitation matrix must be in float32 type
    resampled_precipitation = np.zeros((lead_times, num_time_series, num_ensemble_members)).astype('float32')

    for i, start_index in enumerate(filtered_indices):
        end_index = start_index + lead_times
        resampled_precipitation[:, :, i] = precipitation_data[start_index:end_index]

    return resampled_precipitation












