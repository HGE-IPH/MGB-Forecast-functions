"""
Functions to manipulate binary files from MGB forecast model
@author: Vinícius A. Siqueira (19/07/2023)
IPH-UFRGS
"""

import numpy as np
import os
import datetime


# Reads 2D binary files from disk (unit-catchments, timesteps)
def read_MGB_binary_file(file_path, n_unit_catchments, n_time_steps):
    # Load the binary file
    data = np.fromfile(file_path, dtype=np.float32)

    # Reshape the data into the appropriate format
    data_MGB = data.reshape(n_time_steps, n_unit_catchments)

    return data_MGB


# Saves 2D binary files to disk
def save_MGB_binary_file(file_path, data, dtype='float32'):
    
    # convert the data to the appropriate format
    data = data.astype(dtype)
    
    # Write the binary on disk
    with open(file_path, mode='wb') as f:
         data.tofile(f)



# Reads 3D binary files from disk (unit-catchments, time steps, and ensemble members)
def read_MGB_ensemble_binary_file(file_path, n_unit_catchments, n_time_steps, n_members):
   
    #Compute the number of elements in file
    total_elements = n_unit_catchments * n_time_steps * n_members
    
    # Load the binary file
    with open(file_path, 'rb') as file:
        data = np.fromfile(file, dtype=np.float32, count=total_elements)
    
    #Reshape into a 3D array data using the FORTRAN order
    ensemble_data = data.reshape(n_unit_catchments, n_time_steps, n_members, order='F')
    
    return ensemble_data



# Saves ensemble binary files into disk (unit-catchments, time steps, and ensemble members)
def save_MGB_ensemble_binary_file(ensemble_forecast, file_path, write_order = 'F'):
    
    # write_order can be 'F' (Fortran) or 'C' (Matlab/C)
    # Use F for dimension [unit_catchment, time_steps, ens_members]
    # Use C for dimension [time_steps, unit_catchment, ens_members]
    
    # Precipitation matrix must be a numpy array in float32 type
    # Get the ensemble members
    
    num_ensemble_members = ensemble_forecast.shape[2]
   
    # Write each member by appending to file
    with open(file_path, mode='ba+') as f:
   
        for i in range(num_ensemble_members):          
           slice_ensemble_forecast = ensemble_forecast[:,:,i] 
           
           if write_order == 'F': # Save transposed array
               slice_ensemble_forecast.T.tofile(f)
           elif write_order == 'C':
               slice_ensemble_forecast.tofile(f)



# Reads multiple ensemble binary files from disk  (unit-catchments, time steps, and ensemble members)
# Reads a set of catchments of interest and saves into a 4D array (unit-catchments, time steps, ensemble members, forecast dates) 
def read_ensemble_binary_files_sequentially(file_paths, n_unit_catchments, n_time_steps, n_members, selected_catchments, displace_catchment_ID=True):
    
    # Check if the informed mini IDs must be displaced due to index 0 in python 
    if displace_catchment_ID == True: selected_catchments -= 1
    
    forecast_initializations = len(file_paths)
    n_selected_catchments = len(selected_catchments)
    ensemble_data_files = np.zeros((n_selected_catchments, n_time_steps, n_members, forecast_initializations)).astype('float32')

    for i, file_path in enumerate(file_paths):
        file_name = os.path.basename(file_path)  # Extract the filename from the file path
        print(f"Reading file: {file_name}")  # Print the filename only
        data = read_MGB_ensemble_binary_file(file_path, n_unit_catchments, n_time_steps, n_members)
        for j, selected_catchment in enumerate(selected_catchments):            
            ensemble_data_files[j, :, :, i] = data[selected_catchment, :, :]

    return ensemble_data_files



# Reads the hotstart (i.e., Initial condition) files and return some variables of interest
def read_MGB_hotstart_file(file_WBhot_path, file_hydhot_path, n_unit_catchments, n_HRU):
    
    with open(file_WBhot_path, 'rb') as fid_wb, open(file_hydhot_path, 'rb') as fid_inertial:
       
        # Open Water balance hotstart for read
        nStatVar_wb = (n_unit_catchments * n_HRU) + (2 * n_unit_catchments)
        WB_StateVec = np.fromfile(fid_wb, dtype=np.float32, count=nStatVar_wb)
       
        # Open hydraulic hotstart for read
        nStatVar_inertial = 3 * n_unit_catchments #Set the number of variables
        H_StateVec = np.fromfile(fid_inertial, dtype=np.float64, count=nStatVar_inertial)
       
        # Initialize arrays
        W = np.empty((n_unit_catchments, n_HRU)) #Soil moisture
        Vbas = np.empty(n_unit_catchments)      #Groundwater reservoir
        Vint = np.empty(n_unit_catchments)      #Subsurface reservoir
        q = np.empty(n_unit_catchments)         #Discharge
        hfl = np.empty(n_unit_catchments)       #Water depths
        yfl = np.empty(n_unit_catchments)       #Water Surface Elevations
       
        # Reading water balance variables from StateVars_WB
        counter = 0
        for i in range(n_unit_catchments):
            for iU in range(n_HRU):                
                W[i, iU] = WB_StateVec[counter]  # Reading Soil Moisture for each URH
                counter += 1
                
        for i in range(n_unit_catchments):            
            Vbas[i] = WB_StateVec[counter]  # Reading water stored in groundwater reservoir
            counter += 1

        for i in range(n_unit_catchments):
            Vint[i] = WB_StateVec[counter]  # Reading water stored in subsurface reservoir
            counter += 1
            
            # Other variables currently not included (can be read as above):
            # Surface reservoir (Vsup)
            # Previous Temperature (TA), 
            # Muskingum Cunge Upstream catchment flow (QM2), 
            # Muskingum Cunge Downstream catchment flow (QJ2) 
            # Canopy interception volume (SI), must be read for each HRU, as in W
            # Muskingum Cunge initial flow (QRIOINI)
            # Runoff generated in the catchment (QCEL2)
            
            
        # Reading hydrodynamic variables from StateVars_inertial
        counter = 0
        for i in range(n_unit_catchments):            
            q[i] = H_StateVec[counter]  # Streamflow at catchment (iC)
            counter += 1

        for i in range(n_unit_catchments):
            hfl[i] = H_StateVec[counter]  # Flow depth at catchment (iC)
            counter += 1

        for i in range(n_unit_catchments):
            yfl[i] = H_StateVec[counter]  # Water elevation at catchment (iC)
            counter += 1
        
        # Other variables currently not included (can be read as above):
        # Previous Surface Water storage (Vol1)
        # Current Surface Water storage (Vol2)
        # Flooded Area (Area2)
        # Index of Volume level (jtab)
        # Streamflow coming (or leaving) the catchment through connections (Q2face)
        # Updated flow for interconnections (Q2viz)
        
        # Groundwater volume
        Vground = Vbas + Vint

    return W, Vground, q, yfl
  


#Function to get the dates from binary files    
def get_forecast_file_dates(file_paths):    

    vector_dates = []
    
    for file_path in file_paths:

        # Extract the filename with extension using os.path
        file_name_with_extension = os.path.basename(file_path)

        # Extract the filename without extension
        file_name = os.path.splitext(file_name_with_extension)[0]

        # Extract the last 8 characters from the filename
        date_string = file_name[-8:]

        # Convert the date string to a datetime object
        date_object = datetime.datetime.strptime(date_string, "%Y%m%d")

        vector_dates.append(date_object)
   
    return vector_dates 
