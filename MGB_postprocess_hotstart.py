import numpy as np

# Code to perform spatial averaging for soil moisture (stored in each HRU) and convert Vground from volume to mm
# Work W_state and V_ground_state are variables read from hotstart files 
def postprocess_SM_Vground_states(mini_df, W_state, Vground_state, rescale_HRU_water=True):
    
    #Select columns from mini.gtp dataframe with HRU % information
    cols_HRU=[]
    for col in mini_df.columns:    
        if col[0:3] == 'BLC': cols_HRU.append(col)
    
    nHRU =  len(cols_HRU) 
    nMini = len(mini_df['Mini'])
    
    # Convert to np.array, to facilitate computations
    # Get indvidual catchment areas
    mini_area = np.array(mini_df['Area_(km2)']).astype('float64')
    # Get % of HRUs in each unit-catchment
    urh_perc = np.array(mini_df[cols_HRU]).astype('float64') / 100.0
    
    # When using the inertial routing, % of URH water in FORTRAN goes to zero to avoid duplicating the evaporation from HRU and open water explicitly simulated by the model  
    if rescale_HRU_water == True:
        # Readjusting other HRU proportions for handling the 0% "water" HRU in hydrodynamic module
        urh_perc_rescaled = np.zeros((nMini, nHRU))
        
        for i in range(nMini):
            #Get % of URH water. % of Water is alwyas stored in last HRU
            perc_water = urh_perc[i, -1] 
            
            #Adjust all other % URHs based on current proportions 
            for j in range(nHRU-1):
                urh_perc_rescaled[i, j] = urh_perc[i, j] + (perc_water * urh_perc[i, j]) / (1.0 - perc_water)
        
        #Update urh_percent
        urh_perc = urh_perc_rescaled  
      
    # Aggregate URH soil moisture within unit-catchments
    W_aggregated =  np.sum(urh_perc * W_state, axis=1) #weighted averaging based on % HRU 
    # Transform state Vground from m³ to mm 
    V_subsurface_mm = (0.001 * (Vground_state / mini_area))

    return W_aggregated, V_subsurface_mm


# Compute relative wetness from soil moisture according to the methodology of UK Hydrological Outlook System  
def compute_relative_wetness(S_current, S_mean, S_min, S_max):

    #S storage = Soil moisture (W) or Subsurface Water storage (Vbas + Vint + W) [mm]
    #S_current is the estimated S storage for a given day (typically on the last day of a given month, as in UKHOS methodology) 
    #S mean, min and max are relative to the monthly values   
    
    RW = np.zeros(len(S_current))    

    # Current RW is computed as a % of the maximum positive/negative soil moisture (or subsurface water storage) anomaly
    for i, x in enumerate(S_current):
        
        # Relative Wetness
        if S_current[i] >= S_mean[i]:        
            RW[i] = 100 * (S_current[i] - S_mean[i]) / (S_max[i] - S_mean[i])
        
        # Relative Dryness
        elif S_current[i] < S_mean[i]: 
            RW[i] = 100 * (S_current[i] - S_mean[i]) / (S_mean[i] - S_min[i])
         
    return RW
    
    
    