"""
Functions to analyze MGB ensemble forecasts
@author: Vinícius A. Siqueira (21/07/2023)
IPH-UFRGS
"""
import numpy as np
from scipy.stats import pearsonr

#=============================================================================================
# Metrics section

def anomaly_correlation(obs, fcst, compute_anomalies_obs=True, compute_anomalies_fcst=True):
    """
    Parameters:  
    # obs = numpy array of observations, for a given station
    # fcst = numpy array of forecasts, for a given station and lead time
    """ 
    # Check if anomalies should be computed based on data provided, otherwise values are already given in anomalies
    if compute_anomalies_obs == True:
       # Observed anomalies are computed considering observed climatology
       anomaly_obs = (obs - np.nanmean(obs)) / np.nanstd(obs)
    else: anomaly_obs = obs
    
    if compute_anomalies_fcst == True:
        # Forecast anomalies are computed considering climatology of forecasts
        anomaly_fcst = (fcst - np.nanmean(fcst)) / np.nanstd(fcst)    
    else: anomaly_fcst = fcst    

    # Remove data for indices where observations or forecasts present NaN values
    valid_indices = np.logical_and(np.isnan(anomaly_obs) == 0, np.isnan(anomaly_fcst) == 0)
    valid_anomaly_obs = anomaly_obs[valid_indices]
    valid_anomaly_fcst = anomaly_fcst[valid_indices]

    # Compute correlation between anomalies and p-value
    anomaly_correlation, p_value = pearsonr(valid_anomaly_obs, valid_anomaly_fcst)     

    return anomaly_correlation, p_value



# Ranked probability Score for categorical forecasts
def compute_RPS(obs, fcst, thresholds_obs, thresholds_fcst):
    """
    Calculate the Ranked Probability Score (RPS) for categorical forecasts.
    
    Parameters:
        obs (numpy array): Array of observations for a given station.
        fcst (numpy array): Array of forecasts for a given station and lead time.
                            Columns refer to the ensemble members.
        categories_obs (numpy array): Array of numeric threshold values from observed climatology.
                                      (e.g., below, normal, above)
        categories_fcst (numpy array): Array of numeric threshold values from forecast climatology.
                                       (e.g., below, normal, above)
    
    Returns:
        rps (float): Ranked Probability Score.
    """        
    # If fcst has only one ensemble member (deterministic, 1D array), reshape it to a 2D array
    if fcst.ndim == 1:
       fcst = fcst[:, np.newaxis]
    
    # Get number of time_steps and ens members
    n_time_steps, n_members = fcst.shape
    
    # Compute proportions for each category
    fcst_probabilities = compute_category_proportions(fcst, thresholds_fcst)
    obs_probabilities = compute_category_proportions(obs, thresholds_obs)
    
    # Calculate the cumulative probability for each category
    acc_fcst_probabilities = np.cumsum(fcst_probabilities,axis=1)
    acc_obs_probabilities = np.cumsum(obs_probabilities,axis=1)
    
    # Calculate the RPS for each time step
    rps = np.sum((acc_fcst_probabilities - acc_obs_probabilities)**2, axis=1) / n_time_steps
    
    # Calculate the mean RPS
    mean_fcst_rps = np.nansum(rps)
    
    return mean_fcst_rps
    

# Ranked probability Score for categorical forecasts
def compute_RPS_climatology(obs, thresholds_obs, target_percentiles):
    """
    Calculate the Ranked Probability Score (RPS) for categorical forecasts.
    """
    
    n_time_steps = len(obs)
    
    # Compute proportions for each category
    obs_probabilities = compute_category_proportions(obs, thresholds_obs)
    # Compute proportions for each category based on climatological expected values
    clim_probabilities = categorical_climatology_probs_forecast(target_percentiles, n_time_steps)
    
    # Calculate the cumulative probability for each category
    acc_clim_probabilities = np.cumsum(clim_probabilities,axis=1)
    acc_obs_probabilities = np.cumsum(obs_probabilities,axis=1)
    
    # Calculate the RPS for each time step
    rps = np.sum((acc_clim_probabilities - acc_obs_probabilities)**2, axis=1) / n_time_steps
    
    # Calculate the mean RPS
    mean_clim_rps = np.nansum(rps)
    
    return mean_clim_rps


#=============================================================================================
# Additional functions
# Return the proportions of ensemble/deterministic data inside categories of interest
def compute_category_proportions(ensemble_data, thresholds):

    #reshape it to a 2D array if necessary
    if ensemble_data.ndim == 1:
       ensemble_data = ensemble_data[:, np.newaxis]
    
    n_time_steps, n_members = ensemble_data.shape
    
    category_proportions = np.zeros((n_time_steps, len(thresholds) + 1))

    for time_step in range(n_time_steps):
        for member in range(n_members):
            for i, threshold in enumerate(thresholds):
                if ensemble_data[time_step, member] <= threshold:
                    category_proportions[time_step, i] += 1
                    break
            else:
                category_proportions[time_step, -1] += 1

    category_proportions /= n_members

    return category_proportions


# Function to create climatological probabilities for given categories     
def categorical_climatology_probs_forecast(thresholds, num_forecasts):
    
    # Set limits based on informed thresholds
    prob_clim = np.zeros(len(thresholds)+1)
    prob_clim[0] = thresholds[0]
    prob_clim[-1] = 100 - thresholds[-1]

    # The remaining climatological probabilities are computed from differences
    diff_thresholds = np.diff(thresholds)

    for k, dp in enumerate(diff_thresholds):
        prob_clim[k+1] = dp

    prob_clim = prob_clim / 100
    prob_clim = np.tile(prob_clim,(num_forecasts,1))
     
    return prob_clim   