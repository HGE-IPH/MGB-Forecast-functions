"""
Functions to analyze MGB ensemble forecasts
@author: Vinícius A. Siqueira (21/07/2023)
IPH-UFRGS
"""
import numpy as np
from scipy.stats import pearsonr

#=============================================================================================
# Metrics section

def anomaly_correlation(obs, fcst, compute_anomalies_obs = True, compute_anomalies_fcst = True, n_samples_bootstrap = None):
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

    # Check sampling information
    if n_samples_bootstrap == None: n_samples_bootstrap = 1
    acc_bootstrap_samples = np.zeros(n_samples_bootstrap)

    # Perform bootstrap resampling
    for i_bt in range(n_samples_bootstrap):
        # If None, use all values
        if n_samples_bootstrap == 1:
            bootstrap_indices = np.arange(0, len(valid_anomaly_obs))    
        else:
            # Randomly select indices with replacement for bootstrap sample
            bootstrap_indices = np.random.choice(len(valid_anomaly_obs), len(valid_anomaly_obs), replace=True)

        # Compute correlation between anomalies and p-value
        anomaly_correlation, p_value = pearsonr(valid_anomaly_obs[bootstrap_indices], valid_anomaly_fcst[bootstrap_indices])     

        # Store value in the array of bootstrap samples
        acc_bootstrap_samples[i_bt] = anomaly_correlation
    
    return acc_bootstrap_samples



# Heidke Skill Score for categorical forecasts
def compute_HSS(obs, fcst, bmrk, thresholds_obs, thresholds_fcst, thresholds_bmrk, n_samples_bootstrap = None):

    # If fcst has only one ensemble member (deterministic, 1D array), reshape it to a 2D array
    if fcst.ndim == 1:
       fcst = fcst[:, np.newaxis]
    
    # Get number of time_steps and ens members
    n_time_steps, n_members = fcst.shape
     
    # Remove data for indices where observations present NaN values
    valid_indices = np.isnan(obs) == 0
    valid_obs = obs[valid_indices]
    valid_fcst = fcst[valid_indices]
    valid_bmrk = bmrk[valid_indices]
    
    # Check sampling information
    if n_samples_bootstrap == None: n_samples_bootstrap = 1
    hss_bootstrap_samples = np.zeros(n_samples_bootstrap)
    
    # Perform bootstrap resampling
    for i_bt in range(n_samples_bootstrap):
        # If None, use all values
        if n_samples_bootstrap == 1:
            bootstrap_indices = np.arange(0, len(valid_obs))    
        else:
            # Randomly select indices with replacement for bootstrap sample
            bootstrap_indices = np.random.choice(len(valid_obs), len(valid_obs), replace=True)
        
        # Compute proportions for each category
        fcst_probabilities = compute_category_proportions(valid_fcst[bootstrap_indices], thresholds_fcst)
        bmrk_probabilities = compute_category_proportions(valid_bmrk[bootstrap_indices], thresholds_bmrk)
        obs_categories = compute_category_proportions(valid_obs[bootstrap_indices], thresholds_obs)
        
        # Identify the most likely category of forecast -> convert probabilistic to deterministic (yes/no) forecast             
        fcst_categories = identify_dominant_category(fcst_probabilities)
        bmrk_categories = identify_dominant_category(bmrk_probabilities)        
    
        # Compute the number of hits for forecast and benchmark
        fcst_hits = np.nansum(np.sum(fcst_categories * obs_categories, axis=1))   
        bmrk_hits = np.nansum(np.sum(bmrk_categories * obs_categories, axis=1)) 
        total_hits = np.nansum(np.sum(obs_categories, axis=1))
    
        # Compute Heidke Skill Score
        if (total_hits - bmrk_hits) == 0:
            # Avoid division by 0 if eventually benchmark #hits are perfect
            hss = np.nan
        else:
            hss = (fcst_hits - bmrk_hits) / (total_hits - bmrk_hits)
    
        # Store value in the array of bootstrap samples
        hss_bootstrap_samples[i_bt] = hss
    
    return hss_bootstrap_samples
    


# Ranked Probability Skill Score for categorical forecasts
# Using the debiased version from Weigel et al. (2006). The Discrete Brier and Ranked Probability Skill Scores 
def compute_RPSSclim(obs, fcst, thresholds_obs, thresholds_fcst, target_percentiles, debiased_rps_clim = True, n_samples_bootstrap = None):
    
    # If fcst has only one ensemble member (deterministic, 1D array), reshape it to a 2D array
    if fcst.ndim == 1:
       fcst = fcst[:, np.newaxis]
    
    # Get number of time_steps and ens members
    n_time_steps, n_members = fcst.shape
       
    # Compute proportions for each category based on climatological expected values
    clim_probabilities = categorical_climatology_probs_forecast(target_percentiles, n_time_steps)
       
    # If enabled, use Weigel's analytical formula to compute the D term of debiased RPSclim 
    D = 0.0
    if debiased_rps_clim == True:
        clim_probs = clim_probabilities[0]  
    
        # Apply Weigel's formula for adding the debiased term D to RPS clim        
        for k in range(0, len(clim_probs)):
            for i in range(0,k+1):
                sum_probs = 0.0
                for j in range(i+1,k+1):
                    sum_probs += clim_probs[j]      
                
                # D only depends on ensemble size and climatological probabilities    
                D += clim_probs[i] * (1 - clim_probs[i] - 2 * sum_probs)
        
        # Computing final debiasing term
        D = D / n_members
    
    # Remove data for indices where observations present NaN values
    valid_indices = np.isnan(obs) == 0
    valid_obs = obs[valid_indices]
    valid_fcst = fcst[valid_indices]

    # Check sampling information
    if n_samples_bootstrap == None: n_samples_bootstrap = 1
    rpss_bootstrap_samples = np.zeros(n_samples_bootstrap)
    
    # Perform bootstrap resampling
    for i_bt in range(n_samples_bootstrap):
        # If None, use all values
        if n_samples_bootstrap == 1:
            bootstrap_indices = np.arange(0, len(valid_obs))    
        else:
            # Randomly select indices with replacement for bootstrap sample
            bootstrap_indices = np.random.choice(len(valid_obs), len(valid_obs), replace=True)
        
        # Compute proportions for each category
        fcst_probabilities = compute_category_proportions(valid_fcst[bootstrap_indices], thresholds_fcst)
        obs_probabilities = compute_category_proportions(valid_obs[bootstrap_indices], thresholds_obs)
        
        # Calculate the cumulative probability for each category
        acc_fcst_probabilities = np.cumsum(fcst_probabilities,axis=1)        
        acc_obs_probabilities = np.cumsum(obs_probabilities,axis=1)
        acc_clim_probabilities = np.cumsum(clim_probabilities[valid_indices],axis=1)
        
        # Calculate the RPS [forecast and climatology] for each time step. 
        rps_fcst = np.sum((acc_fcst_probabilities - acc_obs_probabilities)**2, axis=1)
        rps_clim = np.sum((acc_clim_probabilities - acc_obs_probabilities)**2, axis=1) 
        
        # Calculate the mean RPS across samples.
        mean_fcst_rps = np.nanmean(rps_fcst)
        mean_clim_rps = np.nanmean(rps_clim)
    
        # Compute RPSS according to Weigel et al. (2006), to account for ensemble size
        rpss = 1 - (mean_fcst_rps / (mean_clim_rps + D))
    
        # Store value in the array of bootstrap samples
        rpss_bootstrap_samples[i_bt] = rpss

    return rpss_bootstrap_samples




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
                
                # Set nan values for categories if data is nan
                elif np.isnan(ensemble_data[time_step, member]) == 1: 
                    category_proportions[time_step,i] = np.nan   
                    break
                
            else:
                category_proportions[time_step, -1] += 1

    category_proportions /= n_members

    return category_proportions



# Function to identify the most likely category based on forecast probabilities
def identify_dominant_category(categorical_probs):
        
    #categories are represented in columns
    #categorical_probs can be for multiple time steps (rows)
  
    #reshape it to a 2D array if necessary
    if categorical_probs.ndim == 1:
        categorical_probs = categorical_probs[np.newaxis,:]  
  
    # Find the indices of categories with the maximum probabilities 
    max_indices = np.argmax(categorical_probs, axis=1)        
   
    # Set 1 for the categories with maximum probabilities and 0 elsewhere
    dom_category = np.zeros_like(categorical_probs)
    dom_category[np.arange(categorical_probs.shape[0]), max_indices] = 1
    
    return dom_category



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
