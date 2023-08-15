"""
Functions to plot binary files of MGB forecast model
@author: Vinícius A. Siqueira, (15/08/2023)
IPH-UFRGS
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


def plot_ensemble_forecast(ensemble_data, ylabel='Forecasted value', title='Ensemble forecast', include_mean=True):
    n_time_steps, n_members = ensemble_data.shape
    
    plt.figure(dpi=300)
    
    # Plot individual ensemble members in gray color
    ensemble_handle = None
    for member in range(n_members):
        if member == 0:
            ensemble_handle, = plt.plot(range(1, n_time_steps + 1), ensemble_data[:, member], color='red', alpha=0.6)
        else:
            plt.plot(range(1, n_time_steps + 1), ensemble_data[:, member], color='red', alpha=0.6)

    # Calculate and plot the ensemble mean in blue color
    if include_mean:
        ensemble_mean = np.mean(ensemble_data, axis=1)
        mean_handle, = plt.plot(range(1, n_time_steps + 1), ensemble_mean, label='Ensemble Mean', color='blue', linewidth=3)

    # Customize the plot
    plt.xlabel('Lead Time')
    plt.ylabel(ylabel)
    plt.title(title)  

    # Add the ensemble members and the ensemble mean to the legend with their respective colors
    handles = [ensemble_handle] if ensemble_handle else []
    labels = ['Ensemble Members'] if ensemble_handle else []

    if include_mean:
        handles.append(mean_handle)
        labels.append('Ensemble Mean')

    plt.legend(handles, labels)

    plt.grid(True, color='lightgray', alpha = 0.5)  # Set the grid color to lightgray

    # Adjust x-axis limits
    plt.xlim(1, n_time_steps)

    # Show the plot
    plt.show()
    
 
    
def plot_ensemble_forecast_boxplot(ensemble_data, ylabel='Forecasted value', title='Ensemble forecast', include_mean=True):
    n_time_steps, n_members = ensemble_data.shape

    # Create an array for boxplot data
    boxplot_data = []
    for time_step in range(n_time_steps):
        boxplot_data.append(ensemble_data[time_step, :])

    # Define a colormap
    cmap = get_cmap('coolwarm')
    
    plt.figure(dpi=300)
    
    # Plot ensemble members as boxplots with varied colors
    boxplot = plt.boxplot(boxplot_data, positions=range(1, n_time_steps + 1), patch_artist=True)

    # Customize the colors of the boxplots
    for patch in boxplot['boxes']:
        patch.set(facecolor=cmap(0.7))

    # Change the color of the median line to red
    for median in boxplot['medians']:
        median.set(color='red')

    # Calculate and plot the ensemble mean in blue color
    if include_mean:
        ensemble_mean = np.mean(ensemble_data, axis=1)
        mean_handle, = plt.plot(range(1, n_time_steps + 1), ensemble_mean, label='Ensemble Mean', color='blue', linewidth=2)

    # Customize the plot
    plt.xlabel('Lead Time')
    plt.ylabel(ylabel)
    plt.title(title)  

    # Add the ensemble mean to the legend
    handles = [mean_handle] if include_mean else []
    labels = ['Ensemble Mean'] if include_mean else []
    plt.legend(handles, labels)

    plt.grid(True, color='lightgray', alpha = 0.5)  # Set the grid color to lightgray

    # Adjust x-axis limits
    plt.xlim(1, n_time_steps)

    # Show the plot
    plt.show()
    

    
def plot_ensemble_forecast_prediction_intervals(ensemble_data, ylabel='Forecasted value', title='Ensemble forecast', include_mean=True):
    n_time_steps, n_members = ensemble_data.shape

    # Calculate the percentiles for the centered prediction intervals
    percentile_50 = np.percentile(ensemble_data, [25, 75], axis=1)
    percentile_95 = np.percentile(ensemble_data, [2.5, 97.5], axis=1)

    plt.figure(dpi=300)

    # Plot the centered prediction intervals
    interval_50 = plt.fill_between(range(1, n_time_steps + 1), percentile_50[0], percentile_50[1], color='gray', alpha=0.4)
    interval_95 = plt.fill_between(range(1, n_time_steps + 1), percentile_95[0], percentile_95[1], color='gray', alpha=0.2)

    # Calculate and plot the ensemble mean in blue color
    if include_mean:
        ensemble_mean = np.mean(ensemble_data, axis=1)
        mean_handle, = plt.plot(range(1, n_time_steps + 1), ensemble_mean, label='Ensemble Mean', color='blue', linewidth=2)

    # Customize the plot
    plt.xlabel('Lead Time')
    plt.ylabel(ylabel)
    plt.title(title)  

    # Add the ensemble mean and prediction intervals to the legend
    handles = [mean_handle] if include_mean else []
    labels = ['Ensemble Mean'] if include_mean else []

    # Add the prediction interval handles and labels to the legend
    handles.extend([interval_50, interval_95])
    labels.extend(['50% Interval', '95% Interval'])

    # Show the legend
    plt.legend(handles, labels)

    plt.grid(True, color='lightgray', alpha = 0.5)  # Set the grid color to lightgray

    # Adjust x-axis limits
    plt.xlim(1, n_time_steps)

    # Show the plot
    plt.show()
    
        
   
def plot_ensemble_forecast_categories(ensemble_data, thresholds, category_colors, boundary_thresholds = None, ylabel='Forecasted value', title='Ensemble forecast', category_legend = [], alpha_cat = 0.7):
    n_time_steps, n_members = ensemble_data.shape

    # Create an array for boxplot data
    boxplot_data = []
    for time_step in range(n_time_steps):
        boxplot_data.append(ensemble_data[time_step, :])

    plt.figure(dpi=300)

    # Plot ensemble members as boxplots with varied colors
    boxplot = plt.boxplot(boxplot_data, positions=range(1, n_time_steps + 1), patch_artist=True, showfliers=False)

    # Customize the colors of the boxplots
    for patch in boxplot['boxes']:
        #patch.set(facecolor=cmap(0.7))
        patch.set(facecolor='gray', alpha = 0.8)

    # Change the color of the median line to red
    for median in boxplot['medians']:
        median.set(color='black')
   
    # Customize the plot
    plt.xlabel('Lead Time')
    plt.ylabel(ylabel)
    plt.title(title)  
    
    # Get the number of thresholds
    n_thresh = thresholds.shape[1]
    
    legend_handles = []
    
    for i in range(0, n_thresh + 1):    
        #Include thresholds in the figure border
        if boundary_thresholds is not None:           
            if i == 0: 
               legend_handles.append(plt.fill_between(range(0, n_time_steps + 2), np.zeros(n_time_steps+2), 
                                 np.concatenate((boundary_thresholds[0,i], thresholds[:,i], boundary_thresholds[1,i]), axis=None), color=category_colors[i], alpha=alpha_cat))
            
            elif i == max(range(0, n_thresh + 1)):                
                legend_handles.append(plt.fill_between(range(0, n_time_steps + 2), np.concatenate((boundary_thresholds[0,i-1], thresholds[:,i-1], boundary_thresholds[1,i-1]), axis=None), 
                                                             np.ones(n_time_steps+2)*np.max(ensemble_data)*1.2, color=category_colors[i], alpha=alpha_cat))    
                
            else:
                legend_handles.append(plt.fill_between(range(0, n_time_steps + 2), np.concatenate((boundary_thresholds[0,i-1], thresholds[:,i-1], boundary_thresholds[1,i-1]), axis=None),
                                                             np.concatenate((boundary_thresholds[0,i], thresholds[:,i], boundary_thresholds[1,i]), axis=None), color=category_colors[i], alpha=alpha_cat))
             
        else:
            if i == 0:                       
                legend_handles.append(plt.fill_between(range(1, n_time_steps + 1), np.zeros(n_time_steps), thresholds[:,i], color=category_colors[i], alpha=alpha_cat))
            elif i == max(range(0, n_thresh + 1)):
                legend_handles.append(plt.fill_between(range(1, n_time_steps + 1), thresholds[:,i-1], np.ones(n_time_steps)*np.max(ensemble_data)*1.2, color=category_colors[i], alpha=alpha_cat))
            else:
                legend_handles.append(plt.fill_between(range(1, n_time_steps + 1), thresholds[:,i-1], thresholds[:,i], color=category_colors[i], alpha=alpha_cat))


    plt.grid(True, color='lightgray', alpha = 0.5)  # Set the grid color to lightgray

    # Adjust x-axis limits
    plt.xlim(0.5, n_time_steps+0.5)
    plt.ylim(np.min(ensemble_data)*0.5, np.max(ensemble_data))

    # Create the legend using the collected handles and labels
    plt.legend(handles=legend_handles, labels=category_legend, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(category_legend))

    # Show the plot
    plt.show()
    
