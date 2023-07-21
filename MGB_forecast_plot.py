"""
Functions to plot binary files of MGB forecast model
@author: Vinícius A. Siqueira, (19/07/2023)
IPH-UFRGS
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


def plot_ensemble_forecast(ensemble_data, ylabel='Forecasted value', include_mean=True):
    n_time_steps, n_members = ensemble_data.shape
    
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
    plt.title('Ensemble Forecast')  

    # Add the ensemble members and the ensemble mean to the legend with their respective colors
    handles = [ensemble_handle] if ensemble_handle else []
    labels = ['Ensemble Members'] if ensemble_handle else []

    if include_mean:
        handles.append(mean_handle)
        labels.append('Ensemble Mean')

    plt.legend(handles, labels)

    plt.grid(True, color='lightgray')  # Set the grid color to lightgray

    # Adjust x-axis limits
    plt.xlim(1, n_time_steps)

    # Show the plot
    plt.show()
    
 
    
def plot_ensemble_forecast_boxplot(ensemble_data, ylabel='Forecasted value', include_mean=True):
    n_time_steps, n_members = ensemble_data.shape

    # Create an array for boxplot data
    boxplot_data = []
    for time_step in range(n_time_steps):
        boxplot_data.append(ensemble_data[time_step, :])

    # Define a colormap
    cmap = get_cmap('coolwarm')

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
    plt.title('Ensemble Forecast')  # Adjust the catchment index in the title

    # Add the ensemble mean to the legend
    handles = [mean_handle] if include_mean else []
    labels = ['Ensemble Mean'] if include_mean else []
    plt.legend(handles, labels)

    plt.grid(True, color='lightgray')  # Set the grid color to lightgray

    # Adjust x-axis limits
    plt.xlim(1, n_time_steps)

    # Show the plot
    plt.show()
    

    
def plot_ensemble_forecast_prediction_intervals(ensemble_data, ylabel='Forecasted value', include_mean=True):
    n_time_steps, n_members = ensemble_data.shape

    # Calculate the percentiles for the centered prediction intervals
    percentile_50 = np.percentile(ensemble_data, [25, 75], axis=1)
    percentile_95 = np.percentile(ensemble_data, [2.5, 97.5], axis=1)

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
    plt.title('Ensemble Forecast')  # Adjust the catchment index in the title

    # Add the ensemble mean and prediction intervals to the legend
    handles = [mean_handle] if include_mean else []
    labels = ['Ensemble Mean'] if include_mean else []

    # Add the prediction interval handles and labels to the legend
    handles.extend([interval_50, interval_95])
    labels.extend(['50% Interval', '95% Interval'])

    # Show the legend
    plt.legend(handles, labels)

    plt.grid(True, color='lightgray')  # Set the grid color to lightgray

    # Adjust x-axis limits
    plt.xlim(1, n_time_steps)

    # Show the plot
    plt.show()