from MGB_binary_files import read_MGB_binary_file
from MGB_climatology import get_climatological_values
from MGB_forecast_orgdata import *

import os
import numpy as np
import datetime as dt
# import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.cm import get_cmap
mpl.use('Qt5Agg')  # Interative window

def getDatetimes(initial_date, nDT, fmt='%d-%m-%Y'):
    ini_dt = dt.datetime.strptime(initial_date, fmt)
    return np.asarray([ini_dt + dt.timedelta(days=i) for i in range(nDT)])

def get_MGBSimulation(dir_referenceSim, str_variable, str_iniDate, n_nc, n_dt, str_dateFm):
    mgb_data = read_MGB_binary_file(file_path=os.path.abspath(os.path.join(__file__, dir_referenceSim, str_variable)),
                                    n_unit_catchments=n_nc, n_time_steps=n_dt)
    mgb_datetimes = getDatetimes(str_iniDate, n_dt, str_dateFm)
    return mgb_data, mgb_datetimes

def get_analysisValue(mgb_data, mgb_datetimes, analysis_datetimes):
    idx = list()
    for adate in analysis_datetimes:
        idx.append(np.where(mgb_datetimes == adate)[0][0])
    analysis_value = np.nanmean(mgb_data[idx, :], axis=0)
    return analysis_value

def get_percentile_from_values(value, time_series):

    percentiles_array = np.full(time_series.shape[1], fill_value=np.nan)

    for i in range(time_series.shape[1]):  # Loop mini catchments

        itime_series = time_series[:, i]

        ivalue = value[i]

        # Remove NaN values from the time series
        itime_series = itime_series[~np.isnan(itime_series)]

        # Sort the time series data
        sorted_series = np.sort(itime_series)

        # Find the rank of the given value in the sorted time series.
        rank = np.searchsorted(sorted_series, ivalue) + 1

        # Compute percentil according to weibull distribution
        percentile = 100 * (rank / (len(itime_series) + 1))

        percentiles_array[i] = percentile

    return percentiles_array

def get_plot(analysis_percentile, analysisDatetime, dirShp, str_variable, exportPlot=False):

    ianalysisYear, ianalysisMonth, ianalysisDay = analysisDatetime[0].year, analysisDatetime[0].month, analysisDatetime[0].day
    fanalysisYear, fanalysisMonth, fanalysisDay = analysisDatetime[-1].year, analysisDatetime[-1].month, analysisDatetime[-1].day

    # Figure settings
    fsize = (8, 8)
    colors = ['red', 'orange', 'greenyellow', 'lightskyblue', 'royalblue']
    pltTitle = f'Estado hidrológico {ianalysisDay}/{ianalysisMonth}/{ianalysisYear}-{fanalysisDay}/{fanalysisMonth}/{fanalysisYear} '
    fig, ax = plt.subplots(figsize=fsize)

    # If plot for river, also plot gray background
    if 'QTUDO' in str_variable:
        shp = 'River_network.shp'
        gdf_m = gpd.read_file(os.path.abspath(os.path.join(__file__, dirShp, 'Unit_catchments.shp')))
        for index, row in gdf_m.iterrows():
            gpd.GeoSeries(row['geometry']).plot(ax=plt.gca(), color=(0.7, 0.7, 0.7), edgecolor=None)
    else:
        shp = 'Unit_catchments.shp'

    # Store percentile information into shape features
    gdf = gpd.read_file(os.path.abspath(os.path.join(__file__, dirShp, shp)))
    for i, miniid_value in enumerate(gdf['ID_Mini']):
        gdf.loc[gdf['ID_Mini'] == i + 1, 'Percentile'] = analysis_percentile[i]

    # Plot each feature with defined colors
    for index, row in gdf.iterrows():

        pctile = row['Percentile']

        if pctile <= 10:
            color = colors[0]
        elif 10 < pctile <= 33:
            color = colors[1]
        elif 33 < pctile <= 66:
            color = colors[2]
        elif 66 < pctile <= 90:
            color = colors[3]
        elif pctile > 90:
            color = colors[4]
        else:
            pass

        gpd.GeoSeries(row['geometry']).plot(ax=plt.gca(), color=color, edgecolor=(0.7, 0.7, 0.7, 0.5), linewidth=0.5)

    # Settings for labels
    basePosX = ax.get_xlim()[0] + 0.075
    basePosY = ax.get_ylim()[0] + 0.1
    colWidth = 0.2
    colHeight = 0.3

    for i, color in enumerate(reversed(colors)):
        gradient_color = color  # Gradiente de transparência
        rect = mpatches.Rectangle((basePosX + 0.05, basePosY + 0.85*1.5 - (colHeight * i)), colWidth, colHeight, facecolor=gradient_color, edgecolor='none')
        ax.add_patch(rect)

    for i, label in enumerate(reversed(['Notab. inferior', 'Inferior', 'Normal', 'Superior', 'Notab. Superior'])):
        if len(label) > 9:
            gap = 3.2
        else:
            gap = 2.2
        plt.text(basePosX + 0.05 + colWidth*gap, basePosY + 0.85*1.5 + colHeight/4 - (i * colHeight), label,
                 horizontalalignment='center', verticalalignment='bottom', rotation=0, fontsize=7)

    plt.title(pltTitle)

    if exportPlot:
        plt.savefig(os.path.join(os.path.dirname(__file__), f'{str_variable.strip("_1990-2020.MGB")}_State_{ianalysisDay}{ianalysisMonth}{ianalysisYear}-{fanalysisDay}{fanalysisMonth}{fanalysisYear}.png'), dpi=300)
    else:
        plt.show()

    return

def get_csv(mgb_analysisValue, mgb_analysisClimatology, analysis_percentile, str_variable, analysisDatetime, exportCSV=False):

    if not exportCSV:
        return
    # (MINI_ID, CURRENT VALUE, AVERAGE VALUE, CURRENT_PERCENTILE, TARGET PERCENTILES)
    ianalysisYear, ianalysisMonth, ianalysisDay = analysisDatetime[0].year, analysisDatetime[0].month, analysisDatetime[0].day
    fanalysisYear, fanalysisMonth, fanalysisDay = analysisDatetime[-1].year, analysisDatetime[-1].month, analysisDatetime[-1].day

    with open(os.path.join(os.path.dirname(__file__),
                           f'{str_variable.strip("_1990-2020.MGB")}_State_{ianalysisDay}_{ianalysisMonth}_{ianalysisYear}-{fanalysisDay}_{fanalysisMonth}_{fanalysisYear}.csv'), 'w') as f:

        header = 'MINI_ID, Valor Atual, Média climátologica, Percentil Atual, Pclim10, Pclim33, Pclim66, Pclim90\n'
        f.write(header)
        for i in range(mgb_analysisValue.size):
            row = f'{i + 1}, {mgb_analysisValue[i]}, {np.nanmean(mgb_analysisClimatology[:, i], axis=0)}, {analysis_percentile[i]}, {np.nanpercentile(mgb_analysisClimatology[:, i], 10, axis=0)}, {np.nanpercentile(mgb_analysisClimatology[:, i], 33, axis=0)}, {np.nanpercentile(mgb_analysisClimatology[:, i], 66, axis=0)}, {np.nanpercentile(mgb_analysisClimatology[:, i], 90, axis=0)}\n'
            f.write(row)

    return

if __name__ == '__main__':

    # Define directories (relatives)
    DIR_REFERENCE_SIMULATION = '../../../Reference_simulation'
    DIR_SHAPEFILES = r'../../../Shapefiles'

    # Define parameters from reference simulation
    STR_VARIABLE = 'SOILWATER' + '_1991-2020.MGB'  # EVAPTUDO.MGB, QTUDO.MGB, SOILWATER.MGB, TWSTUDO.MGB, VBAS.MGB, VINT.MGB, YTUDO.MGB
    STR_INITIAL_DATE = '01-01-1991'
    STR_DATE_FMT = '%d-%m-%Y'
    N_NC = 1168
    N_NT = 10958

    ### Define parameters for analysis ###
    ANALYSIS_DATETIME = '01-01-2012'
    AGGREGATE_LT = 31
    EXPORT_PLOT = False  # False para apenas visualizar a figura; True para gravar em disco
    EXPORT_CSV = True  # True para gravar tabela com resultados em disco


    # Main functions
    # Read MGB variable
    mgb_data, mgb_datetimes = get_MGBSimulation(DIR_REFERENCE_SIMULATION, STR_VARIABLE, STR_INITIAL_DATE, N_NC, N_NT, STR_DATE_FMT)

    # Compute period averages
    analysis_datetimes = getDatetimes(ANALYSIS_DATETIME, AGGREGATE_LT, fmt=STR_DATE_FMT)
    mgb_analysisClimatology = get_climatological_values(analysis_datetimes, mgb_data, mgb_datetimes, left_right_windows=0)
    mgb_analysisValue = get_analysisValue(mgb_data, mgb_datetimes, analysis_datetimes)

    # Compute analysis percentile from value and climatology (weibull distribution)
    analysis_percentile = get_percentile_from_values(mgb_analysisValue, mgb_analysisClimatology)

    # Plot map
    get_plot(analysis_percentile, analysis_datetimes, DIR_SHAPEFILES, STR_VARIABLE, exportPlot=EXPORT_PLOT)

    # Save Excel file (MINI_ID, CURRENT VALUE, AVERAGE VALUE, CURRENT_PERCENTILE, TARGET PERCENTILES)
    get_csv(mgb_analysisValue, mgb_analysisClimatology, analysis_percentile, STR_VARIABLE, analysis_datetimes, exportCSV=EXPORT_CSV)


# iNC = 1121 #  Chazuta lembrar do -1
