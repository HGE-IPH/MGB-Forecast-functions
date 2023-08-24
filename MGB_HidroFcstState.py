from Auxiliares.MGB_previsao_Huallaga.Scripts.Examples.MGB_binary_files import *
from Auxiliares.MGB_previsao_Huallaga.Scripts.Examples.MGB_climatology import *
from MGB_forecast_analysis import *
from Auxiliares.MGB_previsao_Huallaga.Scripts.Examples.MGB_forecast_orgdata import *

import os
import numpy as np
import datetime as dt
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
mpl.use('Qt5Agg')  # Interative window

def getDatetimes(initial_date, nDT, fmt='%d-%m-%Y'):
    ini_dt = dt.datetime.strptime(initial_date, fmt)
    return np.asarray([ini_dt + dt.timedelta(days=i) for i in range(nDT)])

def get_MGBSimulation(dir_referenceSim, str_variable, str_iniDate, n_nc, n_dt, str_dateFm):
    mgb_data = read_MGB_binary_file(file_path=os.path.abspath(os.path.join(__file__, dir_referenceSim, str_variable)),
                                    n_unit_catchments=n_nc, n_time_steps=n_dt)
    mgb_datetimes = getDatetimes(str_iniDate, n_dt, str_dateFm)
    return mgb_data, mgb_datetimes

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

def get_fcstPercentiles(fcst_data, mgb_analysisClimatology):
    fcstPercentiles = list()
    for iNC in range(fcst_data.shape[0]):
        iNC_climValues = mgb_analysisClimatology[:, iNC]
        iNC_tresholds = np.nanpercentile(iNC_climValues, [10, 33, 66, 90])
        iNC_iLT_fcst_data = fcst_data[iNC, :]
        iNC_iLT_fcst_data = iNC_iLT_fcst_data[np.newaxis, :]
        iNC_iLT_catProb = compute_category_proportions(iNC_iLT_fcst_data, iNC_tresholds)
        iNC_iLT_domProb = identify_dominant_category(iNC_iLT_catProb)
        fcstPercentiles.append(iNC_iLT_catProb)
    return fcstPercentiles

def get_fcstPlot(analysis_percentile, analysisDatetime, dirShp, str_variable, AGGREGATE_LT, ANALYSIS_FCST_LT, exportPlot=False):

    def getColorAlpha(domProb):
        if domProb <= 0.3:
            alpha = 0.3
        elif 0.3 < domProb <= 0.4:
            alpha = 0.4
        elif 0.4 < domProb <= 0.5:
            alpha = 0.5
        elif 0.5 < domProb <= 0.6:
            alpha = 0.6
        elif 0.6 < domProb <= 0.7:
            alpha = 0.7
        elif 0.7 < domProb <= 0.8:
            alpha = 0.8
        elif 0.8 < domProb <= 0.9:
            alpha = 0.9
        elif 0.9 < domProb <= 1.0:
            alpha = 1.0
        return alpha

    ianalysisYear, ianalysisMonth, ianalysisDay = analysisDatetime[0].year, analysisDatetime[0].month, analysisDatetime[0].day
    fanalysisYear, fanalysisMonth, fanalysisDay = analysisDatetime[-1].year, analysisDatetime[-1].month, analysisDatetime[-1].day

    # Figure settings
    fsize = (8, 8)
    colors = ['red', 'orange', 'greenyellow', 'lightskyblue', 'royalblue']
    pltTitle = f'Estado hidrológico {ianalysisDay}/{ianalysisMonth}/{ianalysisYear}-{fanalysisDay}/{fanalysisMonth}/{fanalysisYear}\n' \
               f'Agregado para {AGGREGATE_LT} dias (Lead {ANALYSIS_FCST_LT + 1})'
    fig, ax = plt.subplots(figsize=fsize)

    # If plot for river, also plot gray background
    if 'QTUDO' in str_variable:
        shp = 'River_network.shp'
        gdf_m = gpd.read_file(os.path.abspath(os.path.join(__file__, dirShp, 'Unit_catchments_WGS84.shp')))
        for index, row in gdf_m.iterrows():
            gpd.GeoSeries(row['geometry']).plot(ax=plt.gca(), color=(0.5, 0.5, 0.5), edgecolor=None)
    else:
        shp = 'Unit_catchments_WGS84.shp'

    # Store percentile information into shape features
    gdf = gpd.read_file(os.path.abspath(os.path.join(__file__, dirShp, shp)))
    for i, miniid_value in enumerate(gdf['ID_Mini']):
        domIndex = np.argmax(analysis_percentile[i])
        gdf.loc[gdf['ID_Mini'] == i + 1, 'domIndex'] = domIndex
        gdf.loc[gdf['ID_Mini'] == i + 1, 'Percentile'] = analysis_percentile[i].squeeze()[domIndex]

    # Plot each feature with defined colors
    for index, row in gdf.iterrows():

        pctile = row['Percentile']
        domIndex = row['domIndex']

        if domIndex == 0:
            color = colors[0]
        elif domIndex == 1:
            color = colors[1]
        elif domIndex == 2:
            color = colors[2]
        elif domIndex == 3:
            color = colors[3]
        elif domIndex == 4:
            color = colors[3]
        else:
            pass

        gpd.GeoSeries(row['geometry']).plot(ax=plt.gca(), color=color, edgecolor=(0.7, 0.7, 0.7, 0.5), linewidth=0.8, alpha=getColorAlpha(pctile))

    # Settings for labels
    basePosX = ax.get_xlim()[0] + 0.1 # 0.075
    basePosY = ax.get_ylim()[0] + 0.8
    colWidth = 0.1
    colHeight = 0.2
    ax.set_xlim(ax.get_xlim()[0] - 0.2, ax.get_xlim()[1] + 0.2)

    for i, color in enumerate(reversed(colors)):
        for j in range(8):
            rect = mpatches.Rectangle((basePosX + 0.05 + (i * colWidth), basePosY + 0.85 - colHeight * j), colWidth,
                                      colHeight, facecolor=color, edgecolor='none', alpha=1 - j / 10)
            ax.add_patch(rect)

    # Col labels
    for i, label in enumerate(reversed(['Notab. inferior', 'Inferior', 'Normal', 'Superior', 'Notab. Superior'])):
        plt.text((basePosX + colWidth) + (i * colWidth), basePosY + colHeight + 0.90, label,
                 horizontalalignment='center', verticalalignment='bottom', rotation=90, fontsize=7)

    # Row labels
    for j in range(8):
        plt.text(basePosX + 0.05 - (colWidth / 6), basePosY + (0.85 + colHeight / 2) - colHeight * j,
                 f'{100 - j * 10}%', horizontalalignment='right', verticalalignment='center', fontsize=7)

    plt.title(pltTitle)

    if exportPlot:
        plt.savefig(os.path.join(os.path.dirname(__file__), f'{str_variable.strip("_1990-2020.MGB")}_FcstState_{ianalysisDay}{ianalysisMonth}{ianalysisYear}-{fanalysisDay}{fanalysisMonth}{fanalysisYear}_Agg{AGGREGATE_LT}_LT{ANALYSIS_FCST_LT + 1}.png'), dpi=300)
    else:
        plt.show()

    return

def get_csv(mgb_analysisClimatology, analysis_percentile, str_variable, analysisDatetime, AGGREGATE_LT, ANALYSIS_FCST_LT, exportCSV=False):

    if not exportCSV:
        return
    # (MINI_ID, CURRENT VALUE, AVERAGE VALUE, CURRENT_PERCENTILE, TARGET PERCENTILES)
    ianalysisYear, ianalysisMonth, ianalysisDay = analysisDatetime[0].year, analysisDatetime[0].month, analysisDatetime[0].day
    fanalysisYear, fanalysisMonth, fanalysisDay = analysisDatetime[-1].year, analysisDatetime[-1].month, analysisDatetime[-1].day

    with open(os.path.join(os.path.dirname(__file__),
                           f'{str_variable.strip("_1990-2020.MGB")}_State_{ianalysisDay}_{ianalysisMonth}_{ianalysisYear}-{fanalysisDay}_{fanalysisMonth}_{fanalysisYear}_AGG{AGGREGATE_LT}_LT{ANALYSIS_FCST_LT + 1}.csv'), 'w') as f:

        header = 'MINI_ID, Média climátologica, Pclim10, Pclim33, Pclim66, Pclim90,' \
                 'Ens%_P10, Ens%_P33, Ens%_P66, Ens%_P90, Eens%_P99\n'
        f.write(header)
        for i in range(len(analysis_percentile)):
            row = f'{i + 1}, {np.nanmean(mgb_analysisClimatology[:, i], axis=0)}, {np.nanpercentile(mgb_analysisClimatology[:, i], 10, axis=0)}, {np.nanpercentile(mgb_analysisClimatology[:, i], 33, axis=0)}, {np.nanpercentile(mgb_analysisClimatology[:, i], 66, axis=0)}, {np.nanpercentile(mgb_analysisClimatology[:, i], 90, axis=0)},' \
                  f'{analysis_percentile[i][0][0]}, {analysis_percentile[i][0][1]}, {analysis_percentile[i][0][2]}, {analysis_percentile[i][0][3]}, {analysis_percentile[i][0][4]}\n'
            f.write(row)

    return


if __name__ == '__main__':

    # TODO: Resolver titulos das figuras e informações que serão incluidas. Também traduzir para espanhol
    # TODO: Resolver nomenclatura dos arquivos de saída PNG e CSV

    # Define directories (relatives)
    DIR_REFERENCE_SIMULATION = r'../../../Reference_simulation'
    DIR_SHAPEFILES = r'../../../Shapefiles'
    DIR_FCST = r'../../../Q_forecast'

    # Define parameters from reference simulation
    STR_VARIABLE = 'SOILWATER' + '_1991-2020.MGB'  # EVAPTUDO.MGB, QTUDO.MGB, SOILWATER.MGB, TWSTUDO.MGB, VBAS.MGB, VINT.MGB, YTUDO.MGB
    STR_INITIAL_DATE = '01-01-1991'
    STR_DATE_FMT = '%d-%m-%Y'
    N_NC = 1168
    N_NT = 10958

    # Define parameters for forecasts
    N_NT_FCST = 90
    N_ENS_FCST = 38

    ### Define parameters for analysis ###
    ANALYSIS_FCST_FILE_INDEX = 0        # Seleciona o arquivo de previsão a ser analisado (Atenção a variável referente ao arquivo de previsão, deve ser a mesma da climatologia do MGB)
    AGGREGATE_LT = 1                    # Numero para agregação dos lead-times ou None
    ANALYSIS_FCST_LT = 0                # Seleciona o lead-time alvo para analise (Indexação de python, iniciada em 0)
    EXPORT_PLOT = True                 # False para apenas visualizar a figura; True para gravar em disco
    EXPORT_CSV = True                  # True para gravar tabela com resultados em disco

    # Main functions
    # Read MGB variable
    mgb_data, mgb_datetimes = get_MGBSimulation(DIR_REFERENCE_SIMULATION, STR_VARIABLE, STR_INITIAL_DATE, N_NC, N_NT, STR_DATE_FMT)

    # Get forecasts information
    if AGGREGATE_LT < 1:  # Safeguard to assert one analysis datetime
        AGGREGATE_LT = 1
    fcst_files = os.listdir(os.path.abspath(os.path.join(__file__, DIR_FCST)))
    analysis_datetime = dt.datetime.strptime(fcst_files[ANALYSIS_FCST_FILE_INDEX].split('_')[-1].strip('.MGB'), '%Y%m%d') + dt.timedelta(days=AGGREGATE_LT*(ANALYSIS_FCST_LT - 1))
    fcst_data = read_MGB_ensemble_binary_file(os.path.abspath(os.path.join(__file__, DIR_FCST, fcst_files[ANALYSIS_FCST_FILE_INDEX])), N_NC, N_NT_FCST, N_ENS_FCST)
    if AGGREGATE_LT > 1:
        fcst_data = aggregate_lead_times_matrix(fcst_data, AGGREGATE_LT, axis=1)
    fcst_data = fcst_data[:, ANALYSIS_FCST_LT, :]  # Filter only for selected analysis LT

    # Compute period averages
    analysis_datetimes = getDatetimes(analysis_datetime.strftime(STR_DATE_FMT), AGGREGATE_LT, fmt=STR_DATE_FMT)
    mgb_analysisClimatology = get_climatological_values(analysis_datetimes, mgb_data, mgb_datetimes, left_right_windows=0)

    # Compute analysis percentile from forecast values
    # analysis_percentile = get_percentile_from_values(np.transpose(fcst_data), mgb_analysisClimatology)
    analysis_percentile = get_fcstPercentiles(fcst_data, mgb_analysisClimatology)

    # Plot map
    get_fcstPlot(analysis_percentile, analysis_datetimes, DIR_SHAPEFILES, STR_VARIABLE, AGGREGATE_LT, ANALYSIS_FCST_LT, exportPlot=EXPORT_PLOT)

    # Save Excel file (MINI_ID, CURRENT VALUE, AVERAGE VALUE, CURRENT_PERCENTILE, TARGET PERCENTILES)
    get_csv(mgb_analysisClimatology, analysis_percentile, STR_VARIABLE, analysis_datetimes, AGGREGATE_LT, ANALYSIS_FCST_LT, exportCSV=EXPORT_CSV)
