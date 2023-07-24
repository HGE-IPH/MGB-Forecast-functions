# from MGB_asc_files import *
from MGB_binary_files import *
# from MGB_forecast_orgdata import *
# from MGB_forecast_plot import *
import os
# import pandas as pd
import geopandas as gpd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import get_cmap
mpl.use('Qt5Agg')  # Interative window

def getDatetimes(initial_date, nDT, fmt='%d-%m-%Y'):
    ini_dt = dt.datetime.strptime(initial_date, fmt)
    return np.asarray([ini_dt + dt.timedelta(days=i) for i in range(nDT)])

def get_monthlyAverages(data_array, datetimes_array):
    monthlyAvgs = np.empty(12, dtype=np.float64)
    tmpMonths = np.asarray([i.month for i in datetimes_array])
    for i in range(1, 12 + 1):
        monthIdx = np.where(tmpMonths == i)[0]
        monthlyAvgs[i - 1] = np.nanmean(data_array[monthIdx])
    return monthlyAvgs

def get_monthlyMeanFlow_as_Avgpercentage(data_array, datetimes_array):
    monthlyAvgs = get_monthlyAverages(data_array, datetimes_array)
    tmp_years = np.asarray([i.year for i in datetimes_array])
    tmp_months = np.asarray([i.month for i in datetimes_array])
    unique_years = np.unique(tmp_years)
    unique_months = np.unique(tmp_months)
    monthlyAvgPct = {i: np.empty(12) for i in unique_years}
    for iyear in unique_years:
        for imonth in unique_months:
            iidxs = np.logical_and(tmp_years == iyear, tmp_months == imonth)
            currentMonthlyAvg = np.nanmean(data_array[iidxs])
            currentMonthlyAvgPct = currentMonthlyAvg/monthlyAvgs[imonth - 1]
            monthlyAvgPct[iyear][imonth - 1] = currentMonthlyAvgPct
    return monthlyAvgPct

def get_rankPercentiles(monthlyAvgPct):
    uniqueYears = np.asarray(list(monthlyAvgPct.keys()))
    N = uniqueYears.size
    array_monthlyAvgPct = np.asarray(list(monthlyAvgPct.values()))
    rankedPctiles = {i: np.empty(12) for i in uniqueYears}
    for i in range(N):
        for j in range(12):
            iranks = array_monthlyAvgPct[:, j].argsort().argsort()
            iranks_sorted = np.asarray([iranks.size - i for i in iranks])
            rankedPctle = iranks_sorted[i]/(N + 1)
            rankedPctiles[uniqueYears[i]][j] = rankedPctle
    return rankedPctiles

def add_column_to_shp(input_shapefile, output_shapefile, column_name, column_data):
    # Read the input shapefile
    gdf = gpd.read_file(input_shapefile)
    # Check if the 'MiniID' column exists in the GeoDataFrame
    if 'ID_Mini' not in gdf.columns:
        print("Error: 'MiniID' column not found in the input shapefile.")
        return
    # Make sure the 'column_data' list has the same length as the number of features in the shapefile
    if len(column_data) != len(gdf):
        print("Error: The length of 'column_data' must match the number of features in the shapefile.")
        return
    # Add the new column with the specified data for matching MiniID values
    for i, miniid_value in enumerate(gdf['ID_Mini']):
        gdf.loc[gdf['ID_Mini'] == i, column_name] = column_data[i]
        # if i == miniid_value:
    # Save the modified GeoDataFrame to a new shapefile
    gdf.to_file(output_shapefile)

# funcao para testes e debug
def driver_code(path_MGBOutput, str_MGBVar, str_initialDate, nDT, nNC, stateYear, stateMonth, path_inputSHP, path_outputSHP, _dateFMT='%d-%m-%Y'):
    data_MGBVar = read_MGB_binary_file(file_path=os.path.join(path_MGBOutput, str_MGBVar), n_unit_catchments=nNC, n_time_steps=nDT)  # NT, NC
    datetimes_array = getDatetimes(initial_date=str_initialDate, nDT=nDT, fmt=_dateFMT)
    iCentroid_state = list()
    for iCentroid in range(len(data_MGBVar[0, :])):
        print(f'{iCentroid}')
        iCentroid_data = data_MGBVar[:, iCentroid]
        # 1) Calculate the monthly mean flow
        # 1.1) Note on missing data (nao considerado pois o MGB nao tem falhas)
        # 1.2) Calculated only if expected values are over 50% available (nao considerado pois MGB nao tem falhas)
        monthlyAvgs = get_monthlyAverages(iCentroid_data, datetimes_array)
        # 2) Calculate the monthly mean flow as a percentage of the average
        monthlyAvgs_as_Pctg = get_monthlyMeanFlow_as_Avgpercentage(iCentroid_data, datetimes_array)
        # 3) Calculate rank percentiles (default using eibull distribution)
        monthlyAvgs_rankPctiles = get_rankPercentiles(monthlyAvgs_as_Pctg)
        # 4) Assign percentile to a category

        iCentroid_state.append(monthlyAvgs_rankPctiles[stateYear][stateMonth - 1])

    # Adding column to SHP
    new_columnName = 'RankPctiles'
    add_column_to_shp(path_inputSHP, path_outputSHP, new_columnName, iCentroid_state)

if __name__ == '__main__':

    path_MGBOutput = r'D:\Arquivos\Python\WMO_HydroSOS\Auxiliares\Output_19940101-20221231'
    str_MGBVar = 'QTUDO.MGB'  # EVAPTUDO.MGB, QTUDO.MGB, SOILWATER.MGB, TWSTUDO.MGB, VBAS.MGB, VINT.MGB, YTUDO.MGB

    path_inputSHP = r'D:\Arquivos\WMO_IPH\Huallaga\shapes_MGB_Huallaga\minicuencas_huallaga.shp'
    path_outputSHP = r'D:\Arquivos\WMO_IPH\Huallaga\shapes_MGB_Huallaga\updated_minicuencas_huallaga.shp'

    str_initialDate = '01-01-1994'
    nDT = 10592
    nNC = 1168
    stateYear = 2012
    stateMonth = 12

    driver_code(path_MGBOutput, str_MGBVar, str_initialDate, nDT, nNC, stateYear, stateMonth, path_inputSHP, path_outputSHP)

