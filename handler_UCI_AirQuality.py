# coding: utf8

""" Python file with methods to handle the UCI_AirQuality Data set

Remarks
----------
Information from UCI WebPage:

Data Set Information:
=====================

Data Set Information:

The dataset contains 9358 instances of hourly averaged responses from an array
of 5 metal oxide chemical sensors embedded in an Air Quality Chemical Multisensor
Device. The device was located on the field in a significantly polluted area,
at road level,within an Italian city. Data were recorded from March 2004 to
February 2005 (one year)representing the longest freely available recordings of
on field deployed air quality chemical sensor devices responses. Ground Truth
hourly averaged concentrations for CO, Non Metanic Hydrocarbons, Benzene, Total
Nitrogen Oxides (NOx) and Nitrogen Dioxide (NO2) and were provided by a
co-located reference certified analyzer. Evidences of cross-sensitivities as
well as both concept and sensor drifts are present as described in De Vito et
al., Sens. And Act. B, Vol. 129,2,2008 (citation required) eventually affecting
sensors concentration estimation capabilities. Missing values are tagged with
-200 value.
This dataset can be used exclusively for research purposes. Commercial purposes
are fully excluded.


Attribute Information:
=======================

0 Date	(DD/MM/YYYY)
1 Time	(HH.MM.SS)
2 True hourly averaged concentration CO in mg/m^3 (reference analyzer)
3 PT08.S1 (tin oxide) hourly averaged sensor response (nominally CO targeted)
4 True hourly averaged overall Non Metanic HydroCarbons concentration in microg/m^3 (reference analyzer)
5 True hourly averaged Benzene concentration in microg/m^3 (reference analyzer)
6 PT08.S2 (titania) hourly averaged sensor response (nominally NMHC targeted)
7 True hourly averaged NOx concentration in ppb (reference analyzer)
8 PT08.S3 (tungsten oxide) hourly averaged sensor response (nominally NOx targeted)
9 True hourly averaged NO2 concentration in microg/m^3 (reference analyzer)
10 PT08.S4 (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted)
11 PT08.S5 (indium oxide) hourly averaged sensor response (nominally O3 targeted)
12 Temperature in Â°C
13 Relative Humidity (%)
14 AH Absolute Humidity

References
-----------
[1] S. De Vito, E. Massera, M. Piga, L. Martinotto, G. Di Francia, On field
    calibration of an electronic nose for benzene estimation in an urban
    pollution monitoring scenario, Sensors and Actuators B: Chemical,
    Volume 129, Issue 2, 22 February 2008, Pages 750-757, ISSN 0925-4005,
    [Web Link].
"""
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, scale

# Get basepath such that only relatives paths matter from this folder on
basepath = os.path.dirname(os.path.realpath(__file__))

__exclude_features__ = [
    'NMHC(GT)' #  Many missing values
]
def read_all(return_type = 'np', scaling = 'None'):
    """
    Reads the complete excel sheet and returns it as a 2D Numpy Array or pandas
    DataFrame. If 2D Numpy Array is chosen as return type, the alleged Y
    variable (according to the description) is stored in the last column.

    Parameters
    --------------
    return_type : string ('np' or 'pd')
        Datatype of return object. If 'np', data is returned as a 2D numpy array.
        If 'pd', data is returned as a 2D DataFrame

    scaling : string 'MinMax', 'MeanVar', or 'None'
        Determines the column-wise scaling of the data.

    Returns
    -------------
    Returns the data object containing the entire excel sheet. If return_type is
    'np', return object is a 2D Numpy Array storing the Y-variable in the last
    column. Else it is a pandas dataframe with the descriptors given in the
    excel sheet.
    """
    data = pd.read_csv(basepath + '/UCI_AirQuality/AirQualityUCI.csv', sep = ';',
                       nrows = 9357,
                       usecols = ['CO(GT)',	'PT08.S1(CO)','C6H6(GT)','PT08.S2(NMHC)','NOx(GT)',
                                  'PT08.S3(NOx)','NO2(GT)','PT08.S4(NO2)','PT08.S5(O3)',
                                  'T','RH','AH'])
    # Excluding missing values
    cols = data.columns.tolist()
    for col in cols:
        data = data[data[col] != -200]
        if col in ['CO(GT)', 'C6H6(GT)', 'T', 'RH', 'AH']:
            data[col] = data[col].apply(lambda x: x.replace(',','.'))
            data[col] = pd.to_numeric(data[col])
    if scaling == 'MinMax':
        minmaxscaler = MinMaxScaler(feature_range=(-1, 1))
        data[cols[:-1]] = minmaxscaler.fit_transform(data[cols[:-1]])
    elif scaling == 'MeanVar':
        data[cols[:-1]] = scale(data[cols[:-1]])
    if return_type == 'np':
        return data.values
    elif return_type == 'pd':
        return data
    else:
        raise RuntimeError("Choose return_type = 'np' or 'pd' to read data.")
