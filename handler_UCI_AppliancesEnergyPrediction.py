# coding: utf8

""" Python file with methods to handle the UCI_AppliancesEnergyPrediction DataFile.

Remarks
----------
Information from UCI WebPage:

Data Set Information:
=====================

The data set is at 10 min for about 4.5 months. The house temperature and
humidity conditions were monitored with a ZigBee wireless sensor network.
Each wireless node transmitted the temperature and humidity conditions around
3.3 min. Then, the wireless data was averaged for 10 minutes periods. The energy
data was logged every 10 minutes with m-bus energy meters. Weather from the
nearest airport weather station (Chievres Airport, Belgium) was downloaded
from a public data set from Reliable Prognosis (rp5.ru), and merged together
with the experimental data sets using the date and time column. Two random
variables have been included in the data set for testing the regression models
and to filter out non predictive attributes (parameters).


Attribute Information:
=======================

date time year-month-day hour:minute:second
Appliances, energy use in Wh
lights, energy use of light fixtures in the house in Wh
T1, Temperature in kitchen area, in Celsius
RH_1, Humidity in kitchen area, in %
T2, Temperature in living room area, in Celsius
RH_2, Humidity in living room area, in %
T3, Temperature in laundry room area
RH_3, Humidity in laundry room area, in %
T4, Temperature in office room, in Celsius
RH_4, Humidity in office room, in %
T5, Temperature in bathroom, in Celsius
RH_5, Humidity in bathroom, in %
T6, Temperature outside the building (north side), in Celsius
RH_6, Humidity outside the building (north side), in %
T7, Temperature in ironing room , in Celsius
RH_7, Humidity in ironing room, in %
T8, Temperature in teenager room 2, in Celsius
RH_8, Humidity in teenager room 2, in %
T9, Temperature in parents room, in Celsius
RH_9, Humidity in parents room, in %
To, Temperature outside (from Chievres weather station), in Celsius
Pressure (from Chievres weather station), in mm Hg
RH_out, Humidity outside (from Chievres weather station), in %
Wind speed (from Chievres weather station), in m/s
Visibility (from Chievres weather station), in km
Tdewpoint (from Chievres weather station), Â°C
rv1, Random variable 1, nondimensional
rv2, Random variable 2, nondimensional

Where indicated, hourly data (then interpolated) from the nearest airport weather
station (Chievres Airport, Belgium) was downloaded from a public data set from
Reliable Prognosis, rp5.ru. Permission was obtained from Reliable Prognosis for
the distribution of the 4.5 months of weather data.

References
-----------
[1] Luis M. Candanedo, Veronique Feldheim, Dominique Deramaix, Data driven
    prediction models of energy use of appliances in a low-energy house,
    Energy and Buildings, Volume 140, 1 April 2017, Pages 81-97,
    ISSN 0378-7788, [Web Link].
"""
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, scale

# Get basepath such that only relatives paths matter from this folder on
basepath = os.path.dirname(os.path.realpath(__file__))


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
    data = pd.read_excel(basepath + \
                    '/UCI_AppliancesEnergyPrediction/energydata_complete.xlsx',
                    usecols=range(1,28), skiprows = [0])
    cols = data.columns.tolist()
    cols = cols[1:] + [cols[0]]
    data = data[cols]
    if scaling == 'MinMax':
        minmaxscaler = MinMaxScaler(feature_range=(-1, 1))
        data[cols[:-1]] = minmaxscaler.fit_transform(data[cols[:-1]])
    elif scaling == 'MeanVar':
        data[cols[:-1]] = scale(data[cols[:-1]])
    if return_type == 'np':
        return data.as_matrix()
    elif return_type == 'pd':
        return data
    else:
        raise RuntimeError("Choose return_type = 'np' or 'pd' to read data.")
