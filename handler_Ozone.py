# coding: utf8

""" Python file with methods to handle the R Ozone DataFile.

Remarks
=========================
Data Set Information:

Daily air quality measurements in New York, May to September 1973.

Details
Daily readings of the following air quality values for May 1, 1973 (a Tuesday) to September 30, 1973.
Ozone: Mean ozone in parts per billion from 1300 to 1500 hours at Roosevelt Island
Solar.R: Solar radiation in Langleys in the frequency band 4000–7700 Angstroms from 0800 to 1200 hours at Central Park
Wind: Average wind speed in miles per hour at 0700 and 1000 hours at LaGuardia Airport
Temp: Maximum daily temperature in degrees Fahrenheit at La Guardia Airport.

Attribute Information:
=========================

A data frame with 154 observations on 6 variables.

[,1]	Ozone	numeric	Ozone (ppb)
[,2]	Solar.R	numeric	Solar R (lang)
[,3]	Wind	numeric	Wind (mph)
[,4]	Temp	numeric	Temperature (degrees F)
[,5]	Month	numeric	Month (1--12)
[,6]	Day	numeric	Day of month (1--31)

References
-----------
[1] Chambers, J. M., Cleveland, W. S., Kleiner, B. and Tukey, P. A. (1983)
    Graphical Methods for Data Analysis. Belmont, CA: Wadsworth.
[2] https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/airquality.html
"""


import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, scale


# Get basepath such that only relatives paths matter from this folder on
basepath = os.path.dirname(os.path.realpath(__file__))

__NAN_rows__ = [5,6,10,11,25,26,27,32,33,34,35,36,37,39,42,43,45,46,52,53,54,55,
                56,57,58,59,60,61,65,72,75,83,84,96,97,98,102,103,107,115,119,
                150]

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
    data = pd.read_csv(basepath + '/OzoneDataSet/airquality.csv', sep=',',
                        skiprows = __NAN_rows__)
    # Updated columns
    cols = data.columns.tolist()
    # Rearange cols
    cols = ['Solar.R','Wind','Temp','Ozone']
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


if __name__ == '__main__':
    data = read_all(scaling = 'MeanVar')
