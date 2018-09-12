# coding: utf8

""" Python file with methods to handle the UCI_Airfoil DataFile.

Remarks
=========================
Data Set Information:

The NASA data set comprises different size NACA 0012 airfoils at various wind
tunnel speeds and angles of attack. The span of the airfoil and the
observer position were the same in all of the experiments.



Attribute Information:
=========================

This problem has the following inputs:
1. Frequency, in Hertzs.
2. Angle of attack, in degrees.
3. Chord length, in meters.
4. Free-stream velocity, in meters per second.
5. Suction side displacement thickness, in meters.

The only output is:
6. Scaled sound pressure level, in decibels.

References
-----------
[1] https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise
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
    data = pd.read_csv(basepath + '/UCI_Airfoil/airfoil.csv', sep=';')
    # Updated columns
    cols = data.columns.tolist()
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
    import pdb; pdb.set_trace()
