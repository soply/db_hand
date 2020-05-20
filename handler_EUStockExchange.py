# coding: utf8

""" Python file with methods to handle the EUStockmarkets Data File.

Remarks
=========================
Data Set Information:

Contains the daily closing prices of major European stock indices: Germany
DAX (Ibis), Switzerland SMI, France CAC, and UK FTSE. The data are sampled in
business time, i.e., weekends and holidays are omitted.


Attribute Information:
=========================

A multivariate time series with 1860 observations on 4 variables. The object is of class "mts".
Germany DAX (Ibis), Switzerland SMI, France CAC, and UK FTSE.

References
-----------
[1] https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/EuStockMarkets.html
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
    data = pd.read_csv(basepath + '/EuropeStockExchange/EuStockMarkets.csv', sep=',')
    # Updated columns
    cols = data.columns.tolist()
    # Rearange cols
    cols = ["DAX","SMI","CAC","FTSE"]
    data = data[cols]
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


if __name__ == '__main__':
    data = read_all(scaling = 'MeanVar')
    import pdb; pdb.set_trace()
