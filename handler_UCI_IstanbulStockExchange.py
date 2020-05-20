# coding: utf8

""" Python file with methods to handle the UCI_IstanbulStockExchange DataFile.

Remarks
----------
Information from UCI WebPage:

Data Set Information:
=====================

Data is collected from imkb.gov.tr and finance.yahoo.com. Data is organized
with regard to working days in Istanbul Stock Exchange.


Attribute Information:
=======================

Stock exchange returns. Istanbul stock exchange national 100 index,
Standard & poorâ€™s 500 return index, Stock market return index of Germany,
Stock market return index of UK, Stock market return index of Japan,
Stock market return index of Brazil, MSCI European index, MSCI
emerging markets index

References
-----------
[1] Akbilgic, O., Bozdogan, H., Balaban, M.E., (2013) A novel
    Hybrid RBF Neural Networks model as a forecaster, Statistics and Computing.
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
                    '/UCI_IstanbulStockExchange/istanbul_stock_exchange.xlsx',
                    skiprows = [0], usecols=range(2,10))
    # Put first column as last column
    cols = data.columns.tolist()
    cols = cols[1:] + [cols[0]]
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
