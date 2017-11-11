# coding: utf8

""" Python file with methods to handle the UCI_WineQuality DataFile.

Remarks
----------
Information from UCI WebPage:

Data Set Information:
=====================

The two datasets are related to red and white variants of the Portuguese
"Vinho Verde" wine. For more details, consult: [Web Link] or the reference
[Cortez et al., 2009]. Due to privacy and logistic issues, only physicochemical
(inputs) and sensory (the output) variables are available (e.g. there is no
data about grape types, wine brand, wine selling price, etc.).

These datasets can be viewed as classification or regression tasks. The classes
are ordered and not balanced (e.g. there are munch more normal wines than
excellent or poor ones). Outlier detection algorithms could be used to detect the
few excellent or poor wines. Also, we are not sure if all input variables are
relevant. So it could be interesting to test feature selection methods.


Attribute Information:
=======================

For more information, read [Cortez et al., 2009].
Input variables (based on physicochemical tests):
1 - fixed acidity
2 - volatile acidity
3 - citric acid
4 - residual sugar
5 - chlorides
6 - free sulfur dioxide
7 - total sulfur dioxide
8 - density
9 - pH
10 - sulphates
11 - alcohol
Output variable (based on sensory data):
12 - quality (score between 0 and 10)

References
-----------
[1] P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
    Modeling wine preferences by data mining from physicochemical properties.
    In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

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
    data1 = pd.read_excel(basepath + '/UCI_WineQuality/WineQuality-white.xlsx',
                        header = [1])
    data2 = pd.read_excel(basepath + '/UCI_WineQuality/winequality-red.xlsx',
                          header = [1])
    data = pd.concat([data1, data2])
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
