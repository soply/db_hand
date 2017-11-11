# coding: utf8

""" Python file with methods to handle the UCI_Boston DataFile.

Remarks
----------
Information from UCI WebPage:

Data Set Information:
=====================

Concerns housing values in suburbs of Boston.
Number of Instances: 506
Number of Attributes: 13 continuous attributes (including "class"
                         attribute "MEDV"), 1 binary-valued attribute.

Attribute Information:
======================

1. CRIM      per capita crime rate by town
2. ZN        proportion of residential land zoned for lots over
             25,000 sq.ft.
3. INDUS     proportion of non-retail business acres per town
4. CHAS      Charles River dummy variable (= 1 if tract bounds
             river; 0 otherwise)
5. NOX       nitric oxides concentration (parts per 10 million)
6. RM        average number of rooms per dwelling
7. AGE       proportion of owner-occupied units built prior to 1940
8. DIS       weighted distances to five Boston employment centres
9. RAD       index of accessibility to radial highways
10. TAX      full-value property-tax rate per $10,000
11. PTRATIO  pupil-teacher ratio by town
12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks
             by town
13. LSTAT    % lower status of the population
14. MEDV     Median value of owner-occupied homes in $1000's

References
-----------
   [1]:  Harrison, D. and Rubinfeld, D.L. 'Hedonic prices and the
                 demand for clean air', J. Environ. Economics & Management,
                 vol.5, 81-102, 1978.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.datasets import load_boston


# Get basepath such that only relatives paths matter from this folder on
basepath = os.path.dirname(os.path.realpath(__file__))


def read_all(return_type = 'np', scaling = 'None'):
    """
    Reads the complete excel sheet and returns it as a 2D Numpy Array.
    The alleged Y variable (according to the description) is stored in
    the last column.

    Parameters
    --------------
    return_type : string ('np')
        Datatype of return object. If 'np', data is returned as a 2D numpy array.

    scaling : string 'MinMax', 'MeanVar', or 'None'
        Determines the column-wise scaling of the data.

    Returns
    -------------
    Returns the data object containing the entire excel sheet. If return_type is
    'np', return object is a 2D Numpy Array storing the Y-variable in the last
    column.
    """
    data = load_boston()['data']
    if scaling == 'MinMax':
        minmaxscaler = MinMaxScaler(feature_range=(-1, 1))
        data[:,:-1] = minmaxscaler.fit_transform(data[:,:-1])
    elif scaling == 'MeanVar':
        data[:,:-1] = scale(data[:,:-1])
    if return_type == 'np':
        return data
    else:
        raise RuntimeError("Choose return_type = 'np' to read data.")
