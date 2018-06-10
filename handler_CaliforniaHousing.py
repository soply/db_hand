# coding: utf8

""" Python file with methods to handle the Ames_Housing DataFile.

Remarks
----------
S&P Letters Data
We collected information on the variables using all the block groups in
California from the 1990 Cens us. In this sample a block group on average
 includes 1425.5 individuals living in a geographically co mpact area.
 Naturally, the geographical area included varies inversely with the population
 density. W e computed distances among the centroids of each block group as
 measured in latitude and longitude. W e excluded all the block groups reporting
 zero entries for the independent and dependent variables. T he final data
 contained 20,640 observations on 9 variables. The dependent variable is
 ln(median house value).

                               Bols    tols
INTERCEPT		       11.4939 275.7518
MEDIAN INCOME	       0.4790  45.7768
MEDIAN INCOME2	       -0.0166 -9.4841
MEDIAN INCOME3	       -0.0002 -1.9157
ln(MEDIAN AGE)	       0.1570  33.6123
ln(TOTAL ROOMS/ POPULATION)    -0.8582 -56.1280
ln(BEDROOMS/ POPULATION)       0.8043  38.0685
ln(POPULATION/ HOUSEHOLDS)     -0.4077 -20.8762
ln(HOUSEHOLDS)	       0.0477  13.0792

The file contains all the the variables. Specifically, it contains median house
value, median income, housing median age, total rooms, total bedrooms,
population, households, latitude, and longitude in that order.

Comment TK: Actually the order is:

longitude: continuous.
latitude: continuous.
housingMedianAge: continuous.
totalRooms: continuous.
totalBedrooms: continuous.
population: continuous.
households: continuous.
medianIncome: continuous.
medianHouseValue: continuous.

References
-----------
    [1] Pace, R. Kelley and Ronald Barry, "Sparse Spatial Autoregressions",
        Statistics and Probability Letters, 33 (1997) 291-297.

"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, scale


# Get basepath such that only relatives paths matter from this folder on
basepath = os.path.dirname(os.path.realpath(__file__))

def read_all(return_type = 'np', scaling = 'None', feature_adjustment = True):
    """
    Reads the complete data file and returns it as a 2D Numpy Array.
    The alleged Y variable (according to the description) is stored in
    the last column.

    Parameters
    --------------
    return_type : string ('np')
        Datatype of return object. If 'np', data is returned as a 2D numpy array.

    scaling : string 'MinMax', 'MeanVar', or 'None'
        Determines the column-wise scaling of the data.

    feature_adjustment: If true, we exchange the features by
        longitude
        latitude
        ln(housingMedianAge)
        ln(totalRooms/Population)
        ln(Bedrooms/Population)
        ln(Population/Households)
        ln(Households)
        medianIncome

    Returns
    -------------
    Returns the data object containing the entire excel sheet. If return_type is
    'np', return object is a 2D Numpy Array storing the Y-variable in the last
    column.
    """
    data = pd.read_table(basepath + '/California_Housing/cal_housing.data', sep=',')
    cols = data.columns.tolist()
    if feature_adjustment:
        data = pd.DataFrame({
            '1longitude' : data.ix[:,0],
            '2latitude' : data.ix[:,1],
            '3lnage' : np.log(data.ix[:,2]),
            '4lnroomsbypop' : np.log(data.ix[:,3]/data.ix[:,5]),
            '5lnbedroomsbypop' : np.log(data.ix[:,4]/data.ix[:,5]),
            '6lnpopbyhouseholds' : np.log(data.ix[:,5]/data.ix[:,6]),
            '7lnHouseholds' : np.log(data.ix[:,6]),
            '8income' : data.ix[:,7],
            '9housevalue' : data.ix[:,8]}
        )
        cols = data.columns.tolist()
    if scaling == 'MinMax':
        minmaxscaler = MinMaxScaler(feature_range=(-1, 1))
        data[cols[:-1]] = minmaxscaler.fit_transform(data[cols[:-1]])
    elif scaling == 'MeanVar':
        data[cols[:-1]] = scale(data[cols[:-1]])
    if return_type == 'np':
        return data.as_matrix()
    else:
        raise RuntimeError("Choose return_type = 'np' to read data.")
