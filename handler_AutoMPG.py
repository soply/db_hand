# coding: utf8

""" Python file with methods to handle the Ames_Housing DataFile.

Remarks
----------
This dataset was obtained from the UCI machine learning repository
(http://www.ics.uci.edu/~mlearn/MLSummary.html).
The dataset at UCI had originally 9 attributes (the first being the goal variable,
the mpg attribute). The last attribute (the car model) is unique for all
instances so I've REMOVED IT!
I've also moved the first attribute (the goal variable) to the end, so now is
the last attribute, just for compatibility with my dataset format.
All other things are as in UCI.
In the repository they referred an original dataset were apparently there were
more instances, but they were removed because they had the "mpg" attribute
(the goal) with unknown value.

Characteristics: 398 cases; 4 continuous variables; 3 nominal vars..

The attributes brief description goes as follows :
    1. cylinders:     multi-valued discrete	# the number of cylinders
    2. displacement:  continuous		# ?
    3. horsepower:    continuous		# the power of the engine
    4. weight:        continuous		# obvious meaning
    5. acceleration:  continuous		# obvious meaning
    6. model year:    multi-valued discrete	# obvious meaning
    7. origin:        multi-valued discrete	# obvious meaning
    8. mpg:           continuous		# I think is the fuel consumption (miles per
gallon)

References
-----------
    [1] https://www.dcc.fc.up.pt/~ltorgo/Regression/autompg.html
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, scale


# Get basepath such that only relatives paths matter from this folder on
basepath = os.path.dirname(os.path.realpath(__file__))

def read_all(return_type = 'np', scaling = 'None', features = 'continuous'):
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

    features : string 'all', 'continuous', 'discrete'
        Describes the features that are used. If 'continuous', only features
        with continuous values will be used. If 'discrete', only features with
        discrete values (plus response mpg) will be used.

    Returns
    -------------
    Returns the data object containing the entire excel sheet. If return_type is
    'np', return object is a 2D Numpy Array storing the Y-variable in the last
    column.
    """
    data = pd.read_table(basepath + '/AutoMpg/auto.data', sep=',',
        names = ['1cylinders','2displacement','3horsepower','4weight',
                '5acceleration','6modelyear','7origin','8mpg'])
    # Excluding missing values
    data = data[~((data['3horsepower'] == '?' ))].astype(float)
    if features == 'continuous':
        data = pd.DataFrame({
            '1displacement' : data.ix[:,1],
            '2horsepower' : data.ix[:,2],
            '3weight' : data.ix[:,3],
            '4acceleration' : data.ix[:,4],
            '5mpg' : data.ix[:,7]}
        )
    elif features == 'continuous':
        data = pd.DataFrame({
            '1cylinders' : data.ix[:,0],
            '2modelyear' : data.ix[:,5],
            '3origin' : data.ix[:,6],
            '4mpg' : data.ix[:,7]}
        )
    cols = data.columns.tolist()
    if scaling == 'MinMax':
        minmaxscaler = MinMaxScaler(feature_range=(-1, 1))
        data[cols[:-1]] = minmaxscaler.fit_transform(data[cols[:-1]])
    elif scaling == 'MeanVar':
        data[cols[:-1]] = scale(data[cols[:-1]])
    if return_type == 'np':
        return data.values
    else:
        raise RuntimeError("Choose return_type = 'np' to read data.")
