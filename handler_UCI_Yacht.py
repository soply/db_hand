# coding: utf8

""" Python file with methods to handle the UCI_Yacht DataFile.

Remarks
=========================
Data Set Information:

Prediction of residuary resistance of sailing yachts at the initial design stage
is of a great value for evaluating the shipâ€™s performance and for estimating
the required propulsive power. Essential inputs include the basic hull dimensions
and the boat velocity.

The Delft data set comprises 308 full-scale experiments, which were performed at
the Delft Ship Hydromechanics Laboratory for that purpose.
These experiments include 22 different hull forms, derived from a parent form
closely related to the â€˜Standfast 43â€™ designed by Frans Maas.



Attribute Information:
=========================

Variations concern hull geometry coefficients and the Froude number:

1. Longitudinal position of the center of buoyancy, adimensional.
2. Prismatic coefficient, adimensional.
3. Length-displacement ratio, adimensional.
4. Beam-draught ratio, adimensional.
5. Length-beam ratio, adimensional.
6. Froude number, adimensional.

The measured variable is the residuary resistance per unit weight of displacement:

7. Residuary resistance per unit weight of displacement, adimensional.

References
-----------
[1] http://archive.ics.uci.edu/ml/datasets/yacht+hydrodynamics
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
    data = pd.read_csv(basepath + '/UCI_Yacht/yachts.txt', sep=' ')
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
