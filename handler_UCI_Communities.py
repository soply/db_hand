# coding: utf8

""" Python file with methods to handle the UCI_Communities DataFile.

Remarks
----------
Information from UCI WebPage:

Data Set Information:
=====================

Many variables are included so that algorithms that select or learn weights for
attributes could be tested. However, clearly unrelated attributes were not
included; attributes were picked if there was any plausible connection to
crime (N=122), plus the attribute to be predicted (Per Capita Violent Crimes).
The variables included in the dataset involve the community, such as the percent
of the population considered urban, and the median family income, and involving
law enforcement, such as per capita number of police officers, and percent of
officers assigned to drug units.

The per capita violent crimes variable was calculated using population and the
sum of crime variables considered violent crimes in the United States: murder,
rape, robbery, and assault. There was apparently some controversy in some states
concerning the counting of rapes. These resulted in missing values for rape,
which resulted in incorrect values for per capita violent crime. These cities
are not included in the dataset. Many of these omitted communities were from
the midwestern USA.

Data is described below based on original values. All numeric data was normalized
into the decimal range 0.00-1.00 using an Unsupervised, equal-interval binning
method. Attributes retain their distribution and skew (hence for example the
population attribute has a mean value of 0.06 because most communities are small).
E.g. An attribute described as 'mean people per household' is actually the
normalized (0-1) version of that value.

The normalization preserves rough ratios of values WITHIN an attribute (e.g.
double the value for double the population within the available precision -
xcept for extreme values (all values more than 3 SD above the mean are
normalized to 1.00; all values more than 3 SD below the mean are nromalized
to 0.00)).

However, the normalization does not preserve relationships between values BETWEEN
attributes (e.g. it would not be meaningful to compare the value for whitePerCap
with the value for blackPerCap for a community)

A limitation was that the LEMAS survey was of the police departments with at
least 100 officers, plus a random sample of smaller departments. For our purposes,
communities not found in both census and crime datasets were omitted. Many
ommunities are missing LEMAS data.

Website:
==================
http://archive.ics.uci.edu/ml/datasets/communities+and+crime
"""

# All real features/intuively good features
__exclude_features__ = [
    'state',
    'countyCode',
    'communityCode',
    'fold',
    'LemasSwornFT',
    'LemasSwFTPerPop',
    'LemasSwFTFieldOps',
    'LemasSwFTFieldPerPop',
    'LemasTotalReq',
    'LemasTotReqPerPop',
    'PolicReqPerOffic',
    'PolicPerPop',
    'RacialMatchCommPol',
    'PctPolicWhite',
    'PctPolicBlack',
    'PctPolicHisp',
    'PctPolicAsian',
    'PctPolicMinor',
    'OfficAssgnDrugUnits',
    'NumKindsDrugsSeiz',
    'PolicAveOTWorked',
    'PolicCars',
    'PolicOperBudg',
    'LemasPctPolicOnPatr',
    'LemasGangUnitDeploy',
    'LemasPctOfficDrugUn',
    'PolicBudgPerPop',
    'rapes',
    'rapesPerPop',
    'arsons',
    'arsonsPerPop'
]



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
    data = pd.read_excel(basepath + '/UCI_Communities/crimedata.xls')
    cols = data.columns.tolist()
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
