# coding: utf8

""" Python file with methods to handle the Ames_Housing DataFile.

Remarks
----------
Information from Ames Housing Paper [1]

Data Set Information:
=====================
This paper presents a data set describing the sale of individual residential property in Ames, Iowa
from 2006 to 2010. The data set contains 2930 observations and a large number of explanatory
variables (23 nominal, 23 ordinal, 14 discrete, and 20 continuous) involved in assessing home
values. I will discuss my previous use of the Boston Housing Data Set and I will suggest
methods for incorporating this new data set as a final project in an undergraduate regression
course.

Attribute Information:
======================

See https://ww2.amstat.org/publications/jse/v19n3/decock.pdf and
https://ww2.amstat.org/publications/jse/v19n3/decock/DataDocumentation.txt for
an extensive description of the data set.

From the description:
Potential Pitfalls (Outliers): Although all known errors were corrected in the data, no
observations have been removed due to unusual values and all final residential sales
from the initial data set are included in the data presented with this article. There are
five observations that an instructor may wish to remove from the data set before giving
it to students (a plot of SALE PRICE versus GR LIV AREA will quickly indicate these
points). Three of them are true outliers (Partial Sales that likely don’t represent actual
market values) and two of them are simply unusual sales (very large houses priced
relatively appropriately). I would recommend removing any houses with more than
4000 square feet from the data set (which eliminates these five unusual observations)
before assigning it to students.

References
-----------
    [1] De Cock, Dean. "Ames, Iowa: Alternative to the Boston housing data as an
    end of semester regression project." Journal of Statistics Education
    19.3 (2011).

"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, scale


# Get basepath such that only relatives paths matter from this folder on
basepath = os.path.dirname(os.path.realpath(__file__))

# All Numerical Features (includes categorical numerical features)
__idx_numerical_features__ = [
    'MSSubClass',
    'LotFrontage',
    'LotArea',
    'OverallQual',
    'OverallCond',
    'YearBuilt',
    'YearRemodAdd',
    'MasVnrArea',
    'BsmtFinSF1',
    'BsmtFinSF2',
    'BsmtUnfSF',
    'TotalBsmtSF',
    '1stFlrSF',
    '2ndFlrSF',
    'GrLivArea',
    'LowQualFinSF',
    'BsmtFullBath',
    'BsmtHalfBath',
    'FullBath',
    'HalfBath',
    'BedroomAbvGr',
    'KitchenAbvGr',
    'TotRmsAbvGrd',
    'Fireplaces',
    'GarageYrBlt',
    'GarageCars',
    'GarageArea',
    'WoodDeckSF',
    'OpenPorchSF',
    'EnclosedPorch',
	'3SsnPorch',
    'ScreenPorch',
    'PoolArea',
    'MiscVal',
    'MoSold',
    'YrSold',
    'SalePrice'
]

# All real features/intuively good features
__idx_intuitive_features__ = [
    #'MSSubClass', Actually categorical according to description, type of house
    #'LotFrontage', # Has some NA values
    'LotArea',
    'OverallQual',
    'OverallCond',
    'YearBuilt',
    # 'YearRemodAdd', # Doesn't seem to have large influence
    # 'MasVnrArea', # Has some NA values
    # 'BsmtFinSF1',
    # 'BsmtFinSF2',
    # 'BsmtUnfSF',
    'TotalBsmtSF', # Summary of Bsmt values
    #'1stFlrSF',
    #'2ndFlrSF',
    'GrLivArea', # Captures total Living area
    # 'LowQualFinSF', # All categorical variables
    # 'BsmtFullBath',
    # 'BsmtHalfBath'
    # 'FullBath',
    # 'HalfBath',
    # 'BedroomAbvGr',
    # 'KitchenAbvGr',
    # 'TotRmsAbvGrd', # Discrete number
    # 'Fireplaces',
    # 'GarageYrBlt', # Has some NA values
    #'GarageCars',
    'GarageArea',
    # 'WoodDeckSF', # Exclude these for simplicity at the moment
    # 'OpenPorchSF',
    # 'EnclosedPorch',
	# '3SsnPorch',
    # 'ScreenPorch',
    # 'PoolArea',
    # 'MiscVal',
    # 'MoSold',
    # 'YrSold', # Inspection shows it's not really related to the sales price
    'SalePrice'
]


def read_all(return_type = 'np', scaling = 'None',
             remove_GrLivArea_outliers = True,
             normal_sales_only = True,
             feature_subset = 'all'):
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

    remove_GrLivArea_outliers : Boolean,
        If True, all samples with gross living area > 4000 sqft are removed since
        they constitute outliers according to [1].

    feature_subset : string 'all' (default), 'numerical', 'intuitive'
        String that decides on which features should actually be considered.


    Returns
    -------------
    Returns the data object containing the entire excel sheet. If return_type is
    'np', return object is a 2D Numpy Array storing the Y-variable in the last
    column.
    """
    data = pd.read_csv(basepath + '/Ames_Housing/train.csv')
    # Postprocessing
    if remove_GrLivArea_outliers:
        # See remark in the top
        data = data[data['GrLivArea'] < 4000]
    if normal_sales_only:
        data = data[data['SaleCondition'] == 'Normal']
    if feature_subset == 'numerical':
        data = data[__idx_numerical_features__]
    elif feature_subset == 'intuitive':
        data = data[__idx_intuitive_features__]
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
