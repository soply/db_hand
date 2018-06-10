# coding: utf8

""" Python file with methods to handle the UCI_SkillCraft1 DataFile.

Remarks
=========================
Data Set Information:

-- We aggregated screen movements into screen-fixations using a Salvucci &
Goldberg (2000) dispersion-threshold algorithm, and defined Perception Action
Cycles (PACs) as fixations with at least one action.
-- Time is recorded in terms of timestamps in the StarCraft 2 replay file. When
the game is played on 'faster', 1 real-time second is equivalent to roughly 88.5
timestamps.
-- List of possible game actions is discussed in Thompson, Blair, Chen, & Henrey
(2013)


Attribute Information:
=========================

1. GameID: Unique ID number for each game (integer)
2. LeagueIndex: Bronze, Silver, Gold, Platinum, Diamond, Master, GrandMaster, and Professional leagues coded 1-8 (Ordinal)
3. Age: Age of each player (integer)
4. HoursPerWeek: Reported hours spent playing per week (integer)
5. TotalHours: Reported total hours spent playing (integer)
6. APM: Action per minute (continuous)
7. SelectByHotkeys: Number of unit or building selections made using hotkeys per timestamp (continuous)
8. AssignToHotkeys: Number of units or buildings assigned to hotkeys per timestamp (continuous)
9. UniqueHotkeys: Number of unique hotkeys used per timestamp (continuous)
10. MinimapAttacks: Number of attack actions on minimap per timestamp (continuous)
11. MinimapRightClicks: number of right-clicks on minimap per timestamp (continuous)
12. NumberOfPACs: Number of PACs per timestamp (continuous)
13. GapBetweenPACs: Mean duration in milliseconds between PACs (continuous)
14. ActionLatency: Mean latency from the onset of a PACs to their first action in milliseconds (continuous)
15. ActionsInPAC: Mean number of actions within each PAC (continuous)
16. TotalMapExplored: The number of 24x24 game coordinate grids viewed by the player per timestamp (continuous)
17. WorkersMade: Number of SCVs, drones, and probes trained per timestamp (continuous)
18. UniqueUnitsMade: Unique unites made per timestamp (continuous)
19. ComplexUnitsMade: Number of ghosts, infestors, and high templars trained per timestamp (continuous)
20. ComplexAbilitiesUsed: Abilities requiring specific targeting instructions used per timestamp (continuous)

References
-----------
[1] Thompson JJ, Blair MR, Chen L, Henrey AJ (2013) Video Game Telemetry
    as a Critical Tool in the Study of Complex Skill Learning. PLoS ONE 8(9):
    e75129. [Web Link]
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, scale


# Get basepath such that only relatives paths matter from this folder on
basepath = os.path.dirname(os.path.realpath(__file__))

__exclude_features__ = [
    'GameID',
    'LeagueIndex', # Categorical 1-8 (could be predicted though)
    'TotalHours' # Very related to APM according to SAVE
]

def read_all(return_type = 'np', scaling = 'None', to_predict = 'APM'):
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

    to_predict: string
        Feature that shall be predicted, i.e. is assigned to the last column of the
        output data file.

    Returns
    -------------
    Returns the data object containing the entire excel sheet. If return_type is
    'np', return object is a 2D Numpy Array storing the Y-variable in the last
    column. Else it is a pandas dataframe with the descriptors given in the
    excel sheet.
    """
    data = pd.read_csv(basepath + '/UCI_SkillCraft1/SkillCraft1_Dataset.csv', sep = ',')
    # Excluding missing values
    data = data[~((data['TotalHours'] == '?' ) | (data['HoursPerWeek'] == '?' ) | (data['Age'] == '?' ))].astype(float)
    cols = data.columns.tolist()
    i = cols.index(to_predict)
    cols = cols[0:i] + cols[i+1:] + [cols[i]]
    data = data[cols]
    for feature in __exclude_features__:
        if feature == to_predict:
            pass
        else:
            del data[feature]
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
