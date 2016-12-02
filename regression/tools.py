import numpy as np
ass = ["CMS","Crises","Domicile","Gestion","Gestion - Accueil Telephonique","Gestion Assurances","Gestion Relation Clienteles","Gestion Renault","Japon","Médical","Nuit","RENAULT","Regulation Medicale","SAP","Services","Tech. Axa","Tech. Inter","Téléphonie"]


def split(dataframe):
    """
    Split a dataframe into an array of dataframes
    :param dataframe: a dataframe with an "ASS_ASSIGNMENT" column
    :return : an array of dataframes sorted according to the ass object
    """
    return [dataframe.loc[dataframe["ASS_ASSIGNMENT"]==assignment,dataframe.columns] for assignment in ass]

def splitDummy(dataframe):
    """
    Split a dataframe into an array of dataframes
    :param dataframe: a dataframe with columns like "ASS_ASSIGNMENT_CMS", "ASS_ASSIGNMENT_Crises"
    :return : an array of dataframes sorted according to the ass object
    """
    dataframes = [dataframe[dataframe["ASS_ASSIGNMENT_"+assignment]==1] for assignment in ass]
    for dt in dataframes:
        for assignment in ass:
            del dt["ASS_ASSIGNMENT_"+assignment]
    return dataframes

def trainTestSplit(dataframe, test_size,seed = 0):
    """
    :param dataframe: a dataframe
    :param test_size: a percentage
    :return : two dataframes
    """
    l = len(dataframe);
    i = int(test_size * l)
    np.random.seed(seed)
    permutation = np.random.permutation(l)
    train = dataframe.ix[permutation[i:], :]
    test = dataframe.ix[permutation[:i], :]
    return train,test