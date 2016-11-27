# We should consider the time aspect with the following features:

# – time since epoch
# – time since start of day
# – time as a categorical feature # TODO : How ? -> maybe already available in "TPER_HOUR" feature
# – month
# – week day
# – week end : "WEEK_END" is already available as a feature
# – night/day : "Jours/Nuit" is already available as a feature
# – “day off” : "DAY_OFF" is already available as a feature
# – holidays # TODO : Take holiday data from outside sources.

# Given a date in the format : '2011-04-24 01:30:00.000'
# We therefore use the datetime package to get a date object - easier to work with
import datetime
import numpy as np

def toDatetime(date):
    """
    :param date: A string in a format like :'2011-04-24 01:30:00.000'
    :type date: String
    :return: A suitable time format to work with
    :rtype: pandas.tslib.Timestamp
    """
    return datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S.000')

def datetimeToEpoch(datetime):
    """
        :param datetime: A datetime object
        :type datetime: pandas.tslib.Timestamp
        :return: time since epoch in ms.
    """
    return int(datetime.strftime("%s")) * 1000

def datetimeToStartOfDay(datetime):
    """
        :param datetime: A datetime object
        :type datetime: pandas.tslib.Timestamp
        :return: time start of the day.
    """
    return (datetime-datetime.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()

def createTimeFeatures(data,features_to_create=["EPOCH","START_OF_DAY","MONTH","WEEKDAY"], delete = True):
    """
    A function to create different time features.
    The parameter features_to_create enables to select only the features we want to add.
    :param data: a dataframe with a DATE field.
    :param features_to_create: an array of strings like default ["EPOCH","START_OF_DAY","MONTH","WEEKDAY"]
    :type data: pandas.core.frame.DataFrame.
    :return: data: the modified dataframe.
    """
    data.insert(0, "DATE_TIME", data["DATE"].apply(toDatetime));

    # aF stands for applyFunction
    # c stands for categorical variable
    features = {
        "EPOCH":{'aF' : datetimeToEpoch, 'type':'int'},
        "START_OF_DAY":{'aF' : datetimeToStartOfDay, 'type':'int'},
        "MONTH": {'aF' : lambda datetime: datetime.month, 'type':'category'},
        "WEEKDAY": {'aF' : lambda datetime: datetime.weekday(), 'type':'category'}
    }

    i = 1 # Just for locating features
    for key, aF, type in [(i,features[i]['aF'],features[i]['type']) for i in features_to_create]:
        data.insert(i, key, data["DATE_TIME"].apply(aF));
        if(type): data[key]= data[key].astype(type)
        i += 1
    if(delete):
        del data["DATE"]
    del data["DATE_TIME"]# We no longer need that initial feature
    return data

def createCategoricalFeatures(data,feature,delete=True):
    """
    A function to 'dummy code' and replace categorical feature
    :param data: a dataframe with a feature field.
    :param feature: The name of the feature to 'dummy code'
    :return: data: the modified dataframe.
    """
    index = int(np.where(data.columns == feature)[0][0]) # Get the feature position
    for i in data[feature].unique():
        name = str(feature) + "_" + str(i)
        data.insert(index,name,(data[feature]==i)+0) # +0 : to cast into an array of int
        data[name] = data[name].astype('int')
    # We no longer need that initial feature
    if (delete):
        del data[feature]
    return data

def castToCategorialFeatures(data):
    """
    Categorical features are not automatically recognized by pandas,
    so we manually set them as such.
    :param data: a dataframe with a feature field.
    :return: data: the dataframe with categorical features.
    """
    data["DAY_DS"] = data["DAY_DS"].astype('object')
    for i in ["SPLIT_COD","ACD_COD","WEEK_END", "TPER_HOUR", "DAY_OFF"]:
        data[i] = data[i].astype('int')
    return data

def normalize(data):
    """
       Function to normalize the dataframe (on the right columns)
       :param data: a dataframe.
       :return: data: the dataframe normalized on its 'float64' and 'int64' fields.
    """
    data_tmp = data.select_dtypes(include=['float64', 'int64'])
    data_tmp = (data_tmp - data_tmp.mean()) / data_tmp.std()
    which_rows = (data.dtypes == "int64") | (data.dtypes == "float64")
    data.loc[:, which_rows] = data_tmp
    return data

def createDayNightFeature(data):
    data["NIGHT_TIME"] = (data["START_OF_DAY"]<=7*3600) + (data["START_OF_DAY"]>=23*3600) + 0
    data["LUNCH_TIME"] = (data["START_OF_DAY"] >= 12 * 3600) * (data["START_OF_DAY"] <= 14 * 3600) + 0
    data["MORNING_TIME"] = (data["START_OF_DAY"] >= 10 * 3600) * (data["START_OF_DAY"] <= 12 * 3600) + 0
    data["AFTERNOON_TIME"] = (data["START_OF_DAY"] >= 14 * 3600) * (data["START_OF_DAY"] <= 17 * 3600) + 0
    return data