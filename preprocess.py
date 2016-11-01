# We should consider the time aspect with the following features:

# – time since epoch
# – time since start of day
# – time as a categorical feature # TODO : How ?
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

def createTimeFeatures(data):
    """
    A function to create different time features.
    :param data: a dataframe with a DATE field.
    :type data: pandas.core.frame.DataFrame.
    :return: data: the modified dataframe.
    """
    data.insert(0, "DATE_TIME", data["DATE"].apply(toDatetime))
    data.insert(1, "EPOCH", data["DATE_TIME"].apply(datetimeToEpoch))
    data.insert(2, "START_OF_DAY", data["DATE_TIME"].apply(datetimeToStartOfDay))
    data.insert(3, "MONTH", data["DATE_TIME"].apply(lambda datetime: datetime.month))
    data.insert(4, "WEEKDAY", data["DATE_TIME"].apply(lambda datetime: datetime.weekday()))
    del data["DATE"] # We no longer need that initial feature
    return data

def createCategoricalFeatures(data,feature):
    """
    A function to 'dummy code' and replace categorical feature
    :param data: a dataframe with a feature field.
    :param feature: The name of the feature to 'dummy code'
    :return: data: the modified dataframe.
    """
    index = int(np.where(data.columns == feature)[0][0]) # Get the feature position
    for i in data[feature].unique():
        data.insert(index,i,(data[feature]==i)+0) # +0 : to cast into an array of int
    del data[feature] # We no longer need that initial feature
    return data
