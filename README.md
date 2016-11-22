# challenge-ml

A machine learning challenge from the INF554 course

## What has been done

### Preprocessing

* Loading the data with both read_csv (pandas) and genfromtxt (numpy) function
    * *As suggested in the assignement, we will probably choose the read_csv function*
* Casting some features as categorical ("SPLIT_COD","ACD_COD","WEEK_END", "TPER_HOUR", "DAY_OFF")
* Dummy coding for the 'TPER_TEAM' feature (adding two columns : 'Jours' and 'Nuit')
* Create time features as suggested in the assignement:
    * time since epoch
    * time since start of day
    * time as a categorical feature (maybe already available in "TPER_HOUR" feature)
    * month
    * week day
    * week end : "WEEK_END" is already available as a feature
    * night/day : "Jours/Nuit" is already available as a feature
    * “day off” : "DAY_OFF" is already available as a feature
    * holidays (TODO)
* Normalizing the data
* Create DataProcess class. The huge size of the train data implies to separate the pipeline:
    * we don't want to preprocess **each time** we run our main function
    * the DataProcess class enables to **export in csv** format data already processed
    * importing the newly created file enables us not to compute this preprocess task again

## What do we predict

* One week per month for one year (year 2013)
* List of the intervals to predict:
    * 12/28 : 01/03 (2012 -> 2013)
    * 02/02 : 08/02 (2013)
    * 06/03 : 12/03 (2013)
    * 10/04 : 16/04 (2013)
    * 13/05 : 19/05 (2013)
    * 12/06 : 18/06 (2013)
    * 16/07 : 22/07 (2013)
    * 15/08 : 21/08 (2013)
    * 14/09 : 20/09 (2013)
    * 18/10 : 24/10 (2013)
    * 20/11 : 26/11 (2013)
    * 22/12 : 28/12 (2013)
* Only 18 assignments are necessary!

## What's next

* Dimensionality reduction