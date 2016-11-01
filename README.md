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

## What's next

* Dimensionality reduction