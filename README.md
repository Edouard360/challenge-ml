# challenge-ml

A machine learning challenge from the INF554 course

I forked the scikit-learn repository [here](https://github.com/Edouard360/scikit-learn/). Clone or download it. Build on your machine by doing `cd scikit-learn` and `make`. 
After this, you can either put the sklearn repository at the root of your project, or temporarily replace the sklearn repo.

If you want to see the exact changes, you can go [here](https://github.com/Edouard360/scikit-learn/commit/b00d6f00c3a207fdfe1bb5a5717deeeda0e1a7a8).

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
    * holidays 
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

## Integrating the Linex function

After digging into scikit-learn, I found that the quickest answer to this problem was to add my functions alongside the code. I think the best practice in this case would have been to fork the sklearn repo...

#### Explaining **/help/hackGradientBoosting**:

**LinexEstimator**:

* **fit**: provides one optimal prediction value, for the leaf of the tree (a leaf only predicts one value).

* **predict**: gives a vector of the optimal value computed in fit. We are at the leaf, so the vector is uniform.

**LinexLossFunction**:

 * **\_\_call\_\_**: magic method. Computes the error according to linex function.

 * **negative\_gradient**: gives the negatives gradient of the function (cf. gradient boosting)

  * grad\_factor is a scaling term for the gradient to be significant. (Here grad\_factor = 100)

 * **\_update\_terminal\_region**: Update on leaf to predict its optimal value (*each leaf* predicts only *one value*) for our LinexEstimator. Exactly the same way as LinexEstimator does.

#### Explaining **/help/hackCriterion**:

You will have to recompile sklearn for this to take effect.

Under the hood, the regression trees are built according to a regression criterion, that is **variance-oriented**.

That is far from **how we would like the trees to be built**, ie using a criterion suited to our case, that is, to the linEx loss.

Recomputing impurity according to this loss, gives **significant improvement** in our final linEx score.

This is simply because the tree is built accordingly, seeking to **minimize** that loss.



## What's next

* Handling day-offs (preprocess). ==> Etienne (can easily be done by adding some lines to the holydays csv file)
* Adaboost tests, combined with our optimized trees. Finding the right formula for updating weights - requires computing power ==> Etienne

___

* Dig in the RandomForest code (it looks like RandomForest only take 'mse' as a criterion) ==> Raph
* Autoregressive models (Time series analysis), as suggested by the project ==> Raph
* Maybe SVR (support vector regression) ==> Raph, as you suggested
