# Visualisation

## Tree regression

Just an insight on how decision trees could work in our case. This is for the data average on a semi-hourly basis. 

![1](https://github.com/Edouard360/challenge-ml/blob/master/graphs/1-TreeRegression.png)

## Partial dependence

With the set of graphs below, we can already spot some correlations. *x-axis* is the feature considered, *y-axis* is the number of calls.

On Mondays, the average number of calls is sligthly higher, whereas it's lower on Saturdays, and even worse on Sundays. 

![2](https://github.com/Edouard360/challenge-ml/blob/master/graphs/2-TimeFeatureDependance.png)

## Comparing data for 'Services' and 'Médical' ASS_ASSIGNEMENT

We could first try to separate these two categories doing an LDA.

Let's plot some features averaged per the hour of the day:

![3](https://github.com/Edouard360/challenge-ml/blob/master/graphs/3-AveragePerDay.png)

Let's plot some features averaged per the day of the week:

![4](https://github.com/Edouard360/challenge-ml/blob/master/graphs/3-AveragePerWeek.png)

For doing the LDA, I chose the 5 best features using `SelectKBest(chi2, k=5)`, and then I tried to find the first two best axis for projecting the data `LinearDiscriminantAnalysis(solver = "eigen",n_components=2)`. 

The separation is bad, but we can see that Médical has more variance than Services.

![5](https://github.com/Edouard360/challenge-ml/blob/master/graphs/3-LDA.png)

Maybe we should get rid of the values that accumulate on a line (probably the multiple 0 values)...