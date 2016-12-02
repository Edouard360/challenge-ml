# Preprocessing output

This folder is the container for all the preprocessed data, be it the submission file (submission.txt), or the train data (train_2011_2012_2013.csv). 

They are the result of running the main.py script in the preprocess folder.

You might have to add: 

```
PYTHONPATH=$PYTHONPATH:`dirname $PWD` python main.py
```

Please check in the path.py file (preprocess folder), than the path to the files actually correspond to their location on your computer.

