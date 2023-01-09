# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The Census Income classification model predicts an individuals propensity to have an income
greater or less than $50k based on recorded census data. 

The model uses the [Random Forrest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) model from the scikit-learn library. This is a decision tree based model that uses averages of random samples of the dataset to train the model. 

## Intended Use

The model is intended to be used to make either batch or individual predictions given the census features. 

## Training Data

The model was trained on 26048 rows of census data acquired from the UCI Machine Learning datasets. 

## Evaluation Data

Evaluation was performed on 6513 rows of data from the same dataset as acquired in training. This was a split of 20% of the full dataset at random. 

## Metrics
_Please include the metrics used and your model's performance on those metrics._

## Ethical Considerations

This is self reported information from the census and has opportunity for inaccuracy. 

To predict an individuals propensity for income is based on a lifelong journey that has had historically been affected by outside factors. 

## Caveats and Recommendations
