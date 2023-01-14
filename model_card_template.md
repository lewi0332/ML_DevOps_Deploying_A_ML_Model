# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The Census Income classification model predicts an individuals propensity to 
have an income greater or less than $50k based on recorded census data. 

The model uses the [Random Forrest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) model from the scikit-learn library. This is a decision tree based model that uses averages of random samples of the dataset to train the model. 

## Intended Use

The model is intended to be used to make individual predictions when the following census features are passed: 

| **Column Name** | **Data Type** |
| :-------------: | :-----------: |
|       age       |    integer    |
|    workclass    |    string     |
|      fnlgt      |    integer    |
|    education    |    string     |
|  education-num  |    integer    |
| marital-status  |    string     |
|   occupation    |    string     |
|  relationship   |    string     |
|      race       |    string     |
|       sex       |    string     |
|  capital-gain   |    integer    |
|  capital-loss   |    integer    |
| hours-per-week  |    integer    |
| native-country  |    string     |
|     salary      |    string     |

## Training Data

The model was trained on 26048 rows of census data acquired from the UCI Machine Learning datasets. 

## Evaluation Data

Evaluation was performed on 6513 rows of data from the same dataset as acquired in training. This was a split of 20% of the full dataset at random. 



## Metrics
_Please include the metrics used and your model's performance on those metrics._

The model was scored using typical classification metrics on the evaluation dataset described above. 

- **Precision** - This the number of true positives divided by the number of true positives plus the number of false positives. In other words, how good is the model at correctly guessing the positive cases. 
- **Recall** - Recall is the inverse of precision also know as sensitvity and quantifies the number of positive class predictions made out of all positive examples. 
- **Fbeta** - The Fbeta classification score is a configurable single-score metric for evaluating a binary classification model based on the predictions made for the positive class. It is calculated using precision and recall, and is the weighted harmonic mean of the precision and recall.

| **Precision** | **Recall** | **Fbeta** |
| :-----------: | :--------: | :-------: |
|     0.757     |   0.525    |   0.62    |

## Ethical Considerations

This is self reported information from the census and has opportunity for inaccuracy. 

Predicting an individual's propensity for income is based on a lifelong journey that has had historically been affected by outside factors not present in the data. Even though race, gender and other cultural features are not used to inform the prediction, these bias may be present in secondary data such as location or education.  

## Caveats and Recommendations

This project was built to learn the specifics of deploying a machine learning model with production ready techniques. The model's performance was not greatly considered and should not be used to *actually* make any predictions. It is recommended that you enjoy the component pieces of Continuos Integraion with Github Actions, Data Version control, FastAPI creation and 
Continuous Deployment with Heroku. 