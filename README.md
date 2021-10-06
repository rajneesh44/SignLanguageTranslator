# American Hand Sign Dataset

This project contains the model trained on the American Hand Sign Dataset.

### Dataset
This is an open source dataset taken from kaggle.

Data is in CSV format where both the training set and test set,\
have 785 columns. First column is for the labels of the alphabets,\
with the rest of the columns being the pixel values of the images.

You can find the dataset here. [Kaggle](https://www.kaggle.com/datamunge/sign-language-mnist).

*The dataset only contains 24 classes of alphabets as compared to 26,*\
*the data for J and Z alphabets is missing.*


### Model and Performance
The model used in this project is a XGBClassifier.\
The model gave an Accuracy of 78% with a macro avg F1 score of 0.76.


### Requirements
run the following code to install all the required packages.
```
pip install requirements.txt
```
