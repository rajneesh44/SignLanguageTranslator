import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBClassifier
import seaborn as sns
import pickle as pkl



#import the training data and test data
#train data

print('[NOTE] Importing data set....')
df_train = pd.read_csv('../input/american-sign-language-data/sign_mnist_train.csv')
df_test =  pd.read_csv('../input/american-sign-language-data/sign_mnist_test.csv')



# extract the import features from the dataset
# since the dataset is the pixel values of the images
# there are 784 pixel columns since the images are 28x28
# and one label column containing the label of the image
print('[NOTE] Extracting important features....')
useful_features = [c for c in df_train.columns if c not in ("label")]


# split the data into ytrain and xtrain
# and ytest and xtest

print('[NOTE] split data into labels and data....')
ytrain = df_train['label']
xtrain = df_train[useful_features]

ytest = df_test['label']
xtest = df_test[useful_features]


print('[NOTE] initializing the model....')
model = XGBClassifier(
    random_state = 42,
    tree_method = 'gpu_hist',         # change to 'gpu_hist' if using gpu, to use gpu optimized histogram algorithm
    #gpu_id = 0,                  # uncomment if using gpu
    predictor = "gpu_predictor",  # change it to 'gpu_predictor' if you have cuda supported gpu
    n_estimators=1000,
    n_jobs = -1
)


# training model
print('[NOTE] training the model....')
model.fit(xtrain, ytrain)

#saving the model
print('[NOTE] saving the model....')
model.save_model('sign_model.json')

# getting predictions on test set
print('[NOTE] getting the predictions.....')
preds = model.predict(xtest)

print('[NOTE] confusion matrix and classification report...')
cnf_mtx = confusion_matrix(ytest, preds)
plt.figure(figsize=(25,25))
sns.heatmap(cnf_mtx,annot=True)

print(classification_report(ytest, preds))