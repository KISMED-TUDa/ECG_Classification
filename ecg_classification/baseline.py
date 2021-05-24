from wettbewerb import load_references
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Load dataset with list of leads (numpy arr), labels (str), fs and ecg names
ecg_leads, ecg_labels, fs, ecg_names = load_references(folder="/Users/Vladi/Desktop/training/")

# Transform list to dataframe for XGBoost
ecg_leads = pd.DataFrame(ecg_leads)

# Encode string class labels to int
'''
Label encoding:
A = 0
N = 1
O = 2
~ = 3
'''
label_encoder = preprocessing.LabelEncoder()
label_encoder = label_encoder.fit(ecg_labels)
ecg_labels = label_encoder.transform(ecg_labels)


# Split data into training and validation set
X_train, X_test, y_train, y_test = train_test_split(ecg_leads, ecg_labels, test_size=0.2, random_state=42)

# Init train and test set
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Specify params for xgboost
param = {'max_depth': 10, 'eta': 0.1, 'gamma': 0, 'lambda': 1, 'objective': 'multi:softmax', 'num_classes': 4}

# Init classifier and fit model
classifier_xgb = xgb.XGBClassifier(param)
classifier_xgb.fit(X_train,
                   y_train,
                   verbose=True,
                   eval_set=[(X_test,y_test)])