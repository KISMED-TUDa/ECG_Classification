from wettbewerb import load_references
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


# Load dataset with list of leads (numpy arr), labels (str), fs and ecg names
ecg_leads, ecg_labels, fs, ecg_names = load_references(folder="/Users/Vladi/Desktop/training/")


# Normalize ECG Leads to zero mean and unit variance
ecg_leads = [(ecg_lead - ecg_lead.mean()) / (ecg_lead.std() + 1e-08) for ecg_lead in ecg_leads]

# Transform list to dataframe for XGBoost
ecg_leads = pd.DataFrame(ecg_leads)

'''def counter():
    a_counter = 0
    n_counter = 0
    o_counter = 0
    tilde_counter = 0
    for label in ecg_labels:
        if label == 'A':
            a_counter+=1
        elif label == 'N':
            n_counter+=1
        elif label == 'O':
            o_counter+=1
        elif label == '~':
            tilde_counter+=1
    print(a_counter)
    print(n_counter)
    print(o_counter)
    print(tilde_counter)

counter()'''
# Encode string class labels to int
'''
Label encoding:
A = 0 521x
N = 1 3581x
O = 2 1713x
~ = 3 185x
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
param = {'max_depth': 10, 'eta': 0.1, 'gamma': 0, 'lambda': 1, 'objective': 'multi:softprob', 'num_classes': 4}


# Init classifier and fit model
classifier_xgb = xgb.XGBClassifier(param)
model = classifier_xgb.fit(X_train,
                           y_train,
                           verbose=True,
                           eval_metric='auc',
                           early_stopping_rounds=10,
                           eval_set=[(X_test, y_test)]
)


# Save result evaluation
eval_result = model.evals_result()

'''
# Init saved model
loaded_model = xgb.Booster()
# Load data
loaded_model.load_model('baseline_model')
'''

# Predict labels on training set
train_pred = model.predict(X_train)
print('\nTarget on train data', train_pred)

# Accuray Score on trainingset
train_accuracy = accuracy_score(y_train, train_pred)
print('\naccuracy_score on train dataset : ', train_accuracy)

'''
F1 Score calculated as:
F1 = 2 * (precision * recall) / (precision + recall)
'''

# Compute F1 Score on Training Set
f1_train = f1_score(y_train, train_pred, average='macro')
print('\nF1 Score on training dataset : ', f1_train)

# predict the Labels on the test dataset
test_pred = model.predict(X_test)
print('\nTarget on test data', test_pred)

# Accuracy Score on test dataset
test_accuracy = accuracy_score(y_test, test_pred)
print('\naccuracy_score on test dataset : ', test_accuracy)

# Compute F1 Score on Test Set
f1_test = f1_score(y_test, test_pred, average='macro')
print('\nF1 Score on test dataset : ', f1_test)

#model.save_model("baseline_model")


