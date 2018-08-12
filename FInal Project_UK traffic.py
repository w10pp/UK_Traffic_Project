# %%%%%%%%%%%%% Machine Learning Final Project%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Authors Group 1  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Mengjie Zhang  Email:mengjiezhang@gwmail.gwu.edu
# Xinning Wang   Email:
# Yilin Wang     Email:
# %%%%%%%%%%%%% Date:
# August - 11 - 2018

# Importing the required packages

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

##%%-----------------------------------------------------------------------
# I. Data Preprocessing
# load the data
df = pd.read_csv(
    '/Users/mengjie/Documents/2018Spring/6202MachineLearning/Machine-Learning/Final Project/accidents_2012_to_2014.csv',
    sep=',', header=0)
df.head()
df.shape
df.isnull().sum()

# drop variables that has too many NAs.
df.drop(['Junction_Detail', 'Junction_Control'], axis=1, inplace=True)

# Some features such as longitude, latitude, road class and number appears not useful in this analysis, so we drop them.
df.drop(['Accident_Index', 'Location_Easting_OSGR', 'Location_Northing_OSGR', 'Longitude', 'Latitude', 'Police_Force',
         'Date', 'Local_Authority_(District)', 'Local_Authority_(Highway)', '1st_Road_Class', '1st_Road_Number',
         '2nd_Road_Class', '2nd_Road_Number', 'LSOA_of_Accident_Location', 'Year'], axis=1, inplace=True)

# Since we would like to predict accident severity, we need to drop number of vehicles and casualties
# because they would be unknown until the accident happened.
df.drop(['Number_of_Vehicles', 'Number_of_Casualties'], axis=1, inplace=True)
# 755 NAs in Road_Surface_Condition, since we have over 4000,000 observation, we can safely drop rows with NAs.
df.dropna(inplace=True)

# convert Time to categorical (peak and non-peak time)
df['Hour'] = df.Time.str.slice(0, 2)
df.Hour = pd.to_numeric(df.Hour, errors='coerce')
df.drop('Time', axis=1, inplace=True)

df['Peak'] = df.Hour.map(lambda x: 1 if 7 <= x <= 10 or 16 <= x <= 19 else 0)

# the target Accident_Severity has 3 classes (1, 2, 3), we convert it to binary to make it simpler.
# 1, 2 -- not severe (0) , 3 -- severe(1)
df['Accident_Severity'] = df['Accident_Severity'].map(lambda x: 1 if x == 3 else 0)
df.groupby('Accident_Severity').size()
df.info()
df.describe(include='all')

## df = df.sample(frac=0.1, replace=True)


# encoding the features using get dummies
X_data = pd.get_dummies(df.iloc[:, 1:])
X = X_data.values

# Standard scaling
stdsc = StandardScaler()
stdsc.fit(X)
X = stdsc.transform(X)

y = df.values[:, 0].astype(int)

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)


##%%-----------------------------------------------------------------------
# II. Define function
# Helper method to print metric scores
def confusion_heatmap(y_test, y_test_pred):
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    class_names = df['Accident_Severity'].unique()
    df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    plt.figure(figsize=(5, 5))
    hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                     yticklabels=df_cm.columns, xticklabels=df_cm.columns)
    hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
    hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    plt.tight_layout()
    plt.show()


def get_performance_metrics(y_train, y_train_pred, y_test, y_test_pred, y_train_score, y_test_score, threshold=0.5):
    metric_names = ['AUC', 'Accuracy', 'Precision', 'Recall', 'f1-score']
    metric_values_train = [roc_auc_score(y_train, y_train_score),
                           accuracy_score(y_train, y_train_pred > threshold),
                           precision_score(y_train, y_train_pred > threshold),
                           recall_score(y_train, y_train_pred > threshold),
                           f1_score(y_train, y_train_pred > threshold)
                           ]
    metric_values_test = [roc_auc_score(y_test, y_test_score),
                          accuracy_score(y_test, y_test_pred > threshold),
                          precision_score(y_test, y_test_pred > threshold),
                          recall_score(y_test, y_test_pred > threshold),
                          f1_score(y_test, y_test_pred > threshold)
                          ]
    all_metrics = pd.DataFrame({'metrics': metric_names,
                                'train': metric_values_train,
                                'test': metric_values_test}, columns=['metrics', 'train', 'test']).set_index('metrics')
    print(all_metrics)
    print(classification_report(y_test, y_test_pred))
    print(confusion_matrix(y_test, y_test_pred))


def plot_roc_curve(y_train, y_train_score, y_test, y_test_score):
    roc_auc_train = roc_auc_score(y_train, y_train_score)
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_score)

    roc_auc_test = roc_auc_score(y_test, y_test_score)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_score)
    plt.figure()
    lw = 2
    plt.plot(fpr_train, tpr_train, color='green',
             lw=lw, label='ROC Train (AUC = %0.4f)' % roc_auc_train)
    plt.plot(fpr_test, tpr_test, color='darkorange',
             lw=lw, label='ROC Test (AUC = %0.4f)' % roc_auc_test)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def train_model(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_train_score = clf.predict_proba(X_train)[:, 1]

    y_test_pred = clf.predict(X_test)
    y_test_score = clf.predict_proba(X_test)[:, 1]

    get_performance_metrics(y_train, y_train_pred, y_test, y_test_pred, y_train_score, y_test_score, threshold=0.5)
    confusion_heatmap(y_test, y_test_pred)
    plot_roc_curve(y_train, y_train_score, y_test, y_test_score)

def train_model_SVC(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)

    # the LinearSVC doesn't support predict_proba method to calculate the class probabilities
    y_train_pred = clf.predict(X_train)
    #y_train_score = clf.predict_proba(X_train)[:, 1]
    y_train_score = clf.decision_function(X_train)

    y_test_pred = clf.predict(X_test)
    #y_test_score = clf.predict_proba(X_test)[:, 1]
    y_test_score = clf.decision_function(X_test)

    get_performance_metrics(y_train, y_train_pred, y_test, y_test_pred, y_train_score, y_test_score, threshold=0.5)
    confusion_heatmap(y_test, y_test_pred)
    plot_roc_curve(y_train, y_train_score, y_test, y_test_score)

##%%-----------------------------------------------------------------------
# III. Random Forest

clf_rf = RandomForestClassifier(n_estimators=100)
train_model(clf_rf, X_train, y_train, X_test, y_test)

##%%-----------------------------------------------------------------------
# IV. Neural network

clf_mlp = MLPClassifier(random_state=100)
train_model(clf_mlp, X_train, y_train, X_test, y_test)

##%%-----------------------------------------------------------------------
# V. Naive Bayes

clf_nb = GaussianNB()
train_model(clf_nb, X_train, y_train, X_test, y_test)

##%%-----------------------------------------------------------------------
# IV. SVM

#clf_svm = SVC(kernel='linear', probability=True)

clf_svm = LinearSVC(random_state=100)
train_model_SVC(clf_svm, X_train, y_train, X_test, y_test)


# due to large data size, try boosting ensemble methods
clf_boost = AdaBoostClassifier(LinearSVC(random_state=100), algorithm='SAMME')
train_model_SVC(clf_boost, X_train, y_train, X_test, y_test)