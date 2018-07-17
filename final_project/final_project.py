# credits Cleiton Lima

'''
1. Select the 10 best features from the dataset (you can use, for example, SelectKBest)
2. Choose 3 classifier algorithms
3. Define a set of hyperparameters to be used in the model (at least, 3 hyperparametes), and for each one, choose at least 2 different values
4. Execute GridSearchCV to find the best combination for the classifiers (work with 5 cross validation approach)
5. Run the classifiers with the hyperparameters found using a new subset (unseen examples) - we can use 10% from the original dataset
6. Evaluate your models using classification_report from metrics (show the results)
'''
#from sklearn.preprocessing import Imputer
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier

import numpy as np
import pandas as pd

# classification algorithms from sklearn
etc = ExtraTreeClassifier()
dtc = DecisionTreeClassifier()
lda = LinearDiscriminantAnalysis()
classification_list = [etc, dtc, lda]

# ignore warnings of sklearn (there still runtime error on division fom sklearn, selectkbest)
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# creating tuples to know what is represented by numbers
# and what is represented by strings to change everything to numbers
names = [
    ('encounter_id', 'numeric'),
    ('patient_nbr', 'numeric'),
    ('race', 'nominal'),
    ('gender', 'nominal'),
    ('age', 'nominal'),
    # 'weight',  # attribute removed
    ('admission_type_id', 'nominal'),
    ('discharge_disposition_id', 'nominal'),
    ('admission_source_id', 'nominal'),
    ('time_in_hospital', 'numeric'),
    # 'payer_code',  # attribute removed
    ('medical_specialty', 'nominal'),
    ('num_lab_procedures', 'numeric'),
    ('num_procedures', 'numeric'),
    ('num_medications', 'numeric'),
    ('number_outpatient', 'numeric'),
    ('number_emergency', 'numeric'),
    ('number_inpatient', 'numeric'),
    ('diag_1', 'nominal'),
    ('diag_2', 'nominal'),
    ('diag_3', 'nominal'),
    ('number_diagnoses', ''),
    ('max_glu_serum', 'nominal'),
    ('A1Cresult', 'nominal'),
    ('metformin', 'nominal'),
    ('repaglinide', 'nominal'),
    ('nateglinide', 'nominal'),
    ('chlorpropamide', 'nominal'),
    ('glimepiride', 'nominal'),
    ('acetohexamide', 'nominal'),
    ('glipizide', 'nominal'),
    ('glyburide', 'nominal'),
    ('tolbutamide', 'nominal'),
    ('pioglitazone', 'nominal'),
    ('rosiglitazone', 'nominal'),
    ('acarbose', 'nominal'),
    ('miglitol', 'nominal'),
    ('troglitazone', 'nominal'),
    ('tolazamide', 'nominal'),
    ('examide', 'nominal'),
    ('citoglipton', 'nominal'),
    ('insulin', 'nominal'),
    ('glyburide-metformin', 'nominal'),
    ('glipizide-metformin', 'nominal'),
    ('glimepiride-pioglitazone', 'nominal'),
    ('metformin-rosiglitazone', 'nominal'),
    ('metformin-pioglitazone', 'nominal'),
    ('change', 'nominal'),
    ('diabetesMed', 'nominal'),
    ('readmitted', 'nominal')
]

# cleaning the dataset
def clean_dataset(dataset):
    # put NaN where there are no value
    dataset = dataset.replace('?', np.NaN)

    # removed 'weight' and 'payer_code' because of the lack of values
    dataset.drop(['weight', 'payer_code'], inplace = True, axis = 1)

    # included the value "missing" in the label 'medical_specialty' where the value is null
    dataset['medical_specialty'].fillna('missing', inplace = True)

    # the null values of this features was changed by the most common value of that label
    dataset['race'].fillna(dataset['race'].mode()[0], inplace = True)
    dataset['diag_1'].fillna(dataset['diag_1'].mode()[0], inplace = True)
    dataset['diag_2'].fillna(dataset['diag_2'].mode()[0], inplace = True)
    dataset['diag_3'].fillna(dataset['diag_3'].mode()[0], inplace = True)

    # transforming the nominal values to numerical ones
    numeric_attrs = [n[0] for n in names if n[1] == 'numeric']
    nominal_attrs = [n[0] for n in names if n[1] == 'nominal']
    dataset_num = dataset[numeric_attrs]
    dataset_nom = dataset[nominal_attrs]
    dataset_nom = dataset_nom.apply(LabelEncoder().fit_transform)
    dataset = pd.concat([dataset_num, dataset_nom], axis = 1)

    # keep just the first internation, dump the others
    dataset = dataset.drop_duplicates('patient_nbr', keep = 'first')

    # print(type(dataset))
    # # changing NaN values to the median of that label (because NaN gives much headaches)
    # imp = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 1)
    # dataset = pd.DataFrame(imp.fit_transform(dataset))
    # print(type(dataset))

    return dataset

###################################################
print("The algorithms chosen were Extra Tree Classifier, Decision Tree Classifier and Linear Discriminant Analysis.")
# load dataset
print("Cleaning dataset...")
data = pd.read_csv('diabetic_data.csv', sep=',')
dataset = clean_dataset(data)
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
print("Done.")

# selecting the 10 best features using sklearn function
print("Selecting 10 best features...")
x = SelectKBest(k = 10).fit_transform(x, y)
# creating subsets for training and tests
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .1)
print("Done.")

# hyper parameters:
# Extra Tree Classifier
etc_hp = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'min_samples_split': [2, 3]
}

# Decision Tree Classifier
dtc_hp = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_features': ['auto', 'log2'],
    'random_state' : [2015412]
}

# Linear Discriminant Analysis
lda_hp = {
    'solver': ['lsqr', 'eigen'],
    'shrinkage': ['auto', 0.5],
    'n_components': [1, 3]
}

hyperparameters_list = [etc_hp, dtc_hp, lda_hp]

# run the list of algorithms
print("Running classification algorithms...")
for clf, hparams in zip(classification_list, hyperparameters_list):
    grid = GridSearchCV(clf, hparams, cv = 5)
    print('Running {} algorithm...'.format(clf.__class__.__name__)) # print the name of the algorithm
    # train the model
    model = grid.fit(x_train, y_train)
    print('Best hyperparametes:')
    print(model.best_estimator_)
    # test with predicted values
    y_hat = model.predict(x_test)
    print('Results:')
    print(classification_report(y_test, y_hat))
    print('\n')
