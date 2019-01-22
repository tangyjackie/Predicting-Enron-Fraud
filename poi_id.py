#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import train_test_split

from sklearn.grid_search import GridSearchCV
from matplotlib import pyplot as plt
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list =  ['poi', 'salary', 'deferral_payments', 'total_payments', 
'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 
'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 
'director_fees', 'to_messages', 'from_poi_to_this_person', 
'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] 
# You will need to use more features

#lasso regression and regularization already account for overfitting with many features,
#decision trees (requires feature selection shown in feature selection section, using only most important features)
#try using text learning to predict poi at end
#run lasso regression on all features and see what happens
#try naive bayes


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

import pandas as pd
en_data = pd.DataFrame(data_dict)
en_data = en_data.transpose()
print "The number of data points in data set is", len(en_data)
is_poi = en_data[(en_data["poi"] == 1)]
is_not_poi = en_data[(en_data["poi"] == 0)]
	
### Task 2: Remove outliers

#remove the obvious outlier called total, which should be removed as it is just a sum of each individual financial features
data_dict.pop('TOTAL', 0)
#manual inspection shows that this entity is not a person and should be included in the data
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)


#quick transform of 'NaN' into None for Pandas summary description
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pprint

#dat_dict_pd = [{k: data_dict[p][k] if data_dict[p][k] != 'NaN' else None for k in data_dict[p].keys()} for p in data_dict.keys()]
#dat_pd = pd.DataFrame(dat_dict_pd)
#dat_pd.boxplot('total_payments', by='poi')
#plt.show()

#The code below converts data_dict into a data frame in order to easily access different variables in the features_list
enron_data = pd.DataFrame(data_dict)
enron_data = enron_data.transpose()
enron_data.replace(['NaN'], [None], inplace=True)
#drop email_address as this variable is not used at all and is a text string
enron_data = enron_data.drop('email_address', 1)

#DATA EXPLORATION
print "The number of data points in data set is", len(enron_data)
is_poi = enron_data[(enron_data["poi"] == 1)]
is_not_poi = enron_data[(enron_data["poi"] == 0)]
print "The number of pois in the data set is", len(is_poi)
print "The number of non-pois in the data set is", len(is_not_poi)
#After exploring the data set, it 

#start visualizing some trends between different key variables
#first one is a plot between bonus and total_payments
#colour is also coded for poi and non-poi
#red represents poi (True), blue represents non-poi (False)

#Function below will create plots based on inputs specified
#Purpose of creating these plots to see any general outliers amongst different variables

def view_outliers(x_var, y_var, group):
	cmap = {1 :"red", 0 : "blue"}
	_, ax = plt.subplots()
	for key,group in enron_data.groupby(group):
		group.plot.scatter(ax=ax, x=x_var, y=y_var, label=key, color = cmap[key]);
	plt.xlabel(x_var)
	plt.ylabel(y_var)
	plt.show()
#The code below shows plots across different variables in the features_list grouped by poi
#view_outliers("bonus", "total_payments", "poi")
#view_outliers("expenses", "total_stock_value", "poi")
#view_outliers("bonus", "deferral_payments", "poi")
#view_outliers("salary", "loan_advances", "poi")
#view_outliers("from_messages", "from_poi_to_this_person", "poi")
#view_outliers("to_messages", "from_this_person_to_poi", "poi")

#Outlier Investigation Explanation: 
#1) It seems like there might be an outlier after looking at total_payments
#on closer inspection, we find that this person is a poi and is Kenneth Lay, one of the main people involved in the Enron scandal
print enron_data['total_payments'].loc[enron_data['total_payments'] == enron_data['total_payments'].max()]

#seems like there might be an outlier after looking at loan_advances
#on closer inspection, we find that this person is a poi and is Kenneth Lay, one of the main people involved in the Enron scandal
print enron_data['loan_advances'].loc[enron_data['loan_advances'] == enron_data['loan_advances'].max()]

#Seems like the number of from_messages for Wincenty J Kaminski may be a bit high, however, this could be a plausible number depending on his role in the company
#I have decided to keep it, because although it is an outlier, it is not extraordinarily high
print enron_data['from_messages'].loc[enron_data['from_messages'] == enron_data['from_messages'].max()]



### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

###EXPLANATION: CHOICE OF NEW FEATURES
#One new feature below is the proportion of from_messages that are from_poi_to_this_person
#The rationale behind this variable is that a higher proportion of one's inbox messages coming from pois might indicate
#that the individual might be more likely to be a poi

#Another new feature below is the proportion of to_messages that are from_this_person_to_poi
#Again, more sent messages to pois as a proportion of total sent messages might indicate increased likelihood to be a poi

#A good financial feature might be the proportion of salary to total_payments
#obviously, if this value is very low, then there is a higher likelihood that this person is in a higher-up senior-level position
#This is because other areas such as bonus, loan advances etc. are more likely to be a part of the total compensation package which usually tend to be higher for senior-level people whom may be pois

#One final feature that was created was the sent_received_ratio which looks at the ratio of messages from poi to this person 
#over messages sent from this person to a poi
#A higher ratio might indicate that more messages were sent to a poi which would indicate that the person themselves might be more likely to be a more involved poi in the fraud case
enron_data['prop_from_poi'] = enron_data['from_poi_to_this_person']/enron_data['from_messages']
enron_data['prop_to_poi'] = enron_data['from_this_person_to_poi']/ enron_data['to_messages']
enron_data['total_comp_salary_ratio'] = enron_data['salary']/ enron_data['total_payments']
enron_data['sent_received_ratio'] = enron_data['from_poi_to_this_person']/ enron_data['from_this_person_to_poi']

#Once again, look at the new features to see if there are any outliers
#view_outliers("total_payments", "sent_received_ratio", "poi")
#view_outliers("prop_from_poi", "prop_to_poi", "poi")


#This to check for any potential outliers resulting from the creation of these new variables.
#print enron_data['prop_from_poi'].loc[enron_data['prop_from_poi'] == enron_data['prop_from_poi'].max()]
print enron_data['total_comp_salary_ratio'].loc[enron_data['total_comp_salary_ratio'] == enron_data['total_comp_salary_ratio'].max()]

#First, transform features list back from enron_df now the new features have been added
features_list = list(enron_data.columns.values)

#Switch the position of poi which will be our target variable back to the first variable on the list
features_list = features_list[12:] + features_list[:12]
print features_list

#Looking at the data closely, it seems there are a lot of NaN values in both the email and financial features
print enron_data.head()

#HOW TO HANDLE MISSING VALUES
#I have decided to make an assumption that where financial features are NaN, that they would be zero.
#Taking the mean would skew the values in favour of potential POI classification bias.
#I am assuming that most likely those NaN values for financial features were 0.
#For email features, the median was taken, as it would be improbable that there would be no interaction between non-pois and pois (ex. consider mass company emails)


#create new copy first to work with
enron_im = enron_data


#replace email features with median values
enron_im['from_messages'].fillna(enron_im['from_messages'].median(), inplace=True)
enron_im['to_messages'].fillna(enron_im['to_messages'].median(), inplace=True)
enron_im['from_poi_to_this_person'].fillna(enron_im['from_poi_to_this_person'].median(), inplace=True)
enron_im['from_this_person_to_poi'].fillna(enron_im['from_this_person_to_poi'].median(), inplace=True)

enron_im['sent_received_ratio'].fillna(enron_im['sent_received_ratio'].median(), inplace=True)
enron_im.fillna(0, inplace=True)

#drop inf values for sent_received_ratio variable and for total_comp_salary_ratio
from numpy import inf

enron_im[enron_im['sent_received_ratio'] == inf] = 0
enron_im[enron_im['sent_received_ratio'] == -inf] = 0
enron_im[enron_im['total_comp_salary_ratio'] == inf] = 0
enron_im[enron_im['total_comp_salary_ratio'] == -inf] = 0

#FURTHER DATA EXPLORATION
print enron_im.head()
#After exploring the data set for a bit and having looked at the graphs/plots,
#It seems that variables with a lot of NaN variables are financial variables such as director_fees, loan_advances, and deferral_payments
#This is most likely  because non-pois whom may be at a less senior level would have fewer compensation options/privileges
#Other variables that were found to be NaN were from_poi_to_this_person and from_this_person_to_poi
#and this is most likely that these individuals are not important or high-level enough to deal with pois on a regular basis
#NaN values in these variables mentioned above most likely signify that they are more likely to be non-pois
#However, as explained above, for non-financial features, we took the median as it prevented skewing bias to either zero or very high values
#As mentioned before, there is simply no info for us to make a judgment on whether emails would have been sent or not
#Most likely there would have been at least some communicaton (at least in form of mass emails) between pois and non-pois whom may be more junior-level

###This is the enron data set with imputations having been made
imputed_data = enron_im.to_dict(orient = 'index')


#Below is an overfit decision tree being used in order to gauge feature_importances_ for feature selection

def feature_importance(my_dataset, features_list):
	#The code below calculates the feature importance
	#The inputs are a list of features (features_list) and a data set
	#A random forest is created and calculates the features' importance, and then a list with the feature importancee is returned
	
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.metrics import accuracy_score
	from sklearn.cross_validation import train_test_split
	
	data = featureFormat(my_dataset, features_list, sort_keys = True)
	labels, features = targetFeatureSplit(data)
    
	# new features filtered, NaN values removed
	features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    
	clf = DecisionTreeClassifier()
	clf.fit(features_train, labels_train)
	pred = clf.predict(features_test)
	acc = accuracy_score(labels_test, pred)
	#print "overfitted accuracy", acc
	# The code below is to calculate the importance of each feature in predicting POI
	feature_importance = clf.feature_importances_
	return feature_importance

features_importance_list = []
for i in range(100):
	features_importance = feature_importance(imputed_data, features_list)
	features_importance_list.append(features_importance)
	
#This is to extract all the features except poi
feature_variables = features_list[1:]
feat_imp_dict = {}

for i in feature_variables:
	feat_imp_dict[i] = []

for array in features_importance_list:
	for i, j in enumerate(array):
		feat_imp_dict[feature_variables[i]].append(j)

		
#The threshold of 0.1 was chosen, because thresholds that included other variables or excluded some
# of the existing variables gave lower precision, recall and accuracy levels		
most_important_feature = {}
threshold = 0.05
# getting the average of all runs and storing the ones passing the threshold in a dictionary
for key, value in feat_imp_dict.items():
    mean = np.asarray(value).mean()
    print key, ":", mean
    if mean > threshold:
        most_important_feature[key] = mean

pprint.pprint(most_important_feature)

#once we have the most important features that we want to include, let us update this list
features_list = ['poi'] + most_important_feature.keys()

# This is to prepare the data into training and test sets after we have created the new and updated features list 
enron_data1 = featureFormat(imputed_data, features_list, sort_keys = True)
labels, features = targetFeatureSplit(enron_data1)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

#put all features into a list for easy access later on
list_all_features = [features_train, features_test, labels_train, labels_test]



#Now, the next step is to produce a confusion matrix in order to calculate TP, TN, FP, FN etc.
#This would involve writing a loop that creates a count for each category and then displaying it in a table or list.
#This should be in the form of a function which is later called after the predictions have been made

#create a counter from which each time if statement falls true, this counter is increased by one to 
#arrive at the total number of these values


#the true labels come from labels_test column, while predictions will come from pred column in
def confusion_matrix(real_labels, predictions):
	TN = 0
	TP = 0
	FN = 0
	FP = 0
	zipped = zip(predictions, real_labels)
	#for i in zipped:
	#	if i[0] - i[1] == 0 and i[0] == 1 and i[1] == 1:
	#		TP += 1
	#	if i[0] - i[1] == 1:
	#		TN += 1
	#	if i[1] - i[0] == 1:
	#		FP += 1
	#	elif i[0] - i[1] == 0 and i[0] == 0 and i[1] == 0:
	#		FN += 1
	for prediction, truth in zipped:
		if prediction == 1 and truth == 1:
			TP += 1
		if prediction == 0 and truth == 1:
			FN += 1
		if prediction == 1 and truth == 0:
			FP += 1
		elif prediction == 0 and truth == 0:
			TN += 1
	print """Confusion matrix:                  
              predicted class
              _Yes_|__No_
actual | Yes |  {0}  |  {1}
class  | No  |  {2}  |  {3}""".format(TP, FN, FP, TN)

#This is a function to calculate f1, precision and recall scores
#This function is called after the predictions have been made

def calculate_scores(real_labels, predictions):
	from sklearn.metrics import precision_score
	from sklearn.metrics import recall_score
	from sklearn.metrics import f1_score
	precision = precision_score(real_labels, predictions)
	recall = recall_score(real_labels, predictions)
	f1_score =  f1_score(real_labels, predictions)
	return """precision: {0}
	recall:    {1}
	f1_score:  {2}""".format(precision, recall, f1_score)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#Once the enron_data1 data set has been created and split into the training and test sets (both labels and features)
#Then create the classifier function where the predictions and fit is done
#dataset input should be the list_all_features data set we created earlier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def fit_classify(ml_technique, dataset):
	
	from sklearn.metrics import accuracy_score
	
	features_train = dataset[0]
	features_test = dataset[1]
	labels_train = dataset[2]
	labels_test = dataset[3]
	

	#classify based on the ML technique (ex. SVC, GaussianNB etc.)
	ml_technique.fit(features_train, labels_train)
	pred = ml_technique.predict(features_test)
	
	#find the accuracy from using the ML technique
	acc = accuracy_score(labels_test, pred)
	print "The accuracy is", acc
	
	print confusion_matrix(labels_test, pred)
	print calculate_scores(labels_test, pred)

def fit_classify_scaled(ml_technique, dataset):
#need to scale data for both k-nearest neighbours and svc
	from sklearn.preprocessing import MinMaxScaler
	from sklearn.metrics import accuracy_score
	from sklearn.pipeline import Pipeline
	
	features_train = dataset[0]
	features_test = dataset[1]
	labels_train = dataset[2]
	labels_test = dataset[3]
	
	scaler = MinMaxScaler()
	features_train = scaler.fit_transform(dataset[0])
	features_test = scaler.transform(dataset[1])
	
	clf = Pipeline(steps=[('scaler', MinMaxScaler()), ('ml_technique', ml_technique)])
	
	clf.fit(features_train, labels_train)
	pred = clf.predict(features_test)
	
	#find the accuracy from using the ML technique
	acc = accuracy_score(labels_test, pred)
	
	print "The accuracy is", acc
	print confusion_matrix(labels_test, pred)
	print calculate_scores(labels_test, pred)

#Function is written below to perform the classifications using all the ML techniques
#training_test_list should be "list_all_features" which includes poi and all our most important features for predicting poi

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
	
print "\n Naive Bayes"
fit_classify(GaussianNB(), list_all_features)

print "\n Decision Trees"
fit_classify(DecisionTreeClassifier(), list_all_features)
	
print "\n Random Forests"
fit_classify(RandomForestClassifier(), list_all_features)
	
print "\n K-nearest neighbours"
fit_classify_scaled(KNeighborsClassifier(), list_all_features)

print "\n Support vector machines"
fit_classify_scaled(SVC(), list_all_features)
	
print "\n Linear regression"
fit_classify(LogisticRegression(), list_all_features)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.cross_validation import StratifiedShuffleSplit
from feature_format import featureFormat, targetFeatureSplit

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

def test_classifier(clf, dataset, feature_list, folds = 1000):
	data = featureFormat(dataset, feature_list, sort_keys = True)
	labels, features = targetFeatureSplit(data)
	cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
	true_negatives = 0
	false_negatives = 0
	true_positives = 0
	false_positives = 0
	for train_idx, test_idx in cv: 
		features_train = []
		features_test  = []
		labels_train   = []
		labels_test    = []
		for ii in train_idx:
			features_train.append( features[ii] )
			labels_train.append( labels[ii] )
		for jj in test_idx:
			features_test.append( features[jj] )
			labels_test.append( labels[jj] )
			
		### fit the classifier using training set, and test on test set
		clf.fit(features_train, labels_train)
		predictions = clf.predict(features_test)
		for prediction, truth in zip(predictions, labels_test):
			if prediction == 0 and truth == 0:
				true_negatives += 1
			elif prediction == 0 and truth == 1:
				false_negatives += 1
			elif prediction == 1 and truth == 0:
				false_positives += 1
			elif prediction == 1 and truth == 1:
				true_positives += 1
			else:
				print "Warning: Found a predicted label not == 0 or 1."
				print "All predictions should take value 0 or 1."
				print "Evaluating performance for processed predictions:"
				break
	try:
		total_predictions = true_negatives + false_negatives + false_positives + true_positives
		accuracy = 1.0*(true_positives + true_negatives)/total_predictions
		precision = 1.0*true_positives/(true_positives+false_positives)
		recall = 1.0*true_positives/(true_positives+false_negatives)
		f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
		f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
		print clf
		print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
		print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
		print ""
	except:
		print "Got a divide by zero when trying out:", clf
		print "Precision or recall may be undefined due to a lack of true positive predicitons."

#Decision after trying out several thresholds was that a feature importance of 0.1 as the minimum was the best
#Using this 0.1 threshold, 3 variables: prop_from_poi, shared_receipt_poi, and exercised_stock_options 
	
#Naive bayes has a much higher precision when tuned, though recall is slightly less (declining from 0.285 to 0.235
test_classifier(GaussianNB(), imputed_data, features_list, folds = 3000)

#Decision trees when tuned give precision and recall of 0.3 and above
#when not tuned, the values are actually higher
test_classifier(DecisionTreeClassifier(random_state = 42), imputed_data, features_list, folds = 1000)

#tuning helps when using kNN as recall is substantially better, though precision declines significantly
test_classifier(KNeighborsClassifier(), imputed_data, features_list, folds = 3000)

#Support vector machines do not work for tuning, true positives do not exist with SVMs
test_classifier(SVC(), imputed_data, features_list, folds = 1000)

#No need for tuning in order to obtain precision and recall of 0.3 and above
#tuning actually reduces the recall significantly
test_classifier(RandomForestClassifier(random_state = 42), imputed_data, features_list, folds = 1000)

test_classifier(LogisticRegression(), imputed_data, features_list, folds = 1000)


#Note that for decision trees and random forests, the results are always different
#This is because different partitions of decision boundaries are used




#answer question from Understanding the dataset and question
#Try gridesearchcv and answer why parameter tuning is important
#addresses why validation is important, talk about precision and recall
from sklearn.svm import SVC

#Tuning SVM
print "\n Support vector machines"
parameters = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1]}
sv = SVC(kernel = 'rbf')
#cv = StratifiedShuffleSplit(test_size = 0.2, random_state = 42)
clf_svc = GridSearchCV(sv, parameters, scoring = "f1")
fit_classify_scaled(clf_svc, list_all_features)
#clf_svc.fit(features_train, labels_train)
sv_params = clf_svc.best_params_
print sv_params


#Going through the StratifiedShuffleSplit and then GridSearchCV did not give better results
#clf2 = GridSearchCV(sv, parameters, scoring = "f1")
#test_classifier(clf2, imputed_data, features_list, folds = 1000)

#Tuning Decision trees
from sklearn.tree import DecisionTreeClassifier
print "\n Tuning Decision Trees"
parameters = {'max_depth' : [None, 10, 5, 2],
			'min_samples_split' : [2, 10, 5],
			'min_samples_leaf' : [1, 5, 2],
			'min_weight_fraction_leaf' : [0, 0.25, 0.5],
			'random_state': [42]}
dt = DecisionTreeClassifier()
#cv = StratifiedShuffleSplit(test_size = 0.2, random_state = 42)
clf_dt = GridSearchCV(dt, parameters, scoring = "f1")
#I tried to run the StratifiedShuffleSplit before its fit by using the test_classifier function, but results are same as before
#Both recall and precision were above 0.3 using the StratifiedShuffleSplit before the fit of the best parameters
#test_classifier(clf_dt, imputed_data, features_list, folds = 1000)
fit_classify(clf_dt, list_all_features)
best_clf = clf_dt.best_estimator_
dt_params = clf_dt.best_params_
print dt_params


#Tuning Random Forests
from sklearn.ensemble import RandomForestClassifier
print "\n Tuning Random Forests"
rf = RandomForestClassifier()
clf_rf = GridSearchCV(dt, parameters, scoring = "f1")
#I tried to run the StratifiedShuffleSplit before its fit by using the test_classifier function, but results are same as before
#Both recall and precision were above 0.3 using the StratifiedShuffleSplit before the fit of the best parameters
#test_classifier(clf, imputed_data, features_list, folds = 1000)
fit_classify(clf_rf, list_all_features)
#best_clf = clf_rf.best_estimator_
#clf_rf.fit(features_train, labels_train)
rf_params = clf_rf.best_params_
print rf_params

#Going through the StratifiedShuffleSplit does not give desirable results for the recall value
#test_classifier(clf, imputed_data, features_list, folds= 1000)
#Use the tuned features calculated from the GridSearchCV to fit using Random Forests
#clf_rf = RandomForestClassifier(min_samples_split = 5, min_weight_fraction_leaf= 0, max_depth = 10, min_samples_leaf = 1)
#fit_classify(clf_rf, list_all_features)

#Now, time to tune KNN algorithm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
print "\n Tuning K nearest neighbors"
parameters = {'n_neighbors' : [1, 2, 3, 5, 10],
              'weights' : ['uniform', 'distance'],
              'algorithm' : ['ball_tree', 'kd_tree', 'brute'],
              'leaf_size' : [10, 30, 10, 60, 100],
              }
np.random.seed(42)
knn = KNeighborsClassifier()
clf_knn = GridSearchCV(knn, parameters, scoring = "f1", cv = 3)
fit_classify_scaled(clf_knn, list_all_features)
#best_clf = clf_knn.best_estimator_
#clf_knn.fit(features_train, labels_train)
knn_params = clf_knn.best_params_
print knn_params

#clf_knn = KNearestNeighbors(min_samples_split = 2, min_weight_fraction_leaf= 0, max_depth = None, min_samples_leaf = 1)
#Post-Analysis and write-up on importance of validation and tuning


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#replace clf with the classifier type SVC() GaussianNB(), my_dataset as imputed_data, features_list as poi + the selected ones
#dump_classifier_and_data(clf_rf, imputed_data, features_list)
#dump_classifier_and_data(clf_svc, imputed_data, features_list)
dump_classifier_and_data(best_clf, imputed_data, features_list)
#dump_classifier_and_data(clf_dt, imputed_data, features_list)