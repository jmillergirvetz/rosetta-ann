### MODULE 1 - data clean and model comparison ###

# Imports data structures
import pandas as pd
import numpy as np

# Imports train and test splits for cross-validation
from sklearn.cross_validation import train_test_split

# Imports statistical models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Imports pickle module for data storage files
import cPickle as pickle

# Imports warnings module
import warnings


def import_clean_csv(filename):
	"""Input filename of csv file"""
	"""Output cleaned dataframe"""
	# Imports csv file
	df = pd.read_csv(filename, sep=',')

	# Drops categorical, collinear and id columns
	df = df.drop(['Unnamed: 0', 'Code', 'USDAsoiltexture', '%Silt', 'Sand_Clay'], axis=1) 

	# Returns a randomly shuffled dataframe
	return random_generation(df)

def random_generation(df):
	"""Input dataframe before shuffle"""
	"""Output shuffled dataframe"""
	return df.iloc[np.random.permutation(len(df))]

def split_data(df):
	""""Input dataframe"""
	"""Output cross_validation sets"""
	# Creates X feature and y output dataframes
	y = df.pop('(cmd-1)_Ks')
	X = df

	# Creates train and test cross-validation sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
	return X_train, X_test, y_train, y_test

def model_select(model, X_train, X_test, y_train, y_test, pickle_model=None):
	"""Input model and model parameters"""
	"""Output fitted and scored model"""

	"""
	PARAMETERS

	pickle_model: when set to a pathway inclduing file name, will save a pickled model
	"""
	try:	
		# Checks if linear regression model
		if model.__class__.__name__ == LinearRegression().__class__.__name__:

			# Fits model
			model.fit(X_train, y_train)

			# Prints the results and 
			print model.__class__.__name__, 'Accuracy Is'
			print model.score(X_test, y_test)
			print 
			print "Intercept"
			print model.intercept_
			print
			print "Coefficients"
			print np.array(zip(df_columns, model.coef_))
			print
		else: 
			# Fits model 
			model.fit(X_train, y_train)

			# Prints the results
			print model.__class__.__name__, 'Accuracy Is'
			print model.score(X_test, y_test)
			print 
			print "Feature Importances"
			print np.array(zip(df_columns, model.feature_importances_))
			print
	 	
	 	# pickles model if param is set to pathway
		if pickle_model != None:
			with open(pickle_model, 'wb') as f:
				pickle.dump(data, f)

	except DeprecationWarning:
		import ipdb; ipdb.set_trace()

	except RuntimeWarning:
		import ipdb; ipdb.set_trace()


if __name__ == '__main__':

	# Creates cleaned and randomized dataframe	
	df = import_clean_csv('../data/rosetta-ann_pyformat.csv')

	# Creates a column list to map coefficients and feature importances
	df_columns = df.columns

	# Prints numer of features
	print 'Number of features: ', len(df_columns) - 1

	# Creates cross-validation sets
	X_train, X_test, y_train, y_test = split_data(df)

	# Instantiates models
	lr = LinearRegression()
	rf = RandomForestRegressor(n_estimators=10)
	gbr = GradientBoostingRegressor()

	# Fits, scores and prints feature importances
	# NOTE: add a pathway to the pickle_model parameter to create a pickled model
	model_select(lr, X_train, X_test, y_train, y_test, pickle_model=None)  
	model_select(rf, X_train, X_test, y_train, y_test, pickle_model=None)
	model_select(gbr, X_train, X_test, y_train, y_test, pickle_model=None)

