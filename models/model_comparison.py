### MODULE 1 - data clean and model comparison

# Imports data structures
import pandas as pd
import numpy as np

# Imports statistical models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

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

	# imports and splits data into cross-validation sets
	from sklearn.cross_validation import train_test_split

	X_train, X_test, y_train, y_test = train_test_split(X, y)
	return X_train, X_test, y_train, y_test

def model_select(model, X_train, X_test, y_train, y_test, pickle_model=None):
	"""Input model and model parameters"""
	"""Output fitted and scored model"""

	"""
	PARAMETERS
	pickle_model: when set to a pathway inclduing file name, will save a pickled model
	
	"""
	try:	
		# Fits and transforms model
		if model.__class__.__name__ == LinearRegression().__class__.__name__:
			# Fits model
			model.fit(X_train, y_train)

			# 
			# Prints the results
			print model.__class__.__name__, 'Accuracy Is'
			print model.score(X_test, y_test)
			print 
			print "Intercept"
			print model.intercept_
			print
			print "Coefficients"
			print model.coef_
			print
		else: 
			# Fits model and transforms data
			model.fit(X_train, y_train)

			# Prints the results
			print model.__class__.__name__, 'Accuracy Is'
			print model.score(X_test, y_test)
			print 
			print "Feature Importances"
			print model.feature_importances_
			print
	 	
	 	# pickles model if param is set to pathway
		if pickle_model != None:
			with open(pickle_file, 'wb') as f:
				pickle.dump(data, f)

	except DeprecationWarning:
		import ipdb; ipdb.set_trace()


if __name__ == '__main__':
	
	df = import_clean_csv('../data/rosetta-ann_pyformat.csv')

	X_train, X_test, y_train, y_test = split_data(df)

	lr = LinearRegression(n_jobs=5)
	rf = RandomForestRegressor(n_estimators=50)
	gbr = GradientBoostingRegressor()

	# Fits, scores and prints feature importances
	# NOTE: add a pathway to the pickle_model parameter to create a pickled model file
	model_select(lr, X_train, X_test, y_train, y_test, pickle_model=None)  
	model_select(rf, X_train, X_test, y_train, y_test, pickle_model=None)
	model_select(gbr, X_train, X_test, y_train, y_test, pickle_model=None)

