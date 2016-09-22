### MODULE 1 - data clean and model comparison ###

# Imports data structures
import pandas as pd
import numpy as np

# Imports scoring metrics
from sklearn.metrics import mean_squared_error

# Imports train and test splits for cross-validation
from sklearn.cross_validation import train_test_split

# Imports statistical models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Imports grid search used to determine optimal paramters
from sklearn.grid_search import GridSearchCV

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

def model_select(model, X_train, X_test, y_train, y_test, param_grid=None, pickle_model=None):
	"""Input model and model parameters"""
	"""Output fitted and scored model"""

	"""
	PARAMETERS

	pickle_model: when set to a pathway inclduing filename, will save a pickled model

	param_grid: when set to a parameter grid, returns a grid searched model with \
	optimal parameters
	"""
	try:	
		# Checks if linear regression model
		if model.__class__.__name__ == LinearRegression().__class__.__name__:

			# Fits model
			model.fit(X_train, y_train)

			# Prints the results and 
			print model.__class__.__name__, 'R^2 Is'
			print model.score(X_test, y_test)
			print 
			print "Intercept"
			print model.intercept_
			print
			print "Coefficients"
			print np.array(zip(df_columns, model.coef_))
			print

			# Calculates mean squared error and root mean squared error with predictions
			print "MSE Score"
			mse_score = mean_squared_error(y_test, model.predict(X_test))
			print mse_score
			print
			print "RMSE Score"
			print mse_score**0.5
			print
		else: 

			# Instantiates model with grid_search
			clf = GridSearchCV(model, param_grid=param_grid, cv=5)

			# Fits model 
			clf.fit(X_train, y_train)

			# Prints the results
			print model.__class__.__name__, "R^2 Is"
			print clf.score(X_test, y_test)
			print  
			print "Grid Searched Parameters"
			print convert_dict_2_arr(clf.best_params_)
			print

			# Calculates mean squared error and root mean squared error with predictions
			print "MSE Score"
			mse_score = mean_squared_error(y_test, clf.predict(X_test))
			print mse_score
			print 
			print "RMSE Score"
			print mse_score**0.5
			print
	 	
	 	# pickles model if param is set to pathway
		if pickle_model != None:
			with open(pickle_model, 'wb') as f:
				pickle.dump(data, f)

	except DeprecationWarning:
		import ipdb; ipdb.set_trace()

	except RuntimeWarning:
		import ipdb; ipdb.set_trace()

def convert_dict_2_arr(d):
	"""Input dictionary with keys as strings and values as lists"""
	"""Output numpy ndarray displaying grid searched parameters"""
	arr1 = np.array([d.keys()])
	arr2 = np.array([d.values()])
	return np.concatenate((arr1.T, arr2.T), axis=1)


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
	rf = RandomForestRegressor()
	gbr = GradientBoostingRegressor()

	# Creates parameter grids for lr, rf, gbr
	rf_parameters = {"n_estimators": [1, 2, 3], 
				  "max_depth": [1, 2, 5],
                  "max_features": [1, 2, 5],
                  "min_samples_split": [1, 3, 5],
                  "min_samples_leaf": [1, 3, 5],
                  "bootstrap": [True, False]}

	gbr_parameters = {}

	# Fits, scores and prints feature importances
	# NOTE: add a pathway to the pickle_model parameter to create a pickled model
	model_select(lr, X_train, X_test, y_train, y_test, pickle_model=None)  
	model_select(rf, X_train, X_test, y_train, y_test, param_grid=rf_parameters, \
				 pickle_model=None)
	#model_select(gbr, X_train, X_test, y_train, y_test, pickle_model=None)

