import numpy as np
import math

def adaboost_fit(X, y, n_clf):
	n_samples, n_features = np.shape(X)
	w = np.full(n_samples, (1 / n_samples))
	clfs = []
	# iterate through classifer
	for c in range(n_clf):
		clf = {}
		min_error = float('inf') 

		# iterate through features
		for i_feature in range(n_features):
			feature_v = np.expand_dims(X[:,i_feature],axis =1)
			unique_v = np.unique(feature_v)

			# set the feature values as threshold
			for threshold in unique_v:
				polarity = 1
				prediction = np.ones(np.shape(y))
				prediction[X[:,i_feature] < threshold] = -1
				error = sum(w[y != prediction])
				# if error over 0.5 then change the polarity
				if error > 0.5 :
					error = 1 - error
					polarity = -1
				# find a minum error then update
				if error < min_error :
					min_error = error
					clf['polarity'] = polarity
					clf['threshold'] = threshold
					clf['feature_index'] = i_feature

		# use the min_error of the classifer to calculate the alpha
		alpha = 0.5 * math.log((1.0-min_error)/(min_error+1e-10))
		clf['alpha'] = alpha
		
		# update weights
		prediction = np.ones(np.shape(y))
    # polarity == -1 means oppsite the prediction will get a better performance
		if clf['polarity'] == -1:
			prediction[X[:,clf['feature_index']] >= clf['threshold']] = -1
		else :
			prediction[X[:,clf['feature_index']] < clf['threshold']] = -1
		w = w * np.exp(-alpha * y * prediction)
		w = w/sum(w)
		clfs.append(clf)

	return clfs
