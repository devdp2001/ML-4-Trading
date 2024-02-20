""""""  		  	   		  		 		  		  		    	 		 		   		 		  
"""  		  	   		  		 		  		  		    	 		 		   		 		  
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		  		 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		  		 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		  		 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		  		 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 		  		  		    	 		 		   		 		  
or edited.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		  		 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		  		 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		  		 		  		  		    	 		 		   		 		  
"""

import numpy as np
from scipy import stats

class RTLearner(object):
    """
    This is a Random Tree Learner. It is implemented correctly.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """

    def __init__(self, leaf_size=1, verbose=False):
        """
        Constructor method
        """

        self.leaf_size = leaf_size
        self.verbose = verbose
        # pass  # move along, these aren't the drones you're looking for

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "dpatel426"  # replace tb34 with your Georgia Tech username

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """

        self.tree = self.build_tree(data_x, data_y)

    def query(self, points):
        return self.queryRow(points, 0)

    def queryRow(self, points, row = 0):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """

        total_points = points.shape[0]
        predictions = np.empty(total_points, float)
        for point in range(total_points):
            while ~np.isnan(self.tree[int(row), 0]):
                if points[point, int(self.tree[int(row), 0])] <= self.tree[int(row), 1]:
                    row += int(self.tree[int(row), 2])
                else:
                    row += int(self.tree[int(row), 3])
            predictions[point] = self.tree[int(row), 1]
            row = 0

        return predictions

    def build_tree(self, data_x, data_y):

        if data_x.shape[0] <= self.leaf_size:
            return np.asarray([np.nan, np.mean(data_y), np.nan, np.nan])
        if np.all(np.isclose(data_y, data_y[0])):
            return np.asarray([np.nan, data_y[0], np.nan, np.nan])
        else:
            #Random feature chosen to split on
            totalFeatures = data_x.shape[1]
            index = np.random.randint(0, totalFeatures)
            SplitVal = np.median(data_x[:,index])
            left = (data_x[:,index] <= SplitVal)
            right = ~left

            if np.all(np.isclose(left, left[0])):
                return np.asarray([np.nan, np.mean(data_y), np.nan, np.nan])
            lefttree = self.build_tree(data_x[left], data_y[left])
            righttree = self.build_tree(data_x[right], data_y[right])
            if lefttree.ndim != 1:
                root = np.asarray([index, SplitVal, 1, lefttree.shape[0] + 1])
            else:
                root = np.asarray([index, SplitVal, 1, 2])

            return np.vstack((root, lefttree, righttree))

if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
