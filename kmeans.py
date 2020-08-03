"""
author: Arif Bashar

Use K Means clustering to classify data
"""

import sys
import numpy as np
import matplotlib.pyplot as plt


"""
Goes through all the rows in the entire data set
and picks k number of random points and returns them
in a list
"""
def initClust(data, rows, k):
    centroids = data[np.random.randint(0, rows, k)]
    return centroids


# Gets the euclidian distance between two points
def euclidian(a, b):
    return np.linalg.norm(a-b)


# Checks if a vector already exists in a cluster
def exists(vector, clusters):
    for key in clusters.keys():
        for list in clusters[key]:
            if all(elem in vector for elem in list):
                return True
    return False


# K-means clustering algorithm
def kmeans(data, centroids, k, cols):
    
    # Initializing numpy array so we can use it to hold old centroids
    oldCentroids = np.zeros(centroids.shape)

    while True:
        # Exit loop if updated centroid array == old centroid array
        if np.array_equal(centroids, oldCentroids):
            return centroids, clusters
        
        # Reset clusters dictionary on every iteration
        clusters = {c: [] for c in range(k)}
        
        # We will update centroids later in the loop
        oldCentroids = centroids

        """
        Iterate through each data point
        and calculate the distance between it and each k number of clusters
        """
        for vector in data:
            distances = np.zeros((k, 1))
            
            for index, centroid in enumerate(centroids):
                distances[index] = euclidian(centroid, vector)
                
            # If there is a duplicate vector, we do not want to add it to the cluster  
            if not exists(vector, clusters):
                minIndex = np.argmin(distances)
                clusters[minIndex].append(vector)
                
        # Buffer for holding our new centroids
        tempCentroids = np.zeros((k, cols))
        
        # Calculate the mean for each cluster and make the means the new centroids
        for cluster in range(len(centroids)):
            if clusters[cluster]:
                current = np.array(clusters[cluster])
                tempCentroids[cluster] = np.mean(current, axis=0)
            
        centroids = tempCentroids


# Get the final labels for each data point in each cluster
def getLabels(data, unlabeled, clusters, k):
	
	# Just create a new dictionary where each key corresponds
	# to each key in the clusters dictionary
	labels = {c: [] for c in range(k)}
	
	for key in clusters.keys():
		if clusters[key]:
			for list in clusters[key]:
				listedIndex = np.where((unlabeled == list).all(axis=1))
				index = listedIndex[0][0]
				labels[key].append(data[index][-1])
	return labels


# Return a list of majority labels in each cluster
def getMajority(labels):
	majority = []
	for key in labels.keys():
		if labels[key]:
			majority.append((np.argmax(np.bincount(labels[key]))))
		else:
			majority.append(None)
	return majority


# Run predictions on our testing data
def predict(testingData, centroids, labels, k):
	predictions = []
	majorityLabel = getMajority(labels)
	for vector in testingData:
		distances = np.zeros((k, 1))
		
		for index, centroid in enumerate(centroids):
			distances[index] = euclidian(centroid, vector)
			
		minIndex = np.argmin(distances)
		predictions.append(majorityLabel[minIndex])
	return np.asarray(predictions)


# Calculate the accuracy of our predictions with the actual testing labels
def getAccuracy(testingData, centroids, labels, k, testingLabels):
	predictions = predict(testingData, centroids, labels, k)
	return np.sum(predictions == testingLabels)
	
	
def main():
	seed = int(sys.argv[1])
	k = int(sys.argv[2])
	trainDataName = (sys.argv[3])
	testDataName = (sys.argv[4])
	
	train = np.loadtxt(trainDataName)
	test = np.loadtxt(testDataName)

	if len(train.shape) < 2:
		train = np.array([train])
	if len(test.shape) < 2:
		test = np.array([test])
	
	# Setting the seed for random
	np.random.seed(seed)
    
	# We want to pass unlabeled data, so remove last column
	unlabeled = train[:,:-1]
	unlabeledTest = test[:,:-1]
	
	# We want to run testing data labels against predictions later
	testLabels = test[:,-1]
	
	# Number of rows and columns in the unlabeled training set
	rows, cols = unlabeled.shape

	# Get random initial k centroid points
	centroids = (initClust(unlabeled, rows-1, k))
	
	# Get back our final centroids and clusters
	finalCentroids, clusters = kmeans(unlabeled, centroids, k, cols)
	
	# Find out the labels for each data point in each cluster
	labels = getLabels(train, unlabeled, clusters, k)
	
	# Get accuracy of our predictions
	accuracy = getAccuracy(unlabeledTest, finalCentroids, labels, k, testLabels)
	print(accuracy)
	
main()