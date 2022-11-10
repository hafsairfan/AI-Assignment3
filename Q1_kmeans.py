"""
CS 351 - Artificial Intelligence 
Assignment 3, Question 1

Student 1(Name and ID): Aliza Lakhani-al05435
Student 2(Name and ID): Hafsa Irfan-hi05946

"""
import numpy as np
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image

class KMeansClustering:
    
    def __init__(self, filename: str, K:int):
        self.image = mpimg.imread(filename)     
        self.K = K
        
        self.centroids = []
        self.clusters = {}
        self.labels = []
    def generate_initial_centroids(self) :  # return list
        shape = self.image.shape[0]
        for i in range(self.K): 
            random_var1 = random.randint(0, shape - 1)
            random_var2 = random.randint(0, shape - 1)
            self.centroids.append(self.image[random_var1][random_var2])

        for cen in range(len(self.centroids)):
            self.centroids[cen] = self.centroids[cen].tolist()
            self.clusters[cen] = []
        return(self.centroids)

    def calculate_distance(self, p1: tuple, p2: tuple):   #-> float
        return (np.sqrt(np.sum((p1 - p2)**2)))
 
    def assign_clusters(self):  # return dict; assign each data point to its nearest cluster (centroid)
        self.labels = []
        self.clusters = {}
        for cen in range(len(self.centroids)):
            self.clusters[cen] = []
        for i in self.image:
            for j in i:
                min_distance = 100000000
                centroid = 0
                value = 0
                for k in range(self.K):
                    distance = self.calculate_distance(j, self.centroids[k])
                    if  distance < min_distance:
                        min_distance = distance
                        centroid = k
                        value = j
                self.clusters[centroid].append(j)
                self.labels.append((j, centroid))
    def recompute_centroids(self):  # list: your code here to return new centroids based on cluster formation
        for key in self.clusters:
            self.centroids[key] = np.average(self.clusters[key], axis = 0)
        for cen in range(len(self.centroids)):
            self.centroids[cen] = self.centroids[cen].tolist()
    def apply(self):  #your code here to apply kmeans algorithm to cluster data loaded from the image file.
        iterations = 0
        self.generate_initial_centroids()
        while(iterations != 5):
            self.assign_clusters()
            self.recompute_centroids()
            iterations = iterations + 1
                        
    def save_image(self):  #This function overwrites original image with segmented image to be shown later.
        x, y = self.image.shape[1], self.image.shape[0]
        new_image = Image.new("RGB", size = (x, y))
        pixels = new_image.load()
        for i in range(len(self.image)):
            for j in range(i):
                for k in range(len(self.labels)):
                    print(k)
                    if (self.labels[k][0] == self.image[i][j]).any():
                        pixels[i,j] = (int(self.centroids[self.labels[k][1]][0]) , int(self.centroids[self.labels[k][1]][1]) ,int(self.centroids[self.labels[k][1]][2])) 
        self.image = new_image
    def show_result(self):
        self.save_image()
        plt.imshow(self.image)
        plt.show()
    def print_centroids(self):  #This function prints all centroids formed by Kmeans clustering
       print(self.centroids)
        
kmeans = KMeansClustering("images\sample1.jpg", 5)
kmeans.apply()
kmeans.show_result()
kmeans.print_centroids()  
