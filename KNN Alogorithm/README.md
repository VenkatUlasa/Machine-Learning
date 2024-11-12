# 
## K-Nearest Neighbors (KNN)

The K-Nearest Neighbors (KNN) algorithm is a simple and widely used supervised machine learning algorithm for both classification and regression tasks. It works by finding the "K" training examples that are closest to a new data point and makes predictions based on those nearest neighbors.

![image alt](https://dataaspirant.com/wp-content/uploads/2016/12/Knn-Introduction.jpg)


# How K-Nearest Neighbors Works
### Choose the Number of Neighbors (K): 
K is a user-defined integer that specifies how many neighbors (data points) to consider for making the prediction. A small value of K (e.g., 1 or 3) makes the model sensitive to noise, while a large K value might make the model too smooth.
![image alt](https://miro.medium.com/v2/resize:fit:828/format:webp/1*o65yIH9dbf48dXyQxgBTpw.png)

### Compute Distance: 
For a new data point, KNN calculates the distance between this point and all points in the training dataset. Common distance metrics include:
![image alt](https://miro.medium.com/v2/resize:fit:828/format:webp/1*CCZt1t82V15ITnGzQH6qOg.png)

### Identify Nearest Neighbors: 
After calculating the distances, KNN identifies the K points in the training set that are closest to the new data point.

### Make Prediction:

##### Classification: 
The algorithm assigns the class that is most common among the K neighbors (majority vote).
![image alt](https://intuitivetutorial.com/wp-content/uploads/2023/04/knn-1.png)

from the above image we can observe that the Query datapoint joining with "Blue Dots". because,the mejority of nearest points of Query datapoint is "Blue Dots".

with using the Model we can predict the values of training data. after that we need to find the Accuracy to check the model is giving best results or not.

### Accuracy :



