# Breast Cancer Diagnosis Using K-Nearest Neighbors (KNN) Algorithm
This project focuses on developing a predictive model for diagnosing breast cancer based on cell features. Using the K-Nearest Neighbors (KNN) algorithm, this model classifies breast tumors as either malignant or benign based on a set of features derived from cell nuclei in a digitized image of a fine needle aspirate (FNA) of a breast mass. The goal is to create a reliable diagnostic tool to assist in the early detection and treatment of breast cancer, potentially aiding healthcare providers in making more informed decisions.
<p align="center">
  <img src="https://wisdomml.in/wp-content/uploads/2023/04/breast_cancer.png" alt="Sample Image" width="800" height="400">
</p>

## K-Nearest Neighbors (KNN)

The K-Nearest Neighbors (KNN) algorithm is a simple and widely used supervised machine learning algorithm for both classification and regression tasks. It works by finding the "K" training examples that are closest to a new data point and makes predictions based on those nearest neighbors.

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*pP0zfIM915q4qudY5DkHKg.png" alt="Sample Image" width="500" height="300">
</p>

# How K-Nearest Neighbors Works
### Choose the Number of Neighbors (K): 
K is a user-defined integer that specifies how many neighbors (data points) to consider for making the prediction. A small value of K (e.g., 1 or 3) makes the model sensitive to noise, while a large K value might make the model too smooth.
<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*o65yIH9dbf48dXyQxgBTpw.png" alt="Sample Image" width="400" height="300">
</p>

### Compute Distance: 
For a new data point, KNN calculates the distance between this point and all points in the training dataset. Common distance metrics include:
<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*CCZt1t82V15ITnGzQH6qOg.png" alt="Sample Image" width="500" height="250">
</p>

### Identify Nearest Neighbors: 
After calculating the distances, KNN identifies the K points in the training set that are closest to the new data point.

### Make Prediction:

##### Classification: 
The algorithm assigns the class that is most common among the K neighbors (majority vote).
<p align="center">
  <img src="https://intuitivetutorial.com/wp-content/uploads/2023/04/knn-1.png" alt="Sample Image" width="600" height="300">
</p>
from the above image we can observe that the Query datapoint joining with "Blue Dots". because,the mejority of nearest points of Query datapoint is "Blue Dots".

with using the Model we can predict the values of training data. after that we need to find the Accuracy to check the model is giving best results or not.

### Accuracy :

Accuracy= Total Number of True Predictions / Total Number of all Predictions

##### Accuracy = (TP+TN) / (TP+TN+FP+FN)
<ul>
<li>TP (True Positives): Correctly predicted positive cases</li>
<li>TN (True Negatives): Correctly predicted negative cases</li>
<li>FP (False Positives): Incorrectly predicted positive cases (type I error)</li>
<li>FN (False Negatives): Incorrectly predicted negative cases (type II error)</li>
</ul>

### Classification Report(Precision, Recall, F1-score) :

A Classification Report provides a detailed evaluation of a classification model, showing various metrics such as precision, recall, F1-score, and support. These metrics are useful for understanding how well a model performs across different classes, especially in the context of imbalanced datasets.

The key metrics in a classification report are:
#### Precision: 
The ratio of correctly predicted positive observations to the total predicted positives. It tells you how many of the predicted positives are actually positive.
<p align="center">
  <img src="https://miro.medium.com/max/700/1*pDx6oWDXDGBkjnkRoJS6JA.png" alt="Sample Image" width="400" height="100">
</p>
#### Recall :
The ratio of correctly predicted positive observations to all observations in the actual class. It tells you how many of the actual positives were correctly predicted by the model.
<p align="center">
  <img src="https://images.prismic.io/encord/3c0173c9-409e-4f84-a53f-7073ea00bca9_Recall+-+Mathematical+Formula+-+Encord.png?auto=compress,format" alt="Sample Image" width="400" height="100">
</p>

#### F1-Score: 
The harmonic mean of precision and recall. It is useful when you need to balance precision and recall and is especially valuable when the class distribution is imbalanced.
<p align="center">
  <img src="https://images.prismic.io/encord/0ef9c82f-2857-446e-918d-5f654b9d9133_Screenshot+%2849%29.png?auto=compress,format" alt="Sample Image" width="400" height="100">
</p>

## Skills :
<ul>
  <li>Python</li>
  <li>Machine Learning</li>
  <li>Statistics</li>
  <li>Mathematics</li>
  <li>Numpy</li>
  <li>Pandas</li>
  <li>scikit-learn</li>
</ul>

## Conclusion
This K-Nearest Neighbors (KNN) classification implementation demonstrates a practical approach to solving the breast cancer dataset classification problem. By using the sklearn library, the KNN algorithm is efficiently applied to predict whether a tumor is malignant or benign based on features like radius, texture, and perimeter.

The KNN algorithm is a powerful and easy-to-understand classification method. However, its performance can be influenced by the choice of k, as well as the scale of the input data. For better results in practice, it is recommended to perform scaling (e.g., using StandardScaler) to normalize the feature values before applying KNN.

## Support :
For support, You can Contact : email josh8008venkat13@gmail.com.

## 🔗 Links
You can follow my Profile on ⬇️


[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/venkat-ulasa-645445178?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BspH0MRWkRUyNOsMr2a%2Bfkw%3D%3D)



