import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis, skew
import os
from wordcloud import WordCloud

def compute_image_features(image_path):
    print(f"Processing image: {image_path}")
    # Load the image using OpenCV
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # add preprocessing here: resize, normalize, etc.
        # depending on the tests 
        
        # Compute the mean and variance of the image
        mean = np.mean(image)
        variance = np.var(image)

        # Compute the median of the image
        median = np.median(image)

        # Flatten the image into a 1D array
        flattened_image = image.flatten()

        # Compute the kurtosis and skewness of the image
        kurt = kurtosis(flattened_image)
        skewness = skew(flattened_image)

        # Return the computed features
        return mean, variance, median, kurt, skewness
    except Exception as e:
        print(f"Error processing image: {image_path}")
        print(e)
        return None

def load_filenames(folder_path):
    # Get the list of filenames in the specified folder
    filenames = []
    tmp_list = os.listdir(folder_path)
    for f in tmp_list:
            if f.endswith('.jpg') or f.endswith('.png'):
                filenames.append(f)
    # Return the list of filenames
    return filenames

def compare_images(image1_params, image2_params):
    diff = np.subtract(image1_params, image2_params)

    # Return the sum of the absolute differences
    return diff

folder_path = 'images2'
# Compute image features for each image in images_paths
images_paths = load_filenames(folder_path)
simple_db_features = []
for image_path in images_paths:
    features = compute_image_features(os.path.join(folder_path, image_path))
    with open(f'desc/{image_path}.txt', 'r') as file:
        description = file.read()
    simple_db_features.append({'params': features, 'class': description, 'path': image_path})

print(simple_db_features)

query_image_path = 'query.png'
query = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
query_features = compute_image_features(query_image_path)
print(compare_images(query_features,query_features))
results = []
for image in simple_db_features:
    print(f"Comparing {query_image_path} with {image['path']}")
    diff = compare_images(query_features, image['params'])
    print(f'diff: {diff}')
    sum_diff = np.sum(np.abs(diff))
    print(f'sum_diff: {sum_diff}, kurtosis: {diff[3]}')
    results.append((1.0-np.abs(diff[3]), image['class']))
    
results.sort(key=lambda x: x[0], reverse=True)
print(results)

import matplotlib.pyplot as plt

# Generate bar chart
kurtosis_values = [result[0] for result in results[:3]]
class_labels = [result[1]*100 for result in results[:3]]

plt.figure(figsize=(8, 6))
plt.bar(class_labels, kurtosis_values)
plt.xlabel('Class')
plt.ylabel('Kurtosis')
plt.title('Top 3 Results - Kurtosis Comparison')
plt.show()

# Generate pie chart
class_counts = {}
for result in results[:3]:
    class_label = result[1]
    if class_label in class_counts:
        class_counts[class_label] += 1
    else:
        class_counts[class_label] = 1

class_labels = list(class_counts.keys())
class_values = list(class_counts.values())

plt.figure(figsize=(8, 6))
plt.pie(class_values, labels=class_labels, autopct='%1.1f%%')
plt.title('Top 3 Results - Class Distribution')
plt.show()

# Generate word cloud
wordcloud_text = ' '.join([image['class'] for image in simple_db_features])
wordcloud = WordCloud(width=800, height=400).generate(wordcloud_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Class Descriptions')
plt.show()