import glob
import random

import cv2
import matplotlib.pyplot as plt
from seqeval.metrics import accuracy_score
from sklearn.cluster import KMeans

from LM import LMFilter
import numpy as np

from Schmid import Schmid


def randomSeed():
  return 0.2

def load_dataset():
    dataset = {
        'Bricks': [],
        'Grass': [],
        'Gravel': []
    }
    for img in glob.glob('dataset/*'):
        dataset[img.split('\\')[1].split('_')[0]].append(img.split('\\')[1])

    for key in dataset.keys():
        random.shuffle(dataset[key], randomSeed)

    clssToID = {
        'Bricks': 0,
        'Grass': 1,
        'Gravel': 2
    }
    train_images = []
    train_labels = []
    for key in dataset.keys():
        train_images.extend(dataset[key][0:7])
        train_labels.extend([clssToID[key]] * 7)

    return train_images, train_labels

def get_image(id):
    image = cv2.cvtColor(cv2.imread(f'dataset//{id}'), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    return image

def purity_score(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

n_filter = 12
schmid = Schmid()
filter_bank = schmid.make_filter_bank()
kernels = []
for t in range(12):
    kernels.append(filter_bank[:,:,t])

imgs, labels = load_dataset()
imgs_features = np.zeros([21,n_filter])
for i,label in enumerate(labels):
    f_vector = np.zeros([1,1])
    for filter in kernels:
        f = cv2.filter2D(cv2.cvtColor(get_image(imgs[i]), cv2.COLOR_BGR2GRAY),-1,filter).mean()
        f_vector = np.concatenate((f_vector, f), axis=None)
    f_vector = np.delete(f_vector,0,0)
    imgs_features[i,:] = f_vector


purities = []
for k in range(1,10):
    print(k)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(imgs_features)
    purities.append(purity_score(np.array(labels), np.array(kmeans.labels_)))

plt.plot(range(1,10), purities)
plt.title("Schmid Bank")
plt.xlabel('k')
plt.ylabel('purity')
plt.show()