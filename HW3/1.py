import os

import cv2
import matplotlib.pyplot as plt
import glob
import random

import numpy as np


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
        train_images.extend(dataset[key][0:5])
        train_labels.extend([clssToID[key]] * 5)

    test_images = []
    test_labels = []
    for key in dataset.keys():
        test_images.extend(dataset[key][5:7])
        test_labels.extend([clssToID[key]] * 2)

    return train_images, test_images, train_labels, test_labels

def get_image(id):
    image = cv2.cvtColor(cv2.imread(f'dataset//{id}'), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    return image

def plot_dataset(dataset, title, name=None):
    c = 0
    fig = plt.figure()
    for i,img in enumerate(dataset):
        plt.subplot(3, 5, c+1)
        c += 1
        plt.imshow(get_image(img))
        if name != None:
            plt.title(name[i])
        plt.axis('off')
    plt.tight_layout()
    fig.suptitle(title)

def create_Gabor_Filter_Bank():
    kernels = []
    kernels_parameters = []
    for theta in [0,np.pi/4, np.pi/2, 6*np.pi/8 ,np.pi]:
        for sigma in [1, 1.5]:
            for gamma in [0.4, 0.8]:
                for lambd in [10]:
                    kernel = np.real(cv2.getGaborKernel([20,20], sigma=sigma, theta=theta, lambd=lambd,gamma=gamma))
                    kernels.append(kernel)
                    kernels_parameters.append('t'+str(theta)[0:4] + '_s' + str(sigma) + '_g' + str(gamma) + '_l' + str(lambd))
    return kernels,kernels_parameters

def display_images(h, w, images, names, title):
    fig = plt.figure()
    for i in range(h * w):
        plt.subplot(h, w, i+1)
        plt.imshow(images[i],cmap='gray')
        plt.title(names[i])
        plt.axis('off')
    plt.tight_layout()
    fig.suptitle(title)

def apply_Filter_Bank(kernels,kernels_parameters,dataset, plot=True):
    filtered_images = np.zeros((len(dataset), len(kernels), 128, 128))
    for i, kernel in enumerate(kernels):
        c = 0
        if plot: fig = plt.figure()
        try:
            os.makedirs('outputs/' + kernels_parameters[i])
        except FileExistsError:
            pass
        for k, image in enumerate(dataset):
            filtered_img = cv2.filter2D(cv2.cvtColor(get_image(image), cv2.COLOR_RGB2GRAY),-1,kernel)
            if plot:
                plt.subplot(4, 5, 1)
                plt.imshow(kernel, cmap='gray')
            filtered_images[k, i] = filtered_img
            if plot:
                plt.subplot(4, 5, c + 6)
                plt.imshow(filtered_img,cmap='gray')
                plt.title(image)
                plt.axis('off')
                c += 1
                cv2.imwrite('outputs//' + kernels_parameters[i] + "//" + image + '_' + '.jpg', filtered_img)
        if plot: fig.suptitle(kernels_parameters[i])
    return filtered_images

def test_image(filtered_train, filtered_test, train_label, test_label):
    mse = np.zeros((6, 3))
    for i in range(len(filtered_test)):
        for j in range(len(filtered_train)):
            mse[i, train_label[j]] += np.linalg.norm(filtered_test[i].mean() - filtered_train[j].mean())
    print(mse)
    print(np.argmin(mse, axis=-1))
    print(test_label)
    return np.argmin(mse, axis=-1)

tr, te, tr_label, te_label = load_dataset()
kernels,kernels_parameters = create_Gabor_Filter_Bank()

print(kernels_parameters)
display_images(5, 4, kernels,kernels_parameters, 'Kernels')
f_train = apply_Filter_Bank(kernels, kernels_parameters, tr, True)
f_test = apply_Filter_Bank(kernels, kernels_parameters, te, False)
pred_labels = test_image(f_train, f_test, tr_label, te_label)
pred_labels = [f'pred = {pred}, true = {true}' for pred,true in zip(pred_labels,te_label)]
plot_dataset(te," ", pred_labels)
plt.show()
