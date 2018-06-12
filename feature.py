from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import sklearn.svm
import itertools
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import time
import os
import copy



def plot_confusion_matrix(pred_class, actual_class,option,
                          title='Confusion matrix'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Code from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    cm = confusion_matrix(actual_class, pred_class)

    cmap = plt.cm.Blues
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.nan_to_num(cm)

    print('Confusion matrix')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    if option == 1:
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('C_MATRIX_Feature_extraction_Alexnet')
        plt.show()

    elif option == 2:
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('C_MATRIX_Feature_extraction_VGG')
        plt.show()


print('Project 1:Transfer Learning')
print('Feature Extraction of the following models ')
print('1.AlexNet')
print('2.VGG16')
option=input('Enter model of choice :')
option=int(option)
if option==1:
    print('Alexnet')
    model = torchvision.models.alexnet(pretrained=True)

    normalize = transforms.Normalize(mean=[0.485, 0.486, 0.406],
                                     std=[0.229, 0.224, 0.225])

    resize = transforms.Resize((224, 224))

    preprocessor = transforms.Compose([resize, transforms.ToTensor(), normalize, ])

    new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    model.classifier = new_classifier
    # model = model.features

    data_dir_train = '../art/train'
    data_dir_test = '../art/test'
    batch_size = 5

    train_loader = torch.utils.data.DataLoader(datasets.ImageFolder(data_dir_train, preprocessor),
                                               batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(datasets.ImageFolder(data_dir_test, preprocessor),
                                              batch_size=batch_size, shuffle=True)

    input_big_array = None
    target_big_array = None
    test_input_big_array = None
    test_target_big_array = None

    for i, (in_data, target) in enumerate(train_loader):
        print('Batch_num:', i)
        input_var = torch.autograd.Variable(in_data, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        input_var1 = model(input_var)
        # target_var1=model(target_var)
        input_var1_tensor = input_var1.data
        input_var2 = input_var1_tensor.numpy()
        target_var1_tensor = target_var.data
        target_var2 = target_var1_tensor.numpy()
        if (input_big_array is None):
            input_big_array = input_var2
        else:
            input_big_array = np.append(input_big_array, input_var2, axis=0)
        if (target_big_array is None):
            target_big_array = target_var2
        else:
            target_big_array = np.append(target_big_array, target_var2, axis=0)

    for j, (test_data, test_target) in enumerate(test_loader):
        print('Batch_num:', j)
        test_data_var = torch.autograd.Variable(test_data, volatile=True)
        test_target_var = torch.autograd.Variable(test_target, volatile=True)
        test_input_var1 = model(test_data_var)
        # test_target_var1=model(test_target_var)
        test_input_var1_tensor = test_input_var1.data
        test_input_var2 = test_input_var1_tensor.numpy()
        test_target_var1_tensor = test_target_var.data
        test_target_var2 = test_target_var1_tensor.numpy()
        if (test_input_big_array is None):
            test_input_big_array = test_input_var2
        else:
            test_input_big_array = np.append(test_input_big_array, test_input_var2, axis=0)
            # print("input_big_array_shape:",input_big_array.shape)
        if (test_target_big_array is None):
            test_target_big_array = test_target_var2
        else:
            test_target_big_array = np.append(test_target_big_array, test_target_var2, axis=0)

    model_svm = sklearn.svm.SVC(C=51.0, kernel='rbf')
    model_svm.fit(input_big_array, target_big_array)
    y_pred= model_svm.predict(test_input_big_array)
    print("Accuracy: " + str(np.mean(y_pred == test_target_big_array)))
    plt.figure()
    plot_confusion_matrix(y_pred, test_target_big_array, option, title='Normalized confusion matrix')
    plt.show()

elif option==2:
    print('VGG16')
    model = torchvision.models.vgg16(pretrained=True)
    #print("vgg model:", model)

    normalize = transforms.Normalize(mean=[0.485, 0.486, 0.406],
                                     std=[0.229, 0.224, 0.225])

    resize = transforms.Resize((224, 224))

    preprocessor = transforms.Compose([resize, transforms.ToTensor(), normalize, ])

    new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    model.classifier = new_classifier

    data_dir_train = '../art/train'
    data_dir_test = '../art/test'
    batch_size = 5

    train_loader = torch.utils.data.DataLoader(datasets.ImageFolder(data_dir_train, preprocessor),
                                               batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(datasets.ImageFolder(data_dir_test, preprocessor),
                                              batch_size=batch_size, shuffle=True)

    input_big_array = None
    target_big_array = None
    test_input_big_array = None
    test_target_big_array = None

    for i, (in_data, target) in enumerate(train_loader):
        print('Batch_num:', i)
        input_var = torch.autograd.Variable(in_data, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        input_var1 = model(input_var)
        # target_var1=model(target_var)
        input_var1_tensor = input_var1.data
        input_var2 = input_var1_tensor.numpy()
        target_var1_tensor = target_var.data
        target_var2 = target_var1_tensor.numpy()
        if (input_big_array is None):
            input_big_array = input_var2
        else:
            input_big_array = np.append(input_big_array, input_var2, axis=0)
        if (target_big_array is None):
            target_big_array = target_var2
        else:
            target_big_array = np.append(target_big_array, target_var2, axis=0)

    for j, (test_data, test_target) in enumerate(test_loader):
        print('Batch_num:', j)
        test_data_var = torch.autograd.Variable(test_data, volatile=True)
        test_target_var = torch.autograd.Variable(test_target, volatile=True)
        test_input_var1 = model(test_data_var)
        # test_target_var1=model(test_target_var)
        test_input_var1_tensor = test_input_var1.data
        test_input_var2 = test_input_var1_tensor.numpy()
        test_target_var1_tensor = test_target_var.data
        test_target_var2 = test_target_var1_tensor.numpy()
        if (test_input_big_array is None):
            test_input_big_array = test_input_var2
        else:
            test_input_big_array = np.append(test_input_big_array, test_input_var2, axis=0)
        if (test_target_big_array is None):
            test_target_big_array = test_target_var2
        else:
            test_target_big_array = np.append(test_target_big_array, test_target_var2, axis=0)

    model_svm = sklearn.svm.SVC(C=53.0, kernel='rbf')
    model_svm.fit(input_big_array, target_big_array)
    y_pred= model_svm.predict(test_input_big_array)
    print("Accuracy: " + str(np.mean(y_pred == test_target_big_array)))
    plt.figure()
    plot_confusion_matrix(y_pred, test_target_big_array, option, title='Normalized confusion matrix')
    plt.show()


































