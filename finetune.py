from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools



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

    if option==1:
         plt.tight_layout()
         plt.ylabel('True label')
         plt.xlabel('Predicted label')
         plt.savefig('C_MATRIX_FINETUNE_ALEX')
         plt.show()
    
    elif option==2:
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('C_MATRIX_FINETUNE_VGG')
        plt.show()



def train_model(model, criterion, optimizer, scheduler,option, num_epochs):
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            batch=0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                print("batch %d" %batch)

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                #print(len(inputs))
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                batch=batch+1

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    best_model_wts = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_wts)
    if option==1:
         torch.save(model, 'alexnet_finetune.pt')
    elif option==1:
         torch.save(model, 'vgg16_finetune.pt')
    return model


def predict(model):
    # switch to evaluate mode
    model.eval()

    correct = 0
    total = 0

    all_predicted = []
    t=[]
    for data1 in dataloaders['test']:
        inputs, labels1 = data1
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels1 = Variable(labels1.cuda())
        else:
            inputs, labels1 = Variable(inputs), Variable(labels1)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels1.size(0)
        correct += torch.sum(predicted == labels1.data)
        all_predicted += predicted.cpu().numpy().tolist()
        t += labels1.data.cpu().numpy().tolist()
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

    return t,all_predicted



plt.ion()   # interactive mode

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'animals'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()
dset_classes=image_datasets['train'].classes

print('Project 1:Transfer Learning')
print('Finetuning of the following models for Animals dataset')
print('1.AlexNet')
print('2.VGG16') 
option=input('Enter model of choice :')
option=int(option)
if option==1:
        	
     #Alexnet PART
     print('Finetuning AlexNet for Animal Dataset\n')
     dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=100,
                                             shuffle=True, num_workers=1)
              for x in ['train', 'test']}
     model_ft = torchvision.models.alexnet(pretrained=True)
     num_ftrs=model_ft.classifier[6].in_features
     feature_model=list(model_ft.classifier.children())
     feature_model.pop()
     feature_model.append(nn.Linear(num_ftrs, len(dset_classes)))
     model_ft.classifier=nn.Sequential(*feature_model)
     #print(model_ft)


     if use_gpu:
          model_ft = model_ft.cuda()

     criterion = nn.CrossEntropyLoss()
     if use_gpu:
          criterion = criterion.cuda()

     optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

     exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
     model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,option,num_epochs=30)


     t,pred_labels = predict(model_ft)
     p=np.asarray(pred_labels)
     t=np.asarray(t)
     plt.figure()
     plot_confusion_matrix(p, t, option, title='Normalized confusion matrix')
     plt.show()


elif option==2:

#VGG PART
     print('Finetuning VGG for Animal Dataset\n')
     dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=10,
                                             shuffle=True, num_workers=1)
              for x in ['train', 'test']}
     model_ft = torchvision.models.vgg16(pretrained=True)
     num_ftrs=model_ft.classifier[6].in_features
     feature_model=list(model_ft.classifier.children())
     feature_model.pop()
     feature_model.append(nn.Linear(num_ftrs, len(dset_classes)))
     model_ft.classifier=nn.Sequential(*feature_model)
     #print(model_ft)


     if use_gpu:
          model_ft = model_ft.cuda()

     criterion = nn.CrossEntropyLoss()
     if use_gpu:
          criterion = criterion.cuda()

     optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

     exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

     model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, option, num_epochs=30)


     t,pred_labels = predict(model_ft)
     p=np.asarray(pred_labels)
     t=np.asarray(t)
     plt.figure()
     plot_confusion_matrix(p, t, option, title='Normalized confusion matrix')
     plt.show()


