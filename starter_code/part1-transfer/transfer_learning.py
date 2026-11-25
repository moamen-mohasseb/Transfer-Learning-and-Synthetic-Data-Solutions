# Starter code for Part 1 of the Small Data Solutions Project
# 

#Set up image data for train and test

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms 
from TrainModel import train_model
from TestModel import test_model
from torchvision import models


# use this mean and sd from torchvision transform documentation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

#Set up Transforms (train, val, and test)

# Training: RandomResizedCrop + RandomHorizontalFlip add variety to the small
# training set, which helps mitigate over-fitting.
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Validation / Test: deterministic - resize then center-crop to VGG16's 224x224 input.
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# The test images are evaluated exactly like the validation images.
test_transform = val_transform




#Set up DataLoaders (train, val, and test)
batch_size = 10
num_workers = 4

data_dir = 'imagedata-50'

# ImageFolder infers the class label from each sub-folder name (beach / desert / forest)
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_transform)

# Only the training loader is shuffled
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# class_names comes from the ImageFolder: ['beach', 'desert', 'forest']
class_names = train_dataset.classes



# Using the VGG16 model for transfer learning 
# 1. Get trained model weights
# 2. Freeze layers so they won't all be trained again with our data
# 3. Replace top layer classifier with a classifer for our 3 categories

# 1. Get trained model weights (VGG16 pre-trained on ImageNet)
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

# 2. Freeze every existing layer so its weights are NOT updated by our small dataset
for param in model.parameters():
    param.requires_grad = False

# 3. Replace the last layer of the classifier (1000 ImageNet classes) with a 3-class layer.
#    A freshly created layer has requires_grad=True, so only this layer will be trained.
num_features = model.classifier[6].in_features   # 4096
model.classifier[6] = nn.Linear(num_features, len(class_names))

# Train model with these hyperparameters
# 1. num_epochs 
# 2. criterion 
# 3. optimizer 
# 4. train_lr_scheduler 

# 1. num_epochs: transfer learning converges fast (rubric: should not need more than 5)
num_epochs = 5
# 2. criterion: cross entropy for multi-class (categorical) classification
criterion = nn.CrossEntropyLoss()
# 3. optimizer: Adam, passing ONLY the parameters that are still trainable (the new layer)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
# 4. train_lr_scheduler: halve the learning rate after every epoch (StepLR, step_size=1, gamma=0.5)
train_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)


# When you have all the parameters in place, uncomment these to use the functions imported above
def main():
    trained_model = train_model(model, criterion, optimizer, train_lr_scheduler, train_loader, val_loader, num_epochs=num_epochs)
    test_model(test_loader, trained_model, class_names)

if __name__ == '__main__':
    main()
    print("done")
