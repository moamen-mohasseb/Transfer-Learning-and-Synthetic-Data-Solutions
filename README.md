# Small Data Solutions

**Part 1: Travel Planner Problem**
A travel planning company asks their customers to share pictures of past vacations / holidays so their staff can get a sense of what kind of trips they enjoy. The company offers three basic categories of trips:
- Exploring in the Forest
- Adventure in the Desert
- Relaxing on the Beach 

As part of a new online trip planning software, the company is creating an AI bot that will automatically figure out from the uploaded photos which category is likely to be most appealing to the customer.  The challenge is, the company has fewer than 500 photos that are categorized and they feel it will be difficult to train a model using that little data. 

You will use the starter code and use transfer learning to build a model to categorize images with at least a 90% accuracy rate. 

**Part 2: Loan Funding Prediction Problem**
A loan company has a fairly large dataset that they want to use to train a model that predicts whether or not a loan should be funded.  The problem they face is the dataset they are using has a large class imbalance... they don't have enough examples of loans that were denied. The company wants to augment the "denied loan" class of data in order to train a model that performs better.  

You will use the starter code and use a Variational Autoencoder to generate an additional set of synthetic data for just the denied loan category of data. You will then augment the original dataset with the additional synthetic data and test to see if there were improvements.  

## Getting Started

1. Clone the repository
1. Install the dependencies listed below

### Dependencies

```
Python 3.10.8
Pytorch 1.13.1
torchvision 0.14.1
Numpy 1.24.1
sklearn 1.2.1
```

## Testing

**Part 1: Travel Planner Problem**
There is a test_model function included in TestModel.py. The code is already in place in the transfer_learning.py file to run this, you simply need to provide the correct parameters: a DataLoader for the test data, a trained model, and a list of class names. An image viewer will show the test images and the predicted class.  You should expect at least 9/10 images to be correctly identified.  

**Part 2: Loan Funding Prediction Problem** 
There is a test_model function included in TestModel.py. The function takes a path where the augmented dataset should be placed.  The script produces precision, recall, and an F1-score for both classes of data. You should expect results similar to this:
               
                precision    recall  f1-score   support
           0       0.79      0.55      0.65     61222
           1       0.63      0.84      0.72     56241 


## Project Instructions

**Part 1: Travel Planner Problem**
1. Create 2 Transforms (Train and Validation)
1. Create 2 DataLoaders (Train and Validation)
1. Import the VGG16 model 
1. Freeze the existing layers in the model
1. Replace the classification layer with your own
1. Using the provided train_model function, train a new model with the appropriate criterion, optimizer, and scheduler.
1. Using the set of images labeled "new" categorize them and see how your model performs

**Part 2: Loan Funding Prediction Problem**
1. Load the file "loan_continuous.csv" and split out just the records with Loan Status = 1
1. Use the test_model method to baseline the precision, recall and F1 score of the loan_continuous.csv file
1. Using the provided DataBuilder class, create Datasets for training and validation 
1. Using the provided Autoencoder class and the provided CustomLoss class, write methods to train and validate the model
1. After training the model, use the generate_fake method to generate 50000 additional records
1. Combine the generated data with the original dataset 
1. Use the test_model method to test whether the dataset augmented with synthetic data performs better on precision, recall, and F1 score for both loan statuses


## Results

Screenshots of the actual runs are in the `results/` folder.

### Part 2 (Transfer Learning, VGG16 with frozen layers and a new 3-class classifier)

Hyperparameters: 5 epochs, `CrossEntropyLoss`, `Adam` (lr=0.001, only the new layer is trained), `StepLR(step_size=1, gamma=0.5)`.

| Epoch | Train accuracy | Validation accuracy |
|------:|---------------:|--------------------:|
| 1 | 0.647 | 0.917 |
| 2 | 0.833 | 0.967 |
| 3 | 0.933 | 0.967 |
| 4 | 0.933 | 0.967 |
| 5 | 0.907 | 1.000 |

- Highest validation accuracy: **1.0** (the rubric requires more than 0.90).
- `test_model` displayed the test images with their predicted class. The screenshot shows 3 images, all correctly labelled (2 x beach, 1 x desert).
- Screenshots: `results/part2_training_log.png`, `results/part2_test_predictions.png`

### Part 3 (Synthetic data with a Variational Autoencoder)

- Records with `Loan Status = 1`: 6,241 (of 67,463).
- VAE trained for 1,500 epochs, with training and validation run in every epoch. In the final run the loss fell from 28.60 (train) and 25.55 (validation) at epoch 50 to 14.46 and 15.99 at epoch 1500, and the validation loss flattened after roughly epoch 800.
- 50,000 synthetic records were generated and combined with the original data: 117,463 records (61,222 for class 0 and 56,241 for class 1).

Classification report from `test_model` (the MLP is fitted and evaluated on the same full dataset, as provided in `TestModel.py`):

| Dataset | Class | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| Original (baseline) | 0 | 0.91 | 1.00 | 0.95 | 61,222 |
| Original (baseline) | 1 | 0.00 | 0.00 | 0.00 | 6,241 |
| Augmented | 0 | 0.89 | 0.94 | 0.91 | 61,222 |
| Augmented | 1 | 0.93 | 0.87 | 0.90 | 56,241 |

Overall accuracy: 0.91 (baseline) and 0.91 (augmented). Macro F1: 0.48 (baseline) and 0.91 (augmented).

Interpretation:
- The baseline model never identifies a denied loan (precision, recall and F1 of 0 for class 1). Its 0.91 accuracy only reflects the class imbalance, so it is misleading.
- After augmentation both classes are detected well: class 1 recall goes from 0.00 to 0.87 and F1 from 0.00 to 0.90, while class 0 keeps F1 of 0.91.
- The augmented results are better than the example in the instructions (class 1 recall about 0.84). 
- Caveats: the evaluation uses the same data the MLP was fitted on, and 89% of class 1 in the augmented set is synthetic, so these numbers show the benefit of balancing the classes, not performance on new real loans. Results also vary between runs: an earlier run (different GridSearchCV parameters) gave class 1 recall 0.98 but class 0 recall only 0.16.
- Screenshots: `results/part3_baseline_and_vae_training_start.png`, `results/part3_vae_training_end_and_augmented_counts.png` and `results/part3_augmented_classification_report.png`. These screenshots are from the earlier run; the table above is from the latest run, whose full output is saved in `results/part3_run2_output.txt`.
