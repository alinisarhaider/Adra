# Teeth Root Classification

## Problem Statement
Classify between a 1 root and a multiple root tooth.

## Proposed Solution
Use a Convolution Neural Network (CNN) to perform the classification.

Why CNN..? Best to get most out of image data. Excellent at detecting visual signals 
for both high level and low level details.

## Implementation Approach
An overview of the steps to design and develop the proposed solution.
1. #### Data:
    1. Read the images and store them in numpy arrays.
    2. Normalize the images data so that pixel values ranges between 0 and 1.
    3. Split the data set into three parts as:
        1. 60% Train set (to train the classifier)
        2. 15% Validation set (to use as a reference of unseen data during training)
        3. 25% Test set (to evaluate the classifier on unseen data)
    4. There are only a total of 115 images in the data set. Generate and augment more data by flipping images 
    vertically and horizontally.
2. #### Classifier:
    1. Design a Convolution Neural Network to perform binary classification.
    2. Implement the CNN using Keras / Tensorflow with suitable parameters.
    3. Train the classifier using Train and Validation set and use the callbacks to evaluate and save 
    the best performing model (i.e. neither over-fitting nor under-fitting).
    
3. #### Evaluation:
    1. Evaluate model performance using following metrics:
        1. Accuracy
        2. F1-Score
        3. Precision
        4. Recall
    2. Calculate and plot confusion matrix
    3. For unlabelled testing, save each test image along with its predicted label.

4. #### Results:
    1. Result metrics:
        1. Accuracy -- 96.429 %
        2. F1-Score -- 96.311 %
        3. Precision -- 100 %
        4. Recall -- 91.667 %
    2. Confusion Matrix
    
    <img src="https://github.com/alinisarhaider/Adra/blob/master/cm.png?raw=true"/>

## Instructions to run the code
### Train new model
To train and test a new model: Run 'test_train_CNN' test case in 'test_CNN'. Data set path and parameters
 can be updated in 'config.yaml' file.

### Test existing model
To test unlabelled data on a pre-trained model: Run 'test_predict_CNN' test case in 'test_CNN'.

### Directory Structure
```
├── Adra
│    ├── artifacts
│    │    ├── models                                    
│    │    ├    ├── best_val                             <Best performing model (after training) is saved here>
│    │    ├    ├── last_epoch                           <Last Epoch model (after training) is saved here>
│    │    ├    ├── training_models                      <Improving models during training are saved here>
│    │    ├── predicted labels for test data            <Unlabelled test images along with their predicted labels are stored here>
│    ├── dataset
│    │    ├── xrays database                            
│    │    │    ├── 1 root                               <Contains single root images>
│    │    │    ├── 2 or more roots                      <Contains multi root images>
│    │    │    ├── Test                                 <Contains unlabelled images>
│    ├── evals                                          <Contains modules for model evaluation>
│    │    ├── ...
│    ├── trained_model                                  <Contains pre-trained model to test unlabelled data>

│    ├── __BaseClassifier.py
│    ├── class_CNN.py
│    ├── data_utils.py
│    ├── setup.py
│    ├── test_CNN.py
│    ├── config.yaml
│    ├── README.md
│    ├── requirements.txt
```
### Required Data Files
Place 'dataset/xrays database' folder in root directory that should contain following folders:
1. '/1 root'   (required for training)
2. '/2 or more roots'   (required for training)
3. '/test'  (required for testing)
