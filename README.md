# RecycleVision

## Problem Statement
With the environment issues that society is dealing with today, it's important for us to play our role in resolving what we can. Recycling is important for waste management, however, many individuals struggle to realize when an item can be recycled or not. this is a UI-based projet that utilizes computer vision to identify whether an object is recyclable or not.

## Data Sources
[**Recyclable and Household Waste Classification**](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification): This is a Kaggle dataset that contains 15,000 images containing various recyclable materials, general waste, and household items across 30 different categories. With 500 images per category, the dataset provides a large variety of data in the field of waste classification.

## Reviewing Relevant Previous Efforts and Comparing to this Project

### Previous Efforts
- [**Machine Learning Waste Classification Model**](https://github.com/manuelamc14/waste-classification-model): This project uses a binary classification model with transfer learning to classify waste into either organic or recyclable. The UI is a flask application that allows users to upload an image with the result showing along with a probability score.
- [**Deep Learning Real Life Item Classification**](https://github.com/oscarlee711/deep-learning-recycle-item-classification): This project uses a multi-class classification approach to classify items into multiple recyclable categories. It incorporates both batch normalization and data augmentation to improve validation accuracy while representing metrics in confusion matrices, charts, and more.

### Comparison to this Project
- **Classification Granularity**: This project uses 30 different categories while the other one's use either binary or multi-class options with broader categories. 
- **Model Diversity**: This project uses multiple model implementations while the other projects use only one model each.
- **Evaluation Process**: This project contains a comprehensive approach across all models while the other projects evaluate a single model and focus on validation accuracy with confusion matrices.

## Data Preprocessing Steps
- **Image Loading**: Loading the images from the data/images directory and converting it to RGB format.
- **Image Resizing**: Resizing all the images to 224x224 to ensure that input dimensions are consistent across models.
- **Pixel Normalization**: Normalizing the pixel values from [0, 255] to [0, 1] to improve model training.
- **Data Splitting**: Splits the data into training, validation, and test sets.
- **Data Augmentation**: Applies shifts to the images such as rotating, widht-height shifts, zoom, etc. to increase diversity for the model inputs.
- **Data Storage**: Saves the preprocessed images to the data/processed directory as NumPy arrays and creates generators for efficiency.


## Model Selection and Evaluation Process

### Model Selection
- **Naive Approach**: This approach extracts color-based features including average RGB values, color ratios, deviations, and edge density to create feature vectors for logistic regression classification.
- **Traditional Approach**: This approach combines a few computer vision features such as Histogram of Oriented Gradients (HOG) and Local Binary Pattern (LBP) while implmenting a grid search optimized logistic regression for classification.
- **Deep Learning Approach (Selected Approach)**: This approach uses transfer learning with MobileNetV2 as the base model. Additionally using custom dense layers with dropout regularization and early stopping to automatically learn features from pixel data. This is the approach used for the user flow.

### Evaluation Process
This file loads the test dataset and converts the numeric lables into binary classifications (recyclable/non-recyclable) using the category mapping. Then, it checks the model output files from all three models and extracts the predictions to calculate the accuracy, precision, recall, and F-1 score. Finally, the report is generated/created and saved in the output directory.

## Ethics Statement
This project helps promote environmental sustainability by helping users make recycling decisions, but acknowledges limitations in classification accuracy that could lead to incorrect waste disposal. This project should be used as a supplemental tool, not as an authoritative tool and users should double check with their local governments on recycling guidelines. Model performance is clearly shown for transparency.

## Results and Conclusion

### Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive | 0.6250 | 0.6267 | 0.7344 | 0.6763 |
| Traditional | 0.6917 | 0.7077 | 0.7188 | 0.7132 |
| Deep Learning | 0.9167 | 0.9355 | 0.9062 | 0.9206 |

### Conclusion
The deep learning approach significantly outperforms both the naive and traditional methods across all evaluation metrics. This along with the speed of the model execution validates the selection of the deep learning model for the user interface, as it provides the most reliable classification for recycling decisions.

## Steps to Run Application
```bash
   streamlit run ui.py
```