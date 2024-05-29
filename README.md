# Sentiment Analysis with BERT

This repository contains code for performing sentiment analysis on a review dataset using BERT. The process involves data preprocessing, zero-shot processing, and fine-tuning of the BERT model.

## Data Preprocessing

We used two different techniques for preprocessing the data:

### Zero-Shot Processing

The `preprocess_text` function was used to clean and preprocess text data in the "verified reviews" column. This function performs several steps, including punctuation removal, stopword removal, and lemmatization.

### Few-Shot/Fine-Tuning Processing

A BERT preprocessor was used for fine-tuning the model. This preprocessor tokenizes the input text, adds special tokens required by BERT, and creates input sequences and embeddings.

### Data Split/Preparation

The dataset was divided into training, validation, and test sets. Labels were converted into integers and one-hot encoded for the 'Categorical Cross-Entropy' loss function. TensorFlow datasets were created from the features and labels in each set, and caching and prefetching were used for optimization.

## Ground Truth

The dataset consists of customer reviews with ratings. The ratings are categorized into 5 classes (1 to 5 stars), with a skewed distribution towards higher ratings.

## Network Details

### Zero-Shot

We used the 'bert base multilingual uncased sentiment' model, a variant of BERT specifically designed for sentiment analysis. It includes 12 transformer blocks, 768 hidden layers, and 12 attention heads.

### Fine-Tuning

The model architecture includes an input layer, a pre-processing layer, a BERT encoder, a dropout layer for regularization, and an output layer with 5 units for classification. We used the Categorical Crossentropy loss function and the AdamW optimizer with a learning rate scheduler.

## Results

### Zero-Shot

The model achieved an accuracy of approximately 63.49% and a mean squared error (MSE) of 8.36% on the review dataset.

### Fine-Tuning

We tested varying initial learning rates and found that a learning rate of 5e-5 resulted in the highest testing accuracy of 0.844.

