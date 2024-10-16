# Product Review Sentiment Analysis

## Overview

This project aims to analyze product reviews to determine the sentiment (positive, negative, or neutral) expressed in the text. Using various text preprocessing techniques and machine learning models, we achieve high accuracy in sentiment classification.

## Technologies Used

- Python
- Jupyter Notebook
- Libraries:
  - Pandas
  - NumPy
  - Scikit-learn
  - NLTK
  - SpaCy
  - Transformers (for BERT)



## Features

- **Text Preprocessing**: 
  - **Tokenization**: Splitting text into individual words or tokens.
  - **Stemming**: Reducing words to their root form.
  - **Lemmatization**: Converting words to their base form.
  - **Count Vectorization**: Transforming text into a matrix of token counts.

- **Models Used**:
  - **Logistic Regression**: A statistical model that uses a logistic function to model a binary dependent variable, predicting the probability of positive or negative sentiment based on features extracted from the text.
  
  - **Naive Bayes**: A family of probabilistic algorithms based on Bayes' theorem, assuming independence between features. Effective for text classification due to its simplicity and speed, with variants including Gaussian, Multinomial, and Bernoulli Naive Bayes.
  
  - **Random Forest**: An ensemble method that constructs multiple decision trees and outputs the mode of their predictions. Robust against overfitting and capable of handling high-dimensional data, suitable for text classification tasks.
  
  - **Support Vector Machine (SVM)**: A powerful classification technique that finds the optimal hyperplane to separate different classes in feature space. Effective in high-dimensional spaces for both linear and non-linear classification tasks.
  
  - **Decision Trees**: A tree-like model used for classification and regression that splits data based on feature values, creating branches until reaching a decision point. Easy to interpret but can be prone to overfitting.
  
  - **BERT (Bidirectional Encoder Representations from Transformers)**: A state-of-the-art model using transformer architecture to understand the context of words in a sentence by looking at 

  

- **Model Evaluation**: Performance metrics and accuracy assessment for each model, with BERT achieving an impressive accuracy of **95.53%**.

## Usage

- Load the dataset in the notebook.
- Perform text preprocessing steps.
- Train different models on the processed text.
- Evaluate the models and compare their performance.

## Results

- BERT outperformed all other models with an accuracy of 95.53%.
- Detailed evaluation metrics (precision, recall, F1-score) for all models are included in the notebook.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

Thanks to the authors of the libraries and frameworks used in this project.
Inspiration from various online resources and tutorials on sentiment analysis.

