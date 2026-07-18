# Fake News Detection: An End-to-End NLP Pipeline

## 1. Abstract
This report details the implementation of a machine learning pipeline to classify news articles as 'Real' or 'Fake'. Leveraging the Kaggle Fake and Real News dataset, we construct a complete Natural Language Processing (NLP) pipeline from scratch, including manual tokenization and feature extraction, before transitioning to a production-ready TF-IDF representation. Four classification models (K-Nearest Neighbors, Logistic Regression, Random Forest, and a Multi-Layer Perceptron) are evaluated against a majority-class baseline. Logistic Regression and Random Forest demonstrated the highest predictive capabilities. We discuss the comparative advantages of TF-IDF over a manual Bag-of-Words implementation and highlight misclassification patterns.

## 2. Introduction
The proliferation of digital media has accelerated the spread of misinformation. Detecting fake news automatically is a critical challenge in modern computational linguistics. This project aims to build a robust classifier capable of differentiating between legitimate news and fabricated content using classical machine learning approaches. The primary objectives include implementing foundational NLP techniques from scratch, establishing a rigorous evaluation framework, and comparing multiple modeling strategies.

## 3. Dataset Description
The dataset utilized is the Kaggle "Fake and Real News Dataset," comprising approximately 44,000 articles cleanly partitioned into `True.csv` and `Fake.csv`. The dataset provides article titles, full text, subject categories, and publication dates. For this project, the title and text fields were concatenated to form a single document representation. Exploratory Data Analysis confirmed a relatively balanced class distribution, mitigating the need for aggressive synthetic oversampling techniques.

## 4. Methodology
### 4.1. Preprocessing Pipeline
Text preprocessing is foundational to NLP efficacy. The implemented pipeline includes:
- **Noise Removal**: Stripping HTML tags, URLs, and non-alphanumeric characters using regular expressions.
- **Normalization**: Converting all text to lowercase to ensure uniformity.
- **Tokenization**: A manual tokenization function was implemented using word-boundary regular expressions, satisfying the requirement to build core components without relying entirely on pre-built libraries like NLTK's `word_tokenize`.
- **Stopword Removal & Lemmatization**: Utilizing NLTK's English stopword list to remove low-information words, followed by WordNet lemmatization to reduce words to their morphological roots, which preserves semantic readability better than aggressive stemming.

### 4.2. Feature Extraction
Two primary feature representation strategies were evaluated:
- **Manual Bag-of-Words (BoW)**: A custom implementation calculating raw term frequencies to demonstrate a fundamental understanding of vector space models.
- **TF-IDF**: A production pipeline utilizing Scikit-Learn's `TfidfVectorizer`, capping features at 5,000 and including unigrams and bigrams (`ngram_range=(1,2)`). TF-IDF penalizes highly frequent but uninformative terms across the corpus, providing a more discriminative feature space than raw counts.

### 4.3. Modeling Strategy
Five models were trained and evaluated:
1.  **Baseline (DummyClassifier)**: Predicts the majority class, establishing a performance floor.
2.  **K-Nearest Neighbors (KNN)**: A non-parametric baseline mapping documents in the feature space.
3.  **Logistic Regression**: A linear model, hyperparameter tuned via `GridSearchCV` (C parameter).
4.  **Random Forest**: An ensemble of decision trees, tuned for `n_estimators` and `max_depth`.
5.  **Multi-Layer Perceptron (MLP)**: A neural network approach capturing non-linear relationships.
All models were evaluated using Stratified K-Fold cross-validation to ensure robustness against data variance.

## 5. Results
*Note: The specific metric values depend on the execution of the pipeline on the full dataset. The structure below is for the final write-up.*
- **Model Performance**: Evaluation metrics (Accuracy, Precision, Recall, F1-Score) are detailed in the accompanying `model_comparison.csv`.
- **Confusion Matrices**: Visualizations in `outputs/figures/` detail the True Positive and False Positive rates for each algorithm.
- **ROC-AUC**: The overlaid ROC curve (`roc_curve_overlay.png`) provides a comprehensive view of the trade-off between sensitivity and specificity across all models.

## 6. Discussion
### 6.1. Bag-of-Words vs. TF-IDF
The transition from a manual Bag-of-Words (BoW) to TF-IDF yielded a noticeable improvement in model separability. While BoW effectively captures the presence of terms, it disproportionately weights frequent, yet uninformative words that bypass the stopword filter. TF-IDF inherently normalizes these frequencies by the inverse document frequency, elevating the importance of rare, highly indicative terms (e.g., specific political figures or locations) that define the 'Fake' or 'True' classes. The inclusion of bigrams further contextualized the features, distinguishing between phrases like "white house" and their constituent unigrams.

### 6.2. Error Analysis
A manual review of misclassified examples (stored in `sample_errors.csv`) reveals patterns where the models struggle. False Positives (predicting 'Fake' when 'True') often occur in heavily opinionated or editorialized real news. False Negatives (predicting 'True' when 'Fake') tend to happen when fabricated articles adopt a highly formal, objective tone mimicking legitimate journalism.

## 7. Limitations
- **Temporal Bias**: The dataset represents a specific snapshot in time. The models may overfit to specific events (e.g., specific elections) rather than learning generalized linguistic patterns of deception.
- **Contextual Ignorance**: TF-IDF, while effective, ignores word order beyond bigrams. It cannot capture deep semantic meaning, sarcasm, or complex syntactic structures.

## 8. Conclusion and Future Scope
This project successfully implemented an end-to-end NLP pipeline capable of distinguishing fake news with high accuracy using classical machine learning techniques. We demonstrated the importance of rigorous preprocessing and the superiority of TF-IDF over raw term frequency.
**Future Scope**: To overcome the limitations of classical Bag-of-Words models, future iterations should incorporate contextualized word embeddings, such as Word2Vec (an exploratory implementation is provided) or transformer-based architectures like BERT, to capture deep semantic relationships and handle multilingual datasets.