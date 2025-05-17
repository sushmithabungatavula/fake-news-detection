# Project Title: Fake News Detection Using Natural Language Processing and Machine Learning

Authors: Sai Shishir Koppula, Sushmitha Bungatavula
Date: 04-25-2025

This project aims to detect fake news using Natural Language Processing (NLP) techniques and various machine learning classifiers. The notebook includes text preprocessing, vectorization, model training, and evaluation using classification algorithms such as PassiveAggressiveClassifier, Logistic Regression, Decision Trees, and more.

Description:
-------------
This Jupyter Notebook ('fake_news_detection_code.ipynb') contains the full implementation of the project titled "Fake News Detection Using Natural Language Processing and Machine Learning." The code performs the following tasks:

1. Task 1: Data Preprocessing
   - Reads and merges datasets ('fake.csv' and 'true.csv').
   - Assigns target labels (fake = 0, true = 1).
   - Cleans, formats, and prepares the data for analysis.

2. Task 2: Data Visualization
   - Analyzes the distribution of fake and real news articles.
   - Displays word clouds for fake and true news content.

3. Task 3: Feature Extraction
   - Converts text data into numerical format using TF-IDF Vectorization.
   - Prepares the feature matrix (X) and target vector (y).

4. Task 4: Model Building and Evaluation
   - Splits the dataset into training and testing sets.
   - Builds machine learning models (Logistic Regression, Decision Tree, Random Forest, Support Vector Machine).
   - Evaluates the models using accuracy, confusion matrix, and classification reports.
   - Compares model performances.


## Dataset

- Dataset used: **Fake and real news dataset**
- Format: CSV
- Columns include `title`, `text`, and `label`

You can find similar datasets here: [Kaggle Fake News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)


# File Structure:
---------------
.
├── fake_news_detection_code.ipynb  # Main Jupyter Notebook with implementation
├── readme.txt                      # This readme file
├── News _dataset/                           # Folder containing datasets (to be manually placed)
│   └— [fake.csv]
    └— [true.csv]
└— visualisations/                # Folder for confusion matrices, and graphs
    └— [confusion matrices, plots, etc.]

# Requirements:
-------------
To run the notebook successfully, ensure the following libraries are installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn wordcloud nltk
```

Other libraries like `warnings` (built-in) are also used.

## Installation

Make sure you have Python 3.7+ installed.

You can install required packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk
```

Also, for first-time use, download necessary NLTK data:

```python
import nltk
nltk.download('stopwords')
```

# How to Run:
-----------
1. Download the datasets 'fake.csv' and 'true.csv' from the links below:
   - Fake News: https://drive.google.com/file/d/1LG9iPaDnuH1ykDX2lLnnUHdOqevv8yPN/view?usp=drive_link
   - True News: https://drive.google.com/file/d/1SqP1s3ufn9cdhtIz5G6puFH3y9flm4vR/view?usp=drive_link

2. Create a folder named 'News _dataset' in the same directory as the `.ipynb` file and place both datasets inside it.

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

4. Open 'fake_news_detection_code.ipynb'.

5. Run all cells sequentially (Kernel > Restart & Run All).

Notes:
------
- This notebook is part of the submission for the Feature Engineering course project at the University of North Texas.
- Contact for questions: saishishirkoppula@my.unt.edu or sushmithabungatavula@my.unt.edu