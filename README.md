# 📧 Spam Email Classifier

A Machine Learning project developed during my **Artificial Intelligence Internship at Codec Technologies**. The goal of this project is to automatically classify emails/messages as **Spam** or **Not Spam (Ham)** using Natural Language Processing (NLP) and machine learning techniques.

## 👨‍💻 Internship Project

**Intern:** Suraj Singh
**Organization:** Codec Technologies India
**Role:** Artificial Intelligence Intern
**Internship Duration:** 15 June 2026 – 15 July 2026
**Project:** Spam Email Classifier
**Mode:** Hybrid / India

---

## 📌 Project Overview

Spam emails are unwanted messages that can contain advertisements, scams, malicious links, or other potentially harmful content. Manually identifying spam can be time-consuming, especially when dealing with a large number of emails.

This project uses **Natural Language Processing and Machine Learning** to analyze the text of an email and predict whether it belongs to the **Spam** or **Ham** category.

### 🎯 Objective

The main objectives of this project are:

* Detect spam emails automatically.
* Classify messages into **Spam** and **Ham** categories.
* Perform text preprocessing and feature extraction.
* Train a machine learning classification model.
* Evaluate the model using appropriate performance metrics.
* Provide a simple way to test new email messages.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** – Data manipulation and analysis
* **NumPy** – Numerical computations
* **Scikit-learn** – Machine learning and model evaluation
* **NLTK** – Natural Language Processing
* **Matplotlib / Seaborn** – Data visualization
* **Jupyter Notebook** – Development and experimentation

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Text Preprocessing
   ↓
Feature Extraction
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Spam / Ham Prediction
```

---

## 🧹 Data Preprocessing

The text data is processed before being provided to the machine learning model. The preprocessing pipeline may include:

1. Removing unnecessary characters and formatting.
2. Converting text to lowercase.
3. Removing unwanted punctuation.
4. Removing stopwords where appropriate.
5. Tokenizing text.
6. Converting text into numerical features.

---

## 🔢 Feature Extraction

Since machine learning models cannot directly understand raw text, the email content is transformed into numerical features.

A **TF-IDF (Term Frequency–Inverse Document Frequency)** based representation can be used to assign importance to words based on their occurrence within the dataset.

The resulting feature vectors are then provided to the classification algorithm.

---

## 🤖 Machine Learning Model

The project uses a supervised machine learning approach.

The general process is:

```python
X = email_text
y = email_label

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Transform text into numerical features
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train classifier
model.fit(X_train_vectorized, y_train)
```

The trained model can then be used to classify previously unseen messages.

---

## 📊 Model Evaluation

The classifier can be evaluated using metrics such as:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **Confusion Matrix**

Example:

```text
Accuracy  : XX.XX%
Precision : XX.XX%
Recall    : XX.XX%
F1-Score  : XX.XX%
```

> Replace the `XX.XX%` values with the actual results obtained from your trained model.

---

## 🧪 Example Prediction

The trained classifier can be used to predict new messages.

```python
message = ["Congratulations! You have won a free prize!"]

prediction = model.predict(
    vectorizer.transform(message)
)

print(prediction)
```

Example output:

```text
Spam
```

Another example:

```text
Input:
"Hi, can we meet tomorrow to discuss the project?"

Output:
Ham
```

---

## 📁 Project Structure

```text
Spam-Email-Classifier/
│
├── dataset/
│   └── spam.csv
│
├── notebooks/
│   └── spam_email_classifier.ipynb
│
├── src/
│   └── classifier.py
│
├── models/
│   └── model.pkl
│
├── requirements.txt
├── README.md
└── LICENSE
```

*The structure can be modified according to the actual files in the repository.*

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/Spam-Email-Classifier.git
```

Navigate to the project directory:

```bash
cd Spam-Email-Classifier
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

If the project is implemented in a Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
notebooks/spam_email_classifier.ipynb
```

Run the notebook cells sequentially to:

1. Load the dataset.
2. Clean and preprocess the data.
3. Train the model.
4. Evaluate the model.
5. Test custom email messages.

---

## 📚 Key Learnings

Through this project, I gained practical experience in:

* Natural Language Processing
* Text preprocessing
* Feature extraction using TF-IDF
* Supervised Machine Learning
* Classification algorithms
* Model evaluation
* Python data analysis
* Handling real-world text data
* Building an end-to-end machine learning workflow

---

## 🚀 Future Improvements

Possible improvements include:

* Developing a web interface for real-time predictions.
* Comparing multiple machine learning algorithms.
* Hyperparameter tuning.
* Using advanced NLP techniques.
* Implementing deep learning models.
* Deploying the classifier as a web application or API.
* Continuously updating the training dataset to improve performance.

---

## 👨‍💻 Author

**Suraj Singh**

Artificial Intelligence Intern
Codec Technologies India

---

## 📄 Internship Context

This project was completed as part of my **Artificial Intelligence Internship at Codec Technologies India**, during the internship period of **15 June 2026 to 15 July 2026**.

The project provided practical exposure to machine learning, natural language processing, data preprocessing, model training, and classification.

---

## ⭐ Acknowledgement

I would like to thank **Codec Technologies India** for providing the internship opportunity and allowing me to gain practical experience in Artificial Intelligence and Machine Learning through project-based learning.
