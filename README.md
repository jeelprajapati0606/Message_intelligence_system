<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=25&pause=1000&color=2563EB&center=true&vCenter=true&width=850&lines=Message+Intelligence+System;Spam+Detection+with+Machine+Learning;KNN+%7C+SVM+%7C+Naive+Bayes" alt="Animated project title">

# Message Intelligence System

**A machine-learning project for classifying messages as Spam or Legitimate.**

[![Open Notebook](https://img.shields.io/badge/📓_Open-Notebook-2563EB?style=for-the-badge)](https://github.com/jeelprajapati0606/Message_intelligence_system/blob/main/message_intelligence/Message_Intelligence_system-checkpoint.ipynb)
[![View Dataset](https://img.shields.io/badge/📊_View-Dataset-059669?style=for-the-badge)](https://github.com/jeelprajapati0606/Message_intelligence_system/tree/main/message_intelligence)
[![Explanation Video](https://img.shields.io/badge/▶_Watch-Explanation-DC2626?style=for-the-badge)](#-project-video)

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white) ![Machine Learning](https://img.shields.io/badge/Focus-Machine%20Learning-7C3AED)

<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdWZubTQ1cHdlazNvbG1oZWU0NndxbG84c2d4dWZjOHZ3c2g5b3d3aiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/3oKIPnAiaMCws8nOsE/giphy.gif" width="75%" alt="Animated data and machine learning banner">

</div>

---

## Overview

The **Message Intelligence System** explores how machine-learning models can distinguish spam messages from legitimate ones using numerical message and sender signals. The project compares three complementary approaches:

- **K-Nearest Neighbors (KNN)** — predicts from similar examples.
- **Support Vector Machine (SVM)** — separates classes using a decision boundary.
- **Naive Bayes** — estimates class probabilities using Bayes’ theorem.

In plain terms, the system learns from examples that are already labelled. For a new record, it uses its feature values to estimate whether the message belongs to class **0 (Legitimate)** or **1 (Spam)**.

> **Repository note:** The repository currently provides a dataset and Jupyter notebook. This README explains the project and intended workflow; refer to the notebook for the exact code and implementation details. No model scores are stated here because they have not been verified.

## Dataset

The repository contains a CSV file whose filename includes `5200`. The project description identifies features such as:

| Feature type | Examples |
|---|---|
| Message characteristics | Message length, URLs, special characters |
| Text-derived signals | Spam-keyword frequency or score |
| Sender behavior | Sender activity score |
| Timing | Time-based features |
| Prediction target | `Spam_Label` — 0 = Legitimate, 1 = Spam |

Check the dataset’s actual column names before running the notebook. Keep the target column out of the model inputs to avoid data leakage.

## Workflow

```mermaid
flowchart TD
    A[Load message dataset] --> B[Inspect columns and target labels]
    B --> C[Prepare features X and label y]
    C --> D[Split into training and test data]
    D --> E[Preprocess and scale features]
    E --> F1[Train KNN and test K values]
    E --> F2[Train SVM with selected kernels]
    E --> F3[Train Naive Bayes]
    F1 --> G[Predict test labels]
    F2 --> G
    F3 --> G
    G --> H[Measure accuracy, precision, recall and F1]
    H --> I[Compare models and error trade-offs]
```

## How the models work

### K-Nearest Neighbors
KNN looks at the `K` closest training examples and predicts the class that appears most often among them. Since it relies on distances, feature scaling matters. A small `K` can be sensitive to individual examples; a larger `K` can produce smoother decisions.

### Support Vector Machine
SVM searches for a boundary that separates the classes while maximizing the margin between them. A linear kernel models a straight boundary, while kernels such as RBF can represent more complex relationships. Feature scaling is usually important for SVM.

### Naive Bayes
Naive Bayes applies Bayes’ theorem to estimate the probability that a message belongs to each class. It assumes features are conditionally independent once the class is known. That simplifying assumption makes the model fast, but correlated inputs may affect how well it fits the data.

## Important code concepts

The snippets below show standard patterns for this project. Variable names may differ from those used in the notebook.

### 1. Separate inputs and label

```python
X = df.drop(columns=["Spam_Label"])
y = df["Spam_Label"]
```

`X` contains the signals used to make a prediction. `y` contains the known answer used for training and evaluation.

### 2. Split data for evaluation

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

The training data teaches the model; the held-out test data checks its performance on examples it did not train on. Stratification helps preserve the class proportions in both sets.

### 3. Scale features correctly

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Fit the scaler on training data only, then apply that same transformation to the test data. This avoids leaking test-set statistics into training. Scaling is particularly important for KNN and SVM.

### 4. Evaluate predictions

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1:", f1_score(y_test, y_pred, zero_division=0))
```

- **Accuracy:** fraction of all predictions that are correct.
- **Precision:** among messages flagged as spam, how many are truly spam?
- **Recall:** among all spam messages, how many did the model catch?
- **F1-score:** a balance between precision and recall.

For spam filtering, a false positive (blocking a legitimate message) and a false negative (letting spam through) have different consequences. Review the confusion matrix and choose metrics with that trade-off in mind.

## Run the notebook

1. Clone or download this repository.
2. Install Python 3 and Jupyter Notebook, or open the notebook in Google Colab.
3. Install commonly used dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

4. From the repository root, launch Jupyter:

   ```bash
   jupyter notebook
   ```

5. Open `message_intelligence/Message_Intelligence_system-checkpoint.ipynb` and run its cells in order. If needed, update the CSV path in the notebook to match the file in `message_intelligence/`.

## Project video

[![Watch the project explanation](https://img.shields.io/badge/▶_Watch_the_Project_Explanation-YouTube-red?style=for-the-badge&logo=youtube)](https://youtu.be/REPLACE_WITH_YOUR_VIDEO_ID)

Replace `REPLACE_WITH_YOUR_VIDEO_ID` with the real YouTube video ID before publishing. Until then, this button is a placeholder.

## Limitations and responsible use

- Performance depends on the quality and representativeness of the dataset and labels.
- Correlated inputs may violate the Naive Bayes independence assumption.
- Accuracy alone can hide poor performance on one class; review precision, recall, F1, and the confusion matrix.
- Treat this as an educational project, not a ready-to-deploy security product. A real system needs representative validation, monitoring, privacy safeguards, and human review for consequential decisions.

## Repository structure

```text
Message_intelligence_system/
├── README.md
└── message_intelligence/
    ├── Message_Intelligence_Dataset_5200 - Message_Intelligence_Dataset_5200.csv.csv
    └── Message_Intelligence_system-checkpoint.ipynb
```

## Author

**Jeel Prajapati** · [GitHub profile](https://github.com/jeelprajapati0606)

---

<div align="center">

<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdWZubTQ1cHdlazNvbG1oZWU0NndxbG84c2d4dWZjOHZ3c2g5b3d3aiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/26tn33aiTi1jkl6H6/giphy.gif" width="55%" alt="Animated technology footer">

**Thank you for visiting.** If this project was useful, please consider starring the repository.

[![Back to top](https://img.shields.io/badge/⬆_Back_to_Top-2563EB?style=for-the-badge)](#message-intelligence-system)

</div>
