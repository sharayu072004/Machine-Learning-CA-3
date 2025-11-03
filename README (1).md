# E-Commerce Product Classification using NLP

## Title and Short Description  

The purpose of this project is to categorize e-commerce product titles and descriptions into four groups: Electronics, Household, Books, and Clothing and Accessories. The aim is to automatically classify product information to enhance search outcomes, product recommendations, and inventory management in e-commerce websites. The issue is significant since online marketplaces process large amounts of unstructured text data, and manual classification is time-consuming and inaccurate.  
This project achieves precise classification using effective and interpretable models through Natural Language Processing and Machine Learning techniques. The Support Vector Machine (SVM) model produced the best results, achieving the highest accuracy of 96 percent.

---

## Dataset Source  

The dataset used in this project was obtained from Kaggle, titled *E-commerce Text Classification Dataset* by Saurabh Shahane. It contains over 4000 product samples across four categories. Each sample includes the product title, product description, and category label.  

Data preprocessing steps included:  
- Removing duplicate rows and unnecessary headers  
- Handling missing values by dropping incomplete records  
- Cleaning and normalizing text (lowercasing, removing special characters)  
- Tokenizing text and transforming it into numerical features using TF-IDF (Term Frequency–Inverse Document Frequency)  

These steps ensured that the dataset was consistent and ready for model training.  

**Figure 1: Word Cloud Image**  
*This figure represents the most frequent terms extracted from the dataset after preprocessing such as tokenization and stop-word removal.*

---

## Methods Used  

Four machine learning models were trained and compared to perform text classification: Logistic Regression, Naive Bayes, Support Vector Machine (SVM), and Random Forest. All models used TF-IDF vectorized data to convert text into numerical form.  

The SVM model achieved the best balance between accuracy and generalization. Logistic Regression followed closely with good interpretability, while Naive Bayes and Random Forest also performed reasonably well but struggled with sparse TF-IDF features.  

### Model Comparison Table  

| Model | Key Characteristics | Advantages | Disadvantages |
|--------|---------------------|-------------|----------------|
| Logistic Regression | Linear classifier based on sigmoid function | Easy to interpret, fast to train | Limited to linear relationships |
| Naive Bayes | Probabilistic classifier based on Bayes’ theorem | Works well with text data, simple | Assumes independence between features |
| Support Vector Machine (SVM) | Margin-based classifier for high-dimensional data | Excellent accuracy, robust generalization | Slower for very large datasets |
| Random Forest | Ensemble of decision trees | Handles non-linear data well | Can overfit sparse text features |

**Figure 1(A): ROC curve for Logistic Regression**  
**Figure 1(B): ROC curve for Random Forest Classifier**

---

## Steps to Run the Code  

To execute this project, first download or clone the repository and install all required Python dependencies to ensure compatibility. Once the setup is complete, open the Jupyter Notebook file named *Machine_Learning_Project.ipynb*.  

Run each cell sequentially to perform data loading, preprocessing, and transformation using TF-IDF vectorization. After preprocessing, model training, evaluation, and comparison are executed automatically. The notebook generates visual outputs such as accuracy graphs, confusion matrices, and category distribution plots, helping to analyze classification performance.  

After execution, the notebook displays all performance metrics, comparison results, and final model outputs, demonstrating the complete workflow from raw data to evaluated results.

---

## Summary of Experiments and Results  

This research compared four classical machine learning algorithms — Logistic Regression, Naive Bayes, Support Vector Machine (SVM), and Random Forest. The text data was transformed using TF-IDF vectorization to convert product titles and descriptions into numerical features. Hyperparameter tuning was applied, including regularization strength adjustments for Logistic Regression and SVM, and parameter optimization such as the number of estimators and maximum depth for Random Forest.  

When compared with previously published e-commerce text classification methods, such as those using Naive Bayes and Decision Trees (which achieved 88–91 percent accuracy), this project achieved significantly better results. The SVM model reached an accuracy of 96 percent, outperforming earlier methods based on simpler probabilistic or frequency-based models. Logistic Regression also performed strongly with an accuracy of 94 percent, showing that linear models with proper feature engineering can deliver competitive performance.  

Visualization techniques such as confusion matrices, category distribution plots, and TF-IDF feature importance charts were used to understand model behavior and interpret results. Confusion matrices revealed misclassifications mainly between similar categories like Electronics and Household, while category distribution graphs confirmed that the dataset was balanced.  

**Figure 2: Comparison of Model Accuracies on Test Set**

Overall, the results confirm that the combination of TF-IDF vectorization and SVM provides a powerful, interpretable, and generalizable approach that surpasses existing published baseline models for e-commerce product classification.
