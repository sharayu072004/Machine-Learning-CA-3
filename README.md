*Title and Short Description*

Title: E-Commerce Product Classification using NLP

The purpose of this project is categorizing e-commerce product titles and descriptions into four groups namely Electronics, Household, Books, and Clothing and Accessories. The aim is to automatically classify product information to enhance search outcomes, product recommendations and inventory management in e-commerce websites. The issue is significant since online market places process big amounts of text data that are not in structured form and manual classification is time-consuming and inaccurate. This project has precise classification using effective and interpretable models using Natural Language Processing and Machine Learning. The Support Vector Machine (SVM) model was the best with highest accuracy of 96 percent.

*Dataset Source*

The dataset exploited in this project is obtained at Kaggle, a project by Saurabh Shahane under the name of E-commerce Text Classification Dataset. It has over 4000 samples of products separated in four broad categories. The samples consist of the title of a product, product description, and a label. Preprocessing activities carried on the data include: Elimination of duplicate rows and columns, Dealing with missing values by dropping incomplete records, Cleaning and normalizing text by converting it to lowercase and manually removing special characters, Tokenizing the text and transforming it into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) were used to clean and normalize the data and make it consistent so that it can be used to train the model.  
<img width="940" height="513" alt="image" src="https://github.com/user-attachments/assets/40ee54bf-8f45-4ffa-9cc3-e26520b23760" />


Figure 1 describes how word cloud image has been formed with the help of stop words, tokenization etc.

*Methods Used*

Four models of Machine Learning were run and compared to classify texts. Logistic Regression, Naive Bayes, Support Vector Machine (SVM), and Random Forest are some of the models that were used. All models were trained on TF-IDF vectorized text data. The SVM model demonstrated the best accuracy-generalization balance, because it represents the importance of a word in a document vis-a-vis the total dataset, and thus the TF-IDF technique is an effective feature extraction algorithm in text classification. Logistic Regression came in right behind with good interpretability. Random Forest and Naive Bayes worked fairly well and failed to make use of sparse TF-IDF features.

Steps to Run the Code:

*   To execute this project, we must clone the repository and change his directory to the project one. Then, all the Python requirements are to be installed to make the execution smooth. After the setup is done, it is possible to open the Jupyter Notebook file, called Machine\_Learning\_Project.ipynb.
*   Every cell was executed in a consecutive manner, which means that the whole workflow will be performed beginning with data loading and preprocessing up to model training, evaluation, and visualization.
*   Some of the most important outputs that are represented in the notebook include the accuracy scores, model comparisons and graphical results like the confusion matrices and the category distribution graphs.
*   Steps to Run the Code The first step towards carrying out this project is to download the repository and install all also needed Python dependencies to maintain compatibility. Once the environment is established, it is possible to open the file of the Jupyter Notebook called Machine\_Learning\_Project.ipynb and run it step by step. Data loading, preprocessing, and transformation with the help of TF-IDF vectorization are automatically done by the notebook.
*   After preprocessing, training of the models is completed followed by their evaluation and comparison in the same environment.
*   The notebook produces visual information such as accuracy graphs, confusion matrices and category distribution plots that facilitate in analysing the model performance and quality of classification.
*   Upon completion of the execution, all performance measures, comparison and final output of the model is presented which makes the workflow of the raw data to the evaluated results complete.

*Table:*



| Model                            | Key Characteristics                               | Advantages                                | Disadvantages                         |
| -------------------------------- | ------------------------------------------------- | ----------------------------------------- | ------------------------------------- |
| **Logistic Regression**          | Linear classifier based on sigmoid function       | Easy to interpret, fast to train          | Limited to linear relationships       |
| **Naive Bayes**                  | Probabilistic classifier based on Bayes’ theorem  | Works well with text data, simple         | Assumes independence between features |
| **Support Vector Machine (SVM)** | Margin-based classifier for high-dimensional data | Excellent accuracy, robust generalization | Slower for very large datasets        |
| **Random Forest**                | Ensemble of decision trees                        | Handles non-linear data well              | Can overfit sparse text features      |


<img width="616" height="504" alt="image" src="https://github.com/user-attachments/assets/0e68e8a6-9cbe-40e9-b310-749bc8b9646a" />

<img width="616" height="504" alt="image" src="https://github.com/user-attachments/assets/c7024195-ee78-472a-bcb9-5a1277f310e8" />


*Summary of Experiments / Results.*

In this research, four classic machine learning models that were used extensively included the use of Logistic Regression, naive bayes, the support vector machine (SVM), and the random forest. Text data was initially changed with the help of TF-IDF vectorization to change the names and descriptions of products into valuable numerical variables. Several hyperparameter tuning experiments were conducted including regularization parameter of SVM and Logistic Regression and the number of estimators and depth of Random Forest. The models were all measured with the use of conventional measures such as accuracy, precision, recall, and F1-score. This project had much better accuracy as compared to the already published methodologies in text classification literature such as the one that used Naive Bayes and Decision Trees in e-commerce data and got accuracy of about 88-91 percent. This implementation of the SVM model achieved 96 percent accuracy, exceeding the prior models that were based on frequency-based representations or simple probabilistic classifiers. Logistic Regression also competed effectively with a 94 percent accuracy rate which shows that linear models with an adequate feature engineering can produce good results. Visualization methods have been used to understand the behaviour of the models better. Confusion matrices were used to determine the instances that had been misclassified and particularly the similar types like Electronics and Household whereas the category distribution graphs ensured the dataset was balanced to all four classes. The keywords that had the most influence in the classification could also be seen in the TF-IDF feature importance plots which made the results interpretable. In general, the experimental results provide evidence that the TF-IDF vectorization combined with SVM provides a powerful, balanced, and interpretable model more than the existing published baseline models as well as gives strong generalization in e-commerce product categorization tasks.

Figure 2 Comparison of model Accuracies on Test Set
