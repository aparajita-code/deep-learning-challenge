# deep-learning-challenge



***Analysis of Machine Learning Models to Predict Funding Success for Alphabet Soup***
**Introduction**
This report details the analysis and evaluation of three TensorFlow Keras models designed to predict the success of funding applicants for the nonprofit foundation Alphabet Soup. Each model was tested with varying input neurons, but consistently resulted in an accuracy of 72%. The dataset contains metadata about over 34,000 organizations that received funding from Alphabet Soup.

**Purpose of the Analysis**
The primary goal is to create a binary classifier using machine learning and neural networks to predict whether an applicant will be successful if funded by Alphabet Soup. This tool aims to assist Alphabet Soup in selecting applicants with the highest potential for success.

******Methodology***
Data Preparation:

Loading Data: The provided CSV file containing over 34,000 records was loaded into a DataFrame.

Dropping Columns: Dropped EIN and NAME columns before scaling the data to ensure irrelevant data did not skew results.

Feature Selection: Excluded the identification columns (EIN and NAME). Selected features included APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT, except IS_SUCCESSFUL.

Data Cleaning: Handled missing values, encoded categorical variables, and normalized numerical features.



***Model Architecture:***

**First Attempt:**
nn_model = tf.keras.models.Sequential()
hidden_nodes_layer1 = 8
hidden_nodes_layer2 = 5
Model 2 (Second Attempt):

**Seconnd Attempt**
nn_model = tf.keras.models.Sequential()
input_features_total = len(X_train[0])
hidden_nodes_layer1 = 10
hidden_nodes_layer2 = 10
Model 3 (Third Attempt):

**Third Attempt**
nn_model = tf.keras.models.Sequential()
input_features_total = len(X_train[0])
hidden_nodes_layer1 = 80
hidden_nodes_layer2 = 30

**Training and Evaluation:**

Models were trained using the same parameters: batch size, epochs, and optimizer (Adam).

The primary evaluation metric was accuracy.

**Results**
Model 1: Input layer with 8 neurons.

Accuracy: 72%

Loss: 55%
____________________________________
Model 2: Input layer with 10 neurons.

Accuracy: 72%

Loss: 55%
______________________________________
Model 3: Input layer with 80 neurons.

Accuracy: 72%

Loss: 56%
________________________________________
**Summary of Results**
All three models achieved an accuracy of 72%, indicating that varying the number of input neurons did not significantly impact the model's performance. This consistency suggests that other factors, such as data quality or model architecture, might play a more crucial role.

***Answering Key Questions:***
**What dataset was used?**

A CSV file containing data on over 34,000 organizations funded by Alphabet Soup.

**What features were selected?**

APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT.

**What preprocessing steps were taken?**

Handling missing values, encoding categorical variables, normalizing numerical features. Dropped EIN and NAME columns before scaling the data.

**What models were built?**

Three TensorFlow Keras models with input layers of 8, 10, and 80 neurons.

**What were the results?**

All models achieved an accuracy of 72%.

**What conclusions were drawn?**

Varying input neurons did not significantly impact model accuracy, suggesting other factors might be more critical.

***Alternative Model Approach***
To improve the model's performance, a different approach could be considered, such as using a Random Forest classifier:

Reason for Random Forest: This ensemble method can handle a large number of input features, reduce overfitting, and improve predictive accuracy.

Implementation: Preprocess the data similarly and train a Random Forest classifier. Evaluate its performance and compare it with the neural network models.

***Conclusion***
This analysis demonstrates that while varying the number of input neurons did not significantly affect model accuracy, exploring alternative model architectures and preprocessing techniques might yield better results. A Random Forest classifier is a potential alternative that could provide better performance and robustness.