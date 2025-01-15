# SMS Spam Detector
Module 21 Challenge

This project focuses on building a machine learning application to classify SMS messages as either spam or not spam ("ham"). The solution employs a linear Support Vector Classification (SVC) model for text classification and provides an intuitive, web-based interface using Gradio for real-time user interaction.

The primary goal of this project is to create a functional and efficient tool for detecting spam messages, enabling users to quickly identify and filter out unwanted content. By leveraging machine learning techniques and user-friendly design principles, the application combines technical rigor with practical utility.

---

## Project Objectives

1. **Automated Spam Detection**:  
   Provide an accurate and automated way to identify spam messages, saving users time and effort compared to manual screening.

2. **Interactive User Experience**:  
   Design a simple and accessible interface to make the technology usable for non-technical users.

3. **Scalable Machine Learning Pipeline**:  
   Develop a scalable and maintainable pipeline for training and deploying the spam classification model.

4. **Educational Value**:  
   Serve as an example of how machine learning can be applied to real-world problems, such as filtering unwanted text messages.

---

## Project Features

### 1. **Data Processing and Preparation**
   - **Dataset**:  
     The application uses the `SMSSpamCollection.csv` dataset, which contains labeled SMS messages categorized as "ham" (not spam) or "spam".
   - **Data Transformation**:  
     Text data is preprocessed and transformed into numerical features using the Term Frequency-Inverse Document Frequency (TF-IDF) method. This step converts textual information into a format suitable for machine learning.

### 2. **Machine Learning Model**
   - **Linear Support Vector Classification (SVC)**:  
     The project employs a linear SVC model due to its efficiency and effectiveness in text classification tasks.
   - **Training and Testing**:  
     The dataset is split into training (67%) and testing (33%) sets to ensure robust model evaluation and performance measurement.

### 3. **Spam Prediction Functionality**
   - The model predicts whether a given SMS message is spam or ham.
   - For each prediction, the user receives a detailed message explaining the result.

### 4. **User Interface with Gradio**
   - A Gradio-based interface provides a simple, interactive web application.
   - Users can input SMS messages in a textbox and receive real-time classification results in another textbox.

### 5. **Real-Time Testing**
   - The application processes messages instantly, allowing users to test multiple inputs efficiently.

---

## Technical Highlights

1. **Text Vectorization with TF-IDF**:  
   TF-IDF is used to quantify the importance of words in the dataset, creating feature vectors for classification.
   
2. **Pipeline Integration**:  
   A scikit-learn pipeline integrates vectorization and classification, streamlining the process and ensuring consistency between training and prediction.

3. **Model Deployment**:  
   Gradio enables quick deployment of the machine learning model as a web-based tool, providing a public URL for user access.

---

## Example Use Cases

1. **Personal Use**:  
   Individuals can use the app to screen potentially malicious or unwanted SMS messages.

2. **Educational Tool**:  
   Students and educators in machine learning can explore how text classification models are built and deployed.

3. **Prototype for Businesses**:  
   Organizations dealing with SMS-based communications can use the app as a prototype to integrate spam detection into their systems.

---

## Project Workflow

1. **Data Loading**:
   - Load the SMS dataset into a pandas DataFrame.
   - Inspect the dataset for missing values and ensure it is clean.

2. **Feature Extraction**:
   - Extract features using `TfidfVectorizer` to convert text into numerical data.
   - Set up a pipeline to combine vectorization and classification.

3. **Model Training**:
   - Split the data into training and testing sets.
   - Train the linear SVC model using the training data.

4. **Prediction and Feedback**:
   - Use the trained model to classify new messages as spam or ham.
   - Provide user-friendly feedback based on the classification.

5. **Gradio Interface Development**:
   - Build a Gradio interface to accept SMS input and display predictions.
   - Launch the application locally or share it through a public URL.

---
# Sample View of the Gradio Application

Below is a visual representation of the Gradio interface for this project:

![Gradio Application Example](https://github.com/user-attachments/assets/2e2aae61-c0b6-444d-ae12-0d335f1d83f4)

Below is a visual representation of the "Ham" classification:

![Gradio Application "Ham" Classification Example](https://github.com/user-attachments/assets/33a48174-3b7c-418e-abc8-90572c28113b)

Below is a visual representation of the "Spam" classification:

![Gradio Application "Spam" Classification Example](https://github.com/user-attachments/assets/eb79395f-8d5a-4908-9baa-88903dcbc097)


### Application Features:
- **Input Box**: A textbox where users can enter an SMS message for classification.
- **Output Box**: A textbox that displays the classification result as either "Spam" or "Ham" (not spam).
- **User-Friendly Labels**: Clear labels guide the user on where to enter their message and view results.

---

## Potential Future Applications and Enhancements

1. **Multilingual Support**:  
   Expand the model to detect spam messages in multiple languages.

2. **Integration with Messaging Apps**:  
   Incorporate the model into messaging platforms for real-time spam filtering.

3. **Enhanced Features**:  
   Add capabilities like bulk message classification and message analysis (e.g., detecting phishing attempts).

4. **Model Optimization**:  
   Experiment with other machine learning models, such as ensemble methods or neural networks, to improve accuracy.

5. **Dashboard Features**:  
   Build an analytics dashboard to display classification metrics and insights about user-submitted messages.

---
## Resources

I worked with the tutor to address a few minor issues with the code including setting the stopwords to English and modifying the index.
