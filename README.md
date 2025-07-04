# TalentSol - Applicant Tracking System Learning Project (Machine Learning Component)

This repository contains the machine learning component of the TalentSol project, focusing on training a supervised classifier to predict and recommend candidates for prioritization for a given job role and description. This is a learning project exploring different text embedding techniques and model evaluation for a real-world recruitment use case.

## Project Goal

The primary goal of this project is to train a supervised logistic regression model that can predict which job applicants are the "Best Match" for a given job role and description, thereby helping recruiters prioritize candidates during the screening process.

## Dataset

The project uses the `job_applicant_dataset.csv` dataset, which contains information about job applicants, their resumes, and whether they were considered a "Best Match" for a specific job role and description.

**Dataset Source:** The dataset is sourced from Kaggle: [Recruitment Dataset](https://www.kaggle.com/datasets/surendra365/recruitement-dataset)

## Preprocessing and Feature Engineering

Before we can train a machine learning model on text data like job descriptions and resumes, we need to convert the words and phrases into a numerical format that the model can understand. This process is called **Feature Engineering**, where we create meaningful numerical features from the raw data.

The following preprocessing and feature engineering steps were performed on the dataset:

1.  **Data Loading:** The `job_applicant_dataset.csv` file was loaded into a pandas DataFrame.
2.  **Identifier and Sensitive Feature Removal:** Columns like 'Job Applicant Name', 'Job Applicant ID', 'Age', and 'Gender' were removed early in the preprocessing pipeline. These columns contain **personally identifiable information (PII)** and were removed to mitigate risks of re-identification and ensure data privacy, aligning with responsible AI considerations. They were also not considered relevant or appropriate for the prioritization task in this version of the project, as the focus is on text matching between the resume/CV and the job description/job role.
3.  **Categorical Feature Splitting:** The 'Race' column, containing multiple race values, was split into separate columns ('Race1' and 'Race2') to handle different race categories.
4.  **Missing Value Handling:** Checked for and confirmed no missing data points were present after initial cleaning.
5.  **Handling Class Imbalance:** The dataset had more candidates who were *not* a "Best Match" than those who were. To prevent the model from being biased towards the majority class, we used a technique called **oversampling** to create more examples of the minority class ("Best Match").
6.  **Categorical Encoding:** Non-numerical categories like 'Ethnicity', 'Job Roles', 'Race1', and 'Race2' were converted into numerical representations using One-Hot Encoding.
7.  **Text Feature Embedding:** This is where we convert the text from 'Job Description' and 'Resume' into numbers. We explored different methods:
    *   **TF-IDF Vectorization (Term Frequency-Inverse Document Frequency):** Calculates a score for each word based on its frequency within a document and its rarity across all documents. We also explored and optimized **n-grams** (sequences of words) to capture phrase meaning, finding that bigrams (1,2) improved performance.
    *   **Word2Vec Embeddings:** Learns dense numerical vector representations for words, capturing semantic relationships. Document vectors were created by averaging word vectors.
    *   **Hybrid Embeddings:** Combinations of TF-IDF and Word2Vec features were created to leverage the strengths of both approaches. An optimized hybrid approach used TF-IDF with tuned n-grams and Word2Vec.

## Modeling and Evaluation

A **Logistic Regression** model was chosen as the classifier. Various model configurations (using different text embedding techniques and parameters) were trained and evaluated using **5-fold Stratified Cross-Validation**. This cross-validation approach is particularly suitable for our upsampled dataset as it ensures that each fold has a representative proportion of both classes ('Best Match' and 'Not Best Match').

Hyperparameter tuning for the Logistic Regression model (specifically the regularization strength `C` and the type of penalty `l1` or `l2`) and, where applicable, TF-IDF vectorizer parameters (like `ngram_range`), was performed using **Grid Search Cross-Validation (`GridSearchCV`)**.

We explored and tuned the following six model configurations:

1.  **Logistic Regression with Tuned TF-IDF (Default N-grams):** Tuned `C` and `penalty` for Logistic Regression using TF-IDF features with a default `ngram_range` of (1,1).
2.  **Logistic Regression with Tuned Word2Vec:** Tuned `C` and `penalty` for Logistic Regression using Word2Vec features.
3.  **Logistic Regression with Tuned, N-gram Optimized TF-IDF:** Tuned the `ngram_range` for the TF-IDF vectorizers (exploring (1,1), (1,2), (1,3)) while using the best `C` and `penalty` found in the initial Tuned TF-IDF model tuning.
4.  **Logistic Regression with Hybrid Embeddings (Default TF-IDF + Word2Vec):** Combined default TF-IDF (ngram\_range=(1,1)) and Word2Vec features, then tuned `C` and `penalty` for the Logistic Regression model on this combined feature set.
5.  **Logistic Regression with Optimized Hybrid Embeddings (Optimized TF-IDF + Word2Vec):** Combined optimized TF-IDF (using the tuned `ngram_range` from configuration 3) and Word2Vec features, then tuned `C` and `penalty` for the Logistic Regression model on this combined feature set.
6.  **Weighted Hybrid Model (Tuned TF-IDF + Tuned, N-gram Optimized TF-IDF):** This is a simple ensemble model. Instead of training a single Logistic Regression model on a combined feature set, it averages the predicted probabilities from the separately trained Tuned TF-IDF model (configuration 1) and the Tuned, N-gram Optimized TF-IDF model (configuration 3). The decision threshold for this averaged probability was then optimized.

The models were primarily evaluated based on standard metrics (Accuracy, Precision, Recall, F1-score, ROC AUC). Crucially, **Precision-Recall curves** were analyzed for each model's predicted probabilities, and **decision thresholds were optimized to achieve a target recall level** (approximately 70%). This optimization aligns the model's decision-making with the project's goal of prioritizing candidates and minimizing false negatives. Confusion matrices were visualized at these optimized thresholds to understand the specific performance trade-offs in terms of true positives, false positives, true negatives, and false negatives.

**Methodology for Prediction (How the Model Makes a Prediction):**

For a new applicant, the prediction process involves: applying the *same* preprocessing and feature engineering steps (using the fitted transformers from the chosen trained pipeline, e.g., the Optimized TF-IDF pipeline), scaling the features with the *same* StandardScaler, feeding the scaled features to the trained Logistic Regression model to get a probability score, and finally applying the **optimized decision threshold** (derived during evaluation) to classify the applicant as 'Best Match' (1) or 'Not Best Match' (0). For the Weighted Hybrid model, this would involve getting probabilities from its two constituent models and averaging them before applying the optimized threshold.

**Responsible AI Considerations:**

Responsible AI considerations were incorporated by removing sensitive demographic features ('Age', 'Gender', 'Race' split handled carefully) and using evaluation metrics (like recall and confusion matrices at optimized thresholds) that are relevant to the task's goals while acknowledging potential biases. Logistic Regression's relative interpretability was also a factor. Further steps for a production system would include bias audits, fairness metrics, human oversight, and transparency.

## Results

The models were evaluated at decision thresholds optimized to achieve approximately 70% recall. The key results are summarized below:

*   **Tuned TF-IDF Model:** Precision ~0.62, Recall ~0.70, F1-score ~0.66, ROC AUC ~0.69.
*   **Tuned Word2Vec Model:** Precision ~0.54, Recall ~0.70, F1-score ~0.61, ROC AUC ~0.58.
*   **Hybrid Model (Default TF-IDF + Word2Vec):** Precision ~0.63, Recall ~0.70, F1-score ~0.66, ROC AUC ~0.70.
*   **Optimized TF-IDF Model (Tuned N-grams):** Precision ~0.66, Recall ~0.70, F1-score ~0.68, ROC AUC ~0.73.
*   **Optimized Hybrid Model (Optimized TF-IDF + Word2Vec):** Precision ~0.67, Recall ~0.70, F1-score ~0.68, ROC AUC ~0.75.
*   **Weighted Hybrid Model (TF-IDF + Optimized TF-IDF):** Precision ~0.55, Recall ~0.70, F1-score ~0.62, ROC AUC ~0.59.

The **Optimized TF-IDF Model (Tuned N-grams)** and the **Optimized Hybrid Model (Optimized TF-IDF + Word2Vec)** consistently showed the best performance at the target recall, achieving the highest precision, F1-scores, and ROC AUCs. The Weighted Hybrid model did not outperform the individual optimized models.

Considering the balance between performance, engineering complexity, and **cost efficiency of inference/runtime**, the **Optimized TF-IDF Model (Tuned N-grams)** is selected as the best-performing model for this applicant prioritization task. For high-throughput scenarios like initial candidate screening, the relative simplicity and lower computational overhead of TF-IDF compared to dense embeddings (like Word2Vec) can translate to more cost-effective deployment and faster inference times. While the Optimized Hybrid model shows very similar performance, the pragmatic considerations of engineering effort and operational cost favor the Optimized TF-IDF model.

## Saved Artifacts

The following artifacts from the modeling process have been saved:

*   `upsampled_training_data.csv`: The final upsampled and preprocessed dataset used for training.
*   `optimized_tfidf_logistic_regression_pipeline.joblib`: The trained scikit-learn Pipeline object for the **Optimized TF-IDF Model (Tuned N-grams)**. This pipeline includes the preprocessor (with optimized TF-IDF vectorizers and OneHotEncoder), the scaler, and the Logistic Regression model. This is the recommended artifact for making predictions on new data.

## Inference and Integration

To use the trained model for predicting on new job applicants and integrating into a system:

1.  **Load the Trained Pipeline:** Load the `optimized_tfidf_logistic_regression_pipeline.joblib` file using `joblib.load()`.
2.  **Predict:** Apply the loaded pipeline's `predict_proba()` method to new, unseen applicant data (after ensuring it has the expected column structure). The output probabilities can then be thresholded using the optimized decision threshold (approximately 0.4994 based on the evaluation aiming for 70% recall).

## Engineering Integration Recommendations

For integrating this model into a production ATS (like TalentSol), consider these principles and recommended services:

*   **Start Simple and Iterate:** Begin with a straightforward and lightweight architecture. This is a **key engineering tenet**. For a problem like initial candidate screening, which involves handling a high volume of candidates and prioritizing for high recall, starting simple allows for faster experimentation and testing to quickly determine if the solution is effective before adding complexity. Avoid unnecessary complexity ("bloat") that could negatively impact inference performance or operational costs. Iterate and add complexity only when necessary to meet specific requirements.
*   **Explainability and Transparency Document:** Create a clear and easy-to-understand document explaining how the model works, its limitations, the data used, and how predictions are made. This document is crucial for users of the system (recruiters, HR personnel) and other stakeholders (business leads, legal) to build trust, ensure fair usage, and understand the model's impact on the hiring process.
*   **Real-time Data Pipeline:** Implement an efficient pipeline to preprocess new applicant data using the loaded pipeline's transformers. Recommended services for building data pipelines include serverless functions (e.g., Cloud Functions, Lambda), containerization platforms (e.g., Cloud Run, Kubernetes), or batch processing services (e.g., Dataflow, EMR) depending on latency and throughput requirements.
*   **Data and Model Storage:** Securely store raw data, processed data, and the trained model pipeline. Cloud Object Storage (e.g., Cloud Storage, S3, Blob Storage) is suitable for raw data and model files. Managed databases (e.g., Cloud SQL, RDS, Azure SQL) can store structured applicant data. Consider a Feature Store for managing and serving features consistently.
*   **Latency:** Monitor and optimize prediction latency. The chosen Optimized TF-IDF model's sparse features generally contribute to lower latency compared to dense embeddings, but overall pipeline efficiency is key.
*   **Memory:** Be mindful of the memory footprint of the loaded pipeline, especially for the TF-IDF vectorizer's vocabulary (which can be large for n-grams).
*   **Vector Databases:** For models utilizing dense vector embeddings (like the Word2Vec and Hybrid models), consider the use of a dedicated vector database. This can be essential for efficient storage, indexing, and retrieval of these embeddings at scale, particularly if implementing features beyond simple classification, such as semantic search or candidate similarity matching. While the selected Optimized TF-IDF model generates sparse features and may not require a vector database for core prediction, understanding the architectural implications of embedding-based models is valuable for evaluating the engineering complexity and cost trade-offs.
*   **Scalability:** Deploy the model as a scalable service.
*   **Monitoring:** Implement monitoring for performance tracking (e.g., inference time, error rates), data drift (changes in input data characteristics), and model drift (degradation in model performance over time).
*   **API Endpoint:** Provide a clear and well-documented API for seamless integration with other components of the ATS.
*   **Security:** Ensure secure handling and storage of sensitive applicant data and the model artifact.
*   **Versioning:** Use a system for model versioning to manage updates and rollbacks effectively.

## Resume Optimization Tips (Based on TF-IDF Model)

Based on the selection of the Optimized TF-IDF model as the best performer, here are some tips for job applicants to optimize their resumes/CVs:

*   **Match Keywords from the Job Description:** Since TF-IDF heavily weights terms that are frequent in a document but rare across others, make sure to use keywords and phrases directly from the job description. The model will likely give higher importance to resumes that contain these specific terms.
*   **Consider N-grams:** The optimized model uses n-grams (sequences of words like "project management" or "data analysis"). Ensure your resume includes these multi-word phrases if they are relevant to the job description. Avoid breaking up important phrases.
*   **Tailor Your Resume for Each Application:** A generic resume is less likely to score high with a TF-IDF-based model. Customize your resume to align with the specific language used in each job description you apply for.
*   **Use Relevant Technical and Soft Skills Terminology:** Include industry-specific jargon, technical skills (e.g., programming languages, software), and relevant soft skills (e.g., communication, teamwork) that are mentioned in the job description or are standard for the role.
*   **Structure and Formatting:** While TF-IDF primarily focuses on the text content, clear formatting can help ensure that text extraction processes accurately capture the words and phrases from your resume. Use standard sections (Summary, Experience, Skills, Education).
*   **Quantify Achievements:** While not directly related to TF-IDF weighting, quantifying your achievements (e.g., "Increased sales by 15%") provides concrete evidence of your skills and impact, which is valuable information regardless of the underlying model.

By focusing on using relevant and specific keywords and phrases from the job description, applicants can increase the likelihood of their resume being highly scored by a TF-IDF-based matching system.

Refer to the code notebook for detailed implementation of the preprocessing, modeling, and evaluation steps.
