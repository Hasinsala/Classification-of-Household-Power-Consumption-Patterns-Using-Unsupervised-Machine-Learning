⚡ Classification of Household Power Consumption Patterns Using Unsupervised Machine Learning
🧠 Status: Ongoing | 🔗 Dataset Source:- https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
💡 Applying both unsupervised and supervised learning techniques to uncover behavioral energy usage patterns from real-world household electricity data.

📌 Project Overview
This project focuses on the analysis and classification of household electricity consumption patterns using machine learning. By leveraging a time-series dataset recorded over four years (2006–2010), the aim is to uncover hidden patterns, detect anomalies, and forecast energy usage to promote smart energy management and optimization.

🎯 Objectives
Apply unsupervised learning (e.g., K-Means) to discover clusters of similar energy usage behaviors.

Use supervised models (e.g., Random Forest, SVM, XGBoost) to predict household power consumption.

Perform extensive feature engineering and time-based analysis (hour, weekday, month).

Identify seasonal trends and anomalies in electricity consumption.

Optimize model performance using RFE, Mutual Information, cross-validation, and hyperparameter tuning.

📂 Dataset Description
Source: UCI Machine Learning Repository

Size: ~2 million+ records (one-minute resolution)

Duration: December 2006 – November 2010

Features:

Global_active_power (kW)

Global_reactive_power (kW)

Voltage (V)

Global_intensity (A)

Sub_metering_1 – Kitchen

Sub_metering_2 – Laundry

Sub_metering_3 – Water heater & AC

Date, Time

🧪 Methodology
🔹 Data Preprocessing
Handled missing values (~1.25%)

Extracted temporal features: hour, weekday, month

Normalized continuous features for consistent scaling

🔹 Unsupervised Learning (Clustering)
Applied K-Means clustering

Used Elbow Method to determine optimal cluster count

Analyzed behavioral patterns based on cluster groups

🔹 Supervised Learning (Prediction)
Trained models: Random Forest, SVM, XGBoost, Decision Tree, KNN, Linear Regression

Performed Recursive Feature Elimination (RFE) and Mutual Info Regression for feature selection

Evaluated models using R² Score and MSE

Used K-Fold Cross Validation and Grid Search for robust training

📈 Visualizations & Insights
Distribution plots of key variables (power, voltage, sub-metering zones)

Time-based trend graphs (hourly/weekly/monthly usage)

Correlation heatmaps to identify key relationships

Box plots for sub-metering zone analysis

Cluster visualizations for user behavior segmentation

🚀 Tools & Tech Stack
Languages: Python

Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost

ML Techniques: K-Means Clustering, Random Forest, SVM, Gradient Boosting, RFE, GridSearchCV

Environment: Jupyter Notebook / Google Colab

🔍 Current Progress
✅ Completed data cleaning, preprocessing, and feature extraction

✅ Initial clustering and interpretation completed

✅ Supervised models trained with high performance

⏳ Model refinement, anomaly detection, and final evaluation in progress

📌 Future Work
Incorporate deep learning models for time-series forecasting (e.g., LSTM)

Build an interactive dashboard for real-time pattern visualization

Extend to multi-household datasets for broader generalization

Apply Explainable AI (XAI) methods to interpret model decisions

📊 Results Summary (So far)
Model	MSE	R² Score
Linear Regression	0.00163	0.99855
Decision Tree	0.00227	0.99798
Random Forest	0.00132	0.99882
Gradient Boosting	0.00135	0.99880
K-Nearest Neighbors	0.00155	0.99862

🔥 Random Forest currently provides the best balance of accuracy, robustness, and training speed.

