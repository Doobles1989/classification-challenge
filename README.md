#### Classification-Challenge

## Overview

This project builds a spam detection classifier comparing Logistic Regression and Random Forest models on a dataset of word and character frequencies in emails.

## Dataset

	•	Source: UCI Machine Learning Repository
	•	URL: Spam Data CSV

## Steps

	1.	Retrieve Data: Import using Pandas and display.

data = pd.read_csv("https://static.bc-edx.com/ai/ail-v-1-0/m13/challenge/spam-data.csv")
data.head()


	2.	Prediction: Hypothesis - Random Forest will outperform Logistic Regression.
	3.	Preprocess Data: Split into train/test sets; scale features.

y = data['spam']
X = data.drop('spam', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


	4.	Train and Evaluate Models:
	•	Logistic Regression: Accuracy: 0.9357
	•	Random Forest: Accuracy: 0.9496

log_reg = LogisticRegression(random_state=1)
log_reg.fit(X_train_scaled, y_train)
log_reg_accuracy = accuracy_score(y_test, log_reg.predict(X_test_scaled))

rf_classifier = RandomForestClassifier(random_state=1)
rf_classifier.fit(X_train_scaled, y_train)
rf_accuracy = accuracy_score(y_test, rf_classifier.predict(X_test_scaled))


	5.	Save Predictions:

predictions_df.to_csv('log_reg_predictions.csv', index=False)



Conclusion

Random Forest performed best, supporting its use in complex classification like spam detection.
