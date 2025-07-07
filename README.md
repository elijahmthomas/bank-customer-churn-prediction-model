# Bank Customer Churn Prediction

This project predicts whether a bank customer will churn (leave) based on their profile data using Logistic Regression in Python.
---

## Tools Used
- Python (pandas, scikit-learn, matplotlib)
- VS Code for development

---

## What I did
- Cleaned and prepared the data
- Built and evaluated a logistic regression model
- Tuned prediction threshold to improve recall (default vs custom threshold)
- Visualized precision and recall trade-offs

---

## How to run
1. Install dependencies: `pip install pandas| scikit-learn | matplotlib`
2. Run `bank-churn-ml.py`
3. Check the output predictions in `churn_predictions.csv`

---

## Dataset
[Kaggle](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset?resource=download)

---

## Results and Breakdown

I evaluated the logistic regression model using two different thresholds: the default 0.5 and a custom lower threshold of 0.3 to boost recall for churn detection.

## Default Threshold (0.5)
Accuracy: 71% – the model predicted correctly for 71% of customers.

Churn Recall: 70% – the model correctly identified 70% of customers who actually churned.

Churn Precision: 37% – among those predicted to churn, 37% truly did.

This threshold offers a good balance but seems struggles with precision for churners. (it often mislabels non-churners as churners)

---

## Custom Threshold (0.3)
Churn Recall: 92% – very high recall, meaning we catch almost all churners.

Churn Precision: 26% – low precision, meaning many customers flagged as churners did not churn.

Accuracy: 47% – overall accuracy drops due to more false positives.

Lowering the threshold improves the model's ability to catch at-risk customers, which is valuable if the goal of the business is retention, even at the cost of alerting some false positives.

# Takeaways
Use default threshold for balanced performance.

Use custom threshold (0.3) if catching churners is more important than being precise (e.g. proactive outreach/retention efforts).

## About Me

Hello, I'm Elijah Thomas — a data analyst with a background in finance and a passion for turning messy data into meaningful insights. I created this project to showcase my skills in Python data wrangling, machine learning, and logistic regression. 

I'm actively seeking roles in data analytics, software engineering, or finance, where I can apply clean analytics and technical problem-solving to drive better decisions.

Connect with me on [LinkedIn](https://www.linkedin.com/in/elijahmthomas) or email me directly at [emthomas519@gmail.com]
