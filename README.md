# Heart Failure Prediction

Predicting mortality risk in heart failure patients using machine learning.

## Dataset
- 299 patient records with 12 clinical features
- Target: DEATH_EVENT (survived or died)
- Source: Kaggle Heart Failure Clinical Records

## Models Used
- Logistic Regression
- Random Forest  
- SVM
- XGBoost
- kNN

## Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Random Forest | 85% | 0.89 |
| XGBoost | 83% | 0.87 |
| Logistic Regression | 80% | 0.84 |

## Key Findings
- Ejection fraction, age, and serum creatinine were top risk factors
- Random Forest performed best among all models tested

## Technologies
Python, Pandas, Scikit-learn, XGBoost, Matplotlib, Seaborn

## How to Run

```bash
git clone https://github.com/AbdulMohsin01/Heart-Failure-Prediction-Rate.git
cd Heart-Failure-Prediction-Rate
pip install -r requirements.txt
jupyter notebook Heart_Failure_Prediction.ipynb
Connect
GitHub: AbdulMohsin01

LinkedIn: abdulmohsin01
