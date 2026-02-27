# Logistic Regression Scripts Guide

This project contains three versions of a diabetes logistic regression script:

- `logistic-regression.py`
- `logistic-regression-v2.py`
- `logistic-regression-v3.py`

All scripts expect `diabetes.csv` in the same folder.

## Requirements

- Python 3 installed
- Run commands from this folder:
  - `c:\Users\cholo\OneDrive\Desktop\big-data-act1`

## 1) `logistic-regression.py`

### What it does
- Trains immediately when you run it
- Prints training accuracy
- Starts continuous interactive prediction

### Run
```powershell
python logistic-regression.py
```

### Prediction flow
- Enter 8 values (in this exact order):
1. Pregnancies
2. Glucose
3. BloodPressure
4. SkinThickness
5. Insulin
6. BMI
7. DiabetesPedigreeFunction
8. Age
- Script prints probability and class (`0` or `1`)
- Type `y` to test another patient, `n` to exit

## 2) `logistic-regression-v2.py`

### What it does
- Uses mode selection: `train` or `predict`
- Saves trained model to `logistic_model.csv`

### Run
```powershell
python logistic-regression-v2.py
```

### Train mode
1. Type `train`
2. Wait for training to finish
3. Model file `logistic_model.csv` is created

### Predict mode
1. Run again and type `predict`
2. Enter 8 values in the same order listed above
3. Script prints probability and predicted class

## 3) `logistic-regression-v3.py`

### What it does
- Uses mode selection: `train` or `predict`
- Uses preprocessing + continuous prediction loop
- Saves model/stats to `ogistic_model-v3.csv` (current filename used by script)

### Run
```powershell
python logistic-regression-v3.py
```

### Train mode
1. Type `train`
2. Wait for training to finish
3. Model file is saved

### Predict mode
1. Run again and type `predict`
2. Enter 8 values in the same order listed above
3. Script prints probability and class
4. Type `y` to continue predicting, `n` to exit

## 5 Sample Rows For Testing

Use these predictor-only rows (8 values each), in order:

1. `6,148,72,35,0,33.6,0.627,50`
2. `1,85,66,29,0,26.6,0.351,31`
3. `8,183,64,0,0,23.3,0.672,32`
4. `1,89,66,23,94,28.1,0.167,21`
5. `0,137,40,35,168,43.1,2.288,33`

Tip: for scripts that ask one field at a time, enter each number in sequence.
