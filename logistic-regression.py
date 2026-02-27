import csv
import math

# ==========================================
# PHASE 1: DATA PREPARATION
# ==========================================

dataset = []
with open('diabetes.csv', 'r') as file:
    reader = csv.reader(file)
    headers = next(reader)
    for row in reader:
        dataset.append([float(val) for val in row])

def get_median(values):
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    return sorted_vals[mid]

cols_to_clean = [1, 2, 3, 4, 5]
for col_idx in cols_to_clean:
    non_zero_vals = [row[col_idx] for row in dataset if row[col_idx] != 0]
    col_median = get_median(non_zero_vals)
    for row in dataset:
        if row[col_idx] == 0:
            row[col_idx] = col_median

# We use the whole dataset for training now since we are manually testing
train_set = dataset 

def get_mean_and_std(data, col_idx):
    vals = [row[col_idx] for row in data]
    mean = sum(vals) / len(vals)
    variance = sum((x - mean)**2 for x in vals) / len(vals)
    std = math.sqrt(variance)
    return mean, std

stats = [get_mean_and_std(train_set, i) for i in range(8)]

def standardize(data, stats):
    for row in data:
        for i in range(8):
            if stats[i][1] != 0:
                row[i] = (row[i] - stats[i][0]) / stats[i][1]

standardize(train_set, stats)

# ==========================================
# PHASE 2: MATHEMATICS & TRAINING
# ==========================================

def sigmoid(z):
    z = max(min(z, 250), -250) 
    return 1.0 / (1.0 + math.exp(-z))

def predict(row, slopes, intercept):
    y = intercept 
    for i in range(len(slopes)):
        y += slopes[i] * row[i]
    return sigmoid(y)

def evaluate_accuracy(data, slopes, intercept):
    correct = 0
    for row in data:
        probability = predict(row, slopes, intercept)
        predicted_class = 1 if probability >= 0.5 else 0
        actual_class = int(row[-1])
        if predicted_class == actual_class:
            correct += 1
    return correct / len(data)

print("Training the model... Please wait a moment.")
slopes = [0.0] * 8 
intercept = 0.0    
learning_rate = 0.01
epochs = 1000      

for epoch in range(epochs):
    for row in train_set:
        prediction = predict(row, slopes, intercept)
        actual_outcome = row[-1]
        error = prediction - actual_outcome
        intercept -= learning_rate * error
        for i in range(len(slopes)):
            slopes[i] -= learning_rate * error * row[i]

training_accuracy = evaluate_accuracy(train_set, slopes, intercept)
print(f"Training accuracy: {training_accuracy:.4f}")

# ==========================================
# PHASE 3: INTERACTIVE PREDICTION
# ==========================================

print("\nModel is ready! Let's test a patient.")
while True:
    print("\n--- Enter Patient Laboratory Attributes ---")
    user_inputs = []
    
    # 1. Ask the user for the 8 inputs one by one
    for i in range(8):
        while True:
            try:
                val = float(input(f"Enter {headers[i]}: "))
                user_inputs.append(val)
                break
            except ValueError:
                print("Invalid input. Please enter a valid number.")
    
    # 2. Standardize the user's input so the model can understand it
    standardized_input = []
    for i in range(8):
        if stats[i][1] != 0:
            std_val = (user_inputs[i] - stats[i][0]) / stats[i][1]
        else:
            std_val = 0
        standardized_input.append(std_val)
        
    # 3. Compute the prediction (0 or 1)
    probability = predict(standardized_input, slopes, intercept)
    predicted_class = 1 if probability >= 0.5 else 0
    
    # 4. Output the results
    print("\n=========================================")
    print(f"Computed Probability: {probability:.4f}")
    print(f"Model Output: {predicted_class}")
    
    if predicted_class == 1:
        print("Interpretation: 1 — With Diabetes")
    else:
        print("Interpretation: 0 — No Diabetes")
    print("=========================================")
    
    # Ask if the user wants to test another patient
    cont = input("\nWould you like to test another patient? (y/n): ")
    if cont.lower() != 'y':
        print("Exiting program...")
        break
