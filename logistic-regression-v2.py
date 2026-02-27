import csv

PREDICTORS = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
              "Insulin","BMI","DiabetesPedigreeFunction","Age"]
OUTCOME = "Outcome"


def read_csv(file_path):
    # Read all rows from the CSV into a list of lists.
    data = []
    with open(file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data


def sigmoid(z):
    # Logistic (sigmoid) function: converts any real number to [0, 1].
    return 1.0 / (1.0 + (2.718281828459045 ** (-z)))


def predict_probability(weights, bias, row_values):
    # Compute z = w1*x1 + w2*x2 + ... + w8*x8 + b
    z = bias
    for i in range(len(weights)):
        z += weights[i] * row_values[i]
    # Convert z into probability p.
    return sigmoid(z)


def train_logistic_regression(X_rows, y, learning_rate, epochs):
    # Start with zero weights and zero bias.
    weights = [0.0 for _ in range(len(PREDICTORS))]
    bias = 0.0
    sample_count = len(y)

    # Repeat full-dataset updates for a fixed number of epochs.
    for epoch in range(epochs):
        # These accumulate gradients across all samples in the dataset.
        weight_gradients = [0.0 for _ in range(len(weights))]
        bias_gradient = 0.0

        for i in range(sample_count):
            p = predict_probability(weights, bias, X_rows[i])
            # Difference between actual label and predicted probability.
            adjustment = y[i] - p

            # Accumulate gradient contribution for each weight.
            for j in range(len(weights)):
                weight_gradients[j] += adjustment * X_rows[i][j]
            bias_gradient += adjustment

        # Apply average gradient update once per epoch (batch gradient descent).
        for j in range(len(weights)):
            weights[j] += learning_rate * (weight_gradients[j] / sample_count)
        bias += learning_rate * (bias_gradient / sample_count)

        if (epoch + 1) % 100 == 0:
            print(f"Finished epoch {epoch + 1}/{epochs}")

    return weights, bias


def evaluate_model(weights, bias, X_rows, y):
    correct = 0
    for i in range(len(y)):
        p = predict_probability(weights, bias, X_rows[i])
        # Classification rule requested: p >= 0.5 => 1 else 0
        predicted_class = 1 if p >= 0.5 else 0
        if predicted_class == y[i]:
            correct += 1
    return correct / len(y)


def save_model(model_path, weights, bias):
    # Save weights on line 1 and bias on line 2.
    with open(model_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(weights)
        writer.writerow([bias])


def load_model(model_path):
    # Load weights from line 1 and bias from line 2.
    with open(model_path, 'r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)
        weights = [float(value) for value in rows[0]]
        bias = float(rows[1][0])
    return weights, bias


if __name__ == "__main__":
    model_path = "logistic_model.csv"
    mode = input("Choose mode ('train' or 'predict'): ").strip().lower()

    if mode == "train":
        csv_path = "diabetes.csv"
        rows = read_csv(csv_path)
        print(f"Loaded {len(rows)} rows from {csv_path}")

        header = rows[0]
        data_rows = rows[1:]

        predictor_indices = [header.index(name) for name in PREDICTORS]
        outcome_index = header.index(OUTCOME)

        # Build row-wise data:
        # each X_rows[i] is one patient with 8 predictor values.
        X_rows = []
        y = []

        for row in data_rows:
            predictor_values = []
            for col_idx in predictor_indices:
                predictor_values.append(float(row[col_idx]))
            X_rows.append(predictor_values)
            y.append(int(row[outcome_index]))

        # Hyperparameters (training controls).
        learning_rate = 0.0001
        epochs = 1000

        # Train and evaluate the model.
        weights, bias = train_logistic_regression(X_rows, y, learning_rate, epochs)
        accuracy = evaluate_model(weights, bias, X_rows, y)

        # Save learned parameters for future predictions.
        save_model(model_path, weights, bias)
        print(f"\nModel saved to {model_path}")

        # Show learned model parameters and training accuracy.
        print("\nLearned parameters:")
        for i in range(len(weights)):
            print(f"w{i + 1} ({PREDICTORS[i]}): {weights[i]:.6f}")
        print(f"bias: {bias:.6f}")
        print(f"\nTraining accuracy: {accuracy:.4f}")

    elif mode == "predict":
        weights, bias = load_model(model_path)
        print(f"Loaded model from {model_path}")

        # Ask user for one value per predictor.
        input_values = []
        for predictor in PREDICTORS:
            value = float(input(f"Enter {predictor}: "))
            input_values.append(value)

        p = predict_probability(weights, bias, input_values)
        predicted_class = 1 if p >= 0.5 else 0

        print(f"\nPredicted probability: {p:.4f}")
        print(f"Predicted class: {predicted_class}")
    else:
        print("Invalid mode. Please run again and choose 'train' or 'predict'.")
