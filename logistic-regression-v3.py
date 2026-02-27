import csv

PREDICTORS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
              "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
OUTCOME = "Outcome"

def read_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        return list(reader)

def compute_mean(values):
    values = [float(v) for v in values]
    return sum(values) / len(values)

def cross_deviation(x_values, y_values, x_mean, y_mean):
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))

def squared_deviation(values, mean):
    return sum((v - mean) ** 2 for v in values)

def slope(x_values, y_values):
    x_mean = compute_mean(x_values)
    y_mean = compute_mean(y_values)
    denom = squared_deviation(x_values, x_mean)
    if denom == 0:
        return 0.0
    return cross_deviation(x_values, y_values, x_mean, y_mean) / denom

def intercept(x_values, y_values, m):
    x_mean = compute_mean(x_values)
    y_mean = compute_mean(y_values)
    return y_mean - m * x_mean

def overall_intercept(values):
    y = sum(float(v) for v in values)
    return y / len(values)


def get_median(values):
    # Median is used to fill invalid 0 values in medical columns.
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    return sorted_vals[mid]


def get_mean_and_std(data_rows, predictor_idx):
    # Compute mean/std for one predictor column (used for standardization).
    vals = [row[predictor_idx] for row in data_rows]
    mean = sum(vals) / len(vals)
    variance = sum((x - mean) ** 2 for x in vals) / len(vals)
    std = variance ** 0.5
    return mean, std


def standardize_rows(data_rows, stats):
    # Apply z-score scaling: (x - mean) / std for each predictor.
    for row in data_rows:
        for i in range(len(PREDICTORS)):
            mean, std = stats[i]
            if std != 0:
                row[i] = (row[i] - mean) / std


def sigmoid(z):
    # Logistic (sigmoid) function: converts any real number to [0, 1].
    return 1.0 / (1.0 + (2.718281828459045 ** (-z)))


def predict_probability(weights, bias, row_values):
    z = bias
    for i in range(len(weights)):
        z += weights[i] * row_values[i]
    return sigmoid(z)


def train_logistic_regression(X_rows, y, learning_rate=0.01, epochs=1000):
    # Zero initialization of weights and bias. We will update them iteratively.
    weights = [0.0 for _ in range(len(PREDICTORS))]
    bias = 0.0

    sample_count = len(y)

    for _ in range(epochs):
        for i in range(sample_count):
            p = predict_probability(weights, bias, X_rows[i])
            adjustment = y[i] - p

            # Row-by-row (SGD-style) update
            for j in range(len(weights)):
                weights[j] += learning_rate * adjustment * X_rows[i][j]
            bias += learning_rate * adjustment

    return weights, bias


def evaluate_accuracy(weights, bias, X_rows, y):
    correct = 0
    for i in range(len(y)):
        p = predict_probability(weights, bias, X_rows[i])
        predicted_class = 1 if p >= 0.5 else 0
        if predicted_class == y[i]:
            correct += 1
    return correct / len(y)


def save_model(model_path, weights, bias, predictor_stats):
    # Save model parameters and preprocessing stats for future predictions.
    with open(model_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(weights)
        writer.writerow([bias])
        writer.writerow([stat[0] for stat in predictor_stats])  # means
        writer.writerow([stat[1] for stat in predictor_stats])  # stds


def load_model(model_path):
    # Load model parameters and preprocessing stats.
    with open(model_path, 'r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)
        weights = [float(value) for value in rows[0]]
        bias = float(rows[1][0])
        means = [float(value) for value in rows[2]]
        stds = [float(value) for value in rows[3]]
    predictor_stats = list(zip(means, stds))
    return weights, bias, predictor_stats


def preprocess_training_data(raw_rows, predictor_indices, outcome_index):
    # Convert rows to numeric predictor/outcome table.
    combined_rows = []
    for row in raw_rows:
        predictor_values = []
        for idx in predictor_indices:
            predictor_values.append(float(row[idx]))
        outcome_value = int(row[outcome_index])
        combined_rows.append(predictor_values + [outcome_value])

    # Replace invalid zeros in selected medical columns.
    cols_to_clean = [1, 2, 3, 4, 5]
    for col_idx in cols_to_clean:
        non_zero_vals = [row[col_idx] for row in combined_rows if row[col_idx] != 0]
        col_median = get_median(non_zero_vals)
        for row in combined_rows:
            if row[col_idx] == 0:
                row[col_idx] = col_median

    # Standardize predictors and return stats for future use.
    predictor_stats = [get_mean_and_std(combined_rows, i) for i in range(len(PREDICTORS))]
    standardize_rows(combined_rows, predictor_stats)

    X_predictors = [row[:len(PREDICTORS)] for row in combined_rows]
    y = [int(row[-1]) for row in combined_rows]
    return X_predictors, y, predictor_stats


def standardize_input_values(input_values, predictor_stats):
    standardized = []
    for i in range(len(PREDICTORS)):
        mean, std = predictor_stats[i]
        if std != 0:
            standardized.append((input_values[i] - mean) / std)
        else:
            standardized.append(0.0)
    return standardized


if __name__ == "__main__":
    model_path = "ogistic_model-v3.csv"
    mode = input("Choose mode ('train' or 'predict'): ").strip().lower()

    if mode == "train":
        csv_path = "diabetes.csv"
        rows = read_csv(csv_path)
        print(f"Loaded {len(rows)} rows from {csv_path}")

        header = rows[0]
        data_rows = rows[1:]
        predictor_indices = [header.index(name) for name in PREDICTORS]
        outcome_index = header.index(OUTCOME)

        X_predictors, y, predictor_stats = preprocess_training_data(
            data_rows, predictor_indices, outcome_index
        )

        weights, bias = train_logistic_regression(X_predictors, y)
        accuracy = evaluate_accuracy(weights, bias, X_predictors, y)
        save_model(model_path, weights, bias, predictor_stats)

        print(f"Model saved to {model_path}")
        print(f"Training accuracy: {accuracy:.4f}")

    elif mode == "predict":
        try:
            weights, bias, predictor_stats = load_model(model_path)
        except FileNotFoundError:
            print(f"No trained model found at {model_path}.")
            print("Run in 'train' mode first to create a model before predicting.")
            exit()
        print(f"Loaded model from {model_path}")
        print("Model is ready! Let's test a patient.")

        while True:
            print("\n--- Enter Patient Laboratory Attributes ---")
            input_values = []
            for predictor in PREDICTORS:
                while True:
                    try:
                        val = float(input(f"Enter {predictor}: "))
                        input_values.append(val)
                        break
                    except ValueError:
                        print("Invalid input. Please enter a valid number.")

            standardized_input = standardize_input_values(input_values, predictor_stats)
            probability = predict_probability(weights, bias, standardized_input)
            predicted_class = 1 if probability >= 0.5 else 0

            print("\n=========================================")
            print(f"Computed Probability: {probability:.4f}")
            print(f"Model Output: {predicted_class}")
            if predicted_class == 1:
                print("Interpretation: 1 - With Diabetes")
            else:
                print("Interpretation: 0 - No Diabetes")
            print("=========================================")

            cont = input("\nWould you like to test another patient? (y/n): ").strip().lower()
            if cont != "y":
                print("Exiting program...")
                break
    else:
        print("Invalid mode. Please run again and choose 'train' or 'predict'.")
