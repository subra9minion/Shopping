import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = []
    labels = []

    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:

            # values for each row
            row_evidence = []
            row_label = 0

            # typecasting
            row["Administrative"] = int(row["Administrative"])
            row["Administrative_Duration"] = float(row["Administrative_Duration"])
            row["Informational"] = int(row["Informational"])
            row["Informational_Duration"] = float(row["Informational_Duration"])
            row["ProductRelated"] = int(row["ProductRelated"])
            row["ProductRelated_Duration"] = float(row["ProductRelated_Duration"])
            row["BounceRates"] = float(row["BounceRates"])
            row["ExitRates"] = float(row["ExitRates"])
            row["PageValues"] = float(row["PageValues"])
            row["SpecialDay"] = float(row["PageValues"])
            months = {"Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "June": 5, "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11}
            row["Month"] = months[row["Month"]]
            row["OperatingSystems"] = int(row["OperatingSystems"])
            row["Browser"] = int(row["Browser"])
            row["TrafficType"] = int(row["TrafficType"])
            if row["VisitorType"] == "Returning_Visitor": row["VisitorType"] = 1
            else: row["VisitorType"] = 0
            if row["Weekend"] == "FALSE": row["Weekend"] = 0
            else: row["Weekend"] = 1
            if row["Revenue"] == "TRUE": row["Revenue"] = 1
            else: row["Revenue"] = 0

            # updating values
            for k, v in row.items():
                if k != "Revenue":
                    row_evidence.append(v)
                else:
                    row_label = v
            
            # updating evidence, labels
            evidence.append(row_evidence)
            labels.append(row_label)

    return (evidence, labels)

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    # counts of each label
    true_positives = 0
    true_negatives = 0
    predicted_positives = 0
    predicted_negatives = 0

    # calculating counts of each label
    for predicted, actual in zip(predictions, labels):
        if actual == 1:
            true_positives += 1
            if predicted == 1:
                predicted_positives += 1
        else:
            true_negatives += 1
            if predicted == 0:
                predicted_negatives += 1

    # calculating TPR and TNR
    sensitivity = predicted_positives / true_positives
    specificity = predicted_negatives / true_negatives

    return (sensitivity, specificity)

if __name__ == "__main__":
    main()
