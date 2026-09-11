"""Train a TabPFN classifier on scikit-learn's Iris toy dataset."""

from dotenv import load_dotenv
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier


def main() -> None:
    load_dotenv()

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # CPU keeps this example runnable on any machine. One estimator is enough
    # for a quick toy demonstration; use the default for a stronger ensemble.
    model = TabPFNClassifier(device="cpu", n_estimators=1, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(f"First prediction: class {predictions[0]}")
    print(f"First class probabilities: {probabilities[0].round(3)}")


if __name__ == "__main__":
    main()
