# Loads savedmodel.pth, evaluates accuracy on the stored test set, and prints it
import joblib
from sklearn.metrics import accuracy_score

def main():
    data = joblib.load("savedmodel.pth")
    model = data["model"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[test.py] Test accuracy is: {acc:.4f}")


if __name__ == "__main__":
    main()