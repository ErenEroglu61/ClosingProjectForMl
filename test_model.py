import pickle
import pandas as pd


def math_score():
    with open("Models/closing_project.pkl", "rb") as f:
        saved_data = pickle.load(f)

        model = saved_data["model"]


    X_test_scaled = pd.read_csv("Test_Data/test_data.csv")
    print(model.predict(X_test_scaled))

def writing_score():
    with open("Models/closing_project_writing_score.pkl", "rb") as f:
        saved_data = pickle.load(f)

        model = saved_data["model"]

    X_test_scaled = pd.read_csv("Test_Data/test_data_writing_score.csv")
    print(model.predict(X_test_scaled))

def reading_score():
    with open("Models/closing_project_reading_score.pkl", "rb") as f:
        saved_data = pickle.load(f)

        model = saved_data["model"]

    X_test_scaled = pd.read_csv("Test_Data/test_data_reading_score.csv")
    print(model.predict(X_test_scaled))

if __name__ == "__main__":
    math_score()
    writing_score()
    reading_score()