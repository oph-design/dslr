from sklearn.metrics import accuracy_score
import pandas as pd

predicted_data = pd.read_csv("houses.csv")["Hogwarts House"].values
correct_data = pd.read_csv("test.csv")["Hogwarts House"].values
print(f"{accuracy_score(predicted_data, correct_data) * 100}%")
