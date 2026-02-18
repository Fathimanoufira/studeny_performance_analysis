import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("student_data.csv")

df["Internet_Access"] = df["Internet_Access"].map({"Yes": 1, "No": 0})
df["Parent_Education"] = df["Parent_Education"].map({
    "High School": 0,
    "Graduate": 1,
    "Post Graduate": 2
})

X = df.drop("Final_Score", axis=1)
y = df["Final_Score"]

model = LinearRegression()
model.fit(X, y)

print("Student Performance Model Trained Successfully")