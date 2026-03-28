import pandas as pd
import os

def preprocess_data():
    print("Loading dataset...")

    df = pd.read_csv("data/raw/fitness.csv")

    print("Preview:")
    print(df.head())

    # Lowercase columns
    df.columns = [col.lower() for col in df.columns]

    print("Columns:", df.columns)

    #  Use correct column names from your dataset
    df = df.rename(columns={
        "calories_burned": "calories",
        "session_duration (hours)": "workout",
        "workout_frequency (days/week)": "workout_freq",
        "fat_percentage": "body_fat"
    })

    #  Select features
    features = ["calories", "workout", "workout_freq", "bmi"]

    #  Target variable (use real column instead of fake one)
    target = "body_fat"

    df = df[features + [target]]

    df.dropna(inplace=True)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/processed_data.csv", index=False)

    print(" Preprocessing completed!")
    print("Saved at data/processed/processed_data.csv")

if __name__ == "__main__":
    preprocess_data()