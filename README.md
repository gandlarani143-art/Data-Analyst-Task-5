
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.backends.backend_pdf import PdfPages

# Create output folder
os.makedirs("Task5_EDA_Output", exist_ok=True)

# Load Titanic Dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(url)

print("\n===== TITANIC DATASET =====")
print(df.head())

print("\n===== DATASET INFORMATION =====")
df.info()

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())

print("\n===== SURVIVAL COUNT =====")
print(df["survived"].value_counts())

print("\n===== PASSENGER CLASS COUNT =====")
print(df["pclass"].value_counts())

print("\n===== SEX COUNT =====")
print(df["sex"].value_counts())

# Fill missing values for selected analysis
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(
    df["embarked"].mode()[0]
)

# Set plot style
sns.set_theme(style="whitegrid")

# Save charts and PDF report
pdf_path = "Task5_EDA_Output/EDA_Report.pdf"

with PdfPages(pdf_path) as pdf:

    # Page 1: Report summary
    fig = plt.figure(figsize=(8, 10))
    fig.text(0.1, 0.92, "TITANIC DATASET - EDA REPORT",
             fontsize=18, weight="bold")

    summary = f"""
Total Rows: {df.shape[0]}
Total Columns: {df.shape[1]}

Survived Passengers: {(df['survived'] == 1).sum()}
Not Survived: {(df['survived'] == 0).sum()}

Average Age: {df['age'].mean():.2f}
Average Fare: {df['fare'].mean():.2f}

Missing values before cleaning:
{df.isnull().sum().to_string()}

Observations:
1. Passenger survival varied by passenger class.
2. Female passengers had a higher survival rate.
3. Passenger age and fare distributions were explored.
4. Some columns contained missing values.
5. Correlation analysis helps identify relationships.
"""
    fig.text(0.1, 0.82, summary, fontsize=11,
             verticalalignment="top")
    pdf.savefig(fig)
    plt.close(fig)

    # Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(df["age"], bins=20, kde=True)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.tight_layout()
    pdf.savefig()
    plt.savefig("Task5_EDA_Output/age_histogram.png")
    plt.close()

    # Boxplot
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="survived", y="fare", data=df)
    plt.title("Fare by Survival Status")
    plt.tight_layout()
    pdf.savefig()
    plt.savefig("Task5_EDA_Output/fare_boxplot.png")
    plt.close()

    # Countplot
    plt.figure(figsize=(8, 5))
    sns.countplot(x="survived", data=df)
    plt.title("Passenger Survival Count")
    plt.xlabel("Survived (0 = No, 1 = Yes)")
    plt.tight_layout()
    pdf.savefig()
    plt.savefig("Task5_EDA_Output/survival_count.png")
    plt.close()

    # Scatterplot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x="age", y="fare",
        hue="survived", data=df
    )
    plt.title("Age vs Fare")
    plt.tight_layout()
    pdf.savefig()
    plt.savefig("Task5_EDA_Output/scatterplot.png")
    plt.close()

    # Correlation Heatmap
    plt.figure(figsize=(8, 6))
    numeric_df = df.select_dtypes(include=np.number)
    sns.heatmap(numeric_df.corr(),
                annot=True, cmap="coolwarm",
                fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    pdf.savefig()
    plt.savefig("Task5_EDA_Output/heatmap.png")
    plt.close()

    # Pairplot
    pair = sns.pairplot(
        df[["survived", "pclass", "age", "fare"]],
        hue="survived",
        corner=True
    )
    pair.fig.suptitle("Titanic Pairplot", y=1.02)
    pair.savefig("Task5_EDA_Output/pairplot.png")
    pdf.savefig(pair.fig)
    plt.close("all")

print("\n===== EDA OBSERVATIONS =====")
print("1. Female passengers had a higher survival rate.")
print("2. Passenger class was related to survival.")
print("3. Age and fare distributions were analyzed.")
print("4. Missing values were identified and handled.")
print("5. Heatmap shows correlations between numerical features.")

print("\nEDA ANALYSIS COMPLETED SUCCESSFULLY!")
print("PDF Report:", pdf_path)
print("Charts saved in Task5_EDA_Output folder.")
# Data-Analyst-Task-5