# Required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# 1. Initial Overview
print("Shape:", df.shape)
print("\nInfo:")
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())

# 2. Data Cleaning
# Fill missing Age with median
df['Age'] = df['Age'].fillna(df['Age'].median())
# Fill Embarked with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
# Drop Cabin due to too many missing values
df = df.drop(columns='Cabin')

# 3. Basic Stats
print("\nDescribe:\n", df.describe(include='all'))

# 4. Visualization: Survival Count
sns.countplot(data=df, x='Survived')
plt.title("Survival Count")
plt.show()

# 5. Survival by Gender
sns.countplot(data=df, x='Survived', hue='Sex')
plt.title("Survival by Gender")
plt.show()

# 6. Survival by Pclass
sns.countplot(data=df, x='Survived', hue='Pclass')
plt.title("Survival by Passenger Class")
plt.show()

# 7. Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# 8. Heatmap of Correlation
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# 9. Fare vs Survival
sns.boxplot(x='Survived', y='Fare', data=df)
plt.title("Fare vs Survival")
plt.show()
