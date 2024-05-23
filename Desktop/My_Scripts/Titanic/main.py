import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load data
df = pd.read_csv ('/Users/richardhindley/Desktop/My_Scripts/Titanic/titanic_data/train.csv')

#Display head
print(df.head())

#Data Cleaning and Analysis
df.dropna(inplace=True)
df['Age'] = df['Age'].fillna(df['Age'].median())

#Summaries
# Summary statistics
print(df.describe())

# Value counts for categorical features
print(df['Sex'].value_counts())
print(df['Embarked'].value_counts())


#Visualisations
import matplotlib.pyplot as plt
import seaborn as sns

# Histogram for Age
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Bar plot for Survived
plt.figure(figsize=(10, 6))
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# Scatter plot for Age vs. Fare
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df)
plt.title('Age vs. Fare by Survival Status')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
