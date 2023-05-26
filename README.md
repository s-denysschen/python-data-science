# The Basics of Python for Data Science

# Table of Contents
1. [The Basics of Python](#the-basics-of-python)
2. [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Basic Machine Learning](#basic-machine-learning)
5. [Advanced Machine Learning](#advanced-machine-learning)

## The Basics of Python

### Data structures
Python, being a versatile programming language, offers a wide range of data structures to handle various types of data. Among the fundamental data structures in Python are lists, dictionaries, tuples, and sets. 

* Lists provide a flexible way to store and manipulate an ordered collection of items, allowing for easy indexing and appending of elements. 
* Dictionaries, on the other hand, offer a powerful key-value pairing mechanism, enabling efficient retrieval and modification of data based on unique keys. 
* Tuples, similar to lists, allow for the storage of multiple items, but with the crucial distinction of immutability, making them suitable for situations where data integrity is paramount. 
* Lastly, sets offer an unordered collection of unique elements, providing useful operations like union, intersection, and difference for efficient data manipulation and analysis. 

Python's diverse repertoire of data structures empowers programmers to choose the most suitable structure for their specific needs, enhancing the flexibility and efficiency of their code.
```python
# List
list_data = [1, 2, 3, 4, 5]

# Dictionary
dict_data = {'key1': 'value1', 'key2': 'value2'}

# Tuple
tuple_data = (1, 2, 3, 4, 5)

# Set
set_data = {1, 2, 3, 3, 4, 5}  # duplicate values are ignored
```
### Control flow
Control flow is an essential aspect of programming in Python as it allows for the management and execution of code based on specific conditions. In Python, control flow is primarily achieved through the utilization of conditional statements, loops, and function calls. Conditional statements, such as if-else and switch-case, enable the program to make decisions and choose different paths of execution based on the evaluation of certain conditions. Loops, including for loops and while loops, provide a way to repeat a specific block of code multiple times until a certain condition is met or until the loop is manually terminated. Additionally, function calls allow programmers to execute a specific set of instructions contained within a function, providing modularity and reusability to the code. By mastering control flow in Python, developers can create powerful and dynamic programs that respond intelligently to different scenarios.
```python
# If-else statement
x = 10
if x > 0:
    print('x is positive')
else:
    print('x is not positive')

# For loop
for i in range(5):
    print(i)

# While loop
i = 0
while i < 5:
    print(i)
    i += 1
```
### Functions
Functions play a crucial role in programming as they are modular blocks of code designed to carry out specific tasks. By encapsulating a set of instructions within a function, developers can create reusable units of code that can be easily invoked whenever needed. This not only enhances code organization but also promotes code reusability, making it more efficient and maintainable. Additionally, functions enable developers to break down complex problems into smaller, manageable parts, allowing for easier debugging and troubleshooting. Their ability to accept inputs (parameters) and return outputs further enhances their versatility and usefulness in various programming scenarios. In summary, functions serve as essential building blocks in programming, empowering developers to write clean, modular, and scalable code.
```python
def greet(name):
    print(f'Hello, {name}!')
greet('World')
```
### Classes and object-oriented programming
Python is a programming language that embraces object-oriented programming (OOP) concepts. With OOP, Python enables the creation of classes, which serve as blueprints for defining new types of objects. These classes facilitate the creation of multiple instances, each representing a unique occurrence of that particular type. By encapsulating attributes and behaviors within a class, Python promotes modularity and code reusability. When a class is instantiated, a new object of that type is created, equipped with its own set of attributes and behaviors defined by the class. This object-oriented approach in Python enhances flexibility and allows for more organized and efficient code structure.
```python
class Person:
    def __init__(self, name, age):  # constructor method
        self.name = name  # instance variable
        self.age = age

    def introduce_self(self):  # instance method
        print(f'My name is {self.name} and I am {self.age} years old.')

person1 = Person('Alice', 25)
person1.introduce_self()
```
### Exception handling
In Python, exception handling is an essential feature that helps developers manage unexpected errors and ensure their programs can gracefully recover from exceptional situations. To handle exceptions, Python employs the try/except blocks. The try block is where the potentially problematic code is placed, while the except block contains the code that executes when an exception occurs. By encapsulating code within a try block, developers can anticipate and prepare for potential errors. When an exception arises, Python gracefully transfers control to the appropriate except block, allowing the program to handle the exception in a controlled manner. This approach enhances the robustness and reliability of Python programs, as it prevents unexpected errors from crashing the entire application.
```python
try:
    # code that may raise an exception
    x = 1 / 0
except ZeroDivisionError:
    # what to do when that exception occurs
    print("You can't divide by zero!")
```
### NumPy
##### NumPy arrays
NumPy Arrays are a fundamental component of NumPy (Numerical Python), a powerful library extensively utilized for various array-based computations. With its extensive functionality, NumPy offers not only array manipulation capabilities but also a rich collection of tools for working in domains such as linear algebra, Fourier transform, and matrix operations. By leveraging NumPy, developers and data scientists can efficiently handle complex numerical operations and perform advanced mathematical computations. The library's array-centric design enables users to manipulate large datasets effortlessly, facilitating efficient data processing and analysis. With NumPy's comprehensive suite of functions, programmers can seamlessly tackle a wide range of mathematical tasks while benefiting from optimized performance and enhanced computational efficiency.
```python
import numpy as np

# Creating an array
arr = np.array([1, 2, 3, 4, 5])
print(arr)
```
#### NumPy operations
NumPy arrays offer a versatile range of operations, enabling us to manipulate data effectively. Among the operations available, we can carry out addition, subtraction, multiplication, and much more on NumPy arrays. This functionality empowers us to perform complex calculations and transformations with ease. Whether we need to combine arrays element-wise, compute differences between them, or multiply corresponding elements, NumPy provides the necessary tools. By harnessing these operations, we can efficiently process data, analyze patterns, and derive meaningful insights. Thus, NumPy's array operations empower us to unlock the full potential of numerical computing and data manipulation.
```python
# Arithmetic operations
arr = np.array([1, 2, 3, 4, 5])
print(arr + 2)  # adds 2 to each element of the array
print(arr * 2)  # multiplies each element of the array by 2
```
#### Indexing with NumPy
When working with NumPy arrays, we have the capability to access specific elements within the array through indexing. By utilizing indices, we can conveniently retrieve the desired values from the array. The indexing process involves specifying the position of the element we wish to retrieve, which allows us to extract the corresponding data. This functionality is particularly useful when dealing with large datasets, as it provides a straightforward method to access and manipulate specific array elements. Whether we want to perform computations on individual elements or extract subsets of data, indexing in NumPy arrays empowers us with the flexibility to efficiently work with our data.
```python
arr = np.array([1, 2, 3, 4, 5])
print(arr[0])  # prints the first element of the array
print(arr[-1])  # prints the last element of the array
```
### Pandas
#### Pandas Series
A Pandas Series can be conceptualized as a column within a table, providing a powerful data structure. Its functionality resembles that of a one-dimensional array, capable of storing data of diverse types. By utilizing a Pandas Series, users can efficiently manage and manipulate data, enabling streamlined analysis and computations. This versatile structure is an integral component of the Pandas library, widely used in data science and analysis tasks. Whether it's handling numerical values, categorical data, or even more complex information, the Pandas Series offers a flexible and accessible solution for organizing and working with data.
```python
import pandas as pd

# Creating a series
s = pd.Series([1, 2, 3, 4, 5])
print(s)
```
#### Pandas DataFrame
A DataFrame in pandas is a powerful data structure that represents a two-dimensional table. It organizes data in a tabular format, where each column within the DataFrame holds data of the same type. This structure allows for efficient manipulation and analysis of data, making it a popular choice for data processing tasks. Conceptually, a DataFrame can be thought of as a pandas equivalent of a Numpy 2D ndarray, providing additional functionality and flexibility. With pandas, you can perform various operations on a DataFrame, such as filtering, sorting, grouping, and merging, making it a versatile tool for data manipulation and exploration. By leveraging the DataFrame's tabular structure, you can easily handle large datasets, conduct complex data transformations, and extract valuable insights from your data.
```python
# Creating a DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': ['a', 'b', 'c'],
})
print(df)
```
#### Pandas operations
With pandas, you can perform various tasks to transform, filter, and aggregate data efficiently. One of the fundamental operations is data selection, allowing you to extract specific rows or columns from a dataset based on certain criteria. Additionally, pandas enables powerful data manipulation techniques such as merging, joining, and reshaping datasets, allowing you to combine data from different sources or reorganize it for better analysis. Furthermore, pandas provides extensive support for data cleaning tasks, including handling missing values, removing duplicates, and applying transformations to ensure data consistency and accuracy. Overall, pandas is a versatile library that equips data scientists and analysts with a comprehensive set of operations to tackle complex data challenges effectively.
```python
# Basic operations
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': ['a', 'b', 'c'],
})

# Adding a new column
df['C'] = [4, 5, 6]
print(df)

# Deleting a column
del df['A']
print(df)

# Describing the data
print(df.describe())
```
#### Indexing in Pandas
In the realm of data manipulation using Pandas, accessing and retrieving specific data from DataFrames can be accomplished through the use of indices and column names. This powerful feature allows users to navigate and extract information efficiently. By leveraging the indexing capabilities of Pandas, analysts and data scientists gain the ability to locate and retrieve subsets of data within their DataFrames with ease. Whether it's a specific row or a subset of columns, the combination of indices and column names provides a versatile approach to accessing and manipulating data. With Pandas, the vast landscape of data becomes more accessible, empowering users to perform intricate data operations with precision and flexibility.
```python
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': ['a', 'b', 'c'],
})

# Access a column
print(df['A'])

# Access a row by index
print(df.loc[0])
```

<hr>

## Data Cleaning and Preprocessing
Let's delve into data cleaning and preprocessing, which are crucial steps in any data science project. We will primarily use the pandas library for these tasks.

### Missing values
Missing values in datasets are quite common. They can be filled in with some specified value, or an aggregated value like mean, median, etc. Alternatively, rows or columns containing missing values can also be dropped.
```python
import pandas as pd
import numpy as np

# Creating a DataFrame with some missing values
df = pd.DataFrame({
    'A': [1, np.nan, 3],
    'B': [4, 5, np.nan],
    'C': [7, 8, 9]
})

# Fill missing values with a specified value
df_filled = df.fillna(0)

# Fill missing values with mean of the column
df_filled_mean = df.fillna(df.mean())

# Drop rows with missing values
df_dropped = df.dropna()
```

### Duplicate values
Duplicates in your dataset can negatively impact analysis and prediction results by distorting them. Removing duplicates is crucial for ensuring accurate analyses and predictions.
```python
# Creating a DataFrame with duplicate rows
df = pd.DataFrame({
    'A': ['foo', 'foo', 'bar', 'bar'],
    'B': ['one', 'one', 'two', 'two'],
    'C': [1, 1, 2, 2]
})

# Dropping duplicate rows
df_dropped = df.drop_duplicates()
```

#### Outliers
Outliers Detection and Removal: Outliers are extreme values that deviate significantly from other observations. They might occur due to variability in the data or measurement errors. We can use statistical methods like the IQR method or Z-score method to detect and remove outliers.
```python
# Assume we have a DataFrame df with a column 'data'
Q1 = df['data'].quantile(0.25)
Q3 = df['data'].quantile(0.75)
IQR = Q3 - Q1

# Define criteria for an outlier
filter = (df['data'] >= Q1 - 1.5 * IQR) & (df['data'] <= Q3 + 1.5 * IQR)

# Apply the filter to remove outliers
df_no_outlier = df.loc[filter]
```

#### Normalization
Data normalization involves scaling the data to a specific range, typically between 0 and 1. It is necessary when the dataset contains features with varying scale.
```python
from sklearn.preprocessing import MinMaxScaler

# Assume we have a DataFrame df with columns 'col1' and 'col2'
scaler = MinMaxScaler()
df[['col1', 'col2']] = scaler.fit_transform(df[['col1', 'col2']])
```

#### Categorical encoding
Encoding Categorical Variables: Machine learning models generally require numerical input. Hence categorical variables, both ordinal (with order) and nominal (without order), are often encoded to numerical counterparts.
```python
# Ordinal encoding
df['ordinal_var'] = df['ordinal_var'].map({'low': 1, 'medium': 2, 'high': 3})

# One-hot encoding for nominal variables
df = pd.get_dummies(df, columns=['nominal_var'])
```

### Basic data visualization

#### Matplotlib
Let's start with Matplotlib, a widely-used library for creating static, animated, and interactive visualizations in Python.
```python
import matplotlib.pyplot as plt
import numpy as np

# Line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title('Line Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Scatter plot
x = np.random.rand(50)
y = np.random.rand(50)
plt.scatter(x, y)
plt.title('Scatter Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Bar plot
labels = ['A', 'B', 'C']
values = [10, 35, 50]
plt.bar(labels, values)
plt.title('Bar Plot')
plt.show()
```

#### Seaborn
Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
```python
import seaborn as sns
import pandas as pd

# Creating a sample DataFrame
df = pd.DataFrame({
    'A': np.random.rand(50),
    'B': np.random.rand(50),
    'C': ['X']*25 + ['Y']*25
})

# Box plot
sns.boxplot(x='C', y='A', data=df)
plt.title('Box Plot')
plt.show()

# Violin plot
sns.violinplot(x='C', y='B', data=df)
plt.title('Violin Plot')
plt.show()

# Pair plot
sns.pairplot(df)
plt.title('Pair Plot')
plt.show()
```

<hr>

## Exploratory Data Analysis (EDA)

### Visual inspection
Exploring characteristics of your data is a critical step in any data science project. Using visualization libraries like Matplotlib and Seaborn can greatly assist in this process, making it easier to understand patterns, relationships, and structures within your data. Let's dive into some specifics.

#### Understanding distributions
One of the first steps in exploring your data could be understanding the distribution of various features. Histograms, box plots, and violin plots are commonly used for this purpose.

##### Histogram
Histogram: A histogram shows the frequency of different values in a dataset. In seaborn, you can use sns.histplot() to create histograms.
```python
import seaborn as sns
import pandas as pd
import numpy as np

data = np.random.normal(size=100)
sns.histplot(data)
```

##### Box plot
A box plot is used to depict groups of numerical data through their quartiles. It can give you a better understanding of the spread and skewness of your data. Outliers can also be spotted using box plots. Seaborn's sns.boxplot() can be used to create these.
```python
data = pd.DataFrame(np.random.rand(50, 4), columns=['A', 'B', 'C', 'D'])
sns.boxplot(data=data)
```

##### Violin plot
A violin plot combines the benefits of the previous two plots and simplifies them. It shows the distribution of quantitative data across several levels of one (or more) categorical variables. Use sns.violinplot() to create violin plots.
```python
tips = sns.load_dataset("tips")
sns.violinplot(x=tips["total_bill"])
```

#### Understanding relationships 
If your data has multiple features, it's often useful to understand how these features relate to each other. Scatter plots, line plots, and correlation heatmaps can be useful here.

##### Scatter plot
Scatter plots can help visualize the relationship between two numerical variables. In seaborn, you can use sns.scatterplot() to create scatter plots.
```python
iris = sns.load_dataset("iris")
sns.scatterplot(x='sepal_length', y='sepal_width', data=iris)
```

##### Line plot
A line plot is a way to display data along a number line. Line plots are used to track changes over periods of time. When smaller changes exist, line plots are better to use than bar plots.
```python
years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
revenue = [200, 250, 275, 300, 350, 400, 450, 475, 525, 575]
sns.lineplot(x=years, y=revenue)
```

##### Heatmap
A heatmap is a graphical representation of data that uses a system of color-coding to represent different values. Heatmaps are used in various forms of analytics but are most commonly used to show user behaviour on specific webpages or webpage templates.
```python
# correlation matrix
corr = iris.corr()
sns.heatmap(corr, annot=True)
```

### Formal techniques
EDA is a critical step in the data science pipeline. It also involves examining the data to understand their main characteristics often with visual methods. Here, I will outline some key statistical techniques, both parametric and non-parametric, used during EDA.

#### Parametric methods
Parametric methods assume that data has a specific distribution, typically a Gaussian (normal) distribution. The parameters of the normal distribution, mean and standard deviation, summarize and sufficiently represent the data.

##### Mean: 
It provides the central tendency of the dataset. Mean is the sum of all values divided by the number of values.
```python
import numpy as np
data = np.array([1, 2, 3, 4, 5])
mean = np.mean(data)
print('Mean:', mean)
```

##### Standard Deviation
It quantifies the amount of variation or dispersion of a set of values. A low standard deviation indicates that the values tend to be close to the mean, while a high standard deviation indicates that the values are spread out.
```python
std_dev = np.std(data)
print('Standard Deviation:', std_dev)
```

##### Correlation
It measures the degree to which two variables are linearly related. If we have more than two variables, we typically use a correlation matrix.
```python
import pandas as pd
data = pd.DataFrame({'A': np.random.rand(50), 'B': np.random.rand(50)})
correlation = data.corr()
print('Correlation:\n', correlation)
```

##### T-tests
These are used to determine if there is a significant difference between the means of two groups. In Python, you can use the scipy.stats.ttest_ind() function to conduct a t-test.
```python
import numpy as np
from scipy import stats

# Create three sets of data
np.random.seed(0)  # for reproducibility
group1 = np.random.normal(50, 10, size=50)
group2 = np.random.normal(60, 10, size=50)
group3 = np.random.normal(55, 10, size=50)

# Perform a two-sample t-test on group1 and group2
t_stat, p_val = stats.ttest_ind(group1, group2)

print("t-statistic: ", t_stat)
print("p-value: ", p_val)
```

##### Analysis of Variance (ANOVA): 
This is used to analyze the difference among group means in a sample. In Python, you can use the scipy.stats.f_oneway() function for ANOVA.
```python
# Perform one-way ANOVA
F_stat, p_val = stats.f_oneway(group1, group2, group3)

print("F-statistic: ", F_stat)
print("p-value: ", p_val)
```

#### Non-parametric methods
Non-parametric methods come in handy when the data does not fit a normal distribution. These methods are based on ranks and medians.
##### Median
It is the value separating the higher half from the lower half of a data sample. It provides the central tendency of the dataset.
```python
median = np.median(data)
print('Median:', median)
```

##### Interquartile Range (IQR): 
This is the range between the first quartile (25th percentile) and the third quartile (75th percentile). It is a measure of statistical dispersion.
```python
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1
print('IQR:', IQR)
```

##### Spearman’s Rank Correlation 
It assesses how well the relationship between two variables can be described using a monotonic function.
```python
from scipy.stats import spearmanr
data = pd.DataFrame({'A': np.random.rand(50), 'B': np.random.rand(50)})
correlation, _ = spearmanr(data['A'], data['B'])
print('Spearmans correlation: %.3f' % correlation)
```

##### Mann-Whitney U Test
It is used to compare differences between two independent data samples. In Python, you can use the scipy.stats.mannwhitneyu() function for this test.
```python
# Perform Mann-Whitney U Test on group1 and group2
u_stat, p_val = stats.mannwhitneyu(group1, group2)

print("U-statistic: ", u_stat)
print("p-value: ", p_val)
```

##### Kruskal-Wallis H Test: 
This test is used when the assumptions of one-way ANOVA are not met. It's a rank-based nonparametric test that can be used to determine if there are statistically significant differences between two or more groups.
```python
# Perform Kruskal-Wallis H Test
h_stat, p_val = stats.kruskal(group1, group2, group3)

print("H-statistic: ", h_stat)
print("p-value: ", p_val)
```

### Statistical inference
Statistical inference is the process of making judgments about a population based on sampling properties. It consists of selecting and modeling the data appropriately and interpreting the results correctly. There are two major types of statistical inference: estimation (point estimates and confidence intervals) and hypothesis testing.

#### Estimation

##### Point Estimate
It is a single value estimate of a parameter. For instance, the sample mean is a point estimate of the population mean.
```python
import numpy as np

# Generating a sample
np.random.seed(0)
population = np.random.normal(loc=70, scale=10, size=1000000)
sample = np.random.choice(population, size=1000)

# Point estimate of the mean
point_estimate = np.mean(sample)
print('Point Estimate of Mean:', point_estimate)
```

##### Confidence Interval
A range of values that likely contains the population parameter. For example, a 95% confidence interval implies that if we pull 100 samples and create confidence intervals for each, 95 of those intervals would contain the population mean.
```python
from scipy.stats import sem, t

# Confidence interval
confidence = 0.95
sample_stderr = sem(sample)  # Standard error of the mean
interval = sample_stderr * t.ppf((1 + confidence) / 2., len(sample) - 1)

print('Confidence interval for the mean:', (point_estimate - interval, point_estimate + interval))
```

#### Hypothesis Testing: 
Hypothesis testing is a statistical method that is used in making statistical decisions using experimental data. It is basically an assumption that we make about the population parameter.

Let's take an example, where we have a sample of weights and we are testing if the mean of the weights is significantly different from 70.
```python
# Null Hypothesis: The mean weight is 70
# Alternative Hypothesis: The mean weight is not 70

from scipy.stats import ttest_1samp

t_statistic, p_value = ttest_1samp(sample, 70)

print('t-statistic:', t_statistic)
print('p-value:', p_value)

if p_value < 0.05:  # alpha value is 0.05 or 5%
    print("We are rejecting null hypothesis")
else:
    print("We are accepting null hypothesis")
```

<hr>

## Basic Machine Learning
There are three main types of statistical learning methods: supervised learning, unsupervised learning, and reinforcement learning. Each method has its own problem space and appropriate use cases.

### Supervised Learning: 
This is where you have input variables (X) and an output variable (Y), and you use an algorithm to learn the mapping function from the input to the output. The ultimate goal is to approximate the mapping function so well that when you have new input data (X), you can predict the output variables (Y) for that data. It is called "supervised learning" because the process of an algorithm learning from the training dataset is akin to a teacher supervising the learning process. We know the correct answers; the algorithm iteratively makes predictions on the training data and is corrected by the teacher. The two main types of supervised learning problems are:

* Regression: The output variable is a real or continuous value, such as "salary" or "weight". Algorithms used for these types of problems include Linear Regression, Decision Trees, and Support Vector Regression.
* Classification: The output variable is a category, such as "red" or "blue" or "disease" and "no disease". Algorithms used for these types of problems include Logistic Regression, Naive Bayes, and Random Forest.

### Unsupervised Learning: 
This is where you only have input data (X) and no corresponding output variables. The goal for unsupervised learning is to model the underlying structure or distribution in the data in order to learn more about the data. These are called unsupervised learning because there is no correct answers and there is no teacher. Algorithms are left to their own to discover interesting structures in the data. The main types of unsupervised learning problems are:

* Clustering: The task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar (in some sense) to each other than to those in other groups (clusters). Algorithms include K-Means, Hierarchical Clustering, and DBSCAN.
* Association: The task of finding interesting relationships or associations among a set of items. This is often used for market basket analysis. Algorithms include Apriori and FP-Growth.

### Reinforcement Learning: 
It is about interaction between a learning agent and the environment. The agent takes actions in the environment to reach a certain goal. The environment, in return, gives reward or penalty (reinforcement signal) to the agent. The agent's task is to learn to make optimal decisions. A typical example is learning to play a game like chess. The agent decides the next move, the environment changes (the opponent makes a move), and the agent receives a reward (winning or losing the game).

#### SciKit-learn
Scikit-learn is an open-source machine learning library in Python. It features various machine learning algorithms, including those for classification, regression, and clustering. It also provides tools for model fitting, data preprocessing, model selection and evaluation, and many other utilities.

##### Classification: 
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
clf = DecisionTreeClassifier()

# Fit the model to the training data
clf.fit(X_train, y_train)

# Use the trained model to make predictions on the test data
predictions = clf.predict(X_test)
```

###### Regression:
```python
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

# Load dataset
boston = load_boston()
X, y = boston.data, boston.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
reg = LinearRegression()

# Fit the model
reg.fit(X_train, y_train)

# Make predictions
predictions = reg.predict(X_test)
```

##### Clustering: 
Scikit-learn provides several clustering algorithms like K-Means, Hierarchical clustering, DBSCAN, etc. 
```python
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
X = iris.data

# Initialize the model
kmeans = KMeans(n_clusters=3, random_state=0)

# Fit the model
kmeans.fit(X)

# Get cluster labels for each sample
labels = kmeans.labels_
```

Scikit-learn also provides many utility functions for preprocessing data, tuning hyperparameters, evaluating models, etc. that make the whole process of building and evaluating machine learning models easier and more efficient.

### Model evaluation
Model evaluation is a critical part of the machine learning pipeline. Once we've trained our model, we need to know how well it's performing. Model evaluation is a bit different for classification and regression problems due to the different nature of their output.

#### Classification Model Evaluation Metrics:

##### Accuracy: 
It is the ratio of correctly predicted observations to the total observations. However, it's not a good choice with imbalanced classes.
```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
```

##### Confusion Matrix: 
A table used to describe the performance of a classification model. It presents a clear picture of precision, recall, F1-score, and support.
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
```

##### Precision: 
It is the ratio of correctly predicted positive observations to the total predicted positives.

##### Recall (Sensitivity): 
It is the ratio of correctly predicted positive observations to the all observations in actual class.

##### F1 Score: 
It is the weighted average of Precision and Recall. It tries to find the balance between precision and recall.
```python
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
```

##### ROC-AUC: 
ROC curve is a graph showing the performance of a classification model at all classification thresholds. AUC stands for "Area under the ROC Curve". An excellent model has AUC near to 1, whereas a poor model has AUC near to 0.
```python
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_test, predictions)
```

#### Regression Model Evaluation Metrics:

##### Mean Absolute Error (MAE): 
It is the mean of the absolute value of the errors. It's the easiest to understand, because it's the average error.

##### Mean Squared Error (MSE): 
It is the mean of the squared errors. It's more popular than MAE, because MSE "punishes" larger errors.

##### Root Mean Squared Error (RMSE): 
It is the square root of the mean of the squared errors. It measures the standard deviation of the residuals.

##### R-squared (Coefficient of determination): 
Represents the coefficient of how well the values fit compared to the original values. The value from 0 to 1 interpreted as percentages. A higher value is better.
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)
```

The choice of metric depends on your business objective. Sometimes, we might prefer a model with a higher recall than a high precision, for example in cancer prediction, we want to capture as many positives as possible. In another case like email spam detection, we want to be as precise as possible to not put important emails in the spam folder. In the case of regression, lower values of MAE, MSE, or RMSE suggest a better fit to the data. A higher R-squared indicates a higher proportion of variance explained by the model.

<hr>

## Advanced Machine Learning

### Decision Trees: 
Decision Trees are a flowchart-like type of Supervised Machine Learning where the data is continuously split according to a certain parameter. They are easy to understand and interpret, which is one of their biggest advantages.
```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier

# Initialize the model
clf = DecisionTreeClassifier()

# Fit the model to the training data
clf.fit(X_train, y_train)

# Predict on the test data
predictions = clf.predict(X_test)
```

### Random Forest: 
Random Forest is a popular machine learning algorithm that belongs to the supervised learning technique. It can be used for both Classification and Regression problems in ML. It is based on the concept of ensemble learning, which is a process of combining multiple algorithms to solve a particular problem.
```python
from sklearn.ensemble import RandomForestClassifier

# Initialize the model
forest = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model
forest.fit(X_train, y_train)

# Make predictions
forest_predictions = forest.predict(X_test)
```

### Naive Bayes: 
Naive Bayes classifiers are a collection of classification algorithms based on Bayes’ Theorem. It is not a single algorithm but a family of algorithms where all of them share a common principle, i.e. every pair of features being classified is independent of each other.
```python
from sklearn.naive_bayes import GaussianNB

# Initialize the model
nb = GaussianNB()

# Fit the model
nb.fit(X_train, y_train)

# Make predictions
nb_predictions = nb.predict(X_test)
```

#### K-Nearest Neighbors (KNN): 
K-Nearest Neighbors is one of the most basic yet essential classification algorithms in Machine Learning. It belongs to the supervised learning domain and finds intense application in pattern recognition, data mining and intrusion detection.
```python
from sklearn.neighbors import KNeighborsClassifier

# Initialize the model
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the model
knn.fit(X_train, y_train)

# Make predictions
knn_predictions = knn.predict(X_test)
##### Support Vector Machines (SVM): 
SVM is a supervised machine learning algorithm which can be used for both classification or regression challenges. However, it is mostly used in classification problems. In the SVM algorithm, we plot each data item as a point in n-dimensional space with the value of each feature being the value of a particular coordinate.
from sklearn import svm

# Initialize the model
svc = svm.SVC(kernel='linear')

# Fit the model
svc.fit(X_train, y_train)

# Make predictions
svc_predictions = svc.predict(X_test)
```

### Ensemble methods
Ensemble methods are techniques that combine predictions from multiple machine learning algorithms to deliver more accurate predictions than a single model.

#### Bagging
Bagging, or Bootstrap Aggregating, involves taking multiple subsets of your original dataset, building a separate model for each subset, and then combining the output of all these models. For instance, the Random Forest algorithm is a type of bagging method.
 ```python
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
bagging.fit(X_train, y_train)
predictions = bagging.predict(X_test)
```

#### Boosting
Boosting works by training a model, identifying the mistakes it made, and then building a new model that focuses on the mistakes of the first model. This process is repeated, each time focusing on the mistakes of the last model, until a combined model with a low error rate is obtained. An example is the AdaBoost algorithm.
```python
from sklearn.ensemble import AdaBoostClassifier

adaboost = AdaBoostClassifier(n_estimators=100)
adaboost.fit(X_train, y_train)
predictions = adaboost.predict(X_test)
```

#### Stacking
Stacking involves training multiple different models and then using another machine learning model to combine their outputs.
```python
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC

estimators = [('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
              ('svr', SVC(random_state=42))]

stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacking.fit(X_train, y_train)
predictions = stacking.predict(X_test)
```

### Overfitting and Underfitting:
Overfitting and underfitting refer to the phenomena when a machine learning model performs well on the training data but poorly on the test data (overfitting), or when it performs poorly on both the training data and the test data (underfitting).

* Overfitting occurs when the model learns the detail and noise in the training data to the extent that it negatively impacts the performance of the model on new data. It means the model has learned the data "too well".
* Underfitting occurs when a machine learning model cannot capture the underlying pattern of the data. These models usually have poor predictive performance.

In essence, underfitting is a model with high bias (it makes strong assumptions and oversimplifies the problem), and overfitting is a model with high variance (it models the random noise in the training data, not the intended outputs). Let's demonstrate underfitting and overfitting using the decision tree algorithm. We will use the depth of the tree as our tuning parameter.
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Create a moon-shaped, noisy dataset
X, y = make_moons(n_samples=300, noise=0.25, random_state=42)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Let's set the tree depth to 1 (A stump)
stump = DecisionTreeClassifier(max_depth=1, random_state=42)
stump.fit(X_train, y_train)

# Now let's set the tree depth to 6
tree = DecisionTreeClassifier(max_depth=6, random_state=42)
tree.fit(X_train, y_train)

# And finally, let's not limit the tree depth
deep_tree = DecisionTreeClassifier(max_depth=None, random_state=42)
deep_tree.fit(X_train, y_train)

# Now let's test the accuracy of each model on the training data and test data
models = [stump, tree, deep_tree]
for model in models:
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    print(f'Model with max_depth={model.max_depth}:')
    print(f'Training accuracy: {accuracy_score(y_train, y_train_pred)}')
    print(f'Test accuracy: {accuracy_score(y_test, y_test_pred)}\n')
```

In the output, you might observe that the model with a depth of 1 (the stump) performs poorly on both the training and test sets. This is a classic case of underfitting. On the other hand, the model with no limit on the tree depth might perform very well on the training set but poorly on the test set. This is a classic case of overfitting. The model with a depth of 6 might give the best results, as it might strike a balance and perform well on both the training set and the test set. One way to fine-tune the trade-off between underfitting and overfitting is to use cross-validation to find the optimal model complexity. Moreover, many models have regularization parameters that can be adjusted to avoid overfitting.

### Unsupervised learning
Unsupervised learning is a type of machine learning algorithm used to draw inferences from datasets consisting of input data without labeled responses. The most common unsupervised learning method is cluster analysis, which is used for exploratory data analysis to find hidden patterns or grouping in data. Another common technique is dimensionality reduction, which attempts to reduce the number of features in a dataset while preserving as much statistical information as possible.

#### Clustering

##### DBSCAN
Density-Based Spatial Clustering of Applications with Noise, or DBSCAN, is a density-based clustering algorithm, which has the concept of core samples of high density and expands clusters from them.
```python
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan.fit(X)

# Plot the cluster assignments
plt.scatter(X[:, 0], X[:, 1], c=dbscan.labels_)
plt.show()
```

##### LDA (Latent Dirichlet Allocation): 
This is a generative statistical model that allows sets of observations to be explained by unobserved groups that explain why some parts of the data are similar. It's most commonly used for natural language processing tasks.
```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Sample data
data = ['This is the first document.','This document is the second document.','And this is the third one.','Is this the first document?']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data)

lda = LatentDirichletAllocation(n_components=2, random_state=0)
lda.fit(X)

# Displaying topics
for idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (idx))
    print([(vectorizer.get_feature_names()[i], topic[i])
                    for i in topic.argsort()[:-10 - 1:-1]])
```

#### Dimensionality Reduction

##### PCA (Principal Component Analysis): 
This is a technique used for feature extraction. It combines our input variables in a specific way, and we can drop the “least important” variables while still retaining the most valuable parts of all of the variables.
```python
from sklearn.decomposition import PCA

# Assume X is your matrix with shape (n_samples, n_features)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.show()
```

##### t-SNE (t-Distributed Stochastic Neighbor Embedding): 
This is a tool to visualize high-dimensional data. It converts similarities between data points to joint probabilities and tries to minimize the Kullback-Leibler divergence between the joint probabilities of the low-dimensional embedding and the high-dimensional data.
```python
from sklearn.manifold import TSNE

# Again, assume X is your matrix with shape (n_samples, n_features)
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
plt.show()
```

For both PCA and t-SNE examples, the input X should be a matrix with shape (n_samples, n_features). y is only used for coloring the points in the plot, and represents the true labels of the samples.

### Neural Networks and Deep Learning
Neural networks are a set of algorithms modeled after the human brain, that are designed to recognize patterns. They interpret sensory data through a kind of machine perception, labeling or clustering raw input. Deep Learning is a subfield of machine learning concerned with algorithms inspired by the structure and function of the brain called artificial neural networks. In more concrete terms, deep learning is the name for multilayered neural networks, which are networks composed of several "layers" of nodes — connected in a "deep" structure. Here's an example of a simple feedforward neural network built with PyTorch:
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Set up a simple feedforward neural network

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(20, 50)  # 20 input units to 50 hidden units
        self.fc2 = nn.Linear(50, 1)  # 50 hidden units to 1 output unit

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation function for hidden layer
        x = self.fc2(x)  # No activation function for output layer
        return x

# Assume we have some data in X and targets in Y
X = torch.randn(100, 20)
Y = torch.randn(100, 1)

# Instantiate the network, loss function and optimizer
net = Net()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Train the network
for epoch in range(100):  # loop over the dataset multiple times
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()

print('Finished Training')
```

In this code, we define a simple network with one hidden layer and one output layer. The hidden layer uses the ReLU activation function. We're training this network to minimize the mean squared error between its outputs and the target values Y. We're using stochastic gradient descent (SGD) as our optimization algorithm. This is a simple example, but deep learning can get much more complex! In real-world scenarios, we often use much larger networks and train them on big datasets. This requires more computational resources (especially GPUs), and more sophisticated techniques for managing data and training dynamics. Additionally, you would typically want to separate your data into a training set and a validation set, so that you can monitor your network's performance on unseen data as it trains, and prevent overfitting. Finally, deep learning also includes other types of architectures, such as convolutional neural networks (CNNs) for image tasks, recurrent neural networks (RNNs) for sequential data, transformers for natural language processing, autoencoders for unsupervised learning, and many more.
