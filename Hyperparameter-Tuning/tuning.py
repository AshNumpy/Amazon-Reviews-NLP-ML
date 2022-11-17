import pandas as pd 

path = 'Deployment/Templates/Reviews.csv'
df = pd.read_csv(path)
df = df[['verified_reviews', 'rating']]

# Cleaning the reviews
from cleaning import clean_text
df['cleaned_reviews'] = clean_text(df['verified_reviews'])

# Split the data
X = df.loc[:,'cleaned_reviews']
y = df.loc[:,'rating']

# Vectorization to the reviews text
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(min_df=0., max_df=1.)
X_vector = cv.fit_transform(X.values.astype('U'))

# Split vectored data for ML model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_vector, y, train_size=.8, random_state=0)

# Building machine learning model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Calculating accuracy and error rate
from sklearn.metrics import mean_absolute_percentage_error, accuracy_score
accuracy = accuracy_score(y_test, y_pred)
error = mean_absolute_percentage_error(y_test, y_pred)

# Searching for best hyperparameters
from sklearn.model_selection import RandomizedSearchCV
params = { 
    'n_estimators': [100, 120, 130, 150, 160, 170, 180, 190, 200, 220, 250, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']}

cv = RandomizedSearchCV(rf, params)
cv.fit(X_train, y_train)

best_score = cv.best_score_*100
best_params = cv.best_params_

print(f'''
DEFAULT SETTINGS \n
Average Accuracy Score:{(accuracy*100):.2f}% \n
Average Error Rate: {(error*100):.2f}%
{'*'*50}
''')

print(f"""
Best Parameters: {best_params} \n
Best Score: {best_score:.2f}
""")