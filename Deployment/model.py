import pandas as pd 

path = 'templates/Reviews.csv'
df = pd.read_csv(path)
df = df[['verified_reviews', 'rating']]

# Cleaning the reviews
from cleaning import clean_text
df['cleaned_reviews'] = clean_text(df['verified_reviews'])

# Set dependent and independent cols
X = df.loc[:,'cleaned_reviews']
y = df.loc[:,'rating']

# Vectorization to the reviews text
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(min_df=0., max_df=1.)
X_vector = cv.fit_transform(X.values.astype('U'))

# Split vectored data for ML model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_vector, y, train_size=.85, random_state=123)

# Building machine learning model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=160, max_features='log2', max_depth=4, criterion='gini')
model.fit(X_train, y_train)

# Creating pickle file
import pickle
pickle.dump(model, open("model.pkl", "wb"))