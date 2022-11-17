import pandas as pd 
import numpy as np 

# Load the data
path = 'Deployment/Templates/Reviews.csv'
df = pd.read_csv(path)
df = df[['verified_reviews', 'rating']]
df.columns = ['reviews', 'rates']

# Cleaning the reviews text
from cleaning import clean_text
df['reviews'] = clean_text(df['reviews'])

# Selecting the target and text 
X = df.loc[:,'reviews']
y = df.loc[:,'rates']

print(X)