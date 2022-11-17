# Amazon Reviews Analysis NLP
In this project you will see how to do sentiment analysis and how to import that analysis to machine learning algorithms. 

I examine many machine learning algorithms and calculated the accuracy. End of the examine, I choose the Random Forest Classifier.  

I did hyperparameter tunning and searched best hyperparameters. 

## Read Map
Go to `../Notebooks/..` path and start with reading the notebooks.  
1. `../explore.ipynb`: Exploring and understanding the data.  
1. `../nlp_analysis.ipynb`: Cleaning and analyzing the reviews. Prepare the dataset for machine learning process.  
1. `../machine_learning.ipynb`: Vectorizing the reviews for machine learning. Using *"Multinomial Naive Bayes"* algorithm and checking errors and accuracies.  
1. `../Hyperparameter-Tuning/tuning.py`: Using many machine learning algorithms and searching for the best hyperparameters. Final of the boosting and searching study I decided to algorithm and hyperparameters that is below:
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=160, max_features='log2', max_depth=4, criterion='gini')
``` 

## Requirements & Installation
You can go `../Packages/..` file path and run the `requirements.py` python file for installing and checking the library requirements. If you don't want to do it, you can go below and see the requirements and how to install them.  

<img src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT01Ctpf3nRjz7b9l-om2h2llNA0jL4d_MVtXXXHVF5mWIn5nyMXLgzYscFGZdbhf_LN8M&usqp=CAU' width='100' height='100'> <img src='https://raw.githubusercontent.com/AshNumpy/Amazon-Reviews-Sentiment-Analysis-ML-Project/main/Images/Others/numpy.png' width='100' height='100'> <img src='https://raw.githubusercontent.com/AshNumpy/Amazon-Reviews-Sentiment-Analysis-ML-Project/main/Images/Others/matplotlib.png' width='100' height='100'> <img src='https://raw.githubusercontent.com/AshNumpy/Amazon-Reviews-Sentiment-Analysis-ML-Project/main/Images/Others/plotly.png' width='308' height='100'> <img src='https://raw.githubusercontent.com/AshNumpy/Amazon-Reviews-Sentiment-Analysis-ML-Project/main/Images/Others/nltk.png' width='100' height='100'> <img src='https://raw.githubusercontent.com/AshNumpy/Amazon-Reviews-Sentiment-Analysis-ML-Project/main/Images/Others/scikit-learn.png' width='180' height='100'>

1. Pandas  
```bash
pip install pandas 

Version: 1.4.3
Summary: Powerful data structures for data analysis, time series, and statistics
Home-page: https://pandas.pydata.org
```

1. Numpy  
```bash
pip install numpy 

Version: 1.23.0
Summary: NumPy is the fundamental package for array computing with Python.
Home-page: https://www.numpy.org
```

1. Matplotlib  
```bash
pip install matplotlib 

Version: 3.5.2
Summary: Python plotting package
Home-page: https://matplotlib.org
```

1. Squarify  
```bash
pip install squarify 

Version: 0.4.3
Summary: Pure Python implementation of the squarify treemap layout algorithm
Home-page: https://github.com/laserson/squarify
```

1. Plotly  
```bash
pip install plotly 

Version: 5.9.0
Summary: An open-source, interactive data visualization library for Python
Home-page: https://plotly.com/python/
```

1. Wordcloud  
```bash
pip install wordcloud 

Version: 1.8.2.2
Summary: A little word cloud generator
Home-page: https://github.com/amueller/word_cloud
```

1. Nltk  
```bash
pip install nltk 

Version: 3.7
Summary: Natural Language Toolkit
Home-page: https://www.nltk.org/
```

1. Vader Sentiment  
```bash
pip install vaderSentiment 

Version: 3.3.2
Summary: VADER Sentiment Analysis. VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media, and works well on texts from other domains.
Home-page: https://github.com/cjhutto/vaderSentiment
```

1. Scikit-Learn
```bash
pip install sklearn

Version: 0.0
Summary: A set of python modules for machine learning and data mining
Home-page: https://pypi.python.org/pypi/scikit-learn/
```
