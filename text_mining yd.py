# -*- coding: utf-8 -*-
"""


"""

pip install --upgrade pandas
 
import pandas as pd
df=pd.read_csv("Elon_musk.csv")
df = pd.read_csv('Elon_musk.csv', encoding='latin1')
df


import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load Elon Musk's tweets dataset (replace 'Elon-musk.csv' with the actual file path or URL)
with open('Elon_musk.csv', 'rb') as file:
    content = file.read().decode('latin1')
import io

# Assuming you have the 'content' variable containing the decoded content
elon_tweets = pd.read_csv(io.StringIO(content))

# Perform Sentiment Analysis using TextBlob
elon_tweets['Sentiment'] = elon_tweets['Tweets'].apply(lambda tweet: TextBlob(str(tweet)).sentiment.polarity)

# Categorize sentiment into positive, negative, or neutral
elon_tweets['Sentiment_Category'] = elon_tweets['Sentiment'].apply(lambda score: 'Positive' if score > 0 else 'Negative' if score < 0 else 'Neutral')

# Plot sentiment distribution
plt.figure(figsize=(8, 6))
elon_tweets['Sentiment_Category'].value_counts().plot(kind='bar', color=['green', 'red', 'blue'])
plt.title('Sentiment Analysis of Elon Musk\'s Tweets')
plt.xlabel('Sentiment Category')
plt.ylabel('Number of Tweets')
plt.show()

