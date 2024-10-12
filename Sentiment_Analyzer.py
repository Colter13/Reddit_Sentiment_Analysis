import praw
import re
import random
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def print_comment(reddit, id):  # Prints out information of a particular comment
    post = reddit.submission(id=id)
    comments = post.comments
    for comment in comments[:2]:
        print("Printing comment...")
        print("Comment body - ", comment.body)
        print("Author - ", comment.author)
        print("\n")
def preprocess_text(text):  # Cleans, tokenizes, removes stop words, and lemmatizes
    # Cleaning
    text = re.sub(r'http\S+', '', text) # Removes URLs
    text = re.sub(r'<.*?>', '', text)   # Removes HTML tags
    text = re.sub(r'[^\w\s]', '', text) # Removes punctuation
    text = re.sub('\n', ' ', text)  # Removes newlines
    text = re.sub(r'\s+', ' ', text)    # Removes duplicate spaces
    text = text.lower() # Converts to lowercase

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return " ".join(tokens)
def get_sentiment(text):    # Returns positive, negative, or none sentiment
    scores = sia.polarity_scores(text)
    if scores['compound'] > 0.5:
        return 'positive'
    elif scores['compound'] < -0.5:
        return 'negative'
    else:
        return 'none'
    
def predict_sentiment(sentence):
    # Preprocess the sentence
    preprocessed_sentence = preprocess_text(sentence)
    
    # Transform the sentence using the already fitted vectorizer
    vectorized_sentence = vectorizer.transform([preprocessed_sentence])
    
    # Make a prediction
    prediction = model.predict(vectorized_sentence)
    
    # Map prediction to sentiment
    sentiment = 'positive' if prediction[0] == 1 else 'negative'
    return sentiment

# Reddit API authentication
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent="Sentiment-Analysis/1.0 by /u/Tricky_Language_7068",
)

# Initializing the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Search variables
subreddit = reddit.subreddit('all')
keyword = ''
import_limit = 100
time_filter = 'all'
sort = 'new'

# Import options
im_posts = subreddit.top(limit=import_limit)
#im_posts = subreddit.new(limit=import_limit)
#im_posts = subreddit.search(keyword, limit=import_limit, sort=sort, time_filter=time_filter)

# Preprocesses posts
posts = [preprocess_text(post.title + " " + post.selftext) for post in im_posts]

# Ensures that there are an equal number of positive and negative posts
pos_posts = []
neg_posts = []
for post in posts:
    sentiment = get_sentiment(post)
    if sentiment == 'positive':
        pos_posts.append(post)
    elif sentiment == 'negative':
        neg_posts.append(post)
lower_list = min(len(pos_posts), len(neg_posts))
posts = pos_posts[:lower_list] + neg_posts[:lower_list]
random.shuffle(posts)

# Creates parallel sentiments list
sentiments = [1 if get_sentiment(post) == 'positive' else 0 for post in posts ]

# Initializes a TF-IDF (Term Frequency - Inverse Document Frequency) vectorization using unigrams and bigrams
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

# Vectorizes the data
X = vectorizer.fit_transform(posts)
y = sentiments

# Splits the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Uses a logistic regression model to predict the sentiment based on the tfidf vectorization
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Prints the results
print("Accuracy:", accuracy_score(y_test, predictions))

sentence = input("Enter a sentence: ")
prediction = predict_sentiment(sentence)
print("The sentiment is", prediction)