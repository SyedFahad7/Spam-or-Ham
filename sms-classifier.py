import numpy as np
import pandas as pd

df = pd.read_csv('spam_or_not_spam.csv')
print(df.head())
print(df.shape)
df.info()

# Drop rows with missing values in the 'email' column
df.dropna(subset=['email'], inplace=True)

# Reset the index after dropping rows
df.reset_index(drop=True, inplace=True)

print(df.head())
print(df.shape)

#Dropping Duplicated Columns 
print(df.duplicated().sum())
df = df.drop_duplicates(keep='first')
print(df.shape)

# EDA
import matplotlib.pyplot as plt

# Display the count of each category in the 'spam' column
print(df['label'].value_counts())

# Plot a pie chart to visualize the distribution of 'spam' categories
plt.pie(df['label'].value_counts(), labels=['ham', 'spam'])
plt.title('PIE CHART DISTRIBUTION')
plt.tight_layout()
plt.savefig('pie_chart.png')


import nltk
nltk.download('punkt')

df['num_characters'] = df['email'].apply(len)
print(df.head())

# Count number of words in each email
df['num_words'] = df['email'].apply(lambda x: len(x.split()))

# Display the DataFrame with the new 'num_words' column
print(df.head())

# Count number of sentences in each email
df['num_sentences'] = df['email'].apply(lambda x: len(nltk.sent_tokenize(x)))

# Display the DataFrame with the new 'num_sentences' column
print(df.head())

print(df[['num_sentences','num_characters','num_words']].describe())


#targeting not spam emails
print(df[df['label']==0][['num_sentences','num_characters','num_words']].describe())
#targeting spam emails
print(df[df['label']==1][['num_sentences','num_characters','num_words']].describe())


import matplotlib.pyplot as plt

# Filter data for label 0 (not spam) and label 1 (spam) for number of characters
label_0_data = df[df['label'] == 0]['num_characters']
label_1_data = df[df['label'] == 1]['num_characters']

# Set figure size
plt.figure(figsize=(8, 6))

# Plot histogram for label 0 (not spam)
plt.hist(label_0_data, bins=30, alpha=0.5, label='Not Spam')

# Plot histogram for label 1 (spam)
plt.hist(label_1_data, bins=30, alpha=0.5, color='red', label='Spam')

# Add labels and title
plt.xlabel('Number of Characters')
plt.ylabel('Count')
plt.title('Histogram of Number of Characters for Spam and Not Spam Emails')

plt.legend()
plt.savefig('Histogram_characters.png')

# Filter data for label 0 (not spam) and label 1 (spam) for number of Sentences
label_0_sentences = df[df['label'] == 0]['num_sentences']
label_1_sentences = df[df['label'] == 1]['num_sentences']
plt.figure(figsize=(12, 6))
plt.hist(label_0_sentences, bins=20, alpha=0.5, label='Not Spam')
plt.hist(label_1_sentences, bins=20, alpha=0.5, color='red', label='Spam')
plt.xlabel('Number of Sentences')
plt.ylabel('Count')
plt.title('Histogram of Number of Sentences')
plt.legend()
plt.savefig('Histogram_sentences.png')

# Filter data for label 0 (not spam) and label 1 (spam) for number of Words
label_0_words = df[df['label'] == 0]['num_words']
label_1_words = df[df['label'] == 1]['num_words']
plt.figure(figsize=(12, 6))
plt.hist(label_0_words, bins=20, alpha=0.5, label='Not Spam')
plt.hist(label_1_words, bins=20, alpha=0.5, color='red', label='Spam')
plt.xlabel('Number of Words')
plt.ylabel('Count')
plt.title('Histogram of Number of Words')
plt.legend()
plt.tight_layout()
plt.savefig('Histogram_words.png')

#pairPlot
import seaborn as sns
pairplot_data = df[['num_characters', 'num_words', 'num_sentences', 'label']]
sns.set(style="ticks")
# Plot pair plot
sns.pairplot(pairplot_data, hue='label', diag_kind='hist')
plt.savefig('pairplot.png')

#HeatMaps
corr = df[['num_characters', 'num_words', 'num_sentences']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('Heatmaps.png')


#Data PreProcessing (Lowercase,tokenization,removing special characters,removing stop words and punctutations,stemming)
# Import necessary libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Download stopwords from NLTK
nltk.download('stopwords')

# Initialize PorterStemmer for stemming
ps = PorterStemmer()

# Define a function to preprocess text
def transform_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Tokenize the text (split into words)
    text = nltk.word_tokenize(text)

    # Initialize an empty list to store filtered words
    filtered_text = []

    # Loop through each word in the tokenized text
    for word in text:
        # Check if the word contains only alphanumeric characters
        if word.isalnum():
            # If yes, append it to the filtered_text list
            filtered_text.append(word)
    
    # Update text with the filtered words
    text = filtered_text[:]

    # Clear the filtered_text list for reuse
    filtered_text.clear()

    # Loop through each word in the updated text
    for word in text:
        # Check if the word is not a stopword or punctuation
        if word not in stopwords.words('english') and word not in string.punctuation:
            # If yes, append it to the filtered_text list
            filtered_text.append(word)
    
    # Update text with the filtered words
    text = filtered_text[:]

    # Clear the filtered_text list for reuse
    filtered_text.clear()

    # Stem each word in the updated text using PorterStemmer
    for word in text:
        # Apply stemming to the word and append it to the filtered_text list
        filtered_text.append(ps.stem(word))

    # Join the stemmed words to form a processed text string
    processed_text = " ".join(filtered_text)

    # Return the processed text
    return processed_text

df['tranformed_text']=df['email'].apply(transform_text)
print.head()

from wordcloud import WordCloud #using WordCloud 
import matplotlib.pyplot as plt
wc= WordCloud(width=500,height=500,min_font=10,background_color='white')
spam_wc = wc.generate(df[df['label']==1]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(15,6))
plt.imshow(spam_wc)
