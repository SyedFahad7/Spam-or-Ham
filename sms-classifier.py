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
# Data PreProcessing (Lowercase, tokenization, removing special characters, removing stop words and punctuations, stemming)
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

df['transformed_text'] = df['email'].apply(transform_text)
print(df.head())

from wordcloud import WordCloud  # Importing Word
import matplotlib.pyplot as plt

# Creating WordCloud instance with the correct argument name
wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')

# Generating WordCloud for spam emails
spam_wc = wc.generate(df[df['label'] == 1]['transformed_text'].str.cat(sep=" "))

# Plotting the WordCloud
plt.figure(figsize=(15, 6))
plt.imshow(spam_wc)
plt.show()


#Building Model
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['transformed_text']).toarray()
X.shape
y=df['target'].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score

gnb= GaussianNB()
mnb= MultinomialNB()
bnb= BernoulliNB()

gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))

mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))

bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))

#More ML models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50, random_state=2)
xgb = XGBClassifier(n_estimators=50, random_state=2)

clfs = {
    'SVC': svc,
    'KN': knc,
    'NB': mnb,
    'DT': dtc,
    'LR': lrc,
    'RF': rfc,
    'AdaBoost': abc,
    'BgC': bc,
    'ETC': etc,
    'GBDT': gbdt,
    'xgb': xgb
}

def train_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    return accuracy, precision

print(train_classifier(svc,X_train,y_train,X_test,y_test))
accuracy_scores = []
precision_scores = []
for name, clf in clfs.items():
    current_accuracy, current_precision = train_classifier(clf, X_train, y_train, X_test, y_test)
    print("For", name)
    print("Accuracy:", current_accuracy)
    print("Precision:", current_precision)
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)

performance_df = pd.DataFrame({'Algorithm': clfs.keys(), 'Accuracy': accuracy_scores, 'Precision': precision_scores})
print(performance_df)

performance_df1 = pd.melt(performance_df, id_vars="Algorithm")
print(performance_df1)

sns.catplot(x='Algorithm', y='value', hue='variable', data=performance_df1, kind='bar', height=5)
plt.ylim(0.5, 1.0)
plt.xticks(rotation='vertical')
plt.show()

#model Improve
temp_df = pd.DataFrame({'Algorithm':clf.keys(),'Accuracy_max_ft_3000':accuracy_scores,'precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000x',ascending=False)
new_df = performance_df.merge(temp_df,on='Algorithm')
new_df_scaled = new_df.merge(temp_df,on='Algorithm')
temp_df= pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_num_chars':accuracy_scores,'Precision_num_chars':precision_scores}).sort_values('Precsion_Max_ft_3000y',ascending=False)
print(new_df_scaled.merge(temp_df,on='Algorithm'))

#Voting Classifier
svc =  SVC(kernal='sigmoid',gamma=1.0)
mnb= MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier
voting = VotingClassifier(estimators=[('svm',svc),('nb',mnb),('et',etc)],voting='soft')
voting.fit(X_train,y_train)
print(voting.fit(X_train,y_train))

y_pred =voting.predict(X_test)
print("accurracy",accuracy_score(y_test,y_pred))
print("precision",precision_score(y_test,y_pred))

import pickle
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(mnb, open('model.pkl', 'wb'))

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Assuming you have already defined X_train, y_train, and tfidf
tfidf = TfidfVectorizer(lowercase=True, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)

mnb = MultinomialNB()
mnb.fit(X_train_tfidf, y_train)

# Save the vectorizer
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf, vectorizer_file)

# Save the model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(mnb, model_file)