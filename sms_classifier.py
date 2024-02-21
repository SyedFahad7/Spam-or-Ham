import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources (stopwords and tokenizers)
nltk.download('stopwords')
nltk.download('punkt')

# Load the dataset from the CSV file
df = pd.read_csv('spam_or_not_spam.csv')

# Preprocess the text data
def preprocess_text(text):
    if pd.isnull(text):  # Check for NaN values
        print("NaN value found!")
        return ''
    # Convert to string and remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    # Convert to lowercase
    text = text.lower()
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Join the tokens back into a single string
    preprocessed_text = ' '.join(filtered_tokens)
    return preprocessed_text

# Apply preprocessing to the 'email' column
df['processed_email'] = df['email'].apply(preprocess_text)

# Display the first few rows of the DataFrame after preprocessing
print(df.head())
