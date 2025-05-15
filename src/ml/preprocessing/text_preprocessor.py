import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import List, Optional

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self, 
                 lowercase: bool = True,
                 remove_punctuation: bool = True,
                 remove_stopwords: bool = True,
                 lemmatize: bool = True,
                 language: str = 'english'):
        
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        
        # Initialize components
        self.stopwords = set(stopwords.words(language)) if remove_stopwords else set()
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        
    def preprocess(self, text: str) -> str:
        """Preprocess single text"""
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        
        # Lemmatize
        if self.lemmatize and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Join tokens back
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess batch of texts"""
        return [self.preprocess(text) for text in texts]