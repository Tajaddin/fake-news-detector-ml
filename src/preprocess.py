"""
Text preprocessing module for Fake News Detection
Handles text cleaning, normalization, and augmentation
"""

import re
import string
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import unicodedata
from rich.console import Console
from rich.progress import track

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

from config import PROCESSED_DATA_DIR
from utils import console

class TextPreprocessor:
    """Comprehensive text preprocessing for fake news detection"""
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = False,  # Keep some punctuation for transformers
        remove_stopwords: bool = False,    # Keep for transformers
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_html: bool = True,
        remove_numbers: bool = False,
        normalize_whitespace: bool = True,
        lemmatize: bool = False,
        stem: bool = False,
        max_length: Optional[int] = None
    ):
        """
        Initialize preprocessor with configuration
        
        Args:
            lowercase: Convert text to lowercase
            remove_punctuation: Remove punctuation marks
            remove_stopwords: Remove common stopwords
            remove_urls: Remove URLs
            remove_emails: Remove email addresses
            remove_html: Remove HTML tags
            remove_numbers: Remove numeric values
            normalize_whitespace: Normalize whitespace
            lemmatize: Apply lemmatization
            stem: Apply stemming
            max_length: Maximum text length (chars)
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_html = remove_html
        self.remove_numbers = remove_numbers
        self.normalize_whitespace = normalize_whitespace
        self.lemmatize = lemmatize
        self.stem = stem
        self.max_length = max_length
        
        # Initialize NLP tools
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()
        
        if self.stem:
            self.stemmer = PorterStemmer()
        
        self.console = Console()
    
    def clean_html(self, text: str) -> str:
        """Remove HTML tags from text"""
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Decode HTML entities
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'&#\d+;', '', text)
        return text
    
    def clean_urls(self, text: str) -> str:
        """Remove URLs from text"""
        # Remove URLs starting with http/https/www
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        # Remove domain-like patterns
        text = re.sub(r'\S+\.(com|org|net|gov|edu)\S*', '', text)
        return text
    
    def clean_emails(self, text: str) -> str:
        """Remove email addresses from text"""
        text = re.sub(r'\S+@\S+', '', text)
        return text
    
    def clean_social_media(self, text: str) -> str:
        """Clean social media specific elements"""
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        # Remove hashtags (optional - might be informative)
        # text = re.sub(r'#\w+', '', text)
        # Remove retweet markers
        text = re.sub(r'RT\s+:', '', text)
        return text
    
    def normalize_contractions(self, text: str) -> str:
        """Expand contractions"""
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am",
            "it's": "it is",
            "he's": "he is",
            "she's": "she is",
            "that's": "that is",
            "what's": "what is",
            "where's": "where is",
            "there's": "there is"
        }
        
        for contraction, expansion in contractions.items():
            text = re.sub(contraction, expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters"""
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        # Remove non-ASCII characters
        text = text.encode('ascii', 'ignore').decode('utf-8')
        return text
    
    def process_text(self, text: str) -> str:
        """
        Apply all preprocessing steps to text
        
        Args:
            text: Input text
        
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # HTML cleaning
        if self.remove_html:
            text = self.clean_html(text)
        
        # URL and email cleaning
        if self.remove_urls:
            text = self.clean_urls(text)
        
        if self.remove_emails:
            text = self.clean_emails(text)
        
        # Social media cleaning
        text = self.clean_social_media(text)
        
        # Normalize contractions
        text = self.normalize_contractions(text)
        
        # Normalize unicode
        text = self.normalize_unicode(text)
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Tokenization for word-level processing
        if self.remove_stopwords or self.lemmatize or self.stem:
            try:
                tokens = word_tokenize(text)
                
                # Remove punctuation tokens
                if self.remove_punctuation:
                    tokens = [t for t in tokens if t not in string.punctuation]
                
                # Remove stopwords
                if self.remove_stopwords:
                    tokens = [t for t in tokens if t.lower() not in self.stop_words]
                
                # Lemmatization
                if self.lemmatize:
                    tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
                
                # Stemming
                if self.stem:
                    tokens = [self.stemmer.stem(t) for t in tokens]
                
                text = ' '.join(tokens)
            except Exception:
                # If tokenization fails, continue with original text
                pass
        elif self.remove_punctuation:
            # Simple punctuation removal without tokenization
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Normalize whitespace
        if self.normalize_whitespace:
            text = ' '.join(text.split())
        
        # Truncate if needed
        if self.max_length and len(text) > self.max_length:
            text = text[:self.max_length]
        
        return text.strip()
    
    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
        title_column: Optional[str] = 'title'
    ) -> pd.DataFrame:
        """
        Process all texts in a DataFrame
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            title_column: Name of title column (optional)
        
        Returns:
            DataFrame with processed text
        """
        df = df.copy()
        
        # Process main text
        self.console.print(f"Processing {text_column} column...")
        df[f'{text_column}_processed'] = [
            self.process_text(text) 
            for text in track(df[text_column], description="Processing texts")
        ]
        
        # Process title if exists
        if title_column and title_column in df.columns:
            self.console.print(f"Processing {title_column} column...")
            df[f'{title_column}_processed'] = [
                self.process_text(title) 
                for title in track(df[title_column], description="Processing titles")
            ]
            
            # Combine title and text
            df['combined_processed'] = df[f'{title_column}_processed'] + ' ' + df[f'{text_column}_processed']
        
        # Add text statistics
        df['text_length'] = df[f'{text_column}_processed'].str.len()
        df['word_count'] = df[f'{text_column}_processed'].str.split().str.len()
        df['sentence_count'] = df[text_column].apply(
            lambda x: len(sent_tokenize(str(x))) if pd.notna(x) else 0
        )
        
        # Remove empty texts
        initial_len = len(df)
        df = df[df[f'{text_column}_processed'].str.len() > 0]
        removed = initial_len - len(df)
        
        if removed > 0:
            self.console.print(f"[yellow]Removed {removed} empty texts[/yellow]")
        
        return df
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract additional linguistic features
        
        Args:
            df: DataFrame with processed text
        
        Returns:
            DataFrame with additional features
        """
        df = df.copy()
        
        # Punctuation features
        df['exclamation_count'] = df['text'].str.count('!')
        df['question_count'] = df['text'].str.count(r'\?')
        df['punctuation_density'] = df['text'].apply(
            lambda x: sum(c in string.punctuation for c in str(x)) / max(len(str(x)), 1)
        )
        
        # Capital letters ratio
        df['capital_ratio'] = df['text'].apply(
            lambda x: sum(c.isupper() for c in str(x)) / max(len(str(x)), 1)
        )
        
        # URL and mention counts (before cleaning)
        df['url_count'] = df['text'].str.count(r'http\S+|www\.\S+')
        df['mention_count'] = df['text'].str.count(r'@\w+')
        df['hashtag_count'] = df['text'].str.count(r'#\w+')
        
        # Quotation marks (might indicate cited sources)
        df['quote_count'] = df['text'].str.count('"') + df['text'].str.count("'")
        
        # Numeric content
        df['number_count'] = df['text'].str.count(r'\d+')
        df['numeric_ratio'] = df['number_count'] / df['word_count'].replace(0, 1)
        
        return df


class DataAugmenter:
    """Data augmentation techniques for text"""
    
    def __init__(self, augment_prob: float = 0.1):
        """
        Initialize data augmenter
        
        Args:
            augment_prob: Probability of applying augmentation
        """
        self.augment_prob = augment_prob
        self.console = Console()
    
    def synonym_replacement(self, text: str, n: int = 2) -> str:
        """Replace n words with synonyms"""
        # This is a simplified version - for production use nltk.corpus.wordnet
        words = text.split()
        if len(words) < n:
            return text
        
        # Randomly select words to replace (simplified)
        indices = np.random.choice(len(words), min(n, len(words)), replace=False)
        # In practice, look up synonyms from WordNet
        
        return text
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """Randomly delete words with probability p"""
        words = text.split()
        if len(words) == 1:
            return text
        
        new_words = [w for w in words if np.random.random() > p]
        if len(new_words) == 0:
            return words[0]
        
        return ' '.join(new_words)
    
    def random_swap(self, text: str, n: int = 2) -> str:
        """Randomly swap n word pairs"""
        words = text.split()
        if len(words) < 2:
            return text
        
        for _ in range(n):
            idx1, idx2 = np.random.choice(len(words), 2, replace=False)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def augment_dataset(
        self,
        df: pd.DataFrame,
        text_column: str = 'text_processed',
        target_column: str = 'label'
    ) -> pd.DataFrame:
        """
        Augment dataset to balance classes
        
        Args:
            df: Input DataFrame
            text_column: Column with text
            target_column: Column with labels
        
        Returns:
            Augmented DataFrame
        """
        # Check class balance
        class_counts = df[target_column].value_counts()
        minority_class = class_counts.idxmin()
        majority_class = class_counts.idxmax()
        
        imbalance_ratio = class_counts[majority_class] / class_counts[minority_class]
        
        if imbalance_ratio < 1.5:
            self.console.print("[green]Classes are relatively balanced, no augmentation needed[/green]")
            return df
        
        self.console.print(f"[yellow]Class imbalance detected (ratio: {imbalance_ratio:.2f})[/yellow]")
        
        # Augment minority class
        minority_df = df[df[target_column] == minority_class]
        augmented_samples = []
        
        n_samples_needed = class_counts[majority_class] - class_counts[minority_class]
        n_samples_needed = min(n_samples_needed, len(minority_df))  # Don't over-augment
        
        for _ in range(n_samples_needed):
            # Randomly select a minority sample
            sample = minority_df.sample(1).iloc[0].copy()
            
            # Apply random augmentation
            aug_type = np.random.choice(['deletion', 'swap'])
            if aug_type == 'deletion':
                sample[text_column] = self.random_deletion(sample[text_column])
            else:
                sample[text_column] = self.random_swap(sample[text_column])
            
            augmented_samples.append(sample)
        
        if augmented_samples:
            augmented_df = pd.DataFrame(augmented_samples)
            df = pd.concat([df, augmented_df], ignore_index=True)
            self.console.print(f"[green]Added {len(augmented_samples)} augmented samples[/green]")
        
        return df


def main():
    """Main preprocessing pipeline"""
    import argparse
    from data_io import DataLoader
    
    parser = argparse.ArgumentParser(description="Preprocess fake news dataset")
    parser.add_argument('--minimal', action='store_true',
                       help='Minimal preprocessing (for transformers)')
    parser.add_argument('--augment', action='store_true',
                       help='Apply data augmentation')
    args = parser.parse_args()
    
    # Load data
    loader = DataLoader()
    try:
        train_df, val_df, test_df = loader.load_processed_data()
    except FileNotFoundError:
        console.print("[yellow]No processed data found. Run data_io.py first![/yellow]")
        return
    
    # Configure preprocessor
    if args.minimal:
        # Minimal preprocessing for transformers
        preprocessor = TextPreprocessor(
            lowercase=True,
            remove_punctuation=False,
            remove_stopwords=False,
            remove_urls=True,
            remove_emails=True,
            remove_html=True,
            normalize_whitespace=True
        )
        console.print("[cyan]Using minimal preprocessing for transformers[/cyan]")
    else:
        # Full preprocessing for traditional ML
        preprocessor = TextPreprocessor(
            lowercase=True,
            remove_punctuation=True,
            remove_stopwords=True,
            remove_urls=True,
            remove_emails=True,
            remove_html=True,
            normalize_whitespace=True,
            lemmatize=True
        )
        console.print("[cyan]Using full preprocessing for traditional ML[/cyan]")
    
    # Process each split
    console.print("\n[bold]Processing training data...[/bold]")
    train_df = preprocessor.process_dataframe(train_df)
    train_df = preprocessor.extract_features(train_df)
    
    console.print("\n[bold]Processing validation data...[/bold]")
    val_df = preprocessor.process_dataframe(val_df)
    val_df = preprocessor.extract_features(val_df)
    
    console.print("\n[bold]Processing test data...[/bold]")
    test_df = preprocessor.process_dataframe(test_df)
    test_df = preprocessor.extract_features(test_df)
    
    # Apply augmentation if requested
    if args.augment:
        augmenter = DataAugmenter()
        train_df = augmenter.augment_dataset(train_df)
    
    # Save processed data
    suffix = '_minimal' if args.minimal else '_full'
    train_df.to_csv(PROCESSED_DATA_DIR / f'train_processed{suffix}.csv', index=False)
    val_df.to_csv(PROCESSED_DATA_DIR / f'val_processed{suffix}.csv', index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / f'test_processed{suffix}.csv', index=False)
    
    console.print(f"\n[bold green]✨ Preprocessing complete![/bold green]")
    console.print(f"Processed data saved to {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    main()
