"""
Feature extraction module for Fake News Detection
Implements TF-IDF, n-grams, and metadata features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.sparse import hstack, csr_matrix
import pickle
from pathlib import Path
from rich.console import Console

from config import MODELS_DIR, BASELINE_CONFIG
from utils import console

class FeatureExtractor:
    """
    Comprehensive feature extraction for text classification
    Combines TF-IDF, n-grams, and metadata features
    """
    
    def __init__(
        self,
        use_tfidf: bool = True,
        use_metadata: bool = False,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 10000,
        min_df: int = 3,
        max_df: float = 0.9,
        use_char_ngrams: bool = False,
        metadata_scaler: str = 'standard'
    ):
        """
        Initialize feature extractor
        
        Args:
            use_tfidf: Use TF-IDF (else use count vectorizer)
            use_metadata: Include metadata features
            ngram_range: Range of n-grams to extract
            max_features: Maximum number of features
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            use_char_ngrams: Use character n-grams
            metadata_scaler: Scaler for metadata ('standard' or 'minmax')
        """
        self.use_tfidf = use_tfidf
        self.use_metadata = use_metadata
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.use_char_ngrams = use_char_ngrams
        self.metadata_scaler_type = metadata_scaler
        
        # Initialize vectorizers
        if use_tfidf:
            self.text_vectorizer = TfidfVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
                sublinear_tf=True,
                use_idf=True,
                smooth_idf=True,
                stop_words='english',
                analyzer='word' if not use_char_ngrams else 'char'
            )
            
            # Optional title vectorizer
            self.title_vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=1000,
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
                use_idf=True,
                stop_words='english'
            )
        else:
            self.text_vectorizer = CountVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
                stop_words='english',
                analyzer='word' if not use_char_ngrams else 'char'
            )
            
            self.title_vectorizer = CountVectorizer(
                ngram_range=(1, 2),
                max_features=1000,
                min_df=2,
                max_df=0.95,
                stop_words='english'
            )
        
        # Metadata scaler
        if metadata_scaler == 'standard':
            self.metadata_scaler = StandardScaler()
        else:
            self.metadata_scaler = MinMaxScaler()
        
        # Feature names tracking
        self.feature_names_ = None
        self.n_features_ = None
        
        self.console = Console()
    
    def extract_metadata_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract metadata/linguistic features from DataFrame
        
        Args:
            df: DataFrame with text and computed features
        
        Returns:
            Metadata feature matrix
        """
        metadata_features = []
        
        # Text statistics
        if 'text_length' in df.columns:
            metadata_features.append(df['text_length'].values.reshape(-1, 1))
        
        if 'word_count' in df.columns:
            metadata_features.append(df['word_count'].values.reshape(-1, 1))
        
        if 'sentence_count' in df.columns:
            metadata_features.append(df['sentence_count'].values.reshape(-1, 1))
        
        # Linguistic features
        if 'punctuation_density' in df.columns:
            metadata_features.append(df['punctuation_density'].values.reshape(-1, 1))
        
        if 'capital_ratio' in df.columns:
            metadata_features.append(df['capital_ratio'].values.reshape(-1, 1))
        
        if 'exclamation_count' in df.columns:
            metadata_features.append(df['exclamation_count'].values.reshape(-1, 1))
        
        if 'question_count' in df.columns:
            metadata_features.append(df['question_count'].values.reshape(-1, 1))
        
        # Social media features
        if 'url_count' in df.columns:
            metadata_features.append(df['url_count'].values.reshape(-1, 1))
        
        if 'mention_count' in df.columns:
            metadata_features.append(df['mention_count'].values.reshape(-1, 1))
        
        if 'hashtag_count' in df.columns:
            metadata_features.append(df['hashtag_count'].values.reshape(-1, 1))
        
        # Readability features (if available)
        if 'avg_word_length' in df.columns:
            metadata_features.append(df['avg_word_length'].values.reshape(-1, 1))
        
        if metadata_features:
            return np.hstack(metadata_features)
        else:
            return np.zeros((len(df), 0))
    
    def fit(self, df: pd.DataFrame, y: Optional[np.ndarray] = None):
        """
        Fit feature extractors on training data
        
        Args:
            df: Training DataFrame
            y: Target labels (optional)
        
        Returns:
            self
        """
        self.console.print("[cyan]Fitting feature extractors...[/cyan]")
        
        # Determine text column
        text_col = 'text_processed' if 'text_processed' in df.columns else 'text'
        
        # Fit text vectorizer
        self.console.print(f"  Fitting text vectorizer (TF-IDF={self.use_tfidf})...")
        self.text_vectorizer.fit(df[text_col].fillna(''))
        
        # Fit title vectorizer if available
        if 'title' in df.columns or 'title_processed' in df.columns:
            title_col = 'title_processed' if 'title_processed' in df.columns else 'title'
            self.console.print("  Fitting title vectorizer...")
            self.title_vectorizer.fit(df[title_col].fillna(''))
        
        # Fit metadata scaler if using metadata
        if self.use_metadata:
            self.console.print("  Fitting metadata scaler...")
            metadata_features = self.extract_metadata_features(df)
            if metadata_features.shape[1] > 0:
                self.metadata_scaler.fit(metadata_features)
        
        # Store feature names
        self._update_feature_names()
        
        self.console.print(f"[green]✓ Feature extraction fitted[/green]")
        return self
    
    def transform(self, df: pd.DataFrame) -> csr_matrix:
        if not hasattr(self, 'text_vectorizer') or not hasattr(self.text_vectorizer, 'vocabulary_'):
            raise RuntimeError("Feature extractor must be fitted before transform!")
        """
        Transform data into feature matrix
        
        Args:
            df: DataFrame to transform
        
        Returns:
            Sparse feature matrix
        """
        features = []
        
        # Text features
        text_col = 'text_processed' if 'text_processed' in df.columns else 'text'
        text_features = self.text_vectorizer.transform(df[text_col].fillna(''))
        features.append(text_features)
        
        # Title features (if available)
        if 'title' in df.columns or 'title_processed' in df.columns:
            title_col = 'title_processed' if 'title_processed' in df.columns else 'title'
            if hasattr(self, 'title_vectorizer'):
                title_features = self.title_vectorizer.transform(df[title_col].fillna(''))
                features.append(title_features)
        
        # Metadata features
        if self.use_metadata:
            metadata_features = self.extract_metadata_features(df)
            if metadata_features.shape[1] > 0:
                metadata_features = self.metadata_scaler.transform(metadata_features)
                features.append(csr_matrix(metadata_features))
        
        # Combine all features
        if len(features) > 1:
            X = hstack(features)
        else:
            X = features[0]
        
        self.n_features_ = X.shape[1]
        return X
    
    def fit_transform(self, df: pd.DataFrame, y: Optional[np.ndarray] = None) -> csr_matrix:
        """
        Fit and transform in one step
        
        Args:
            df: Training DataFrame
            y: Target labels (optional)
        
        Returns:
            Sparse feature matrix
        """
        self.fit(df, y)
        return self.transform(df)
    
    def _update_feature_names(self):
        """Update feature names after fitting"""
        feature_names = []
        
        # Text feature names
        text_names = [f"text_{name}" for name in self.text_vectorizer.get_feature_names_out()]
        feature_names.extend(text_names)
        
        # Title feature names
        if hasattr(self, 'title_vectorizer') and hasattr(self.title_vectorizer, 'get_feature_names_out'):
            title_names = [f"title_{name}" for name in self.title_vectorizer.get_feature_names_out()]
            feature_names.extend(title_names)
        
        # Metadata feature names
        if self.use_metadata:
            metadata_names = [
                'text_length', 'word_count', 'sentence_count',
                'punctuation_density', 'capital_ratio',
                'exclamation_count', 'question_count',
                'url_count', 'mention_count', 'hashtag_count'
            ]
            feature_names.extend([f"meta_{name}" for name in metadata_names])
        
        self.feature_names_ = feature_names
    
    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        if self.feature_names_ is None:
            self._update_feature_names()
        return self.feature_names_
    
    def get_top_features(self, indices: np.ndarray, top_k: int = 10) -> List[str]:
        """
        Get top feature names by importance indices
        
        Args:
            indices: Feature importance indices
            top_k: Number of top features
        
        Returns:
            List of feature names
        """
        feature_names = self.get_feature_names()
        top_indices = indices[:top_k]
        return [feature_names[i] for i in top_indices if i < len(feature_names)]
    
    def save(self, path: Path):
        """
        Save only picklable state to avoid RLock/thread objects.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "use_tfidf": self.use_tfidf,
            "use_metadata": self.use_metadata,
            "ngram_range": self.ngram_range,
            "max_features": self.max_features,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "use_char_ngrams": self.use_char_ngrams,
            "metadata_scaler_type": self.metadata_scaler_type,
            "text_vectorizer": self.text_vectorizer,
            "title_vectorizer": getattr(self, "title_vectorizer", None),
            "metadata_scaler": self.metadata_scaler,
            "feature_names_": getattr(self, "feature_names_", None),
            "n_features_": getattr(self, "n_features_", None),
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        self.console.print(f"[green]✓ Feature extractor saved to {path}[/green]")

    @classmethod
    def load(cls, path: Path):
        """
        Reconstruct FeatureExtractor from saved state.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        obj = cls(
            use_tfidf=state["use_tfidf"],
            use_metadata=state["use_metadata"],
            ngram_range=state["ngram_range"],
            max_features=state["max_features"],
            min_df=state["min_df"],
            max_df=state["max_df"],
            use_char_ngrams=state["use_char_ngrams"],
            metadata_scaler=state["metadata_scaler_type"],
        )
        obj.text_vectorizer = state["text_vectorizer"]
        if state.get("title_vectorizer") is not None:
            obj.title_vectorizer = state["title_vectorizer"]
        obj.metadata_scaler = state["metadata_scaler"]
        obj.feature_names_ = state.get("feature_names_")
        obj.n_features_ = state.get("n_features_")
        return obj
    
    def get_params(self) -> Dict:
        """Get configuration parameters"""
        return {
            'use_tfidf': self.use_tfidf,
            'use_metadata': self.use_metadata,
            'ngram_range': self.ngram_range,
            'max_features': self.max_features,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'use_char_ngrams': self.use_char_ngrams,
            'n_features': self.n_features_
        }


class FeatureCombiner:
    """Combine multiple feature extractors"""
    
    def __init__(self, extractors: List[FeatureExtractor]):
        """
        Initialize feature combiner
        
        Args:
            extractors: List of feature extractors
        """
        self.extractors = extractors
        self.n_features_ = None
    
    def fit(self, df: pd.DataFrame, y: Optional[np.ndarray] = None):
        """Fit all extractors"""
        for extractor in self.extractors:
            extractor.fit(df, y)
        return self
    
    def transform(self, df: pd.DataFrame) -> csr_matrix:
        """Transform using all extractors"""
        features = []
        for extractor in self.extractors:
            features.append(extractor.transform(df))
        
        X = hstack(features)
        self.n_features_ = X.shape[1]
        return X
    
    def fit_transform(self, df: pd.DataFrame, y: Optional[np.ndarray] = None) -> csr_matrix:
        """Fit and transform"""
        self.fit(df, y)
        return self.transform(df)


def create_baseline_features() -> FeatureExtractor:
    """Create baseline feature extractor with default configuration"""
    config = BASELINE_CONFIG['tfidf']
    
    return FeatureExtractor(
        use_tfidf=True,
        use_metadata=False,  # Start without metadata
        ngram_range=config['ngram_range'],
        max_features=config['max_features'],
        min_df=config['min_df'],
        max_df=config['max_df']
    )


def create_advanced_features() -> FeatureExtractor:
    """Create advanced feature extractor with metadata"""
    config = BASELINE_CONFIG['tfidf']
    
    return FeatureExtractor(
        use_tfidf=True,
        use_metadata=True,
        ngram_range=config['ngram_range'],
        max_features=config['max_features'],
        min_df=config['min_df'],
        max_df=config['max_df']
    )


def main():
    """Test feature extraction"""
    import argparse
    from data_io import DataLoader
    
    parser = argparse.ArgumentParser(description="Extract features from text data")
    parser.add_argument('--metadata', action='store_true',
                       help='Include metadata features')
    parser.add_argument('--save', action='store_true',
                       help='Save feature extractor')
    args = parser.parse_args()
    
    # Load data
    loader = DataLoader()
    train_df, val_df, test_df = loader.load_processed_data()
    
    # Create feature extractor
    if args.metadata:
        console.print("[cyan]Creating advanced feature extractor with metadata[/cyan]")
        extractor = create_advanced_features()
    else:
        console.print("[cyan]Creating baseline feature extractor[/cyan]")
        extractor = create_baseline_features()
    
    # Fit and transform
    X_train = extractor.fit_transform(train_df)
    X_val = extractor.transform(val_df)
    X_test = extractor.transform(test_df)
    
    # Display information
    console.print(f"\n[bold]Feature extraction complete:[/bold]")
    console.print(f"  Train features: {X_train.shape}")
    console.print(f"  Val features: {X_val.shape}")
    console.print(f"  Test features: {X_test.shape}")
    console.print(f"  Sparsity: {1 - X_train.nnz / (X_train.shape[0] * X_train.shape[1]):.2%}")
    
    # Get top features by frequency
    feature_sums = X_train.sum(axis=0).A1
    top_indices = feature_sums.argsort()[-20:][::-1]
    top_features = extractor.get_top_features(top_indices, top_k=10)
    
    console.print(f"\n[bold]Top features by frequency:[/bold]")
    for i, feature in enumerate(top_features, 1):
        console.print(f"  {i:2d}. {feature}")
    
    # Save if requested
    if args.save:
        save_path = MODELS_DIR / "feature_extractor.pkl"
        extractor.save(save_path)


if __name__ == "__main__":
    main()

