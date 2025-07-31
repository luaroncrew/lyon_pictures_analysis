from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import string
import logging
import re
from typing import Optional, List, Set
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """Configuration class for text preprocessing parameters."""
    
    # Text cleaning parameters
    min_token_length: int = 2
    use_pos_tagging: bool = True
    remove_numbers: bool = True
    normalize_whitespace: bool = True
    remove_duplicates: bool = True
    
    # Language and stopwords
    languages: List[str] = field(default_factory=lambda: [
        'english', 'french', 'italian', 'spanish', 'portuguese', 'german'
    ])
    custom_stopwords: List[str] = field(default_factory=lambda: [
        "france", "lyon", "69", "rhônealpes", "french", "français", "ville"
    ])
    
    # Processing parameters
    chunk_size: int = 10000
    validate_output: bool = True
    
    # File parameters
    input_file: str = "initial_data.csv"
    output_file: str = "lemmatized_data.csv"
    text_column: str = " title"

def get_stop_words(config: PreprocessingConfig) -> Set[str]:
    """Load and combine stop words from multiple languages with custom terms."""
    try:
        stop_words = set()
        
        for lang in config.languages:
            try:
                stop_words.update(stopwords.words(lang))
            except Exception as e:
                logger.warning(f"Could not load {lang} stopwords: {e}")
        
        stop_words.update(config.custom_stopwords)
        
        logger.info(f"Loaded {len(stop_words)} stop words")
        return stop_words
    except Exception as e:
        logger.error(f"Error loading stop words: {e}")
        return set()


lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(word_pos: str) -> str:
    """Convert NLTK POS tag to WordNet POS tag for better lemmatization."""
    tag = word_pos[0].upper()
    tag_dict = {
        'J': wordnet.ADJ,
        'N': wordnet.NOUN, 
        'V': wordnet.VERB,
        'R': wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)

def clean_and_lemmatize(text: Optional[str], 
                       config: PreprocessingConfig,
                       stop_words: Set[str]) -> str:
    """Advanced text cleaning and lemmatization with configurable parameters.
    
    Args:
        text: Input text to process
        config: Preprocessing configuration object
        stop_words: Set of stop words to remove
        
    Returns:
        Cleaned and lemmatized text
    """
    if not text or not isinstance(text, str) or text.isspace():
        return ""
    
    try:
        # Normalize whitespace and lowercase
        if config.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text.strip()).lower()
        else:
            text = text.strip().lower()
        
        # Remove URLs and email addresses
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep accented characters
        text = re.sub(r'[^\w\s\u00C0-\u017F]', ' ', text)
        
        # Remove extra spaces created by cleaning
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove numbers if specified
        if config.remove_numbers:
            tokens = [token for token in tokens if not token.isdigit()]
        
        # POS tagging for better lemmatization
        if config.use_pos_tagging:
            pos_tags = pos_tag(tokens)
            cleaned = []
            
            for token, pos in pos_tags:
                if (token.isalpha() 
                    and len(token) >= config.min_token_length 
                    and token not in stop_words):
                    
                    # Use POS tag for more accurate lemmatization
                    wordnet_pos = get_wordnet_pos(pos)
                    lemmatized = lemmatizer.lemmatize(token, pos=wordnet_pos)
                    cleaned.append(lemmatized)
        else:
            # Standard lemmatization without POS tagging
            cleaned = [
                lemmatizer.lemmatize(token)
                for token in tokens
                if (token.isalpha() 
                    and len(token) >= config.min_token_length 
                    and token not in stop_words)
            ]
        
        # Remove duplicates while preserving order
        if config.remove_duplicates:
            seen = set()
            result = []
            for word in cleaned:
                if word not in seen:
                    seen.add(word)
                    result.append(word)
            return " ".join(result)
        else:
            return " ".join(cleaned)
    
    except Exception as e:
        logger.warning(f"Error processing text: {e}")
        return ""


def validate_data_quality(df: pd.DataFrame, text_column: str) -> dict:
    """Validate data quality and return metrics.
    
    Args:
        df: DataFrame to validate
        text_column: Name of text column to check
        
    Returns:
        Dictionary with validation metrics
    """
    metrics = {
        'total_rows': len(df),
        'empty_text': 0,
        'null_text': 0,
        'duplicate_rows': 0,
        'avg_text_length': 0,
        'min_text_length': 0,
        'max_text_length': 0
    }
    
    if text_column in df.columns:
        text_series = df[text_column].fillna('')
        metrics['empty_text'] = (text_series == '').sum()
        metrics['null_text'] = df[text_column].isnull().sum()
        metrics['duplicate_rows'] = df.duplicated().sum()
        
        text_lengths = text_series.str.len()
        metrics['avg_text_length'] = text_lengths.mean()
        metrics['min_text_length'] = text_lengths.min()
        metrics['max_text_length'] = text_lengths.max()
    
    return metrics

def process_data_efficiently(config: Optional[PreprocessingConfig] = None) -> None:
    """Process data in chunks for better memory management.
    
    Args:
        config: Preprocessing configuration object (uses defaults if None)
    """
    if config is None:
        config = PreprocessingConfig()
    
    stop_words = get_stop_words(config)
    try:
        logger.info(f"Starting data processing: {config.input_file} -> {config.output_file}")
        
        # Check if input file exists and get basic info
        try:
            total_rows = sum(1 for _ in open(config.input_file)) - 1  # Subtract header
            logger.info(f"Processing {total_rows:,} rows in chunks of {config.chunk_size:,}")
        except FileNotFoundError:
            logger.error(f"Input file not found: {config.input_file}")
            return
        
        first_chunk = True
        processed_rows = 0
        
        # Process in chunks
        total_processed = 0
        total_valid = 0
        
        for chunk_num, chunk in enumerate(pd.read_csv(config.input_file, chunksize=config.chunk_size)):
            # Validate required column exists
            if config.text_column not in chunk.columns:
                logger.error(f"Column '{config.text_column}' not found in data")
                return
            
            # Validate chunk quality
            if first_chunk and config.validate_output:
                metrics = validate_data_quality(chunk, config.text_column)
                logger.info(f"Data quality metrics: {metrics}")
                
                # Warn about quality issues
                if metrics['empty_text'] > len(chunk) * 0.1:
                    logger.warning(f"High percentage of empty text: {metrics['empty_text']}/{len(chunk)}")
                if metrics['duplicate_rows'] > 0:
                    logger.warning(f"Found {metrics['duplicate_rows']} duplicate rows")
            
            # Process chunk
            original_size = len(chunk)
            chunk['text_data'] = chunk[config.text_column].fillna('').astype(str)
            
            # Apply preprocessing with configuration
            chunk['cleaned_text'] = chunk['text_data'].apply(
                lambda x: clean_and_lemmatize(x, config, stop_words)
            )
            
            # Remove rows with empty cleaned text to save space
            valid_chunk = chunk[chunk['cleaned_text'].str.len() > 0]
            removed_rows = original_size - len(valid_chunk)
            
            if removed_rows > 0:
                logger.info(f"Chunk {chunk_num}: Removed {removed_rows} rows with empty cleaned text")
            
            # Write to output file if we have valid data
            if len(valid_chunk) > 0:
                mode = 'w' if first_chunk else 'a'
                header = first_chunk
                valid_chunk.to_csv(config.output_file, mode=mode, header=header, index=False)
                total_valid += len(valid_chunk)
            
            total_processed += original_size
            logger.info(f"Processed {total_processed:,} rows, {total_valid:,} valid")
            first_chunk = False
        
        logger.info(f"Data processing completed. Output saved to: {config.output_file}")
        
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise

if __name__ == '__main__':
    process_data_efficiently()




