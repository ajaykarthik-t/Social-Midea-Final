#!/usr/bin/env python3
"""
dataset.py - Simple dataset creation for fake social media account detection.
"""

import os
import re
import json
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


class SimpleDataset:
    """Class for creating and processing text data for fake account detection."""
    
    def __init__(self, data_dir='data'):
        """Initialize the dataset class."""
        self.data_dir = data_dir
        self.vectorizer = None
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"Created directory: {data_dir}")
    
    def generate_data(self, n_samples=2000):
        """
        Generate synthetic social media text data with labels.
        
        Args:
            n_samples: Number of examples to generate
            
        Returns:
            DataFrame with 'text' and 'label' columns (0 for real, 1 for fake)
        """
        print(f"Generating {n_samples} synthetic text samples...")
        
        # Characteristics of real vs. fake accounts for synthetic data
        real_patterns = [
            "Just watched {movie} with friends. Highly recommend!",
            "Excited to announce that I'm starting a new job at {company}!",
            "Happy birthday to my {relation}! Love you lots!",
            "Just finished reading {book}. What should I read next?",
            "The weather in {city} is beautiful today. Going for a walk!",
            "Made {food} for dinner tonight. It turned out pretty good!",
            "Can't believe it's already {month}. This year is flying by!",
            "Feeling grateful for my amazing friends and family today.",
            "Just adopted a new {pet}! Any advice for a first-time owner?",
            "Spent the weekend hiking at {location}. The views were incredible!"
        ]
        
        fake_patterns = [
            "MAKE $$$$ FROM HOME!!! Click here: {url}",
            "I made $5000 in just one week using this amazing system! DM me for details!",
            "FREE {product} giveaway! Like, share, and follow to win!",
            "BREAKING NEWS: {celebrity} reveals shocking secret about {topic}!",
            "WARNING: Don't drink water from {source} - government is hiding the truth!",
            "I lost 20 pounds in just 2 days with this miracle pill! {url}",
            "EXCLUSIVE: {politician} caught in scandal with {person}. Share before it's deleted!",
            "I'm giving away 100 free {product} to the first 100 people who retweet this!",
            "SECRET SALE: 90% OFF luxury {item} - Limited time only! Click now: {url}",
            "ATTENTION EVERYONE: {company} is giving away free money! Just click: {url}"
        ]
        
        # Sample data for template filling
        fill_data = {
            "movie": ["The Avengers", "Inception", "The Shawshank Redemption", "Pulp Fiction", "The Dark Knight"],
            "company": ["Google", "Microsoft", "Apple", "Amazon", "Tesla", "Facebook"],
            "relation": ["sister", "brother", "mom", "dad", "best friend", "cousin", "grandma"],
            "book": ["The Great Gatsby", "To Kill a Mockingbird", "1984", "Pride and Prejudice"],
            "city": ["New York", "Los Angeles", "Chicago", "Seattle", "Boston", "Austin"],
            "food": ["pasta", "pizza", "tacos", "sushi", "curry", "stir fry", "lasagna"],
            "month": ["January", "February", "March", "April", "May", "June", "July", "August"],
            "pet": ["dog", "cat", "hamster", "fish", "rabbit", "guinea pig", "parrot"],
            "location": ["Yellowstone", "Grand Canyon", "Yosemite", "Zion National Park"],
            "url": ["bit.ly/2xScam", "tinyurl.com/fakeoffer", "shorturl.at/scamnow"],
            "product": ["iPhone", "MacBook", "AirPods", "PlayStation 5", "Xbox Series X"],
            "celebrity": ["Taylor Swift", "Brad Pitt", "Jennifer Lopez", "Tom Cruise", "Beyoncé"],
            "topic": ["aliens", "illuminati", "the government", "Hollywood", "secret health cure"],
            "source": ["tap water", "bottled water", "public fountains", "filtered water"],
            "politician": ["Biden", "Trump", "Obama", "Clinton", "Bush", "Sanders"],
            "person": ["intern", "aide", "secret agent", "alien", "Russian spy", "FBI informant"],
            "item": ["Gucci bags", "Rolex watches", "Louis Vuitton wallets", "Nike sneakers"],
            "company": ["Amazon", "Google", "Facebook", "Apple", "Microsoft", "PayPal"]
        }
        
        texts = []
        labels = []
        
        for _ in range(n_samples):
            if random.random() < 0.5:  # 50% real, 50% fake
                # Generate real account post
                template = random.choice(real_patterns)
                for key in re.findall(r'\{([^}]+)\}', template):
                    if key in fill_data:
                        template = template.replace(f"{{{key}}}", random.choice(fill_data[key]))
                texts.append(template)
                labels.append(0)  # Real
            else:
                # Generate fake account post
                template = random.choice(fake_patterns)
                for key in re.findall(r'\{([^}]+)\}', template):
                    if key in fill_data:
                        template = template.replace(f"{{{key}}}", random.choice(fill_data[key]))
                texts.append(template)
                labels.append(1)  # Fake
        
        df = pd.DataFrame({"text": texts, "label": labels})
        print(f"Generated dataset with {len(df)} samples")
        return df
    
    def preprocess_text(self, df, text_column='text'):
        """
        Simple text preprocessing.
        
        Args:
            df: DataFrame containing text data
            text_column: Name of the column containing text
            
        Returns:
            DataFrame with added 'cleaned_text' column
        """
        print("Preprocessing text...")
        
        # Make a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Apply basic preprocessing to each text
        processed_df['cleaned_text'] = processed_df[text_column].apply(self._clean_text)
        
        return processed_df
    
    def _clean_text(self, text):
        """Basic text cleaning."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove mentions (@username)
        text = re.sub(r'@\S+', '', text)
        
        # Remove hashtags as symbols but keep the text
        text = re.sub(r'#(\S+)', r'\1', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_features(self, df, max_features=1000):
        """
        Extract simple bag-of-words features.
        
        Args:
            df: DataFrame with a 'cleaned_text' column
            max_features: Maximum number of features to extract
            
        Returns:
            Feature matrix and DataFrame with features
        """
        print(f"Extracting features (max_features={max_features})...")
        
        # Initialize vectorizer
        self.vectorizer = CountVectorizer(max_features=max_features)
        
        # Extract features
        features = self.vectorizer.fit_transform(df['cleaned_text'])
        
        # Convert to array for easier handling
        feature_array = features.toarray()
        
        print(f"Extracted {feature_array.shape[1]} features")
        return feature_array, df
    
    def save_dataset(self, df, features, test_size=0.2):
        """
        Save the dataset and features.
        
        Args:
            df: DataFrame with text data
            features: Feature matrix
            test_size: Fraction of data for testing
            
        Returns:
            Dictionary with paths to saved files
        """
        # Create output folder if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Split data
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            features, df['label'], np.arange(len(df)), 
            test_size=test_size, random_state=42, stratify=df['label']
        )
        
        # Save features
        np.save(os.path.join(self.data_dir, "train_features.npy"), X_train)
        np.save(os.path.join(self.data_dir, "test_features.npy"), X_test)
        
        # Save data
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        train_df.to_csv(os.path.join(self.data_dir, "train_data.csv"), index=False)
        test_df.to_csv(os.path.join(self.data_dir, "test_data.csv"), index=False)
        
        # Save vectorizer
        with open(os.path.join(self.data_dir, "vectorizer.json"), 'w') as f:
            # Convert vocabulary to native Python types
            vocabulary = {str(k): int(v) for k, v in self.vectorizer.vocabulary_.items()}
            feature_names = [str(name) for name in self.vectorizer.get_feature_names_out().tolist()]
            
            json.dump({
                'vocabulary': vocabulary,
                'feature_names': feature_names
            }, f)
        
        print(f"Dataset saved to {self.data_dir}")
        
        return {
            'train_features': os.path.join(self.data_dir, "train_features.npy"),
            'test_features': os.path.join(self.data_dir, "test_features.npy"),
            'train_data': os.path.join(self.data_dir, "train_data.csv"),
            'test_data': os.path.join(self.data_dir, "test_data.csv"),
            'vectorizer': os.path.join(self.data_dir, "vectorizer.json")
        }


def main():
    """Main function to create and save a dataset."""
    # Create a dataset object
    dataset = SimpleDataset(data_dir='data')
    
    # Generate synthetic data
    df = dataset.generate_data(n_samples=2000)
    
    # Preprocess text
    processed_df = dataset.preprocess_text(df)
    
    # Extract features
    features, _ = dataset.extract_features(processed_df, max_features=1000)
    
    # Save dataset
    dataset.save_dataset(processed_df, features)
    
    print("Dataset creation completed!")


if __name__ == "__main__":
    main()