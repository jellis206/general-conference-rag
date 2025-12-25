"""
Semantic Search Module for General Conference Talks

This module provides functionality to:
1. Load embeddings from CSV files
2. Create embeddings from query strings
3. Calculate cosine similarity
4. Return top-k most similar results
"""

import pandas as pd
import numpy as np
import ast
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()


class SemanticSearch:
    """Handles semantic search across different embedding strategies."""

    def __init__(self, embedding_type: str = "free", granularity: str = "talks"):
        """
        Initialize the semantic search engine.

        Args:
            embedding_type: "free" or "openai"
            granularity: "talks", "paragraphs", or "clusters"
        """
        self.embedding_type = embedding_type
        self.granularity = granularity
        self.df = None
        self.embeddings = None

        # Initialize models
        if embedding_type == "free":
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = 384
        else:  # openai
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            self.client = OpenAI(api_key=api_key)
            self.embedding_dim = 1536

        # Load the appropriate CSV file
        self._load_data()

    def _load_data(self):
        """Load the CSV file with embeddings."""
        if self.granularity == "clusters":
            filename = f"{self.embedding_type}/{self.embedding_type}_3_clusters.csv"
        else:
            filename = f"{self.embedding_type}/{self.embedding_type}_{self.granularity}.csv"

        try:
            self.df = pd.read_csv(filename)
            print(f"Loaded {len(self.df)} entries from {filename}")

            # Convert string embeddings to numpy arrays
            self.df['embedding'] = self.df['embedding'].apply(
                lambda x: np.array(ast.literal_eval(x))
            )

            # Stack embeddings into a matrix for efficient similarity computation
            self.embeddings = np.stack(self.df['embedding'].values)

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find {filename}. "
                "Please run the data generation scripts first:\n"
                "  python scraper.py\n"
                f"  python {'free_embeddings.py' if self.embedding_type == 'free' else 'openai_embeddings.py'}\n"
                "  python clusters.py"
            )

    def create_query_embedding(self, query: str) -> np.ndarray:
        """
        Create an embedding for the query string.

        Args:
            query: The search query

        Returns:
            numpy array of the query embedding
        """
        if self.embedding_type == "free":
            embedding = self.model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        else:  # openai
            response = self.client.embeddings.create(
                input=query,
                model="text-embedding-3-small"
            )
            embedding = np.array(response.data[0].embedding)

        return embedding

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Perform semantic search to find the most similar entries.

        Args:
            query: The search query
            top_k: Number of results to return

        Returns:
            List of dictionaries containing the top-k results
        """
        # Create query embedding
        query_embedding = self.create_query_embedding(query)

        # Calculate cosine similarity with all embeddings
        # Reshape query_embedding to 2D array for sklearn
        query_embedding_2d = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_embedding_2d, self.embeddings)[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Prepare results
        results = []
        for idx in top_indices:
            result = {
                'title': self.df.iloc[idx]['title'],
                'speaker': self.df.iloc[idx]['speaker'],
                'calling': self.df.iloc[idx]['calling'],
                'year': self.df.iloc[idx]['year'],
                'season': self.df.iloc[idx]['season'],
                'url': self.df.iloc[idx]['url'],
                'similarity': float(similarities[idx]),
            }

            # Add text field (different for clusters)
            if self.granularity == "clusters":
                result['cluster_id'] = self.df.iloc[idx]['cluster_id']
                result['text'] = self.df.iloc[idx]['text']  # This is a list of paragraphs
            else:
                result['text'] = self.df.iloc[idx]['text']

            # Add paragraph number if applicable
            if self.granularity == "paragraphs":
                result['paragraph_number'] = self.df.iloc[idx]['paragraph_number']

            results.append(result)

        return results

    def format_results(self, results: List[Dict], show_text: bool = False) -> str:
        """
        Format search results for display.

        Args:
            results: List of result dictionaries
            show_text: Whether to include the full text

        Returns:
            Formatted string
        """
        output = []
        for i, result in enumerate(results, 1):
            output.append(f"\n{i}. {result['title']}")
            output.append(f"   Speaker: {result['speaker']}")
            output.append(f"   Calling: {result['calling']}")
            output.append(f"   Date: {result['season']} {result['year']}")
            output.append(f"   Similarity: {result['similarity']:.4f}")
            output.append(f"   URL: {result['url']}")

            if self.granularity == "paragraphs":
                output.append(f"   Paragraph: {result['paragraph_number']}")
            elif self.granularity == "clusters":
                output.append(f"   Cluster: {result['cluster_id']}")

            if show_text:
                text = result['text']
                if isinstance(text, list):  # Cluster results
                    output.append(f"   Text (top paragraphs):")
                    for j, para in enumerate(text, 1):
                        # Truncate long paragraphs
                        truncated = para[:200] + "..." if len(para) > 200 else para
                        output.append(f"      {j}. {truncated}")
                else:
                    # Truncate long text
                    truncated = text[:500] + "..." if len(text) > 500 else text
                    output.append(f"   Text: {truncated}")

        return "\n".join(output)


def compare_strategies(query: str, top_k: int = 3):
    """
    Compare all search strategies for a given query.

    Args:
        query: The search query
        top_k: Number of results per strategy
    """
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"{'='*80}")

    strategies = [
        ("free", "talks", "Free Embeddings - Full Talks"),
        ("free", "paragraphs", "Free Embeddings - Paragraphs"),
        ("free", "clusters", "Free Embeddings - 3-Clusters"),
        ("openai", "talks", "OpenAI Embeddings - Full Talks"),
        ("openai", "paragraphs", "OpenAI Embeddings - Paragraphs"),
        ("openai", "clusters", "OpenAI Embeddings - 3-Clusters"),
    ]

    for embedding_type, granularity, label in strategies:
        print(f"\n{'-'*80}")
        print(f"{label}")
        print(f"{'-'*80}")

        try:
            searcher = SemanticSearch(embedding_type, granularity)
            results = searcher.search(query, top_k)
            print(searcher.format_results(results, show_text=False))
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    # Example usage
    test_queries = [
        "How can I gain a testimony of Jesus Christ?",
        "What are some ways to deal with challenges in life and find a purpose?",
        "How can I fix my car if it won't start?",
    ]

    # Test with first query
    query = test_queries[0]
    print(f"Testing with query: {query}\n")

    # Example: Free embeddings with full talks
    searcher = SemanticSearch(embedding_type="free", granularity="talks")
    results = searcher.search(query, top_k=3)
    print(searcher.format_results(results, show_text=False))

    # Uncomment to compare all strategies:
    # compare_strategies(query)
