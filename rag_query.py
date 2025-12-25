"""
RAG (Retrieval-Augmented Generation) Module

This module integrates semantic search with ChatGPT to generate
contextually relevant answers based on retrieved General Conference talks.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict
from semantic_search import SemanticSearch
import ast

# Load environment variables
load_dotenv()


class RAGSystem:
    """Retrieval-Augmented Generation system for General Conference talks."""

    def __init__(self, embedding_type: str = "free", granularity: str = "talks"):
        """
        Initialize the RAG system.

        Args:
            embedding_type: "free" or "openai"
            granularity: "talks", "paragraphs", or "clusters"
        """
        self.searcher = SemanticSearch(embedding_type, granularity)

        # Initialize OpenAI client for generation
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=api_key)

    def format_context(self, results: List[Dict]) -> str:
        """
        Format retrieved results into context for the LLM.

        Args:
            results: List of search results

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, result in enumerate(results, 1):
            context_parts.append(f"\n--- Talk {i} ---")
            context_parts.append(f"Title: {result['title']}")
            context_parts.append(f"Speaker: {result['speaker']}, {result['calling']}")
            context_parts.append(f"Date: {result['season']} {result['year']}")
            context_parts.append(f"URL: {result['url']}")
            context_parts.append(f"\nContent:")

            # Handle different text formats
            text = result['text']
            if isinstance(text, str):
                # For talks or paragraphs
                context_parts.append(text)
            elif isinstance(text, list):
                # For clusters (list of paragraphs)
                try:
                    # Try to parse if it's a string representation of a list
                    if isinstance(text, str):
                        text = ast.literal_eval(text)
                    context_parts.append("\n\n".join(text))
                except:
                    context_parts.append(str(text))
            else:
                context_parts.append(str(text))

            context_parts.append("\n")

        return "\n".join(context_parts)

    def generate_answer(
        self,
        query: str,
        top_k: int = 3,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict:
        """
        Generate an answer to the query using RAG.

        Args:
            query: The user's question
            top_k: Number of talks to retrieve
            model: OpenAI model to use for generation
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Dictionary with query, retrieved talks, and generated answer
        """
        # Step 1: Retrieve relevant talks
        print(f"Retrieving top {top_k} relevant talks...")
        results = self.searcher.search(query, top_k)

        # Step 2: Format context
        context = self.format_context(results)

        # Step 3: Create prompt
        system_prompt = """You are a knowledgeable assistant about Latter-day Saint teachings and General Conference talks.

Your task is to answer questions using ONLY the provided General Conference talks as your source material.

Important instructions:
1. Base your answer exclusively on the content of the provided talks
2. Cite specific talks and speakers when referencing their teachings
3. If the provided talks don't contain relevant information, clearly state this
4. Do not use outside knowledge or make assumptions beyond what is stated in the talks
5. Maintain a respectful and thoughtful tone
6. Organize your answer in a clear, structured way

Remember: You must stay within the bounds of the provided talks."""

        user_prompt = f"""Based on the following General Conference talks, please answer this question:

Question: {query}

{context}

Please provide a comprehensive answer based solely on these talks, citing specific speakers and their teachings."""

        # Step 4: Generate answer
        print(f"Generating answer using {model}...")
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            answer = response.choices[0].message.content

        except Exception as e:
            answer = f"Error generating answer: {e}"

        return {
            "query": query,
            "embedding_type": self.searcher.embedding_type,
            "granularity": self.searcher.granularity,
            "retrieved_talks": results,
            "answer": answer,
            "model": model
        }

    def print_result(self, result: Dict):
        """
        Pretty print a RAG result.

        Args:
            result: Dictionary from generate_answer()
        """
        print(f"\n{'='*80}")
        print(f"QUERY: {result['query']}")
        print(f"{'='*80}")
        print(f"Strategy: {result['embedding_type']} embeddings, {result['granularity']} granularity")
        print(f"Model: {result['model']}")
        print(f"\n{'-'*80}")
        print("RETRIEVED TALKS:")
        print(f"{'-'*80}")

        for i, talk in enumerate(result['retrieved_talks'], 1):
            print(f"\n{i}. {talk['title']}")
            print(f"   {talk['speaker']}, {talk['calling']}")
            print(f"   {talk['season']} {talk['year']}")
            print(f"   Similarity: {talk['similarity']:.4f}")

        print(f"\n{'-'*80}")
        print("GENERATED ANSWER:")
        print(f"{'-'*80}")
        print(result['answer'])
        print(f"\n{'='*80}\n")


def main():
    """Example usage of the RAG system."""

    # Test queries
    queries = [
        "How can I gain a testimony of Jesus Christ?",
        "What are some ways to deal with challenges in life and find a purpose?",
        "How can I fix my car if it won't start?",
    ]

    # Initialize RAG system with a specific strategy
    # You can experiment with different combinations:
    # - embedding_type: "free" or "openai"
    # - granularity: "talks", "paragraphs", or "clusters"

    print("Initializing RAG system...")
    rag = RAGSystem(embedding_type="openai", granularity="talks")

    # Generate answer for the first query
    query = queries[0]
    result = rag.generate_answer(query, top_k=3)
    rag.print_result(result)


if __name__ == "__main__":
    main()
