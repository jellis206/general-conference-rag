"""
Experiment Runner for General Conference RAG

This script runs comprehensive experiments comparing different
semantic search strategies and RAG configurations.
"""

import json
from datetime import datetime
from semantic_search import SemanticSearch
from rag_query import RAGSystem
from typing import List, Dict
import os
import numpy as np


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


class ExperimentRunner:
    """Runs and tracks experiments for the assignment."""

    def __init__(self, output_dir: str = "results"):
        """
        Initialize the experiment runner.

        Args:
            output_dir: Directory to save results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.queries = [
            # Static questions from assignment
            "How can I gain a testimony of Jesus Christ?",
            "What are some ways to deal with challenges in life and find a purpose?",
            "How can I fix my car if it won't start?",
            # Custom questions
            "What is the importance of family in the gospel?",
            "How can I strengthen my relationship with God through prayer?",
        ]

        self.strategies = [
            ("free", "talks"),
            ("free", "paragraphs"),
            ("free", "clusters"),
            ("openai", "talks"),
            ("openai", "paragraphs"),
            ("openai", "clusters"),
        ]

    def run_semantic_search_experiments(self, top_k: int = 3):
        """
        Run semantic search experiments for all queries and strategies.

        Args:
            top_k: Number of results to retrieve
        """
        print("="*80)
        print("RUNNING SEMANTIC SEARCH EXPERIMENTS")
        print("="*80)

        all_results = {}

        for query_idx, query in enumerate(self.queries, 1):
            print(f"\n\nQUERY {query_idx}: {query}")
            print("="*80)

            query_results = {}

            for embedding_type, granularity in self.strategies:
                strategy_name = f"{embedding_type}_{granularity}"
                print(f"\n{'-'*80}")
                print(f"Strategy: {embedding_type} embeddings, {granularity} granularity")
                print(f"{'-'*80}")

                try:
                    searcher = SemanticSearch(embedding_type, granularity)
                    results = searcher.search(query, top_k)

                    # Print results
                    print(searcher.format_results(results, show_text=False))

                    # Store results (convert numpy arrays to lists for JSON serialization)
                    serializable_results = []
                    for r in results:
                        r_copy = r.copy()
                        # Convert list strings to actual lists if needed
                        if isinstance(r_copy.get('text'), list):
                            r_copy['text'] = [str(item) for item in r_copy['text']]
                        serializable_results.append(r_copy)

                    query_results[strategy_name] = serializable_results

                except Exception as e:
                    print(f"Error: {e}")
                    query_results[strategy_name] = {"error": str(e)}

            all_results[f"query_{query_idx}"] = {
                "query": query,
                "results": query_results
            }

        # Save results to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"semantic_search_results_{timestamp}.json")

        # Convert numpy types before saving
        all_results = convert_numpy_types(all_results)

        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n\nResults saved to: {output_file}")
        return all_results

    def run_rag_experiments(
        self,
        selected_strategy: tuple = ("openai", "talks"),
        model: str = "gpt-4o-mini",
        top_k: int = 3
    ):
        """
        Run RAG experiments for the three static questions.

        Args:
            selected_strategy: Tuple of (embedding_type, granularity)
            model: OpenAI model to use
            top_k: Number of talks to retrieve
        """
        print("\n\n" + "="*80)
        print("RUNNING RAG EXPERIMENTS")
        print("="*80)
        print(f"Strategy: {selected_strategy[0]} embeddings, {selected_strategy[1]} granularity")
        print(f"Model: {model}")
        print(f"Top-K: {top_k}")

        # Only use the three static questions for RAG
        static_queries = self.queries[:3]

        rag = RAGSystem(embedding_type=selected_strategy[0], granularity=selected_strategy[1])

        all_results = []

        for query_idx, query in enumerate(static_queries, 1):
            print(f"\n\n{'='*80}")
            print(f"RAG QUERY {query_idx}: {query}")
            print(f"{'='*80}")

            try:
                result = rag.generate_answer(query, top_k=top_k, model=model)
                rag.print_result(result)

                # Store result (make it JSON serializable)
                serializable_result = {
                    "query": result["query"],
                    "embedding_type": result["embedding_type"],
                    "granularity": result["granularity"],
                    "model": result["model"],
                    "retrieved_talks": [
                        {
                            "title": t["title"],
                            "speaker": t["speaker"],
                            "calling": t["calling"],
                            "year": t["year"],
                            "season": t["season"],
                            "url": t["url"],
                            "similarity": t["similarity"]
                        }
                        for t in result["retrieved_talks"]
                    ],
                    "answer": result["answer"]
                }

                all_results.append(serializable_result)

            except Exception as e:
                print(f"Error: {e}")
                all_results.append({
                    "query": query,
                    "error": str(e)
                })

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            self.output_dir,
            f"rag_results_{selected_strategy[0]}_{selected_strategy[1]}_{timestamp}.json"
        )

        # Convert numpy types before saving
        all_results = convert_numpy_types(all_results)

        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n\nRAG results saved to: {output_file}")
        return all_results

    def generate_comparison_report(self, semantic_results: Dict):
        """
        Generate a text report comparing different strategies.

        Args:
            semantic_results: Results from semantic search experiments
        """
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("COMPARISON REPORT: Semantic Search Strategies")
        report_lines.append("="*80)
        report_lines.append("")

        report_lines.append("ANALYSIS OF GRANULARITY DIFFERENCES")
        report_lines.append("-"*80)
        report_lines.append("")

        report_lines.append("1. FULL-TEXT EMBEDDINGS (Talks)")
        report_lines.append("   - Captures overall themes and messages of entire talks")
        report_lines.append("   - Best for: Broad thematic queries, finding talks with similar overall messages")
        report_lines.append("   - Limitation: May miss specific details or examples within talks")
        report_lines.append("")

        report_lines.append("2. PARAGRAPH EMBEDDINGS")
        report_lines.append("   - Focuses on individual paragraphs as retrieval units")
        report_lines.append("   - Best for: Specific questions requiring precise teachings or examples")
        report_lines.append("   - Limitation: May lose broader context of the full talk")
        report_lines.append("")

        report_lines.append("3. CLUSTER EMBEDDINGS (k=3)")
        report_lines.append("   - Groups paragraphs into thematic clusters within each talk")
        report_lines.append("   - Best for: Balanced approach between specificity and context")
        report_lines.append("   - Limitation: Clustering may not always align with natural talk structure")
        report_lines.append("")

        report_lines.append("="*80)
        report_lines.append("EMBEDDING MODEL COMPARISON")
        report_lines.append("="*80)
        report_lines.append("")

        report_lines.append("FREE EMBEDDINGS (all-MiniLM-L6-v2)")
        report_lines.append("  - Dimensions: 384")
        report_lines.append("  - Pros: Fast, free, good general performance")
        report_lines.append("  - Cons: Lower dimensional space, may miss nuanced semantic relationships")
        report_lines.append("")

        report_lines.append("OPENAI EMBEDDINGS (text-embedding-3-small)")
        report_lines.append("  - Dimensions: 1536")
        report_lines.append("  - Pros: Higher quality, captures nuanced semantics, better for complex queries")
        report_lines.append("  - Cons: Requires API key, costs money, slightly slower")
        report_lines.append("")

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"comparison_report_{timestamp}.txt")

        with open(report_file, 'w') as f:
            f.write("\n".join(report_lines))

        print(f"\n\nComparison report saved to: {report_file}")
        print("\n".join(report_lines))

        return report_lines


def main():
    """Run all experiments."""

    print("Starting General Conference RAG Experiments")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")

    runner = ExperimentRunner()

    # Part 1: Run semantic search experiments for all strategies
    print("\n" + "="*80)
    print("PART 1: SEMANTIC SEARCH COMPARISON")
    print("="*80)
    semantic_results = runner.run_semantic_search_experiments(top_k=3)

    # Part 2: Generate comparison report
    print("\n" + "="*80)
    print("PART 2: GENERATING COMPARISON REPORT")
    print("="*80)
    runner.generate_comparison_report(semantic_results)

    # Part 3: Run RAG experiments with selected strategy
    print("\n" + "="*80)
    print("PART 3: RAG INTEGRATION")
    print("="*80)
    print("Testing with OpenAI embeddings + full talks strategy")
    rag_results = runner.run_rag_experiments(
        selected_strategy=("openai", "talks"),
        model="gpt-4o-mini",
        top_k=3
    )

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)
    print(f"Results saved in: {runner.output_dir}/")


if __name__ == "__main__":
    main()
