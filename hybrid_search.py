"""
Hybrid Search Strategy - A Novel Approach

This module implements advanced retrieval strategies that combine
multiple embedding models and granularities for improved results.

Novel Strategy: Two-Stage Retrieval with Score Fusion
1. Stage 1: Use talk-level embeddings to identify candidate talks (coarse search)
2. Stage 2: Use paragraph-level embeddings to find specific content (fine search)
3. Fusion: Combine scores from both free and OpenAI embeddings
4. Diversity: Ensure results come from different talks for broader coverage
"""

import numpy as np
from semantic_search import SemanticSearch
from typing import List, Dict, Tuple
from collections import defaultdict


class HybridSearch:
    """Implements hybrid search strategies."""

    def __init__(self):
        """Initialize all searchers for hybrid approach."""
        # We'll load these lazily as needed
        self.searchers = {}

    def _get_searcher(self, embedding_type: str, granularity: str) -> SemanticSearch:
        """Get or create a searcher (lazy loading)."""
        key = f"{embedding_type}_{granularity}"
        if key not in self.searchers:
            self.searchers[key] = SemanticSearch(embedding_type, granularity)
        return self.searchers[key]

    def two_stage_retrieval(
        self,
        query: str,
        stage1_k: int = 10,
        stage2_k: int = 3,
        embedding_type: str = "openai"
    ) -> List[Dict]:
        """
        Two-stage retrieval: first find candidate talks, then find best paragraphs.

        Args:
            query: Search query
            stage1_k: Number of talks to retrieve in stage 1
            stage2_k: Final number of results
            embedding_type: Which embeddings to use

        Returns:
            List of results with combined scores
        """
        print(f"Two-Stage Retrieval:")
        print(f"  Stage 1: Finding top {stage1_k} candidate talks...")

        # Stage 1: Coarse search - find relevant talks
        talk_searcher = self._get_searcher(embedding_type, "talks")
        candidate_talks = talk_searcher.search(query, top_k=stage1_k)

        # Extract URLs of candidate talks
        candidate_urls = {talk['url'] for talk in candidate_talks}

        print(f"  Stage 2: Finding best paragraphs within candidates...")

        # Stage 2: Fine search - find best paragraphs within those talks
        para_searcher = self._get_searcher(embedding_type, "paragraphs")
        all_paragraphs = para_searcher.search(query, top_k=100)  # Get many paragraphs

        # Filter to only paragraphs from candidate talks
        filtered_paragraphs = [
            p for p in all_paragraphs
            if p['url'] in candidate_urls
        ][:stage2_k]

        # Combine information
        results = []
        for para in filtered_paragraphs:
            # Find the corresponding talk to get its score
            talk_match = next((t for t in candidate_talks if t['url'] == para['url']), None)
            talk_similarity = talk_match['similarity'] if talk_match else 0.0

            # Combined score (weighted average)
            combined_score = 0.4 * talk_similarity + 0.6 * para['similarity']

            result = para.copy()
            result['talk_similarity'] = talk_similarity
            result['paragraph_similarity'] = para['similarity']
            result['combined_score'] = combined_score
            result['strategy'] = 'two_stage_retrieval'

            results.append(result)

        # Re-sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)

        return results

    def ensemble_fusion(
        self,
        query: str,
        top_k: int = 3,
        granularity: str = "talks"
    ) -> List[Dict]:
        """
        Combine scores from both free and OpenAI embeddings.

        Uses reciprocal rank fusion to combine rankings from both models.

        Args:
            query: Search query
            top_k: Number of results
            granularity: Level of granularity

        Returns:
            List of fused results
        """
        print(f"Ensemble Fusion: Combining free and OpenAI embeddings...")

        # Get results from both embedding types
        free_searcher = self._get_searcher("free", granularity)
        openai_searcher = self._get_searcher("openai", granularity)

        free_results = free_searcher.search(query, top_k=20)
        openai_results = openai_searcher.search(query, top_k=20)

        # Build URL to result mapping
        url_to_scores = defaultdict(lambda: {'free_sim': 0, 'openai_sim': 0, 'free_rank': 1000, 'openai_rank': 1000})

        # Track rankings
        for rank, result in enumerate(free_results):
            key = result['url']
            if granularity == "paragraphs":
                key += f"_para_{result['paragraph_number']}"
            elif granularity == "clusters":
                key += f"_cluster_{result['cluster_id']}"

            url_to_scores[key]['free_sim'] = result['similarity']
            url_to_scores[key]['free_rank'] = rank + 1
            url_to_scores[key]['result'] = result

        for rank, result in enumerate(openai_results):
            key = result['url']
            if granularity == "paragraphs":
                key += f"_para_{result['paragraph_number']}"
            elif granularity == "clusters":
                key += f"_cluster_{result['cluster_id']}"

            url_to_scores[key]['openai_sim'] = result['similarity']
            url_to_scores[key]['openai_rank'] = rank + 1
            if 'result' not in url_to_scores[key]:
                url_to_scores[key]['result'] = result

        # Reciprocal Rank Fusion
        # Score = sum(1/(k + rank_i)) where k=60 is a constant
        k = 60
        fused_results = []

        for key, scores in url_to_scores.items():
            rrf_score = (1 / (k + scores['free_rank'])) + (1 / (k + scores['openai_rank']))

            # Weighted average of similarity scores
            avg_similarity = 0.5 * scores['free_sim'] + 0.5 * scores['openai_sim']

            result = scores['result'].copy()
            result['free_similarity'] = scores['free_sim']
            result['openai_similarity'] = scores['openai_sim']
            result['fused_score'] = rrf_score
            result['average_similarity'] = avg_similarity
            result['strategy'] = 'ensemble_fusion'

            fused_results.append(result)

        # Sort by fused score
        fused_results.sort(key=lambda x: x['fused_score'], reverse=True)

        return fused_results[:top_k]

    def diverse_retrieval(
        self,
        query: str,
        top_k: int = 3,
        embedding_type: str = "openai",
        granularity: str = "paragraphs",
        diversity_penalty: float = 0.3
    ) -> List[Dict]:
        """
        Retrieve diverse results from different talks using MMR-like approach.

        Uses Maximal Marginal Relevance to balance relevance and diversity.

        Args:
            query: Search query
            top_k: Number of results
            embedding_type: Which embeddings to use
            granularity: Level of granularity
            diversity_penalty: How much to penalize similarity to already selected results

        Returns:
            List of diverse results
        """
        print(f"Diverse Retrieval: Using MMR approach for diversity...")

        searcher = self._get_searcher(embedding_type, granularity)
        candidates = searcher.search(query, top_k=50)  # Get many candidates

        if not candidates:
            return []

        # MMR algorithm
        selected = []
        remaining = candidates.copy()

        # Select first result (highest similarity)
        selected.append(remaining.pop(0))

        # Select remaining results to maximize diversity
        while len(selected) < top_k and remaining:
            max_mmr_score = -float('inf')
            max_mmr_idx = 0

            for idx, candidate in enumerate(remaining):
                # Relevance score
                relevance = candidate['similarity']

                # Diversity score (minimum similarity to selected items)
                if granularity == "talks":
                    # For talks, ensure different talks
                    diversity = 1.0 if candidate['url'] not in {s['url'] for s in selected} else 0.0
                else:
                    # For paragraphs/clusters, prefer different talks
                    same_talk_penalty = 0.5 if candidate['url'] in {s['url'] for s in selected} else 0.0
                    diversity = 1.0 - same_talk_penalty

                # MMR score
                mmr_score = (1 - diversity_penalty) * relevance + diversity_penalty * diversity

                if mmr_score > max_mmr_score:
                    max_mmr_score = mmr_score
                    max_mmr_idx = idx

            selected_candidate = remaining.pop(max_mmr_idx)
            selected_candidate['mmr_score'] = max_mmr_score
            selected_candidate['strategy'] = 'diverse_retrieval'
            selected.append(selected_candidate)

        return selected

    def format_results(self, results: List[Dict], strategy_name: str) -> str:
        """Format hybrid search results."""
        output = [f"\nStrategy: {strategy_name}"]
        output.append("=" * 80)

        for i, result in enumerate(results, 1):
            output.append(f"\n{i}. {result['title']}")
            output.append(f"   Speaker: {result['speaker']}")
            output.append(f"   Date: {result['season']} {result['year']}")

            if 'combined_score' in result:
                output.append(f"   Combined Score: {result['combined_score']:.4f}")
                output.append(f"     - Talk Similarity: {result['talk_similarity']:.4f}")
                output.append(f"     - Paragraph Similarity: {result['paragraph_similarity']:.4f}")
            elif 'fused_score' in result:
                output.append(f"   Fused Score: {result['fused_score']:.4f}")
                output.append(f"     - Free Similarity: {result['free_similarity']:.4f}")
                output.append(f"     - OpenAI Similarity: {result['openai_similarity']:.4f}")
            elif 'mmr_score' in result:
                output.append(f"   MMR Score: {result['mmr_score']:.4f}")
                output.append(f"   Similarity: {result['similarity']:.4f}")
            else:
                output.append(f"   Similarity: {result['similarity']:.4f}")

            if 'paragraph_number' in result:
                output.append(f"   Paragraph: {result['paragraph_number']}")

        return "\n".join(output)


def compare_hybrid_strategies(query: str):
    """Compare all hybrid strategies for a given query."""
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"{'='*80}")

    hybrid = HybridSearch()

    # Strategy 1: Two-stage retrieval
    print(f"\n{'-'*80}")
    print("HYBRID STRATEGY 1: Two-Stage Retrieval")
    print(f"{'-'*80}")
    results1 = hybrid.two_stage_retrieval(query, stage1_k=10, stage2_k=3)
    print(hybrid.format_results(results1, "Two-Stage Retrieval"))

    # Strategy 2: Ensemble fusion
    print(f"\n{'-'*80}")
    print("HYBRID STRATEGY 2: Ensemble Fusion (Free + OpenAI)")
    print(f"{'-'*80}")
    results2 = hybrid.ensemble_fusion(query, top_k=3, granularity="talks")
    print(hybrid.format_results(results2, "Ensemble Fusion"))

    # Strategy 3: Diverse retrieval
    print(f"\n{'-'*80}")
    print("HYBRID STRATEGY 3: Diverse Retrieval (MMR)")
    print(f"{'-'*80}")
    results3 = hybrid.diverse_retrieval(query, top_k=3)
    print(hybrid.format_results(results3, "Diverse Retrieval"))


if __name__ == "__main__":
    # Test the hybrid strategies
    query = "How can I gain a testimony of Jesus Christ?"

    print("Testing Hybrid Search Strategies")
    print("="*80)

    compare_hybrid_strategies(query)
