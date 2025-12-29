# General Conference RAG System

A Retrieval-Augmented Generation (RAG) system for semantic search and question answering using Latter-day Saint General Conference talks.

## Overview

This project implements a complete RAG pipeline that:
1. Scrapes General Conference talks from churchofjesuschrist.org
2. Generates embeddings using both free (SentenceTransformers) and OpenAI models
3. Creates clustered representations for efficient semantic search
4. Enables question answering using retrieved talks as context for LLMs

## Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key (for embeddings and generation)
- Optional: PostgreSQL 18 with pgvector extension

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd general-conference-rag
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
# venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Usage

Run the complete pipeline in order:

```bash
# Step 1: Scrape conference talks (takes 5-10 minutes)
python scraper.py

# Step 2: Generate free embeddings (takes 2-5 minutes)
python free_embeddings.py

# Step 3: Generate OpenAI embeddings (takes 5-10 minutes)
python openai_embeddings.py

# Step 4: Create clustered representations (takes 1-2 minutes)
python clusters.py
```

Or run all at once:
```bash
python scraper.py && python free_embeddings.py && python openai_embeddings.py && python clusters.py
```

### Running Experiments

After data generation, verify your setup:
```bash
python verify_setup.py
```

Run experiments:

```bash
# Test semantic search with one strategy
python semantic_search.py

# Test RAG with ChatGPT integration
python rag_query.py

# Run all experiments (for assignment submission)
python experiment_runner.py

# Test novel hybrid strategies
python hybrid_search.py
```

## Project Structure

```
.
├── scraper.py              # Scrapes conference talks
├── free_embeddings.py      # Generates free embeddings
├── openai_embeddings.py    # Generates OpenAI embeddings
├── clusters.py             # Creates k-means clusters
├── semantic_search.py      # Semantic search module
├── rag_query.py            # RAG integration with ChatGPT
├── experiment_runner.py    # Runs all experiments for assignment
├── hybrid_search.py        # Novel hybrid search strategies
├── verify_setup.py         # Verifies setup before experiments
├── .env                    # Environment configuration (create from .env.example)
├── .env.example            # Template for environment variables
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── claude.md               # Detailed technical documentation
├── free/                   # Free embedding outputs
│   ├── free_talks.csv
│   ├── free_paragraphs.csv
│   └── free_3_clusters.csv
├── openai/                 # OpenAI embedding outputs
│   ├── openai_talks.csv
│   ├── openai_paragraphs.csv
│   └── openai_3_clusters.csv
└── results/                # Experiment results (created automatically)
    ├── semantic_search_results_*.json
    ├── rag_results_*.json
    └── comparison_report_*.txt
```

## Configuration

Edit `.env` file:

```bash
OPENAI_API_KEY=your_api_key_here
YEARS=7
```

- `OPENAI_API_KEY`: Your OpenAI API key
- `YEARS`: Number of years to scrape (from current year backwards)

## Output Files

### Scraped Data (temporary)
- `SCRAPED_TALKS.csv`: Full talk texts with metadata
- `SCRAPED_PARAGRAPHS.csv`: Individual paragraphs with metadata

### Embeddings
- `free/free_talks.csv`: Full-text embeddings (384-dim)
- `free/free_paragraphs.csv`: Paragraph embeddings (384-dim)
- `free/free_3_clusters.csv`: 3-cluster representations (384-dim)
- `openai/openai_talks.csv`: Full-text embeddings (1536-dim)
- `openai/openai_paragraphs.csv`: Paragraph embeddings (1536-dim)
- `openai/openai_3_clusters.csv`: 3-cluster representations (1536-dim)

## Semantic Search Strategies

### 1. Full-Text Search
Search entire talks for overall thematic similarity.

### 2. Paragraph Search
Search individual paragraphs for specific teachings or examples.

### 3. Cluster Search
Search k-means cluster centroids (k=3) for balanced granularity.

### 4. Embedding Models
- **Free**: SentenceTransformers 'all-MiniLM-L6-v2' (384-dim)
- **OpenAI**: 'text-embedding-3-small' (1536-dim)

## Assignment Requirements

### Key Tasks
1. Compare semantic search results across strategies
2. Integrate with ChatGPT/Claude for answer generation
3. Design and test a novel retrieval strategy
4. Document results and comparisons

## PostgreSQL Setup (Optional)

If using PostgreSQL 18 with pgvector:

```bash
# Ensure PostgreSQL 18 is in PATH
source ~/.zshrc
pg_use 18

# Connect to database
psql -d your_database

# Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
```

## Performance Notes

- Scraping uses ThreadPoolExecutor for parallelization
- Free embeddings can use GPU if CUDA is available
- OpenAI embeddings process in batches respecting token limits
- CSV files are used for simplicity; consider PostgreSQL for production

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Run `pip install -r requirements.txt`
2. **API Errors**: Check your OpenAI API key in `.env`
3. **Out of Memory**: Reduce batch sizes in embedding scripts
4. **PostgreSQL not found**: Run `source ~/.zshrc && pg_use 18`

### Getting Help

- Review script comments and docstrings
- Verify environment variables in `.env`

## License

Educational project for CS 452 at BYU.

## Acknowledgments

- Starter code from BYU CS 452 course materials
- Conference talks from The Church of Jesus Christ of Latter-day Saints
