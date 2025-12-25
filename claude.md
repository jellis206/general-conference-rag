# General Conference RAG System - Technical Documentation

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system for finding and answering questions about Latter-day Saint General Conference talks. The system combines semantic search using embeddings with generative AI to provide contextually relevant answers based on conference talk content.

## Architecture

### System Components

1. **Data Collection** (scraper.py)
   - Scrapes General Conference talks from churchofjesuschrist.org
   - Extracts metadata (title, speaker, calling, year, season)
   - Splits talks into full text and individual paragraphs
   - Outputs: `SCRAPED_TALKS.csv` and `SCRAPED_PARAGRAPHS.csv`

2. **Embedding Generation**
   - **Free Embeddings** (free_embeddings.py): Uses SentenceTransformer 'all-MiniLM-L6-v2' model
   - **OpenAI Embeddings** (openai_embeddings.py): Uses OpenAI's 'text-embedding-3-small' model
   - Generates vector representations for semantic similarity search
   - Outputs organized into `free/` and `openai/` directories

3. **Clustering** (clusters.py)
   - Groups paragraphs within each talk into k clusters (default k=3)
   - Uses K-means clustering on paragraph embeddings
   - Creates centroid embeddings representing thematic groups
   - Outputs: `{prefix}_{k}_clusters.csv`

4. **Retrieval & Generation** (to be implemented)
   - Semantic search to find relevant talks/paragraphs
   - Integration with ChatGPT/Claude for answer generation
   - Query embedding and similarity calculation

## Data Flow

```
Input Query → Embed Query → Semantic Search → Top-K Results → LLM Context → Generated Answer
```

### Processing Pipeline

```
Raw Web Content (scraper.py)
    ↓
CSV Files (SCRAPED_TALKS.csv, SCRAPED_PARAGRAPHS.csv)
    ↓
Embedding Generation (free_embeddings.py, openai_embeddings.py)
    ↓
Embedding CSVs (free/openai directories)
    ↓
Clustering (clusters.py)
    ↓
Cluster CSVs (3-cluster representations)
    ↓
Semantic Search (to be implemented)
    ↓
Answer Generation (to be implemented)
```

## Embedding Strategies

### 1. Full-Text Embeddings
- **Granularity**: Entire talk as single embedding
- **Pros**: Captures overall themes and message
- **Cons**: May miss specific details, less precise matching
- **Use Case**: Finding talks with similar overall themes

### 2. Paragraph Embeddings
- **Granularity**: Individual paragraphs
- **Pros**: More precise matching, captures specific points
- **Cons**: May lose broader context, more embeddings to search
- **Use Case**: Finding specific teachings or examples

### 3. Cluster Embeddings (k=3)
- **Granularity**: k thematic clusters per talk
- **Pros**: Balance between specificity and context, reduces dimensionality
- **Cons**: Requires tuning k parameter, clustering overhead
- **Use Case**: Finding talks with similar thematic sections

## Semantic Search Algorithm

### Cosine Similarity

The system uses cosine similarity to compare embeddings:

```
similarity = (A · B) / (||A|| × ||B||)
```

Where:
- A is the query embedding
- B is each document embedding
- Result ranges from -1 (opposite) to 1 (identical)

### Search Process

1. Convert user query to embedding using same model
2. Calculate cosine similarity between query and all document embeddings
3. Sort by similarity score (highest first)
4. Return top-k most similar results (typically k=3)

## Implementation Details

### File Structure

```
project/
├── scraper.py              # Web scraper for conference talks
├── free_embeddings.py      # Free embedding generation
├── openai_embeddings.py    # OpenAI embedding generation
├── clusters.py             # K-means clustering
├── .env                    # Environment variables (not in git)
├── .env.example            # Template for environment variables
├── requirements.txt        # Python dependencies
├── SCRAPED_TALKS.csv       # Raw talk data (temporary)
├── SCRAPED_PARAGRAPHS.csv  # Raw paragraph data (temporary)
├── free/                   # Free embeddings directory
│   ├── free_talks.csv
│   ├── free_paragraphs.csv
│   └── free_3_clusters.csv
└── openai/                 # OpenAI embeddings directory
    ├── openai_talks.csv
    ├── openai_paragraphs.csv
    └── openai_3_clusters.csv
```

### Environment Variables

The system uses a `.env` file for configuration:

```
OPENAI_API_KEY=your_openai_api_key_here
YEARS=7
```

- `OPENAI_API_KEY`: Your OpenAI API key for embeddings and generation
- `YEARS`: Number of years of conference talks to scrape (from current year backwards)

## Assignment Requirements

### Part 1: Semantic Search Comparison

For each query, find top-3 similar results across different strategies:

**Static Queries:**
1. How can I gain a testimony of Jesus Christ?
2. What are some ways to deal with challenges in life and find a purpose?
3. How can I fix my car if it won't start?

**Compare:**
- Free embeddings vs OpenAI embeddings
- Full-text vs paragraph vs cluster models

### Part 2: RAG Integration

- Integrate ChatGPT/Claude for answer generation
- Send top-3 talks as context
- Instruct model to answer using only provided talks
- Implement for 3 static questions

### Part 3: Novel Strategy

Design and implement an alternative matching strategy:
- Could involve: hybrid search, re-ranking, multi-stage retrieval, etc.
- Compare effectiveness against baseline strategies
- Document rationale and results

## Model Information

### Free Embeddings: all-MiniLM-L6-v2
- **Provider**: SentenceTransformers (HuggingFace)
- **Dimensions**: 384
- **Max Tokens**: 256 word pieces
- **Performance**: Fast, good for general semantic similarity
- **Cost**: Free

### OpenAI Embeddings: text-embedding-3-small
- **Provider**: OpenAI
- **Dimensions**: 1536
- **Max Tokens**: 8191
- **Performance**: High quality, captures nuanced semantics
- **Cost**: ~$0.02 per 1M tokens

## Performance Considerations

### Optimization Strategies

1. **Batch Processing**: Process embeddings in batches to reduce API calls
2. **Token Limits**: Respect OpenAI's 300K token limit per request
3. **GPU Acceleration**: Use CUDA for free embeddings when available
4. **Caching**: Reuse embeddings across experiments
5. **Parallel Processing**: Use ThreadPoolExecutor for web scraping

### Expected Runtimes

- **Scraping** (7 years): ~5-10 minutes
- **Free Embeddings**: ~2-5 minutes
- **OpenAI Embeddings**: ~5-10 minutes (API dependent)
- **Clustering**: ~1-2 minutes

## Future Enhancements

### PostgreSQL Integration

The project currently uses CSV files for storage. For production use, consider:

1. **Database Schema**:
```sql
CREATE TABLE talks (
    id SERIAL PRIMARY KEY,
    title TEXT,
    speaker TEXT,
    calling TEXT,
    year INTEGER,
    season TEXT,
    url TEXT UNIQUE,
    text TEXT,
    embedding VECTOR(384)  -- or VECTOR(1536) for OpenAI
);

CREATE TABLE paragraphs (
    id SERIAL PRIMARY KEY,
    talk_id INTEGER REFERENCES talks(id),
    paragraph_number INTEGER,
    text TEXT,
    embedding VECTOR(384)
);

CREATE INDEX ON talks USING ivfflat (embedding vector_cosine_ops);
```

2. **Benefits**:
   - Faster similarity search with pgvector
   - Better data integrity and relationships
   - Scalability for larger datasets
   - Built-in indexing for performance

3. **Migration Path**:
   - Install PostgreSQL 18 with pgvector extension
   - Create schema and indexes
   - Import CSV data using COPY commands
   - Update Python scripts to use psycopg2/SQLAlchemy

## Prompt Engineering for RAG

### Effective System Prompts

```
You are a knowledgeable assistant about Latter-day Saint teachings.
Answer the following question using ONLY the provided General Conference talks.
Do not use outside knowledge or make assumptions beyond what is stated in the talks.
If the talks don't contain relevant information, say so.

[Insert top-3 talks here]

Question: [User question]
```

### Key Principles

1. **Constraint**: Explicitly limit model to provided context
2. **Attribution**: Ask model to cite specific talks/speakers
3. **Honesty**: Allow model to say "not found in provided talks"
4. **Formatting**: Structure output for readability

## Troubleshooting

### Common Issues

1. **Empty Results**: Check that embeddings were generated correctly
2. **Low Similarity Scores**: Query may be too specific or out of domain
3. **API Errors**: Verify OpenAI API key and rate limits
4. **Memory Issues**: Process in smaller batches or use smaller model
5. **PostgreSQL Connection**: Run `pg_use 18` after sourcing `~/.zshrc`

## References

- [SentenceTransformers Documentation](https://www.sbert.net/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
- [RAG Paper](https://arxiv.org/abs/2005.11401)
