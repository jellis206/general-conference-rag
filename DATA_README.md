# Data Files

## What's Included

Due to GitHub file size limitations, only the following embedding files are included in the repository:

- `free/free_talks.csv` (9.3 MB) - Full talk embeddings using free model
- `free/free_3_clusters.csv` (16 MB) - Cluster embeddings using free model
- `openai/openai_talks.csv` (23 MB) - Full talk embeddings using OpenAI

## What's Not Included

The following large files are **not** included but can be regenerated:

- `free/free_paragraphs.csv` (145 MB)
- `openai/openai_paragraphs.csv` (563 MB)
- `openai/openai_3_clusters.csv` (58 MB)

## How to Regenerate Missing Files

If you need the full dataset, run the complete pipeline:

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Run the data generation pipeline
python scraper.py              # Scrape conference talks
python free_embeddings.py      # Generate free embeddings (includes paragraphs)
python openai_embeddings.py    # Generate OpenAI embeddings (includes paragraphs)
python clusters.py             # Generate cluster embeddings
```

**Total time**: ~10-15 minutes
**Requirements**: OpenAI API key in `.env` file

## Alternative: Use Smaller Datasets

The included files (talks and free clusters) are sufficient for:
- Basic semantic search testing
- Understanding the system architecture
- Running experiments with talk-level retrieval

For full paragraph-level and OpenAI cluster experiments, you'll need to regenerate the data.

## File Sizes Reference

| File | Size | Included | Purpose |
|------|------|----------|---------|
| free_talks.csv | 9.3 MB | ✅ Yes | Full talk embeddings (free) |
| free_paragraphs.csv | 145 MB | ❌ No | Paragraph embeddings (free) |
| free_3_clusters.csv | 16 MB | ✅ Yes | Cluster embeddings (free) |
| openai_talks.csv | 23 MB | ✅ Yes | Full talk embeddings (OpenAI) |
| openai_paragraphs.csv | 563 MB | ❌ No | Paragraph embeddings (OpenAI) |
| openai_3_clusters.csv | 58 MB | ❌ No | Cluster embeddings (OpenAI) |

**Total included**: ~48 MB
**Total if regenerated**: ~814 MB
