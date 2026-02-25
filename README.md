# Semantic Scholar MCP Server for ML Researcher Recruiting

A Model Context Protocol (MCP) server that provides tools to find and evaluate Machine Learning researchers using the Semantic Scholar Academic Graph API and GitHub API.

## 🎯 Purpose

This server is designed for **recruiting ML researchers** by combining:
- **Academic metrics** from Semantic Scholar (h-index, citations, venues, influential citations)
- **Code activity** from GitHub (repos, stars, activity level)
- **Conference filtering** for top ML venues (NeurIPS, ICML, ICLR, CVPR, ACL, etc.)

## ⚡ Key Features

### Paper Search
- Search papers by topic with advanced filtering
- Filter by venue (NeurIPS, ICML, ICLR, etc.)
- Filter by year range, minimum citations
- Sort by citations or recency

### Author Discovery
- Search authors by name
- Get detailed author profiles with h-index
- Find top authors at specific conferences
- **Find Rising Stars** - high recent impact, lower h-index
- Batch lookup for multiple authors

### Recruiting Tools
- `find_venue_top_authors` - Find prolific researchers at top conferences
- `find_rising_stars` - Discover talented researchers before they're famous
- `combined_researcher_profile` - Full academic + GitHub profile
- `github_activity_score` - Recruitability scoring (0-100)

## 🛠️ Tools Available

| Tool | Description |
|------|-------------|
| `search_papers` | Search papers with venue/year/citation filters |
| `get_paper_details` | Full paper info with abstract, authors, citations |
| `get_paper_citations` | Papers that cite a specific paper |
| `search_authors` | Search authors by name |
| `get_author_details` | Full author profile with metrics |
| `get_author_papers` | List papers by a specific author |
| `find_venue_top_authors` | **Recruiting:** Top authors at a conference |
| `find_rising_stars` | **Recruiting:** High impact, early career researchers |
| `search_researcher_github` | Find GitHub profiles by name |
| `github_activity_score` | **Recruiting:** GitHub activity score (0-100) |
| `combined_researcher_profile` | **Recruiting:** Combined academic + GitHub profile |
| `batch_author_lookup` | Look up multiple authors at once |
| `list_ml_venues` | Show supported venue shortcuts |

## 🏛️ Supported Venues

Use these shortcuts in venue filters:

**General ML:**
- `neurips` → NeurIPS
- `icml` → ICML
- `iclr` → ICLR
- `aaai` → AAAI
- `ijcai` → IJCAI
- `jmlr` → JMLR

**Computer Vision:**
- `cvpr` → CVPR
- `iccv` → ICCV
- `eccv` → ECCV

**NLP:**
- `acl` → ACL
- `emnlp` → EMNLP
- `naacl` → NAACL

**Applied ML / Robotics:**
- `kdd` → KDD/SIGKDD
- `icra` → ICRA
- `corl` → CoRL

**High Impact Journals:**
- `nature` → Nature
- `science` → Science
- `tpami` → IEEE TPAMI

## 📦 Installation

### Option 1: Docker (Recommended)

```bash
# Build the image
docker build -t semantic-scholar-mcp .

# Run with environment variables
docker run -e S2_API_KEY="your_key" -e GITHUB_TOKEN="your_token" semantic-scholar-mcp
```

### Option 2: Local Python

```bash
# Install dependencies
pip install mcp httpx fastmcp

# Set environment variables
export S2_API_KEY="your_key"  # Optional but recommended
export GITHUB_TOKEN="your_token"  # Required for GitHub features

# Run
python semantic_scholar_server.py
```

## 🔑 API Keys

### Semantic Scholar API Key (Optional but Recommended)
- Without key: 100 requests / 5 minutes (shared with all users)
- With key: 1 request/second (dedicated)
- Request at: https://www.semanticscholar.org/product/api#api-key-form

### GitHub Token (Required for GitHub features)
- Create at: https://github.com/settings/tokens
- Scopes needed: `public_repo` (read public repos)

## 📋 Example Usage

### Find Top Researchers at NeurIPS
```
find_venue_top_authors(venue="neurips", query="transformers", year_from=2022)
```

### Find Rising Stars in LLMs
```
find_rising_stars(topic="large language models", year_from=2023, max_h_index=25)
```

### Search Papers at ICML
```
search_papers(query="reinforcement learning", venue="icml", year_from=2022, min_citations=100)
```

### Get Combined Profile
```
combined_researcher_profile(
    author_name="John Doe",
    s2_author_id="12345",
    github_username="johndoe"
)
```

## 🔄 Integration with Docker MCP Gateway

Add to your `~/.docker/mcp/registry.yaml`:

```yaml
servers:
  semantic-scholar:
    description: "Semantic Scholar API for ML researcher recruiting"
    transport: stdio
    command: docker
    args:
      - run
      - --rm
      - -i
      - -e
      - S2_API_KEY=${S2_API_KEY}
      - -e  
      - GITHUB_TOKEN=${GITHUB_TOKEN}
      - semantic-scholar-mcp
```

## 📊 Rate Limits

| Condition | Rate Limit |
|-----------|------------|
| No API key | 100 req / 5 min (shared) |
| With API key | 1 req / second (dedicated) |
| Batch endpoints | Up to 500-1000 IDs per request |

## 🤝 Combining with Other Data Sources

This server pairs well with:
- **OpenAlex** - For institution-level filtering and additional metrics
- **Papers With Code** - To find researchers with working code
- **arXiv** - For the latest preprints
- **DBLP** - For comprehensive CS bibliography

## 📄 License

MIT License
