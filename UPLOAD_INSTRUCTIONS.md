# Quick Upload to GitHub

## Option 1: GitHub Web UI (easiest, no git needed)
1. Go to https://github.com/pranavgandhirecruiting-star/semantic-scholar-mcp
2. Click "Add file" → "Upload files"
3. Drag all 4 files into the upload area:
   - semantic_scholar_server.py
   - Dockerfile
   - requirements.txt
   - README.md (will replace the default one)
4. Click "Commit changes"

## Option 2: Git CLI
```bash
git clone https://github.com/pranavgandhirecruiting-star/semantic-scholar-mcp.git
cd semantic-scholar-mcp
# Copy the 4 files into this directory, then:
git add .
git commit -m "Add Semantic Scholar MCP server for ML researcher recruiting"
git push
```
