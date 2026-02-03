FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    mcp \
    httpx \
    fastmcp

# Copy the server
COPY semantic_scholar_server.py .

# Set environment variables (can be overridden)
ENV S2_API_KEY=""
ENV GITHUB_TOKEN=""

# Run the server
CMD ["python", "semantic_scholar_server.py"]
