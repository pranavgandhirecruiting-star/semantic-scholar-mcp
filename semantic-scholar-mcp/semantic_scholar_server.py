#!/usr/bin/env python3
"""
Semantic Scholar MCP Server for ML Researcher Recruiting

This MCP server provides tools to find and evaluate ML researchers using:
- Semantic Scholar API for academic metrics (h-index, citations, venues)
- GitHub API for code activity and recruitability scoring

Key Features:
- Search papers by topic with venue filtering (NeurIPS, ICML, ICLR, CVPR, ACL)
- Find top authors by conference/venue
- Get detailed author profiles with h-index
- Track influential citations
- Combined academic + GitHub recruitability scoring
"""

import os
import json
import asyncio
import httpx
from typing import Optional, List, Dict, Any
from datetime import datetime
from mcp.server.fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP("semantic-scholar")

# API Configuration
S2_BASE_URL = "https://api.semanticscholar.org/graph/v1"
S2_API_KEY = os.environ.get("S2_API_KEY", "")  # Optional but recommended
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

# Rate limiting - S2 allows 100 req/5min without key, 1 RPS with key
REQUEST_DELAY = 1.0 if S2_API_KEY else 3.0

# Top ML Conference/Venue mappings
TOP_ML_VENUES = {
    "neurips": ["NeurIPS", "Neural Information Processing Systems", "NIPS"],
    "icml": ["ICML", "International Conference on Machine Learning"],
    "iclr": ["ICLR", "International Conference on Learning Representations"],
    "cvpr": ["CVPR", "Computer Vision and Pattern Recognition"],
    "iccv": ["ICCV", "International Conference on Computer Vision"],
    "eccv": ["ECCV", "European Conference on Computer Vision"],
    "acl": ["ACL", "Association for Computational Linguistics"],
    "emnlp": ["EMNLP", "Empirical Methods in Natural Language Processing"],
    "naacl": ["NAACL", "North American Chapter of the Association for Computational Linguistics"],
    "aaai": ["AAAI", "Association for the Advancement of Artificial Intelligence"],
    "ijcai": ["IJCAI", "International Joint Conference on Artificial Intelligence"],
    "kdd": ["KDD", "SIGKDD", "Knowledge Discovery and Data Mining"],
    "icra": ["ICRA", "International Conference on Robotics and Automation"],
    "corl": ["CoRL", "Conference on Robot Learning"],
    "jmlr": ["JMLR", "Journal of Machine Learning Research"],
    "tpami": ["TPAMI", "IEEE Transactions on Pattern Analysis and Machine Intelligence"],
    "nature": ["Nature", "Nature Machine Intelligence"],
    "science": ["Science"],
}


def get_headers() -> Dict[str, str]:
    """Get API headers with optional API key."""
    headers = {"Content-Type": "application/json"}
    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY
    return headers


async def make_s2_request(
    client: httpx.AsyncClient,
    method: str,
    endpoint: str,
    params: Optional[Dict] = None,
    json_data: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Make a request to Semantic Scholar API with rate limiting."""
    url = f"{S2_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = await client.get(url, params=params, headers=get_headers())
        else:
            response = await client.post(url, params=params, json=json_data, headers=get_headers())
        
        if response.status_code == 429:
            return {"error": "Rate limited. Please wait and try again."}
        elif response.status_code == 404:
            return {"error": "Not found"}
        elif response.status_code != 200:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
        
        await asyncio.sleep(REQUEST_DELAY)  # Rate limiting
        return response.json()
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# PAPER SEARCH TOOLS
# =============================================================================

@mcp.tool()
async def search_papers(
    query: str,
    venue: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    min_citations: Optional[int] = None,
    fields_of_study: Optional[str] = None,
    open_access_only: bool = False,
    limit: int = 20,
    sort: str = "citationCount:desc"
) -> str:
    """
    Search for academic papers on Semantic Scholar.
    
    IDEAL FOR: Finding influential papers in specific areas/venues.
    
    Args:
        query: Search terms (searches title and abstract)
        venue: Filter by venue - use shorthand like 'neurips', 'icml', 'iclr', 'cvpr', 'acl'
               or full name like 'NeurIPS', 'Nature'
        year_from: Minimum publication year
        year_to: Maximum publication year
        min_citations: Minimum citation count
        fields_of_study: Comma-separated: Computer Science, Mathematics, etc.
        open_access_only: Only return papers with free PDFs
        limit: Max results (1-100, default 20)
        sort: Sort order - 'citationCount:desc', 'publicationDate:desc', 'paperId'
    
    Returns:
        List of papers with titles, authors, citations, venues, and links
    """
    async with httpx.AsyncClient(timeout=30) as client:
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": "paperId,title,year,venue,publicationVenue,authors,citationCount,influentialCitationCount,abstract,externalIds,openAccessPdf,publicationDate",
        }
        
        # Handle venue filtering
        if venue:
            venue_lower = venue.lower()
            if venue_lower in TOP_ML_VENUES:
                # Use the first (canonical) name
                params["venue"] = TOP_ML_VENUES[venue_lower][0]
            else:
                params["venue"] = venue
        
        # Year filtering
        if year_from and year_to:
            params["year"] = f"{year_from}-{year_to}"
        elif year_from:
            params["year"] = f"{year_from}-"
        elif year_to:
            params["year"] = f"-{year_to}"
        
        if min_citations:
            params["minCitationCount"] = str(min_citations)
        
        if fields_of_study:
            params["fieldsOfStudy"] = fields_of_study
        
        if open_access_only:
            params["openAccessPdf"] = ""
        
        # Use bulk search for sorting capability
        params["sort"] = sort
        
        result = await make_s2_request(client, "GET", "/paper/search/bulk", params)
        
        if "error" in result:
            return f"❌ Error: {result['error']}"
        
        papers = result.get("data", [])
        total = result.get("total", 0)
        
        if not papers:
            return f"No papers found for query: {query}"
        
        output = [f"📚 Found {total:,} papers (showing top {len(papers)})\n"]
        output.append(f"   Query: '{query}'")
        if venue:
            output.append(f"   Venue: {venue}")
        output.append("")
        
        for i, paper in enumerate(papers, 1):
            title = paper.get("title", "Unknown")
            year = paper.get("year", "N/A")
            venue_name = paper.get("venue") or (paper.get("publicationVenue", {}) or {}).get("name", "")
            citations = paper.get("citationCount", 0)
            influential = paper.get("influentialCitationCount", 0)
            paper_id = paper.get("paperId", "")
            
            # Get first 3 authors
            authors = paper.get("authors", [])[:3]
            author_names = ", ".join([a.get("name", "") for a in authors])
            if len(paper.get("authors", [])) > 3:
                author_names += f" +{len(paper['authors']) - 3} more"
            
            # External links
            ext_ids = paper.get("externalIds", {}) or {}
            arxiv_id = ext_ids.get("ArXiv", "")
            doi = ext_ids.get("DOI", "")
            
            output.append(f"{i}. 📄 {title}")
            output.append(f"   👥 {author_names}")
            output.append(f"   📅 {year} | 🏛️ {venue_name or 'N/A'}")
            output.append(f"   📊 Citations: {citations:,} ({influential} influential)")
            
            if arxiv_id:
                output.append(f"   🔗 arXiv: https://arxiv.org/abs/{arxiv_id}")
            if doi:
                output.append(f"   🔗 DOI: https://doi.org/{doi}")
            
            output.append(f"   🆔 S2: {paper_id}")
            output.append("")
        
        return "\n".join(output)


@mcp.tool()
async def get_paper_details(paper_id: str) -> str:
    """
    Get detailed information about a specific paper.
    
    Args:
        paper_id: Semantic Scholar paper ID, DOI (prefix with 'DOI:'), 
                  or arXiv ID (prefix with 'ARXIV:')
    
    Returns:
        Detailed paper info including abstract, all authors, citations
    """
    async with httpx.AsyncClient(timeout=30) as client:
        params = {
            "fields": "paperId,corpusId,title,year,venue,publicationVenue,authors,abstract,citationCount,influentialCitationCount,referenceCount,externalIds,openAccessPdf,publicationDate,tldr,fieldsOfStudy,s2FieldsOfStudy"
        }
        
        result = await make_s2_request(client, "GET", f"/paper/{paper_id}", params)
        
        if "error" in result:
            return f"❌ Error: {result['error']}"
        
        title = result.get("title", "Unknown")
        year = result.get("year", "N/A")
        venue = result.get("venue") or (result.get("publicationVenue", {}) or {}).get("name", "N/A")
        citations = result.get("citationCount", 0)
        influential = result.get("influentialCitationCount", 0)
        refs = result.get("referenceCount", 0)
        abstract = result.get("abstract", "No abstract available")
        tldr = result.get("tldr", {})
        tldr_text = tldr.get("text", "") if tldr else ""
        
        # Authors with IDs
        authors = result.get("authors", [])
        
        output = [f"📄 {title}\n"]
        output.append(f"📅 Year: {year}")
        output.append(f"🏛️ Venue: {venue}")
        output.append(f"📊 Citations: {citations:,} ({influential} influential) | References: {refs}")
        
        # External IDs
        ext_ids = result.get("externalIds", {}) or {}
        if ext_ids.get("ArXiv"):
            output.append(f"🔗 arXiv: https://arxiv.org/abs/{ext_ids['ArXiv']}")
        if ext_ids.get("DOI"):
            output.append(f"🔗 DOI: https://doi.org/{ext_ids['DOI']}")
        
        # Open access PDF
        pdf_info = result.get("openAccessPdf", {})
        if pdf_info and pdf_info.get("url"):
            output.append(f"📥 PDF: {pdf_info['url']}")
        
        output.append(f"\n👥 Authors ({len(authors)}):")
        for author in authors[:10]:
            author_id = author.get("authorId", "")
            name = author.get("name", "Unknown")
            output.append(f"   • {name} [ID: {author_id}]")
        if len(authors) > 10:
            output.append(f"   ... and {len(authors) - 10} more")
        
        # Fields of study
        fields = result.get("s2FieldsOfStudy", []) or []
        if fields:
            field_names = [f.get("category", "") for f in fields[:5]]
            output.append(f"\n📚 Fields: {', '.join(field_names)}")
        
        # TLDR
        if tldr_text:
            output.append(f"\n📝 TL;DR: {tldr_text}")
        
        # Abstract
        output.append(f"\n📋 Abstract:\n{abstract[:1000]}{'...' if len(abstract) > 1000 else ''}")
        
        return "\n".join(output)


@mcp.tool()
async def get_paper_citations(paper_id: str, limit: int = 20) -> str:
    """
    Get papers that cite a specific paper.
    
    Useful for: Finding follow-up work, tracking research impact.
    
    Args:
        paper_id: Semantic Scholar paper ID
        limit: Max citations to return (default 20, max 100)
    
    Returns:
        List of citing papers with their details
    """
    async with httpx.AsyncClient(timeout=30) as client:
        params = {
            "fields": "paperId,title,year,venue,authors,citationCount",
            "limit": min(limit, 100)
        }
        
        result = await make_s2_request(client, "GET", f"/paper/{paper_id}/citations", params)
        
        if "error" in result:
            return f"❌ Error: {result['error']}"
        
        citations = result.get("data", [])
        
        if not citations:
            return "No citations found for this paper."
        
        output = [f"📊 Citations ({len(citations)} shown)\n"]
        
        for i, cite in enumerate(citations, 1):
            paper = cite.get("citingPaper", {})
            title = paper.get("title", "Unknown")
            year = paper.get("year", "N/A")
            venue = paper.get("venue", "")
            cite_count = paper.get("citationCount", 0)
            authors = paper.get("authors", [])[:2]
            author_names = ", ".join([a.get("name", "") for a in authors])
            
            output.append(f"{i}. {title}")
            output.append(f"   👥 {author_names} | 📅 {year} | 🏛️ {venue}")
            output.append(f"   📊 {cite_count:,} citations")
            output.append("")
        
        return "\n".join(output)


# =============================================================================
# AUTHOR SEARCH & ANALYSIS TOOLS
# =============================================================================

@mcp.tool()
async def search_authors(name: str, limit: int = 10) -> str:
    """
    Search for authors by name.
    
    Args:
        name: Author name to search for
        limit: Max results (default 10, max 100)
    
    Returns:
        List of matching authors with their metrics
    """
    async with httpx.AsyncClient(timeout=30) as client:
        params = {
            "query": name,
            "limit": min(limit, 100),
            "fields": "authorId,name,affiliations,homepage,paperCount,citationCount,hIndex"
        }
        
        result = await make_s2_request(client, "GET", "/author/search", params)
        
        if "error" in result:
            return f"❌ Error: {result['error']}"
        
        authors = result.get("data", [])
        total = result.get("total", 0)
        
        if not authors:
            return f"No authors found matching: {name}"
        
        output = [f"👥 Found {total:,} authors matching '{name}' (showing {len(authors)})\n"]
        
        for i, author in enumerate(authors, 1):
            author_id = author.get("authorId", "")
            author_name = author.get("name", "Unknown")
            affiliations = author.get("affiliations", []) or []
            affil_str = affiliations[0] if affiliations else "No affiliation"
            homepage = author.get("homepage", "")
            papers = author.get("paperCount", 0)
            citations = author.get("citationCount", 0)
            h_index = author.get("hIndex", 0)
            
            output.append(f"{i}. 👤 {author_name} [ID: {author_id}]")
            output.append(f"   🏢 {affil_str}")
            output.append(f"   📊 h-index: {h_index} | Papers: {papers:,} | Citations: {citations:,}")
            if homepage:
                output.append(f"   🌐 {homepage}")
            output.append("")
        
        return "\n".join(output)


@mcp.tool()
async def get_author_details(author_id: str) -> str:
    """
    Get detailed information about an author.
    
    Args:
        author_id: Semantic Scholar author ID
    
    Returns:
        Detailed author profile with metrics and recent papers
    """
    async with httpx.AsyncClient(timeout=30) as client:
        # Get author info
        params = {
            "fields": "authorId,externalIds,name,aliases,affiliations,homepage,paperCount,citationCount,hIndex,papers,papers.year,papers.title,papers.venue,papers.citationCount"
        }
        
        result = await make_s2_request(client, "GET", f"/author/{author_id}", params)
        
        if "error" in result:
            return f"❌ Error: {result['error']}"
        
        name = result.get("name", "Unknown")
        affiliations = result.get("affiliations", []) or []
        homepage = result.get("homepage", "")
        paper_count = result.get("paperCount", 0)
        citations = result.get("citationCount", 0)
        h_index = result.get("hIndex", 0)
        papers = result.get("papers", []) or []
        ext_ids = result.get("externalIds", {}) or {}
        
        output = [f"👤 {name}\n"]
        output.append(f"🆔 Semantic Scholar ID: {author_id}")
        
        if ext_ids.get("ORCID"):
            output.append(f"🔗 ORCID: https://orcid.org/{ext_ids['ORCID']}")
        if ext_ids.get("DBLP"):
            output.append(f"🔗 DBLP: {ext_ids['DBLP']}")
        
        if affiliations:
            output.append(f"🏢 Affiliations: {', '.join(affiliations)}")
        
        if homepage:
            output.append(f"🌐 Homepage: {homepage}")
        
        output.append(f"\n📊 Metrics:")
        output.append(f"   h-index: {h_index}")
        output.append(f"   Total papers: {paper_count:,}")
        output.append(f"   Total citations: {citations:,}")
        
        # Sort papers by citations and show top ones
        if papers:
            sorted_papers = sorted(papers, key=lambda x: x.get("citationCount", 0) or 0, reverse=True)
            output.append(f"\n📄 Top Papers (by citations):")
            for paper in sorted_papers[:5]:
                title = paper.get("title", "Unknown")
                year = paper.get("year", "N/A")
                venue = paper.get("venue", "")
                cites = paper.get("citationCount", 0)
                output.append(f"   • [{year}] {title[:60]}{'...' if len(title) > 60 else ''}")
                output.append(f"     🏛️ {venue} | 📊 {cites:,} citations")
        
        return "\n".join(output)


@mcp.tool()
async def get_author_papers(
    author_id: str,
    limit: int = 20,
    year_from: Optional[int] = None,
    sort_by: str = "citations"
) -> str:
    """
    Get papers by a specific author.
    
    Args:
        author_id: Semantic Scholar author ID
        limit: Max papers to return (default 20, max 100)
        year_from: Only papers published on or after this year
        sort_by: 'citations' (default), 'year', or 'influential'
    
    Returns:
        List of author's papers with details
    """
    async with httpx.AsyncClient(timeout=30) as client:
        params = {
            "fields": "paperId,title,year,venue,citationCount,influentialCitationCount,externalIds",
            "limit": min(limit, 100)
        }
        
        result = await make_s2_request(client, "GET", f"/author/{author_id}/papers", params)
        
        if "error" in result:
            return f"❌ Error: {result['error']}"
        
        papers = result.get("data", [])
        
        if not papers:
            return "No papers found for this author."
        
        # Filter by year if specified
        if year_from:
            papers = [p for p in papers if (p.get("year") or 0) >= year_from]
        
        # Sort
        if sort_by == "citations":
            papers.sort(key=lambda x: x.get("citationCount", 0) or 0, reverse=True)
        elif sort_by == "influential":
            papers.sort(key=lambda x: x.get("influentialCitationCount", 0) or 0, reverse=True)
        elif sort_by == "year":
            papers.sort(key=lambda x: x.get("year", 0) or 0, reverse=True)
        
        output = [f"📚 Papers by author {author_id} ({len(papers)} found)\n"]
        
        for i, paper in enumerate(papers[:limit], 1):
            title = paper.get("title", "Unknown")
            year = paper.get("year", "N/A")
            venue = paper.get("venue", "")
            citations = paper.get("citationCount", 0)
            influential = paper.get("influentialCitationCount", 0)
            ext_ids = paper.get("externalIds", {}) or {}
            
            output.append(f"{i}. 📄 {title}")
            output.append(f"   📅 {year} | 🏛️ {venue}")
            output.append(f"   📊 Citations: {citations:,} ({influential} influential)")
            if ext_ids.get("ArXiv"):
                output.append(f"   🔗 arXiv: {ext_ids['ArXiv']}")
            output.append("")
        
        return "\n".join(output)


# =============================================================================
# RECRUITING-FOCUSED TOOLS
# =============================================================================

@mcp.tool()
async def find_venue_top_authors(
    venue: str,
    query: Optional[str] = None,
    year_from: int = 2020,
    min_papers: int = 2,
    limit: int = 20
) -> str:
    """
    Find top authors who publish at a specific venue/conference.
    
    IDEAL FOR RECRUITING: Find prolific researchers at top ML conferences.
    
    Args:
        venue: Conference/venue - 'neurips', 'icml', 'iclr', 'cvpr', 'acl', etc.
        query: Optional topic filter (e.g., 'transformers', 'reinforcement learning')
        year_from: Minimum year (default 2020)
        min_papers: Minimum papers at venue (default 2)
        limit: Max authors to return (default 20)
    
    Returns:
        Ranked list of authors with their venue publications and metrics
    """
    async with httpx.AsyncClient(timeout=60) as client:
        # Resolve venue name
        venue_lower = venue.lower()
        venue_name = TOP_ML_VENUES.get(venue_lower, [venue])[0]
        
        # Search for papers at venue
        search_query = query if query else "*"
        params = {
            "query": search_query,
            "venue": venue_name,
            "year": f"{year_from}-",
            "limit": 100,
            "fields": "paperId,title,year,authors,citationCount",
            "sort": "citationCount:desc"
        }
        
        result = await make_s2_request(client, "GET", "/paper/search/bulk", params)
        
        if "error" in result:
            return f"❌ Error: {result['error']}"
        
        papers = result.get("data", [])
        
        if not papers:
            return f"No papers found at {venue_name} since {year_from}"
        
        # Aggregate authors
        author_stats: Dict[str, Dict] = {}
        
        for paper in papers:
            citations = paper.get("citationCount", 0) or 0
            year = paper.get("year", 0)
            for author in paper.get("authors", []):
                author_id = author.get("authorId")
                if not author_id:
                    continue
                
                if author_id not in author_stats:
                    author_stats[author_id] = {
                        "name": author.get("name", "Unknown"),
                        "papers": 0,
                        "total_citations": 0,
                        "years": []
                    }
                
                author_stats[author_id]["papers"] += 1
                author_stats[author_id]["total_citations"] += citations
                if year:
                    author_stats[author_id]["years"].append(year)
        
        # Filter and rank
        qualified = [
            (aid, stats) for aid, stats in author_stats.items()
            if stats["papers"] >= min_papers
        ]
        qualified.sort(key=lambda x: (x[1]["papers"], x[1]["total_citations"]), reverse=True)
        
        # Get detailed info for top authors
        top_authors = qualified[:limit]
        author_ids = [a[0] for a in top_authors]
        
        # Batch fetch author details
        detail_params = {"fields": "authorId,name,affiliations,hIndex,citationCount,paperCount"}
        detail_result = await make_s2_request(
            client, "POST", "/author/batch", 
            params=detail_params,
            json_data={"ids": author_ids}
        )
        
        author_details = {}
        if "error" not in detail_result and isinstance(detail_result, list):
            for author in detail_result:
                if author:
                    author_details[author.get("authorId")] = author
        
        output = [f"🏆 Top Authors at {venue_name} (since {year_from})\n"]
        if query:
            output.append(f"   Topic filter: '{query}'")
        output.append(f"   Found {len(qualified)} authors with {min_papers}+ papers\n")
        
        for i, (author_id, stats) in enumerate(top_authors, 1):
            name = stats["name"]
            venue_papers = stats["papers"]
            venue_citations = stats["total_citations"]
            years = sorted(stats["years"]) if stats["years"] else []
            year_range = f"{min(years)}-{max(years)}" if years else "N/A"
            
            # Get additional details if available
            details = author_details.get(author_id, {})
            h_index = details.get("hIndex", "?")
            total_citations = details.get("citationCount", "?")
            affiliations = details.get("affiliations", []) or []
            affil = affiliations[0] if affiliations else "Unknown"
            
            output.append(f"{i}. 👤 {name} [ID: {author_id}]")
            output.append(f"   🏢 {affil}")
            output.append(f"   📊 h-index: {h_index} | Total citations: {total_citations:,}" if isinstance(total_citations, int) else f"   📊 h-index: {h_index}")
            output.append(f"   🏛️ {venue_name}: {venue_papers} papers ({venue_citations:,} citations)")
            output.append(f"   📅 Active: {year_range}")
            output.append("")
        
        return "\n".join(output)


@mcp.tool()
async def find_rising_stars(
    topic: str,
    year_from: int = 2022,
    min_citations: int = 50,
    max_h_index: int = 30,
    limit: int = 15
) -> str:
    """
    Find rising star researchers - high recent impact but not yet established.
    
    IDEAL FOR RECRUITING: Find talented researchers before they're famous.
    
    Args:
        topic: Research area (e.g., 'large language models', 'diffusion models')
        year_from: Only consider papers from this year onwards (default 2022)
        min_citations: Minimum citations for papers to consider (default 50)
        max_h_index: Maximum h-index (to filter out established researchers)
        limit: Max results (default 15)
    
    Returns:
        Researchers with high recent impact but lower overall h-index
    """
    async with httpx.AsyncClient(timeout=60) as client:
        # Find highly-cited recent papers
        params = {
            "query": topic,
            "year": f"{year_from}-",
            "minCitationCount": str(min_citations),
            "fieldsOfStudy": "Computer Science",
            "limit": 100,
            "fields": "paperId,title,year,authors,citationCount,influentialCitationCount",
            "sort": "citationCount:desc"
        }
        
        result = await make_s2_request(client, "GET", "/paper/search/bulk", params)
        
        if "error" in result:
            return f"❌ Error: {result['error']}"
        
        papers = result.get("data", [])
        
        if not papers:
            return f"No highly-cited papers found for '{topic}' since {year_from}"
        
        # Collect unique authors
        author_ids = set()
        author_papers: Dict[str, List[Dict]] = {}
        
        for paper in papers:
            for author in paper.get("authors", []):
                aid = author.get("authorId")
                if aid:
                    author_ids.add(aid)
                    if aid not in author_papers:
                        author_papers[aid] = []
                    author_papers[aid].append({
                        "title": paper.get("title"),
                        "year": paper.get("year"),
                        "citations": paper.get("citationCount", 0)
                    })
        
        # Get author details
        author_id_list = list(author_ids)[:500]  # API limit
        detail_params = {"fields": "authorId,name,affiliations,hIndex,citationCount,paperCount,homepage"}
        
        detail_result = await make_s2_request(
            client, "POST", "/author/batch",
            params=detail_params,
            json_data={"ids": author_id_list}
        )
        
        if "error" in detail_result:
            return f"❌ Error getting author details: {detail_result['error']}"
        
        # Filter for rising stars (lower h-index but high recent impact)
        rising_stars = []
        for author in detail_result:
            if not author:
                continue
            h_index = author.get("hIndex", 0) or 0
            if h_index > max_h_index:
                continue
            
            author_id = author.get("authorId")
            recent_papers = author_papers.get(author_id, [])
            recent_citations = sum(p["citations"] for p in recent_papers)
            
            # Score: recent citations / (h_index + 1) to favor high impact relative to career stage
            score = recent_citations / (h_index + 1)
            
            rising_stars.append({
                "author": author,
                "recent_papers": recent_papers,
                "recent_citations": recent_citations,
                "score": score
            })
        
        rising_stars.sort(key=lambda x: x["score"], reverse=True)
        rising_stars = rising_stars[:limit]
        
        output = [f"⭐ Rising Stars in '{topic}'\n"]
        output.append(f"   Criteria: Papers since {year_from} with {min_citations}+ citations")
        output.append(f"   Max h-index: {max_h_index} (filtering out established researchers)\n")
        
        for i, star in enumerate(rising_stars, 1):
            author = star["author"]
            name = author.get("name", "Unknown")
            author_id = author.get("authorId", "")
            h_index = author.get("hIndex", 0)
            total_citations = author.get("citationCount", 0)
            affiliations = author.get("affiliations", []) or []
            affil = affiliations[0] if affiliations else "Unknown"
            homepage = author.get("homepage", "")
            
            recent_citations = star["recent_citations"]
            recent_papers = star["recent_papers"]
            
            output.append(f"{i}. ⭐ {name} [ID: {author_id}]")
            output.append(f"   🏢 {affil}")
            output.append(f"   📊 h-index: {h_index} | Total citations: {total_citations:,}")
            output.append(f"   🚀 Recent impact: {recent_citations:,} citations from {len(recent_papers)} papers")
            if homepage:
                output.append(f"   🌐 {homepage}")
            
            # Top recent paper
            if recent_papers:
                top_paper = max(recent_papers, key=lambda x: x["citations"])
                output.append(f"   📄 Top paper: {top_paper['title'][:50]}... ({top_paper['citations']} cites)")
            output.append("")
        
        return "\n".join(output)


# =============================================================================
# GITHUB INTEGRATION (if token available)
# =============================================================================

@mcp.tool()
async def search_researcher_github(name: str) -> str:
    """
    Search for a researcher's GitHub profile by name.
    
    Args:
        name: Researcher's name
    
    Returns:
        Matching GitHub profiles with activity metrics
    """
    if not GITHUB_TOKEN:
        return "❌ GITHUB_TOKEN not configured. Add it to your MCP server environment."
    
    async with httpx.AsyncClient(timeout=30) as client:
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        response = await client.get(
            "https://api.github.com/search/users",
            params={"q": f"{name} in:name", "per_page": 10},
            headers=headers
        )
        
        if response.status_code != 200:
            return f"❌ GitHub API error: {response.status_code}"
        
        data = response.json()
        users = data.get("items", [])
        
        if not users:
            return f"No GitHub users found matching: {name}"
        
        output = [f"🐙 GitHub profiles matching '{name}':\n"]
        
        for i, user in enumerate(users, 1):
            login = user.get("login", "")
            profile_url = user.get("html_url", "")
            
            # Get detailed user info
            user_resp = await client.get(
                f"https://api.github.com/users/{login}",
                headers=headers
            )
            
            if user_resp.status_code == 200:
                details = user_resp.json()
                full_name = details.get("name", login)
                bio = details.get("bio", "")
                company = details.get("company", "")
                location = details.get("location", "")
                repos = details.get("public_repos", 0)
                followers = details.get("followers", 0)
                
                output.append(f"{i}. 👤 {full_name} (@{login})")
                output.append(f"   🔗 {profile_url}")
                if company:
                    output.append(f"   🏢 {company}")
                if location:
                    output.append(f"   📍 {location}")
                if bio:
                    output.append(f"   📝 {bio[:100]}...")
                output.append(f"   📊 {repos} repos | {followers} followers")
                output.append("")
        
        return "\n".join(output)


@mcp.tool()
async def github_activity_score(username: str) -> str:
    """
    Analyze a GitHub user's activity and calculate a recruitability score.
    
    Args:
        username: GitHub username
    
    Returns:
        Detailed activity analysis with recruitability score (0-100)
    """
    if not GITHUB_TOKEN:
        return "❌ GITHUB_TOKEN not configured. Add it to your MCP server environment."
    
    async with httpx.AsyncClient(timeout=30) as client:
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        # Get user info
        user_resp = await client.get(
            f"https://api.github.com/users/{username}",
            headers=headers
        )
        
        if user_resp.status_code == 404:
            return f"❌ GitHub user '{username}' not found"
        elif user_resp.status_code != 200:
            return f"❌ GitHub API error: {user_resp.status_code}"
        
        user = user_resp.json()
        
        # Get repos
        repos_resp = await client.get(
            f"https://api.github.com/users/{username}/repos",
            params={"per_page": 100, "sort": "updated"},
            headers=headers
        )
        repos = repos_resp.json() if repos_resp.status_code == 200 else []
        
        # Get recent events
        events_resp = await client.get(
            f"https://api.github.com/users/{username}/events",
            params={"per_page": 100},
            headers=headers
        )
        events = events_resp.json() if events_resp.status_code == 200 else []
        
        # Calculate metrics
        name = user.get("name", username)
        bio = user.get("bio", "")
        company = user.get("company", "")
        followers = user.get("followers", 0)
        following = user.get("following", 0)
        public_repos = user.get("public_repos", 0)
        
        # Repo analysis
        total_stars = sum(r.get("stargazers_count", 0) for r in repos if isinstance(r, dict))
        total_forks = sum(r.get("forks_count", 0) for r in repos if isinstance(r, dict))
        languages = {}
        for r in repos:
            if isinstance(r, dict):
                lang = r.get("language")
                if lang:
                    languages[lang] = languages.get(lang, 0) + 1
        
        # Recent activity
        recent_events = len([e for e in events if isinstance(e, dict)])
        push_events = len([e for e in events if isinstance(e, dict) and e.get("type") == "PushEvent"])
        
        # Calculate score (0-100)
        score = 0
        score += min(followers / 10, 20)  # Up to 20 points for followers
        score += min(total_stars / 50, 25)  # Up to 25 points for stars
        score += min(public_repos / 5, 15)  # Up to 15 points for repos
        score += min(recent_events / 10, 20)  # Up to 20 points for activity
        score += min(push_events / 5, 10)  # Up to 10 points for commits
        score += 10 if bio else 0  # 10 points for having a bio
        score = min(int(score), 100)
        
        # Rating
        if score >= 80:
            rating = "🌟 Excellent - Highly active, influential developer"
        elif score >= 60:
            rating = "⭐ Good - Active developer with solid presence"
        elif score >= 40:
            rating = "👍 Moderate - Shows regular activity"
        elif score >= 20:
            rating = "📊 Low - Limited public activity"
        else:
            rating = "❓ Minimal - Very limited public presence"
        
        output = [f"🐙 GitHub Analysis: {name} (@{username})\n"]
        output.append(f"🎯 Recruitability Score: {score}/100")
        output.append(f"   {rating}\n")
        
        output.append("📊 Profile Stats:")
        output.append(f"   Followers: {followers:,} | Following: {following:,}")
        output.append(f"   Public repos: {public_repos}")
        output.append(f"   Total stars: {total_stars:,} | Forks: {total_forks:,}")
        
        if company:
            output.append(f"   Company: {company}")
        if bio:
            output.append(f"   Bio: {bio[:100]}...")
        
        if languages:
            top_langs = sorted(languages.items(), key=lambda x: x[1], reverse=True)[:5]
            lang_str = ", ".join([f"{l[0]} ({l[1]})" for l in top_langs])
            output.append(f"\n💻 Top Languages: {lang_str}")
        
        output.append(f"\n🔥 Recent Activity (last 100 events):")
        output.append(f"   Total events: {recent_events}")
        output.append(f"   Push events: {push_events}")
        
        # Top repos
        if repos:
            output.append("\n⭐ Top Repositories:")
            sorted_repos = sorted(
                [r for r in repos if isinstance(r, dict)],
                key=lambda x: x.get("stargazers_count", 0),
                reverse=True
            )[:3]
            for repo in sorted_repos:
                name = repo.get("name", "")
                stars = repo.get("stargazers_count", 0)
                desc = repo.get("description", "")[:50] if repo.get("description") else ""
                output.append(f"   • {name} ⭐{stars} - {desc}")
        
        return "\n".join(output)


@mcp.tool()
async def combined_researcher_profile(
    author_name: str,
    s2_author_id: Optional[str] = None,
    github_username: Optional[str] = None
) -> str:
    """
    Get a combined academic + GitHub profile for a researcher.
    
    IDEAL FOR RECRUITING: Full picture of a researcher's capabilities.
    
    Args:
        author_name: Researcher's name (used to search S2 if no ID provided)
        s2_author_id: Optional Semantic Scholar author ID
        github_username: Optional GitHub username
    
    Returns:
        Combined profile with academic metrics and GitHub activity
    """
    output = [f"🎯 Combined Profile: {author_name}\n"]
    output.append("=" * 50)
    
    async with httpx.AsyncClient(timeout=60) as client:
        # Get Semantic Scholar profile
        if s2_author_id:
            params = {
                "fields": "authorId,name,affiliations,homepage,paperCount,citationCount,hIndex,externalIds"
            }
            s2_result = await make_s2_request(client, "GET", f"/author/{s2_author_id}", params)
        else:
            # Search by name
            params = {
                "query": author_name,
                "limit": 1,
                "fields": "authorId,name,affiliations,homepage,paperCount,citationCount,hIndex,externalIds"
            }
            search_result = await make_s2_request(client, "GET", "/author/search", params)
            s2_result = search_result.get("data", [{}])[0] if "error" not in search_result else search_result
        
        if "error" not in s2_result and s2_result:
            output.append("\n📚 ACADEMIC PROFILE (Semantic Scholar)")
            output.append("-" * 40)
            
            name = s2_result.get("name", author_name)
            affiliations = s2_result.get("affiliations", []) or []
            h_index = s2_result.get("hIndex", 0)
            citations = s2_result.get("citationCount", 0)
            papers = s2_result.get("paperCount", 0)
            homepage = s2_result.get("homepage", "")
            ext_ids = s2_result.get("externalIds", {}) or {}
            
            output.append(f"👤 {name}")
            if affiliations:
                output.append(f"🏢 {', '.join(affiliations)}")
            output.append(f"📊 h-index: {h_index} | Citations: {citations:,} | Papers: {papers:,}")
            if homepage:
                output.append(f"🌐 {homepage}")
            if ext_ids.get("ORCID"):
                output.append(f"🔗 ORCID: https://orcid.org/{ext_ids['ORCID']}")
        else:
            output.append("\n📚 ACADEMIC PROFILE: Not found on Semantic Scholar")
        
        # Get GitHub profile
        if github_username and GITHUB_TOKEN:
            headers = {
                "Authorization": f"token {GITHUB_TOKEN}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            user_resp = await client.get(
                f"https://api.github.com/users/{github_username}",
                headers=headers
            )
            
            if user_resp.status_code == 200:
                gh_user = user_resp.json()
                
                output.append("\n🐙 GITHUB PROFILE")
                output.append("-" * 40)
                
                gh_name = gh_user.get("name", github_username)
                bio = gh_user.get("bio", "")
                company = gh_user.get("company", "")
                followers = gh_user.get("followers", 0)
                repos = gh_user.get("public_repos", 0)
                
                output.append(f"👤 {gh_name} (@{github_username})")
                if company:
                    output.append(f"🏢 {company}")
                if bio:
                    output.append(f"📝 {bio}")
                output.append(f"📊 Repos: {repos} | Followers: {followers:,}")
                output.append(f"🔗 https://github.com/{github_username}")
            else:
                output.append(f"\n🐙 GITHUB PROFILE: User '{github_username}' not found")
        elif not GITHUB_TOKEN:
            output.append("\n🐙 GITHUB PROFILE: Token not configured")
        else:
            output.append("\n🐙 GITHUB PROFILE: No username provided")
        
        # Overall assessment
        output.append("\n" + "=" * 50)
        output.append("📋 RECRUITMENT ASSESSMENT")
        
        academic_score = min(100, (s2_result.get("hIndex", 0) or 0) * 3 + (s2_result.get("citationCount", 0) or 0) / 100)
        
        if academic_score >= 50:
            output.append(f"   🎓 Academic: Strong (estimated {int(academic_score)}/100)")
        elif academic_score >= 20:
            output.append(f"   🎓 Academic: Moderate (estimated {int(academic_score)}/100)")
        else:
            output.append(f"   🎓 Academic: Early career (estimated {int(academic_score)}/100)")
        
        return "\n".join(output)


# =============================================================================
# UTILITY TOOLS
# =============================================================================

@mcp.tool()
async def list_ml_venues() -> str:
    """
    List supported ML conference/venue shortcuts.
    
    Returns:
        List of venue shortcuts and their full names
    """
    output = ["🏛️ Supported ML Venue Shortcuts\n"]
    output.append("Use these shortcuts in venue filters:\n")
    
    categories = {
        "General ML": ["neurips", "icml", "iclr", "aaai", "ijcai", "jmlr"],
        "Computer Vision": ["cvpr", "iccv", "eccv"],
        "NLP": ["acl", "emnlp", "naacl"],
        "Applied ML": ["kdd"],
        "Robotics": ["icra", "corl"],
        "High Impact Journals": ["nature", "science", "tpami"]
    }
    
    for category, venues in categories.items():
        output.append(f"\n{category}:")
        for v in venues:
            names = TOP_ML_VENUES.get(v, [v])
            output.append(f"   '{v}' → {names[0]}")
    
    return "\n".join(output)


@mcp.tool()
async def batch_author_lookup(author_ids: str) -> str:
    """
    Look up multiple authors at once (up to 500).
    
    Args:
        author_ids: Comma or pipe-separated author IDs
    
    Returns:
        Details for all requested authors
    """
    # Parse IDs
    ids = [id.strip() for id in author_ids.replace("|", ",").split(",")]
    ids = [id for id in ids if id][:500]
    
    if not ids:
        return "❌ No valid author IDs provided"
    
    async with httpx.AsyncClient(timeout=30) as client:
        params = {
            "fields": "authorId,name,affiliations,homepage,paperCount,citationCount,hIndex,externalIds"
        }
        
        result = await make_s2_request(
            client, "POST", "/author/batch",
            params=params,
            json_data={"ids": ids}
        )
        
        if "error" in result:
            return f"❌ Error: {result['error']}"
        
        if not isinstance(result, list):
            return "❌ Unexpected response format"
        
        output = [f"👥 Batch Author Lookup ({len(result)} results)\n"]
        
        for i, author in enumerate(result, 1):
            if not author:
                output.append(f"{i}. ❓ Not found")
                continue
            
            name = author.get("name", "Unknown")
            author_id = author.get("authorId", "")
            affiliations = author.get("affiliations", []) or []
            h_index = author.get("hIndex", 0)
            citations = author.get("citationCount", 0)
            papers = author.get("paperCount", 0)
            
            output.append(f"{i}. 👤 {name} [ID: {author_id}]")
            if affiliations:
                output.append(f"   🏢 {affiliations[0]}")
            output.append(f"   📊 h-index: {h_index} | Citations: {citations:,} | Papers: {papers:,}")
            output.append("")
        
        return "\n".join(output)


if __name__ == "__main__":
    mcp.run()
