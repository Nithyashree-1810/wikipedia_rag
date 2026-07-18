import httpx
import re
from backend.rag.chunker import chunk_by_paragraph

WIKI_API = "https://en.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": "WikiRAGApp/2.0 (educational project; contact@example.com) httpx/python",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

def clean_query(query: str) -> str:
    """Remove typos-prone words, normalize query for Wikipedia search."""
    query = query.strip()
    # remove question marks and common filler words
    query = re.sub(r'[?!]', '', query)
    query = re.sub(r'\b(who is|what is|what are|tell me about|explain|how does|when was|where is)\b', '', query, flags=re.IGNORECASE)
    query = re.sub(r'\s+', ' ', query).strip()
    return query

def simplify_query(query: str) -> str:
    """Fall back to first 3 meaningful words if full query fails."""
    words = [w for w in query.split() if len(w) > 2]
    return " ".join(words[:3])

async def search_wikipedia(query: str) -> list[str]:
    cleaned = clean_query(query)
    params = {
        "action": "query",
        "list": "search",
        "srsearch": cleaned,
        "srlimit": 3,
        "format": "json",
        "utf8": 1
    }
    async with httpx.AsyncClient(timeout=10.0) as http:
        r = await http.get(WIKI_API, params=params, headers=HEADERS)
        if r.status_code != 200 or not r.text.strip():
            return []
        data = r.json()
        results = data.get("query", {}).get("search", [])

        # if no results, retry with simplified query
        if not results:
            simple = simplify_query(cleaned)
            if simple and simple != cleaned:
                params["srsearch"] = simple
                r2 = await http.get(WIKI_API, params=params, headers=HEADERS)
                if r2.status_code == 200 and r2.text.strip():
                    results = r2.json().get("query", {}).get("search", [])

        return [x["title"] for x in results]


async def fetch_article(title: str) -> str:
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "titles": title,
        "format": "json",
        "utf8": 1
    }
    async with httpx.AsyncClient(timeout=15.0) as http:
        r = await http.get(WIKI_API, params=params, headers=HEADERS)
        if r.status_code != 200 or not r.text.strip():
            return ""
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return ""
        page = list(pages.values())[0]
        # -1 pageid means article not found
        if page.get("pageid") == -1:
            return ""
        return page.get("extract", "")


async def retrieve_context(query: str) -> tuple[list[dict], list[dict]]:
    titles = await search_wikipedia(query)
    if not titles:
        return [], []

    all_chunks = []
    sources = []

    for title in titles:
        content = await fetch_article(title)
        if content:
            chunks = chunk_by_paragraph(content)
            all_chunks += [{"text": c, "title": title} for c in chunks]
            sources.append({
                "title": title,
                "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            })

    return all_chunks, sources