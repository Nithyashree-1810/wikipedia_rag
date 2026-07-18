import httpx
from backend.rag.chunker import chunk_by_paragraph

WIKI_API = "https://en.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": "WikiRAGApp/2.0 (educational project; contact@example.com) httpx/python",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

async def search_wikipedia(query: str) -> list[str]:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 3,
        "format": "json",
        "utf8": 1
    }
    async with httpx.AsyncClient(timeout=10.0) as http:
        r = await http.get(WIKI_API, params=params, headers=HEADERS)
        if r.status_code != 200 or not r.text.strip():
            return []
        data = r.json()
        return [x["title"] for x in data.get("query", {}).get("search", [])]


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
        return list(pages.values())[0].get("extract", "")


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