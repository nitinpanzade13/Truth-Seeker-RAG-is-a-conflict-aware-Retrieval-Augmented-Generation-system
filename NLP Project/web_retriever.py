import requests
from bs4 import BeautifulSoup
from ddgs import DDGS


def web_search(query, max_results=5):
    """
    Retrieve real-time web results using DuckDuckGo search API.
    Returns list of dicts with 'title', 'snippet', 'url'.
    """
    try:
        results_raw = DDGS().text(query, max_results=max_results)
    except Exception as e:
        print(f"[WebRetriever] Search failed: {e}")
        return []

    results = []
    for item in results_raw:
        results.append({
            "title": item.get("title", ""),
            "snippet": item.get("body", ""),
            "url": item.get("href", ""),
        })

    return results


def fetch_page_text(url, max_chars=3000):
    """
    Fetch and extract main text content from a URL.
    """
    headers = {"User-Agent": "TruthSeekerRAG/1.0"}

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
    except requests.RequestException:
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove script/style elements
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    return text[:max_chars]


def web_retrieve(query, max_results=3):
    """
    End-to-end web retrieval: search + fetch page content.
    Returns list of dicts with 'title', 'content', 'url', 'source'.
    """
    search_results = web_search(query, max_results=max_results)
    documents = []

    for result in search_results:
        content = result["snippet"]

        # Try to fetch full page text for richer content
        if result["url"]:
            page_text = fetch_page_text(result["url"])
            if page_text and len(page_text) > len(content):
                content = page_text

        if len(content.strip()) > 50:
            documents.append({
                "title": result["title"],
                "content": content,
                "url": result["url"],
                "source": "web",
            })

    return documents
