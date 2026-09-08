import os
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("google-search")

GOOGLE_API_BASE = "https://www.googleapis.com/customsearch/v1"


@mcp.tool()
async def google_search(query: str, num_results: int = 5) -> str:
    """Search the web using Google Custom Search.

    Args:
        query: Search query string.
        num_results: Number of results to return (max 10).
    """
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    engine_id = os.environ.get("GOOGLE_SEARCH_ENGINE_ID", "")

    if not api_key or not engine_id:
        return "Missing GOOGLE_API_KEY or GOOGLE_SEARCH_ENGINE_ID environment variables."

    params = {
        "key": api_key,
        "cx": engine_id,
        "q": query,
        "num": min(num_results, 10),
    }

    async with httpx.AsyncClient(verify=False) as client:
        try:
            response = await client.get(GOOGLE_API_BASE, params=params, timeout=15.0)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            return f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            return f"Request failed: {e}"

    items = data.get("items", [])
    if not items:
        return "No results found."

    results = []
    for item in items:
        results.append(f"Title: {item.get('title')}\nURL: {item.get('link')}\nSnippet: {item.get('snippet')}")

    return "\n\n---\n\n".join(results)


if __name__ == "__main__":
    mcp.run()
