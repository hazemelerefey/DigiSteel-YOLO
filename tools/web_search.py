"""Simple web search using DuckDuckGo HTML (no API key)."""
import urllib.request
import urllib.parse
import re
import json
import html
from pathlib import Path


def search_ddg(query, max_results=10):
    """Search DuckDuckGo and return list of {title, link, snippet}."""
    q = urllib.parse.quote(query)
    url = f"https://html.duckduckgo.com/html/?q={q}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        text = resp.read().decode("utf-8", errors="ignore")

    results = []
    # DuckDuckGo HTML results
    pattern = re.compile(
        r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S
    )
    for m in pattern.finditer(text):
        link_raw, title_raw = m.group(1), m.group(2)
        title = re.sub(r"<[^>]+>", "", title_raw).strip()
        title = html.unescape(title)
        # Extract actual URL from DDG redirect
        link = urllib.parse.unquote(link_raw)
        if link.startswith("/"):
            link = "https://html.duckduckgo.com" + link
        results.append({"title": title, "link": link, "snippet": ""})
        if len(results) >= max_results:
            break
    return results


def fetch_page(url, timeout=15):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return f"ERROR: {e}"


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "NEU-DET YOLOv11 transfer learning 2025"
    out = Path(".firecrawl")
    out.mkdir(exist_ok=True)
    results = search_ddg(query, max_results=10)
    for r in results:
        r["html"] = fetch_page(r["link"])[:5000]
    (out / "search_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))
