from duckduckgo_search import DDGS

try:
    with DDGS(proxies={"http": "http://103.155.223.116:8080", "https": "http://103.155.223.116:8080"}, timeout=10) as ddgs:
        results = list(ddgs.text("python programming", max_results=2))
        print(results)
except Exception as e:
    print(f"Failed: {e}")
