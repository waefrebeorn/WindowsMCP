import webbrowser
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def open_url_in_default_browser(url: str) -> Dict[str, Any]:
    """
    Opens the given URL in the system's default web browser.

    Args:
        url: The URL to open (e.g., "http://www.google.com").
             It should be a complete URL including the scheme (http/https).

    Returns:
        A dictionary with "url" and "message" on success, or "error".
    """
    if not url:
        logger.error("Empty URL provided.")
        return {"error": "URL cannot be empty."}

    # Attempt to prepend https:// if scheme is missing
    if not (url.startswith("http://") or url.startswith("https://")):
        logger.info(f"URL '{url}' is missing a scheme. Prepending 'https://'.")
        url = "https://" + url

    logger.info(f"Attempting to open URL: {url}")
    try:
        # webbrowser.open_new_tab(url) # Opens in a new tab if possible, otherwise new window
        # For simplicity and broader compatibility, webbrowser.open() is often sufficient.
        # open_new_tab is generally preferred if available.
        success = webbrowser.open_new_tab(url)
        if success:
            logger.info(f"Successfully requested to open URL: {url}")
            return {"url": url, "message": f"Attempted to open URL '{url}' in the default browser."}
        else:
            # This 'else' case for webbrowser.open/open_new_tab returning False is rare
            # on desktop systems but indicates an issue finding/launching a browser.
            logger.warning(f"webbrowser.open_new_tab({url}) returned False. Browser might not have launched.")
            return {"url": url, "error": "Failed to open URL. The browser could not be launched or the action was blocked."}
    except Exception as e:
        logger.error(f"Error opening URL '{url}': {e}", exc_info=True)
        return {"error": f"An unexpected error occurred while opening URL: {e}"}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    logger.info("Web Interaction Module Example")

    test_url_valid = "https://www.google.com"
    test_url_needs_scheme = "www.bing.com"
    test_url_empty = ""

    print(f"\n--- Testing open_url_in_default_browser with valid URL: {test_url_valid} ---")
    result_valid = open_url_in_default_browser(test_url_valid)
    print(result_valid)
    assert "error" not in result_valid, f"Valid URL test failed: {result_valid.get('error')}"

    print(f"\n--- Testing open_url_in_default_browser with URL needing scheme: {test_url_needs_scheme} ---")
    result_needs_scheme = open_url_in_default_browser(test_url_needs_scheme)
    print(result_needs_scheme)
    assert "error" not in result_needs_scheme, f"URL needing scheme test failed: {result_needs_scheme.get('error')}"
    # Expected to open https://www.bing.com

    print(f"\n--- Testing open_url_in_default_browser with empty URL: {test_url_empty} ---")
    result_empty = open_url_in_default_browser(test_url_empty)
    print(result_empty)
    assert "error" in result_empty, f"Empty URL test failed to produce an error: {result_empty}"
    if "error" in result_empty:
         assert "URL cannot be empty" in result_empty["error"], "Empty URL test failed with wrong error message"


    logger.info("Web Interaction module example finished. If in a desktop environment, some browser tabs might have opened.")


def search_web(query: str, search_engine_url: str = "https://www.google.com/search?q=") -> Dict[str, Any]:
    """
    Performs a web search using the default browser.

    Args:
        query: The search query string.
        search_engine_url: The base URL for the search engine (query term will be appended).
                           Defaults to Google.

    Returns:
        A dictionary with "query", "search_url", and "message" on success, or "error".
    """
    import urllib.parse

    if not query:
        logger.error("Empty search query provided.")
        return {"error": "Search query cannot be empty."}

    try:
        search_url = f"{search_engine_url}{urllib.parse.quote_plus(query)}"
        logger.info(f"Attempting to perform web search for '{query}' by opening URL: {search_url}")

        success = webbrowser.open_new_tab(search_url)
        if success:
            logger.info(f"Successfully requested web search for: {query}")
            return {"query": query, "search_url": search_url, "message": f"Attempted to search for '{query}' in the default browser."}
        else:
            logger.warning(f"webbrowser.open_new_tab({search_url}) returned False for search query '{query}'.")
            return {"query": query, "search_url": search_url, "error": "Failed to open search URL. The browser could not be launched or the action was blocked."}
    except Exception as e:
        logger.error(f"Error performing web search for '{query}': {e}", exc_info=True)
        return {"error": f"An unexpected error occurred while performing web search for '{query}': {e}"}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    # Previous tests for open_url_in_default_browser are now part of the main block.
    # Re-running them here for clarity if this block is executed.
    print("\n--- Re-testing open_url_in_default_browser (if run as __main__) ---")
    open_url_in_default_browser("https://example.com")
    open_url_in_default_browser("neverssl.com") # Test http auto-prefixing

    print("\n--- Testing search_web ---")
    search_query = "latest AI news"
    print(f"Attempting to search for: '{search_query}'")
    search_result = search_web(search_query)
    print(search_result)
    assert "error" not in search_result, f"Search web test failed: {search_result.get('error')}"

    search_query_empty = ""
    print(f"\nAttempting to search with empty query: '{search_query_empty}'")
    search_result_empty = search_web(search_query_empty)
    print(search_result_empty)
    assert "error" in search_result_empty, f"Empty search query test failed to produce an error: {search_result_empty}"
    if "error" in search_result_empty:
        assert "query cannot be empty" in search_result_empty["error"], "Empty search test failed with wrong error message"

    logger.info("Web Interaction module extended example finished.")
