from urllib.parse import urljoin, urlparse
import tldextract
from langchain.schema import Document
from playwright.sync_api import sync_playwright

def normalize_url(base_url, link):
    return urljoin(base_url, link)

def is_internal(base_domain, link_url):
    link_domain = tldextract.extract(link_url).domain
    return base_domain == link_domain

def get_internal_links(base_url, max_links=20):
    visited = set()
    to_visit = [base_url]
    base_domain = tldextract.extract(base_url).domain
    internal_links = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        while to_visit and len(internal_links) < max_links:
            url = to_visit.pop(0)
            if url in visited:
                continue

            try:
                page.goto(url, timeout=10000)
                page.wait_for_load_state("networkidle", timeout=5000)

                visited.add(url)
                anchors = page.query_selector_all("a[href]")
                for anchor in anchors:
                    href = anchor.get_attribute("href")
                    if not href:
                        continue
                    full_url = normalize_url(url, href)
                    parsed = urlparse(full_url)
                    if parsed.scheme in {"http", "https"} and is_internal(base_domain, full_url):
                        if full_url not in visited and full_url not in internal_links:
                            internal_links.append(full_url)
                            to_visit.append(full_url)
            except Exception as e:
                print(f"[!] Failed to crawl {url}: {e}")
                continue

        browser.close()

    return internal_links[:max_links]

def load_docs_from_links(links):
    docs = []
    # Start Playwright session
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Loop through the links and scrape the data
        for url in links:
            try:
                print(f"Visiting: {url}")
                # Go to the URL
                page.goto(url, timeout=10000)

                # Wait for the page to load and for all network requests to complete
                page.wait_for_load_state("networkidle", timeout=5000)

                # Capture all visible text content from the page
                visible_text = page.inner_text("body")

                # Add this document to the list
                docs.append(Document(page_content=visible_text, metadata={"source": url}))

            except Exception as e:
                print(f"[!] Failed to load {url}: {e}")
                continue

        # Close the browser after crawling
        browser.close()

    return docs