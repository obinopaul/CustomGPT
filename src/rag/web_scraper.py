import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import SSLError, HTTPError

class WebScraper:
    def __init__(self):
        pass

    def normalize_url(self, url):
        parsed_url = urlparse(url)
        scheme = parsed_url.scheme or "http"
        netloc = parsed_url.netloc
        path = parsed_url.path
        normalized_url = f"{scheme}://{netloc}{path}"
        return normalized_url

    def get_base_domain(self, url):
        parsed_url = urlparse(url)
        base_domain = parsed_url.netloc.split(".")[-2:]
        return ".".join(base_domain)

    def get_page_urls(self, url, base_url):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            if 'text/html' not in response.headers.get('Content-Type', '').lower():
                return []

            soup = BeautifulSoup(response.content, "html.parser")
            urls = []
            for link in soup.find_all("a"):
                href = link.get("href")
                if href:
                    absolute_url = urljoin(base_url, href)
                    normalized_url = self.normalize_url(absolute_url)
                    if self.is_valid_url(normalized_url, base_url):
                        urls.append(normalized_url)
            return urls
        except (requests.exceptions.RequestException, SSLError, HTTPError):
            # Silently catch and ignore the exceptions
            return []

    def is_valid_url(self, url, base_url):
        parsed_url = urlparse(url)
        base_domain = self.get_base_domain(base_url)
        return parsed_url.scheme in ["http", "https"] and base_domain in parsed_url.netloc

    def get_weblinks(self, base_url, num_of_urls=None, time_limit=5, max_workers=30):
        start_time = time.time()
        base_url = self.normalize_url(base_url)
        visited_urls = set()
        urls_to_visit = [base_url]
        all_urls = set()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while urls_to_visit and (num_of_urls is None or len(all_urls) < num_of_urls) and time.time() - start_time < time_limit:
                futures = []
                while urls_to_visit and len(futures) < max_workers:
                    url = urls_to_visit.pop(0)
                    if url not in visited_urls:
                        visited_urls.add(url)
                        all_urls.add(url)
                        futures.append(executor.submit(self.get_page_urls, url, base_url))

                for future in as_completed(futures):
                    try:
                        child_urls = future.result()
                        for child_url in child_urls:
                            if child_url not in visited_urls and child_url not in urls_to_visit:
                                urls_to_visit.append(child_url)
                    except Exception as e:
                        print(f"Error processing URL: {future}")
                        print(f"Error message: {str(e)}")

                time.sleep(0.05)

        return sorted(all_urls)

    def api_get_weblinks(self, base_url, num_of_urls=None, time_limit=5, max_workers=30):
        try:
            urls = self.get_weblinks(base_url, num_of_urls, time_limit, max_workers)
            return {"urls": urls}
        except Exception as e:
            return {"error": str(e)}










#-------------------------------------------------------------
import os
import sys
import psutil
import asyncio
import requests
from xml.etree import ElementTree
import asyncio
from typing import List
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
import sys
sys.stdout.reconfigure(encoding='utf-8')



__location__ = os.path.dirname(os.path.abspath(__file__))
__output__ = os.path.join(__location__, "output")

# Append parent directory to system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)



async def crawl_sequential(urls: List[str]):
    print("\n=== Sequential Crawling with Session Reuse ===")

    browser_config = BrowserConfig(
        headless=True,
        # For better performance in Docker or low-memory environments:
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )

    crawl_config = CrawlerRunConfig(
        markdown_generator=DefaultMarkdownGenerator()
    )

    # Create the crawler (opens the browser)
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        session_id = "session1"  # Reuse the same session across all URLs
        results = []
        for url in urls:
            result = await crawler.arun(
                url=url,
                config=crawl_config,
                session_id=session_id
            )
            if result.success:
                print(f"Successfully crawled: {url}")
                # E.g. check markdown length
                print(f"Markdown length: {len(result.markdown_v2.raw_markdown)}")
                results.append({"url": url, "result": result.markdown_v2.raw_markdown})
            else:
                print(f"Failed: {url} - Error: {result.error_message}")
    finally:
        # After all URLs are done, close the crawler (and the browser)
        await crawler.close()
        
    return result
        
        
async def crawl_parallel(urls: List[str], max_concurrent: int = 3):
    print("\n=== Parallel Crawling with Browser Reuse + Memory Check ===")

    # We'll keep track of peak memory usage across all tasks
    peak_memory = 0
    process = psutil.Process(os.getpid())

    def log_memory(prefix: str = ""):
        nonlocal peak_memory
        current_mem = process.memory_info().rss  # in bytes
        if current_mem > peak_memory:
            peak_memory = current_mem
        print(f"{prefix} Current Memory: {current_mem // (1024 * 1024)} MB, Peak: {peak_memory // (1024 * 1024)} MB")

    # Minimal browser config
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,   # corrected from 'verbos=False'
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # We'll chunk the URLs in batches of 'max_concurrent'
        success_count = 0
        fail_count = 0
        for i in range(0, len(urls), max_concurrent):
            batch = urls[i : i + max_concurrent]
            tasks = []

            for j, url in enumerate(batch):
                # Unique session_id per concurrent sub-task
                session_id = f"parallel_session_{i + j}"
                task = crawler.arun(url=url, config=crawl_config, session_id=session_id)
                tasks.append(task)

            # Check memory usage prior to launching tasks
            log_memory(prefix=f"Before batch {i//max_concurrent + 1}: ")

            # Gather results
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check memory usage after tasks complete
            log_memory(prefix=f"After batch {i//max_concurrent + 1}: ")

            output = {}
            # Evaluate results
            for url, result in zip(batch, results):
                print(f"\nCrawled: {url}")
                print(f"Result: {result}")
                
                if isinstance(result, Exception):
                    print(f"Error crawling {url}: {result}")
                    fail_count += 1
                elif result.success:
                    output[url] = result
                    success_count += 1
                else:
                    fail_count += 1

        print(f"\nSummary:")
        print(f"  - Successfully crawled: {success_count}")
        print(f"  - Failed: {fail_count}")

    finally:
        print("\nClosing crawler...")
        await crawler.close()
        # Final memory log
        log_memory(prefix="Final: ")
        print(f"\nPeak memory usage (MB): {peak_memory // (1024 * 1024)}")
        
    return result

def get_pydantic_ai_docs_urls():
    """
    Fetches all URLs from the Pydantic AI documentation.
    Uses the sitemap (https://ai.pydantic.dev/sitemap.xml) to get these URLs.
    
    Returns:
        List[str]: List of URLs
    """            
    sitemap_url = "https://ai.pydantic.dev/sitemap.xml"
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        
        # Parse the XML
        root = ElementTree.fromstring(response.content)
        
        # Extract all URLs from the sitemap
        # The namespace is usually defined in the root element
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []        

async def main():
    urls = get_pydantic_ai_docs_urls()
    if urls:
        print(f"Found {len(urls)} URLs to crawl")
        result = await crawl_parallel(urls, max_concurrent=10)  # Run in parallel
        result = await crawl_sequential(urls)   # Run sequentially
        print(f"results: {result}")
    else:
        print("No URLs found to crawl")    

if __name__ == "__main__":
    asyncio.run(main())

