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
