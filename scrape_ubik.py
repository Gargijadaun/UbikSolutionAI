import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from collections import deque
import time
import logging
import json
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Base website URL and starting points (including potential page URLs)
base_url = "https://www.ubiksolution.com"
start_urls = [
    "/",
    "/product-category/",
    "/services/",
    "/global-presence/",  # Possible URL for Our Global Presence
    "/resources/",
    "/investor/",
    "/contact-us/",      # Possible URL for Contact Us
    "/employees-corner/", # Possible URL for Employees Corner
    "/about/",
    "/blog/"
]

# Set to store visited URLs and queue for crawling
visited_urls = set()
url_queue = deque(start_urls)
max_pages = 300  # Increased to cover all pages and subpages

# Set up requests session with retries
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

# Optional: Selenium setup for dynamic content
def setup_selenium():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(options=options)
    return driver

# Normalize and validate URLs
def normalize_url(url):
    return urljoin(base_url, url.strip())

def is_valid_url(url):
    # Exclude invalid URLs (e.g., Cloudflare email protection, anchors, non-content files)
    if url in visited_urls or "cdn-cgi/l/email-protection" in url or url.endswith(('#', '#top')):
        return False
    if url.startswith(('mailto:', 'tel:', 'javascript:')):
        return False
    if url.endswith(('.jpg', '.png', '.pdf', '.css', '.js')):
        return False
    return url.startswith(base_url)

# Extract links from a page
def extract_links(soup, current_url):
    links = set()
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        full_url = normalize_url(href)
        if is_valid_url(full_url):
            links.add(full_url)
    return links

# Scrape a page with structured product, service, and content extraction
def scrape_page(url, use_selenium=False):
    try:
        if use_selenium:
            driver = setup_selenium()
            driver.get(url)
            time.sleep(5)  # Wait for JavaScript to load
            page_source = driver.page_source
            driver.quit()
            soup = BeautifulSoup(page_source, "html.parser")
        else:
            response = session.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

        # Structured data dictionary
        data = {
            'url': url,
            'title': soup.title.get_text(strip=True) if soup.title else '',
            'meta_description': soup.find('meta', attrs={'name': 'description'})['content'] if soup.find('meta', attrs={'name': 'description'}) else '',
            'category': url.split('/product-category/')[-1].strip('/') or 'general' if 'product-category' in url else url.split('/')[-2] or 'general',
            'products': [],
            'services': [],
            'sections': [],
            'links': []
        }

        # Extract product details (for /product-category/ and product pages)
        product_sections = soup.find_all(['div', 'article', 'li'], class_=re.compile('product|item|card|woocommerce|entry|product-card', re.I))
        for section in product_sections:
            product = {}
            # Product name
            name = section.find(['h1', 'h2', 'h3', 'h4', 'span'], class_=re.compile('title|name|product-title|woocommerce-loop-product__title|entry-title', re.I))
            product['name'] = name.get_text(strip=True) if name else ''
            # Product description
            desc = section.find(['p', 'div'], class_=re.compile('description|content|excerpt|summary|woocommerce-product-details__short-description', re.I))
            product['description'] = desc.get_text(strip=True) if desc else ''
            # Product price
            price = section.find(class_=re.compile('price|amount|woocommerce-Price-amount', re.I))
            product['price'] = price.get_text(strip=True) if price else ''
            # Product ingredients
            ingredients = section.find(['p', 'div', 'ul'], class_=re.compile('ingredients|composition|key-ingredients', re.I))
            product['ingredients'] = ingredients.get_text(strip=True) if ingredients else ''
            # Product category
            product['category'] = data['category']
            if product['name']:  # Only add if product name exists
                data['products'].append(product)

        # Extract service details (for /services/ and subpages like /idoc-academy/)
        service_sections = soup.find_all(['div', 'section', 'article'], class_=re.compile('service|solutions|idoc|brandyou|vistaderm|academy', re.I))
        for section in service_sections:
            service = {}
            # Service name
            name = section.find(['h1', 'h2', 'h3', 'h4'], class_=re.compile('title|name|service-title', re.I))
            service['name'] = name.get_text(strip=True) if name else ''
            # Service description
            desc = section.find(['p', 'div'], class_=re.compile('description|content|summary', re.I))
            service['description'] = desc.get_text(strip=True) if desc else ''
            # Service features
            features = section.find(['ul', 'div'], class_=re.compile('features|benefits', re.I))
            service['features'] = features.get_text(strip=True) if features else ''
            if service['name']:  # Only add if service name exists
                data['services'].append(service)

        # Extract general page content (for Global Presence, Resources, Investor, etc.)
        for header in soup.find_all(['h1', 'h2', 'h3']):
            section_content = []
            next_element = header.find_next()
            while next_element and next_element.name not in ['h1', 'h2', 'h3']:
                if next_element.name in ['p', 'div', 'span', 'ul', 'li'] and next_element.get_text(strip=True):
                    section_content.append(next_element.get_text(strip=True))
                next_element = next_element.find_next()
            if section_content:
                data['sections'].append({
                    'header': header.get_text(strip=True),
                    'content': ' '.join(section_content)
                })

        # Extract contact information (for /contact-us/)
        contact_info = soup.find(['div', 'section'], class_=re.compile('contact|footer|info', re.I))
        if contact_info:
            data['contact'] = {
                'text': contact_info.get_text(strip=True),
                'emails': re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', contact_info.get_text()),
                'phones': re.findall(r'\+?\d{1,4}[\s.-]?\d{3}[\s.-]?\d{3,4}[\s.-]?\d{3,4}', contact_info.get_text())
            }

        # Extract pagination links
        pagination = soup.find_all('a', class_=re.compile('next|page-numbers|pagination|woocommerce-pagination', re.I))
        for page in pagination:
            page_url = normalize_url(page['href'])
            if is_valid_url(page_url):
                data['links'].append(page_url)

        # Extract all links for further crawling
        data['links'].extend(extract_links(soup, url))
        return data
    except Exception as e:
        logging.error(f"Failed to scrape {url}: {str(e)}")
        return None

# Crawl the website
scraped_data = {
    'products': [],  # Consolidated product list
    'services': [],  # Consolidated service list
    'pages': {}     # Other page content
}
while url_queue and len(visited_urls) < max_pages:
    path = url_queue.popleft()
    full_url = normalize_url(path)
    if full_url in visited_urls:
        continue

    logging.info(f"Scraping {full_url}")
    page_data = scrape_page(full_url, use_selenium=False)  # Set to True if dynamic content is needed
    if page_data:
        # Add products to consolidated list
        for product in page_data['products']:
            if product not in scraped_data['products']:  # Avoid duplicates
                scraped_data['products'].append(product)
        # Add services to consolidated list
        for service in page_data['services']:
            if service not in scraped_data['services']:  # Avoid duplicates
                scraped_data['services'].append(service)
        # Store page content
        scraped_data['pages'][full_url] = {
            'title': page_data['title'],
            'meta_description': page_data['meta_description'],
            'category': page_data['category'],
            'sections': page_data['sections'],
            'contact': page_data.get('contact', {}),
            'links': page_data['links']
        }
        visited_urls.add(full_url)
        for link in page_data['links']:
            if is_valid_url(link) and link not in url_queue:
                url_queue.append(link)
    time.sleep(1.5)  # Polite delay

# Save comprehensive data to JSON
with open("ubik_data.json", "w", encoding="utf-8") as f:
    json.dump(scraped_data, f, indent=2, ensure_ascii=False)

logging.info(f"Scraping complete. Data saved to ubik_all_data.json")