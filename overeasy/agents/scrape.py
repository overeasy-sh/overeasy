import base64
from urllib.parse import urlparse, urljoin
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from lxml import html
import os

def _is_valid_xml_char(c):
    codepoint = ord(c)
    return (
        0x20 <= codepoint <= 0xD7FF or
        codepoint in (0x9, 0xA, 0xD) or
        0xE000 <= codepoint <= 0xFFFD or
        0x10000 <= codepoint <= 0x10FFFF
    )

def _sanitize_string(s):
    return ''.join(c for c in s if _is_valid_xml_char(c))

def _get_page_source(url):
    options = Options()
    options.add_argument('--headless')
    service = Service(ChromeDriverManager().install())
    
    with webdriver.Chrome(service=service, options=options) as driver:
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        WebDriverWait(driver, 10).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        return driver.page_source

def _download_resource(url, base_url):
    full_url = urljoin(base_url, url) if not url.startswith(('http://', 'https://')) else url
    response = requests.get(full_url)
    response.raise_for_status()
    return response.content

def _embed_stylesheets(tree, base_url):
    for link in tree.xpath('//link[@rel="stylesheet"]'):
        href = link.get('href')
        if href:
            css_content = _download_resource(href, base_url).decode('utf-8')
            style = html.Element('style')
            style.text = css_content
            link.getparent().replace(link, style)

def _embed_scripts(tree, base_url):
    for script in tree.xpath('//script[@src]'):
        src = script.get('src')
        if src:
            js_content = _download_resource(src, base_url)
            script_tag = html.Element('script')
            script_tag.set('type', 'text/javascript')
            script_tag.text = _sanitize_string(js_content.decode('utf-8', errors='ignore'))
            script.getparent().replace(script, script_tag)
    
    for link in tree.xpath('//link[@rel="modulepreload"]'):
        href = link.get('href')
        if href:
            js_content = _download_resource(href, base_url)
            script_tag = html.Element('script')
            script_tag.set('type', 'module')
            script_tag.text = _sanitize_string(js_content.decode('utf-8', errors='ignore'))
            link.getparent().replace(link, script_tag)

def _embed_images(tree, base_url):
    for img in tree.xpath('//img'):
        src = img.get('src')
        if src and not src.startswith('data:'):
            img_content = _download_resource(src, base_url)
            img_base64 = base64.b64encode(img_content).decode('utf-8')
            img.set('src', f"data:image/{src.split('.')[-1]};base64,{img_base64}")

def _remove_footers(tree):
    for footer in tree.xpath('//footer'):
        footer.getparent().remove(footer)

def _process_webpage(url):
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    html_content = _get_page_source(url)
    tree = html.fromstring(html_content)
    
    _embed_stylesheets(tree, base_url)
    _embed_scripts(tree, base_url)
    _embed_images(tree, base_url)
    _remove_footers(tree)
    
    for button in tree.xpath('//button[@aria-label="Download"]'):
        button.getparent().remove(button)
    
    dark_mode_path = os.path.join(os.path.dirname(__file__), 'dark_mode.txt')
    with open(dark_mode_path, 'r') as f:
        dark_mode_content = f.read()
    
    # Append dark_mode_content as the last tag before </html>
    html_root = tree.getroottree().getroot()
    dark_mode_element = html.fromstring(dark_mode_content)
    html_root.append(dark_mode_element)

    return html.tostring(tree, pretty_print=True, encoding='utf-8').decode('utf-8')

def scrape_and_inline_to_buffer(url):
    """
    Scrape a webpage, inline all resources, remove footers, and return the result as a string.

    Args:
    url (str): The URL of the webpage to scrape.

    Returns:
    str: The processed HTML content as a string.
    """
    return _process_webpage(url)

def scrape_and_inline_to_file(url, output_file='gradio_visualization.html'):
    """
    Scrape a webpage, inline all resources, remove footers, and save the result to a file.

    Args:
    url (str): The URL of the webpage to scrape.
    output_file (str): The name of the file to save the result. Defaults to 'requested.html'.

    Returns:
    None
    """
    processed_html = _process_webpage(url)
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(processed_html)