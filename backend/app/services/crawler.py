import time
import requests
from urllib.parse import urlparse
import urllib.robotparser
from typing import List
import logging
from app.core.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = 'SearchOneBot/1.0'


def allowed_by_robots(url: str, user_agent: str = DEFAULT_USER_AGENT) -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = urllib.robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        # on failure, be conservative and allow (or choose to disallow)
        return True


def crawl_and_ingest(urls: List[str], api_base: str = 'http://127.0.0.1:2001', delay: float = 1.0):
    """Crawl a list of URLs and POST them to the backend /ingest/url endpoint.

    This function respects robots.txt when possible and waits `delay` seconds between requests.
    """
    results = []
    for url in urls:
        try:
            if not allowed_by_robots(url):
                logger.info("URL blocked by robots.txt: %s", url)
                results.append({'url': url, 'status': 'blocked_by_robots'})
                continue
            logger.info("Posting URL to ingest endpoint: %s", url)
            resp = requests.post(f"{api_base}/ingest/url", params={'url': url})
            if resp.status_code == 200:
                results.append({'url': url, 'status': 'ok', 'data': resp.json()})
            else:
                results.append({'url': url, 'status': 'error', 'code': resp.status_code, 'detail': resp.text})
        except Exception as e:
            logger.exception("Exception while crawling %s: %s", url, e)
            results.append({'url': url, 'status': 'exception', 'detail': str(e)})
        time.sleep(delay)
    return results
