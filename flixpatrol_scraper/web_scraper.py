#!/usr/bin/env python3
"""
FlixPatrol Web Scraper
Scrapes FlixPatrol.com for streaming views/rankings data
Used as fallback when API doesn't provide complete data
EXCLUDES Netflix - focuses on all other platforms
"""

import requests
from bs4 import BeautifulSoup
import time
import re
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, wait_exponential, stop_after_attempt
import random

from config import (
    COUNTRIES,
    FLIXPATROL_COUNTRY_CODES,
    PLATFORMS_TO_TRACK,
    PLATFORM_CANONICAL_MAP,
    CONTENT_TYPES,
    REQUEST_DELAY_SECONDS,
    MAX_RETRIES,
    TIMEOUT_SECONDS,
    OUTPUT_DIR,
    HEADERS,
    START_DATE,
    END_DATE,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{OUTPUT_DIR}/web_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FlixPatrolWebScraper:
    """
    Web scraper for FlixPatrol.com
    Collects streaming viewership data from public pages
    """

    BASE_URL = "https://flixpatrol.com"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.collected_data = []

        # Create output directory
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

        # User agents rotation for anti-detection
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        ]

    def _rotate_user_agent(self):
        """Rotate user agent for anti-detection"""
        self.session.headers["User-Agent"] = random.choice(self.user_agents)

    @retry(wait=wait_exponential(multiplier=1, min=2, max=60), stop=stop_after_attempt(MAX_RETRIES))
    def _fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a page"""
        self._rotate_user_agent()

        try:
            response = self.session.get(url, timeout=TIMEOUT_SECONDS)

            if response.status_code == 200:
                return BeautifulSoup(response.text, 'lxml')
            elif response.status_code == 429:
                logger.warning(f"Rate limited on {url}, waiting 60s...")
                time.sleep(60)
                raise Exception("Rate limited")
            elif response.status_code == 403:
                logger.warning(f"Forbidden (403) on {url}, rotating agent...")
                time.sleep(10)
                raise Exception("Forbidden")
            else:
                logger.warning(f"HTTP {response.status_code} on {url}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            raise

    def scrape_top_chart(self, platform: str, country: str, content_type: str,
                         date: str = None) -> List[Dict]:
        """
        Scrape top chart for a specific platform/country/content_type/date

        URL format: https://flixpatrol.com/top10/{platform}/{country}/{date}/{content_type}
        Example: https://flixpatrol.com/top10/amazon-prime-video/united-states/2024-01-15/full/
        """
        results = []

        # Build URL
        if content_type == "movies":
            type_suffix = "full"  # Full chart shows both
        else:
            type_suffix = "full"

        if date:
            url = f"{self.BASE_URL}/top10/{platform}/{country}/{date}/{type_suffix}/"
        else:
            url = f"{self.BASE_URL}/top10/{platform}/{country}/{type_suffix}/"

        logger.debug(f"Scraping: {url}")

        soup = self._fetch_page(url)
        if not soup:
            return results

        # Parse the chart table
        # FlixPatrol uses various table structures
        tables = soup.find_all('table', class_=re.compile(r'.*table.*'))

        for table in tables:
            rows = table.find_all('tr')

            for row in rows:
                try:
                    # Get rank
                    rank_cell = row.find('td', class_=re.compile(r'.*rank.*|.*position.*'))
                    rank = None
                    if rank_cell:
                        rank_text = rank_cell.get_text(strip=True)
                        rank_match = re.search(r'\d+', rank_text)
                        if rank_match:
                            rank = int(rank_match.group())

                    # Get title info
                    title_cell = row.find('td', class_=re.compile(r'.*title.*|.*name.*'))
                    if not title_cell:
                        title_cell = row.find('a', href=re.compile(r'/title/'))

                    title = None
                    title_url = None
                    flixpatrol_id = None

                    if title_cell:
                        link = title_cell.find('a') if title_cell.name != 'a' else title_cell
                        if link:
                            title = link.get_text(strip=True)
                            title_url = link.get('href', '')
                            # Extract FlixPatrol ID from URL
                            id_match = re.search(r'/title/([^/]+)', title_url)
                            if id_match:
                                flixpatrol_id = id_match.group(1)

                    # Get points/score
                    points_cell = row.find('td', class_=re.compile(r'.*points.*|.*score.*'))
                    points = None
                    if points_cell:
                        points_text = points_cell.get_text(strip=True)
                        points_match = re.search(r'[\d,]+', points_text.replace(',', ''))
                        if points_match:
                            points = int(points_match.group().replace(',', ''))

                    # Get content type (movie/tv)
                    type_indicator = row.find(class_=re.compile(r'.*movie.*|.*series.*|.*tv.*'))
                    detected_type = "movie"
                    if type_indicator:
                        type_text = type_indicator.get_text(strip=True).lower()
                        if 'series' in type_text or 'tv' in type_text or 'show' in type_text:
                            detected_type = "tv_show"

                    if title and (rank or points):
                        results.append({
                            "flixpatrol_id": flixpatrol_id,
                            "title": title,
                            "flixpatrol_rank": rank,
                            "flixpatrol_points": points,
                            "content_type": detected_type,
                            "_collection_date": date or datetime.now().strftime("%Y-%m-%d"),
                            "_country_code": country,
                            "_platform": platform,
                            "_platform_canonical": PLATFORM_CANONICAL_MAP.get(platform, platform),
                            "_source_url": url,
                        })

                except Exception as e:
                    logger.debug(f"Error parsing row: {e}")
                    continue

        # Alternative parsing: div-based structure
        if not results:
            chart_items = soup.find_all('div', class_=re.compile(r'.*chart.*item.*|.*top.*entry.*'))

            for i, item in enumerate(chart_items):
                try:
                    title_elem = item.find('a', href=re.compile(r'/title/'))
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)
                    title_url = title_elem.get('href', '')

                    id_match = re.search(r'/title/([^/]+)', title_url)
                    flixpatrol_id = id_match.group(1) if id_match else None

                    # Try to find points
                    points_elem = item.find(class_=re.compile(r'.*point.*|.*score.*'))
                    points = None
                    if points_elem:
                        points_text = points_elem.get_text(strip=True)
                        points_match = re.search(r'[\d,]+', points_text.replace(',', ''))
                        if points_match:
                            points = int(points_match.group().replace(',', ''))

                    results.append({
                        "flixpatrol_id": flixpatrol_id,
                        "title": title,
                        "flixpatrol_rank": i + 1,
                        "flixpatrol_points": points,
                        "content_type": content_type.replace("-", "_").rstrip("s"),
                        "_collection_date": date or datetime.now().strftime("%Y-%m-%d"),
                        "_country_code": country,
                        "_platform": platform,
                        "_platform_canonical": PLATFORM_CANONICAL_MAP.get(platform, platform),
                        "_source_url": url,
                    })

                except Exception as e:
                    logger.debug(f"Error parsing div item: {e}")
                    continue

        return results

    def scrape_title_details(self, flixpatrol_id: str) -> Optional[Dict]:
        """Scrape detailed information for a specific title"""
        url = f"{self.BASE_URL}/title/{flixpatrol_id}/"

        soup = self._fetch_page(url)
        if not soup:
            return None

        details = {"flixpatrol_id": flixpatrol_id}

        try:
            # Title
            title_elem = soup.find('h1')
            if title_elem:
                details["title"] = title_elem.get_text(strip=True)

            # IMDB ID
            imdb_link = soup.find('a', href=re.compile(r'imdb\.com/title/tt\d+'))
            if imdb_link:
                imdb_match = re.search(r'tt\d+', imdb_link.get('href', ''))
                if imdb_match:
                    details["imdb_id"] = imdb_match.group()

            # TMDB ID
            tmdb_link = soup.find('a', href=re.compile(r'themoviedb\.org/(movie|tv)/\d+'))
            if tmdb_link:
                tmdb_match = re.search(r'/(movie|tv)/(\d+)', tmdb_link.get('href', ''))
                if tmdb_match:
                    details["tmdb_id"] = int(tmdb_match.group(2))

            # Release date
            release_elem = soup.find(text=re.compile(r'Release|Premiere'))
            if release_elem:
                parent = release_elem.find_parent()
                if parent:
                    date_match = re.search(r'\d{4}-\d{2}-\d{2}|\d{4}', parent.get_text())
                    if date_match:
                        details["premiere_date"] = date_match.group()

            # Runtime
            runtime_elem = soup.find(text=re.compile(r'Runtime|Duration|Length'))
            if runtime_elem:
                parent = runtime_elem.find_parent()
                if parent:
                    runtime_match = re.search(r'(\d+)\s*min', parent.get_text())
                    if runtime_match:
                        details["runtime_minutes"] = int(runtime_match.group(1))

            # Seasons (for TV shows)
            seasons_elem = soup.find(text=re.compile(r'Season|Episodes'))
            if seasons_elem:
                parent = seasons_elem.find_parent()
                if parent:
                    seasons_match = re.search(r'(\d+)\s*season', parent.get_text(), re.I)
                    if seasons_match:
                        details["max_seasons"] = int(seasons_match.group(1))

            # Description/Overview
            desc_elem = soup.find('div', class_=re.compile(r'.*description.*|.*overview.*|.*synopsis.*'))
            if desc_elem:
                details["overview"] = desc_elem.get_text(strip=True)[:1000]  # Limit length

            # Production countries
            country_elem = soup.find(text=re.compile(r'Country|Countries|Production'))
            if country_elem:
                parent = country_elem.find_parent()
                if parent:
                    details["production_country"] = parent.get_text(strip=True)

        except Exception as e:
            logger.error(f"Error parsing title details for {flixpatrol_id}: {e}")

        return details

    def scrape_historical_charts(self, start_date: str, end_date: str,
                                  platforms: List[str] = None,
                                  countries: List[str] = None) -> List[Dict]:
        """
        Scrape historical chart data for date range
        This is the main data collection method
        """
        all_data = []

        platforms = platforms or PLATFORMS_TO_TRACK
        countries = countries or list(FLIXPATROL_COUNTRY_CODES.values())

        # Filter out Netflix
        platforms = [p for p in platforms if 'netflix' not in p.lower()]

        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        final_date = datetime.strptime(end_date, "%Y-%m-%d")

        total_days = (final_date - current_date).days + 1
        day_count = 0

        while current_date <= final_date:
            date_str = current_date.strftime("%Y-%m-%d")
            day_count += 1

            logger.info(f"[{day_count}/{total_days}] Scraping date: {date_str}")

            for platform in platforms:
                for country in countries:
                    for content_type in CONTENT_TYPES:
                        try:
                            chart_data = self.scrape_top_chart(
                                platform=platform,
                                country=country,
                                content_type=content_type,
                                date=date_str
                            )

                            all_data.extend(chart_data)

                            # Rate limiting
                            time.sleep(REQUEST_DELAY_SECONDS + random.uniform(0.5, 1.5))

                        except Exception as e:
                            logger.error(f"Error scraping {platform}/{country}/{date_str}: {e}")
                            time.sleep(5)  # Extra delay on error

            # Save checkpoint every week
            if current_date.weekday() == 0:  # Monday
                self._save_checkpoint(all_data, f"web_checkpoint_{date_str}")
                logger.info(f"Checkpoint saved. Total records: {len(all_data)}")

            current_date += timedelta(days=1)

        return all_data

    def scrape_weekly_aggregates(self, start_date: str, end_date: str) -> List[Dict]:
        """
        Scrape weekly aggregate data (faster than daily)
        FlixPatrol provides weekly summaries
        """
        all_data = []

        # FlixPatrol weekly URLs
        # https://flixpatrol.com/top10/{platform}/{country}/this-week/full/
        # For historical: use date parameter

        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        final_date = datetime.strptime(end_date, "%Y-%m-%d")

        while current_date <= final_date:
            # Move to Monday of current week
            days_since_monday = current_date.weekday()
            week_start = current_date - timedelta(days=days_since_monday)
            week_str = week_start.strftime("%Y-%m-%d")

            logger.info(f"Scraping week of: {week_str}")

            for platform in PLATFORMS_TO_TRACK:
                if 'netflix' in platform.lower():
                    continue

                for country in FLIXPATROL_COUNTRY_CODES.values():
                    try:
                        # Weekly aggregate URL
                        url = f"{self.BASE_URL}/top10/{platform}/{country}/{week_str}/full/"

                        soup = self._fetch_page(url)
                        if soup:
                            # Parse using same logic as scrape_top_chart
                            chart_data = self._parse_chart_page(soup, platform, country, week_str)
                            all_data.extend(chart_data)

                        time.sleep(REQUEST_DELAY_SECONDS)

                    except Exception as e:
                        logger.error(f"Error scraping weekly {platform}/{country}/{week_str}: {e}")

            # Move to next week
            current_date = week_start + timedelta(days=7)

        return all_data

    def _parse_chart_page(self, soup: BeautifulSoup, platform: str, country: str,
                          date: str) -> List[Dict]:
        """Generic chart page parser"""
        results = []

        # Try multiple selectors
        selectors = [
            ('table tr', 'table'),
            ('div.chart-item', 'div'),
            ('[data-rank]', 'data'),
        ]

        for selector, parse_type in selectors:
            elements = soup.select(selector)

            if not elements:
                continue

            for i, elem in enumerate(elements):
                try:
                    record = self._extract_record(elem, i + 1, parse_type)
                    if record and record.get("title"):
                        record.update({
                            "_collection_date": date,
                            "_country_code": country,
                            "_platform": platform,
                            "_platform_canonical": PLATFORM_CANONICAL_MAP.get(platform, platform),
                        })
                        results.append(record)
                except Exception:
                    continue

            if results:
                break

        return results

    def _extract_record(self, elem, default_rank: int, parse_type: str) -> Dict:
        """Extract a single record from an element"""
        record = {"flixpatrol_rank": default_rank}

        # Find title link
        title_link = elem.find('a', href=re.compile(r'/title/'))
        if title_link:
            record["title"] = title_link.get_text(strip=True)
            href = title_link.get('href', '')
            id_match = re.search(r'/title/([^/]+)', href)
            if id_match:
                record["flixpatrol_id"] = id_match.group(1)

        # Find points
        points_text = elem.get_text()
        points_match = re.search(r'(\d{1,3}(?:,\d{3})*)\s*(?:pts?|points?)', points_text, re.I)
        if points_match:
            record["flixpatrol_points"] = int(points_match.group(1).replace(',', ''))

        # Find rank if explicitly marked
        rank_match = re.search(r'#?(\d+)', elem.get_text()[:20])
        if rank_match:
            record["flixpatrol_rank"] = int(rank_match.group(1))

        return record

    def _save_checkpoint(self, data: List[Dict], filename: str):
        """Save checkpoint data"""
        if not data:
            return

        df = pd.DataFrame(data)
        checkpoint_path = f"{OUTPUT_DIR}/{filename}.parquet"
        df.to_parquet(checkpoint_path, index=False)
        logger.info(f"Checkpoint: {checkpoint_path} ({len(df)} rows)")

    def enrich_with_details(self, data: List[Dict]) -> List[Dict]:
        """Enrich chart data with title details"""
        # Get unique title IDs
        title_ids = set(r.get("flixpatrol_id") for r in data if r.get("flixpatrol_id"))
        logger.info(f"Enriching {len(title_ids)} unique titles...")

        title_details = {}

        for i, title_id in enumerate(title_ids):
            logger.info(f"Fetching details {i+1}/{len(title_ids)}: {title_id}")

            details = self.scrape_title_details(title_id)
            if details:
                title_details[title_id] = details

            time.sleep(REQUEST_DELAY_SECONDS)

        # Merge details into data
        enriched = []
        for record in data:
            title_id = record.get("flixpatrol_id")
            if title_id and title_id in title_details:
                record.update(title_details[title_id])
            enriched.append(record)

        return enriched

    def save_results(self, data: List[Dict], filename: str):
        """Save final results"""
        if not data:
            logger.warning("No data to save")
            return

        df = pd.DataFrame(data)

        # Save as parquet
        parquet_path = f"{OUTPUT_DIR}/{filename}.parquet"
        df.to_parquet(parquet_path, index=False)

        # Save as CSV for inspection
        csv_path = f"{OUTPUT_DIR}/{filename}.csv"
        df.to_csv(csv_path, index=False)

        logger.info(f"Saved {len(df)} records to {parquet_path}")
        logger.info(f"Columns: {list(df.columns)}")


def main():
    """Main execution for web scraper"""
    print("=" * 60)
    print("FLIXPATROL WEB SCRAPER")
    print("Excludes Netflix - All other platforms")
    print("=" * 60)

    scraper = FlixPatrolWebScraper()

    print(f"\nDate range: {START_DATE} to {END_DATE}")
    print(f"Countries: {len(COUNTRIES)}")
    print(f"Platforms: {len(PLATFORMS_TO_TRACK)}")

    # Option 1: Daily scraping (comprehensive but slow)
    # all_data = scraper.scrape_historical_charts(START_DATE, END_DATE)

    # Option 2: Weekly aggregates (faster)
    print("\nStarting weekly aggregate scraping...")
    all_data = scraper.scrape_weekly_aggregates(START_DATE, END_DATE)

    if all_data:
        print(f"\nCollected {len(all_data)} chart entries")

        # Enrich with title details
        print("\nEnriching with title details...")
        enriched_data = scraper.enrich_with_details(all_data)

        # Save results
        scraper.save_results(enriched_data, "flixpatrol_web_scraped_data")

    print("\nScraping complete!")


if __name__ == "__main__":
    main()
