#!/usr/bin/env python3
"""
FlixPatrol VIEWS Web Scraper
Uses Playwright to render JavaScript and extract actual rankings data
Bypasses API limitations by scraping the website directly
"""

import asyncio
import pandas as pd
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re

from playwright.async_api import async_playwright, Page, Browser

# Add Parallel Engine to path
sys.path.insert(0, str(Path(r"C:\Users\RoyT6\Downloads\Parallel Engine")))

from system_capability_engine.quick_config import IO_WORKERS

# Configuration
OUTPUT_DIR = Path(r"C:\Users\RoyT6\Downloads\Views TRaining Data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Target platforms (FlixPatrol URL slugs)
TARGET_PLATFORMS = [
    "amazon-prime",
    "disney-plus",
    "max",  # HBO Max
    "paramount-plus",
    "hulu",
    "peacock",
    "apple-tv-plus",
    "starz",
    "amc-plus",
    "discovery-plus",
    "crunchyroll",
    "mubi",
    "britbox",
    "crave",
    "stan",
    "binge",
    "shudder",
    "curiositystream",
]

# Target countries (FlixPatrol URL slugs)
TARGET_COUNTRIES = [
    "united-states",
    "united-kingdom",
    "germany",
    "france",
    "canada",
    "australia",
    "brazil",
    "mexico",
    "spain",
    "italy",
    "japan",
    "india",
    "netherlands",
    "sweden",
    "world",  # Global aggregate
]

# Content types
CONTENT_TYPES = ["", "movies", "shows"]  # "" = overall


@dataclass
class ScrapingStats:
    total_records: int = 0
    pages_scraped: int = 0
    errors: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class FlixPatrolWebScraper:
    """
    Scrapes FlixPatrol website using Playwright for JavaScript rendering
    Extracts actual views/rankings data
    """

    def __init__(self):
        self.output_dir = OUTPUT_DIR
        self.stats = ScrapingStats()
        self.all_data: List[Dict[str, Any]] = []
        self.browser: Optional[Browser] = None

    async def init_browser(self):
        """Initialize Playwright browser"""
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(headless=True)
        print("Browser initialized")

    async def close_browser(self):
        """Close browser"""
        if self.browser:
            await self.browser.close()

    async def scrape_top10_page(
        self,
        page: Page,
        platform: str,
        country: str,
        content_type: str = "",
    ) -> List[Dict[str, Any]]:
        """Scrape a single TOP 10 page"""
        records = []

        # Build URL
        if content_type:
            url = f"https://flixpatrol.com/top10/{platform}/{country}/{content_type}/"
        else:
            url = f"https://flixpatrol.com/top10/{platform}/{country}/"

        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            await asyncio.sleep(1)  # Wait for dynamic content

            # Extract rankings from page
            # Look for ranking items
            rows = await page.query_selector_all("div.tabular-nums, tr[data-id], .ranking-item, [class*='rank']")

            if not rows:
                # Try alternative selectors
                rows = await page.query_selector_all("table tbody tr, .chart-item, .top10-item")

            for row in rows:
                try:
                    # Extract rank
                    rank_elem = await row.query_selector(".rank, [class*='rank'], td:first-child")
                    rank_text = await rank_elem.inner_text() if rank_elem else ""
                    rank = int(re.search(r"\d+", rank_text).group()) if re.search(r"\d+", rank_text) else None

                    # Extract title
                    title_elem = await row.query_selector("a[href*='/title/'], .title, td:nth-child(2) a")
                    title = await title_elem.inner_text() if title_elem else ""
                    title_link = await title_elem.get_attribute("href") if title_elem else ""

                    # Extract flixpatrol ID from link
                    fp_id = None
                    if title_link:
                        id_match = re.search(r"/title/([^/]+)/", title_link)
                        fp_id = id_match.group(1) if id_match else None

                    # Extract points if available
                    points_elem = await row.query_selector(".points, [class*='point'], td:nth-child(3)")
                    points_text = await points_elem.inner_text() if points_elem else ""
                    points = int(re.search(r"[\d,]+", points_text.replace(",", "")).group()) if re.search(r"\d+", points_text) else None

                    if title and rank:
                        records.append({
                            "flixpatrol_id": fp_id,
                            "title": title.strip(),
                            "flixpatrol_rank": rank,
                            "flixpatrol_points": points,
                            "platform": platform,
                            "country": country,
                            "content_type": content_type or "overall",
                            "scrape_date": datetime.now().isoformat(),
                        })
                except Exception as e:
                    continue

            # If no records found with selectors, try to extract from page text
            if not records:
                content = await page.content()

                # Look for structured data
                title_pattern = r'<a[^>]*href="/title/([^"]+)/"[^>]*>([^<]+)</a>'
                matches = re.findall(title_pattern, content)

                for idx, (fp_id, title) in enumerate(matches[:20], 1):  # Top 20
                    records.append({
                        "flixpatrol_id": fp_id,
                        "title": title.strip(),
                        "flixpatrol_rank": idx,
                        "flixpatrol_points": None,
                        "platform": platform,
                        "country": country,
                        "content_type": content_type or "overall",
                        "scrape_date": datetime.now().isoformat(),
                    })

        except Exception as e:
            print(f"Error scraping {url}: {e}")
            self.stats.errors += 1

        return records

    async def scrape_most_watched_page(
        self,
        page: Page,
        platform: str,
        country: str,
    ) -> List[Dict[str, Any]]:
        """Scrape a most-watched page for historical data"""
        records = []
        url = f"https://flixpatrol.com/streaming-services/{platform}/most-watched/{country}/"

        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            await asyncio.sleep(1)

            content = await page.content()

            # Extract titles and points from the page
            # Pattern for title entries with points
            pattern = r'href="/title/([^"]+)/"[^>]*>([^<]+)</a>.*?(\d[\d,]*)\s*(?:points?|pts)'
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)

            for idx, (fp_id, title, points_str) in enumerate(matches[:100], 1):
                points = int(points_str.replace(",", "")) if points_str else None
                records.append({
                    "flixpatrol_id": fp_id,
                    "title": title.strip(),
                    "flixpatrol_rank": idx,
                    "flixpatrol_points": points,
                    "platform": platform,
                    "country": country,
                    "content_type": "most_watched",
                    "scrape_date": datetime.now().isoformat(),
                })

            # If no matches, try simpler title extraction
            if not records:
                title_pattern = r'<a[^>]*href="/title/([^"]+)/"[^>]*>([^<]+)</a>'
                title_matches = re.findall(title_pattern, content)

                for idx, (fp_id, title) in enumerate(title_matches[:100], 1):
                    records.append({
                        "flixpatrol_id": fp_id,
                        "title": title.strip(),
                        "flixpatrol_rank": idx,
                        "flixpatrol_points": None,
                        "platform": platform,
                        "country": country,
                        "content_type": "most_watched",
                        "scrape_date": datetime.now().isoformat(),
                    })

        except Exception as e:
            print(f"Error scraping {url}: {e}")
            self.stats.errors += 1

        return records

    async def scrape_all_data(self):
        """Main scraping method"""
        print("=" * 70)
        print("FLIXPATROL WEB VIEWS SCRAPER")
        print("Scraping actual rankings data from website")
        print(f"Platforms: {len(TARGET_PLATFORMS)}")
        print(f"Countries: {len(TARGET_COUNTRIES)}")
        print("=" * 70)

        self.stats.start_time = datetime.now()

        await self.init_browser()
        page = await self.browser.new_page()

        # Set user agent
        await page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

        total_combos = len(TARGET_PLATFORMS) * len(TARGET_COUNTRIES) * len(CONTENT_TYPES)
        current = 0

        for platform in TARGET_PLATFORMS:
            print(f"\n[Platform] {platform}")

            for country in TARGET_COUNTRIES:
                for content_type in CONTENT_TYPES:
                    current += 1
                    print(f"  [{current}/{total_combos}] {country}/{content_type or 'overall'}...", end=" ")

                    # Scrape TOP 10 page
                    records = await self.scrape_top10_page(page, platform, country, content_type)
                    self.all_data.extend(records)
                    self.stats.pages_scraped += 1

                    print(f"{len(records)} records")

                    # Rate limiting
                    await asyncio.sleep(0.5)

                # Also scrape most-watched for this platform/country
                mw_records = await self.scrape_most_watched_page(page, platform, country)
                self.all_data.extend(mw_records)
                if mw_records:
                    print(f"    Most watched: {len(mw_records)} records")

        await self.close_browser()

        self.stats.end_time = datetime.now()
        self.stats.total_records = len(self.all_data)

        return self.all_data

    def save_data(self):
        """Save scraped data"""
        if not self.all_data:
            print("No data to save!")
            return

        df = pd.DataFrame(self.all_data)
        df["source"] = "flixpatrol_web_scrape"
        df["data_type"] = "views_rankings"

        # Deduplicate
        df = df.drop_duplicates(subset=["flixpatrol_id", "platform", "country", "content_type"])

        # Save Parquet
        parquet_path = self.output_dir / "flixpatrol_web_views.parquet"
        df.to_parquet(parquet_path, index=False)
        print(f"\nSaved: {parquet_path}")

        # Save CSV
        csv_path = self.output_dir / "flixpatrol_web_views.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

        # Stats
        elapsed = (self.stats.end_time - self.stats.start_time).total_seconds()

        summary = {
            "data_type": "VIEWS DATA (web scraped)",
            "total_records": int(len(df)),
            "unique_titles": int(df["flixpatrol_id"].nunique()),
            "platforms": df["platform"].nunique(),
            "countries": df["country"].nunique(),
            "pages_scraped": self.stats.pages_scraped,
            "errors": self.stats.errors,
            "elapsed_seconds": round(elapsed, 2),
            "scrape_date": datetime.now().isoformat(),
        }

        summary_path = self.output_dir / "flixpatrol_web_scrape_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 70)
        print("WEB SCRAPING COMPLETE")
        print("=" * 70)
        print(f"Total Records: {summary['total_records']:,}")
        print(f"Unique Titles: {summary['unique_titles']:,}")
        print(f"Platforms: {summary['platforms']}")
        print(f"Countries: {summary['countries']}")
        print(f"Pages Scraped: {summary['pages_scraped']}")
        print(f"Errors: {summary['errors']}")
        print(f"Time: {elapsed:.1f} seconds")


async def main():
    scraper = FlixPatrolWebScraper()
    await scraper.scrape_all_data()
    scraper.save_data()


if __name__ == "__main__":
    asyncio.run(main())
