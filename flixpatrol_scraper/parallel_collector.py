#!/usr/bin/env python3
"""
FlixPatrol Parallel Data Collector
Optimized for Ryzen 9 3950X + 128GB RAM using Parallel Engine
Uses async I/O with 150 concurrent connections for maximum throughput
"""

import asyncio
import aiohttp
import pandas as pd
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import base64

# Add Parallel Engine to path
sys.path.insert(0, str(Path(r"C:\Users\RoyT6\Downloads\Parallel Engine")))

from system_capability_engine.quick_config import (
    SCRAPING_LIMIT,
    AIOHTTP_CONNECTOR,
    AIOHTTP_TIMEOUT,
    IO_WORKERS,
    HTTP_TIMEOUT,
)
from system_capability_engine.optimizer import get_optimal_config, TaskType

# Configuration
API_KEY = "aku_4bXKmMWPSCaKxwn2tVzTXmcg"
BASE_URL = "https://api.flixpatrol.com/v2"
OUTPUT_DIR = Path(r"C:\Users\RoyT6\Downloads\Views TRaining Data")

# Get optimized config for scraping
SCRAPING_CONFIG = get_optimal_config(TaskType.SCRAPING)

@dataclass
class CollectionStats:
    total_fetched: int = 0
    total_errors: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class FlixPatrolParallelCollector:
    """
    High-performance FlixPatrol data collector using async I/O
    Optimized for Ryzen 9 3950X with 128GB RAM
    """

    def __init__(self):
        self.api_key = API_KEY
        self.base_url = BASE_URL
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Auth header (Basic auth with API key as username, empty password)
        auth_string = base64.b64encode(f"{self.api_key}:".encode()).decode()
        self.headers = {
            "Authorization": f"Basic {auth_string}",
            "Accept": "application/json",
        }

        self.stats = CollectionStats()
        self.all_titles: List[Dict[str, Any]] = []

    async def fetch_page(
        self,
        session: aiohttp.ClientSession,
        url: str,
        semaphore: asyncio.Semaphore,
    ) -> Optional[Dict]:
        """Fetch a single page with rate limiting"""
        async with semaphore:
            try:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:
                        # Rate limited - wait and retry
                        print(f"Rate limited, waiting 30s...")
                        await asyncio.sleep(30)
                        return await self.fetch_page(session, url, semaphore)
                    else:
                        print(f"Error {response.status}: {url}")
                        self.stats.total_errors += 1
                        return None
            except Exception as e:
                print(f"Exception: {e}")
                self.stats.total_errors += 1
                return None

    def parse_title(self, item: Dict) -> Dict[str, Any]:
        """Parse a title from API response"""
        title_info = item.get("data", {})

        record = {
            "flixpatrol_id": title_info.get("id"),
            "title": title_info.get("title"),
            "link": title_info.get("link"),
            "premiere_date": title_info.get("premiere"),
            "premiere_online": title_info.get("premiereOnline"),
            "description": title_info.get("description"),
            "runtime_minutes": title_info.get("length"),
            "budget": title_info.get("budget"),
            "imdb_id": f"tt{str(title_info.get('imdbId', '')).zfill(7)}" if title_info.get("imdbId") else None,
            "imdb_link": title_info.get("imdbLink"),
            "tmdb_id": title_info.get("tmdbId"),
            "tmdb_link": title_info.get("tmdbLink"),
            "content_type": "movie" if title_info.get("type") == 1 else "tv_show",
            "updated_at": title_info.get("updatedAt"),
        }

        # Extract nested IDs
        if title_info.get("country"):
            record["country_id"] = title_info["country"].get("data", {}).get("id")
        if title_info.get("company"):
            record["company_id"] = title_info["company"].get("data", {}).get("id")
        if title_info.get("genre"):
            record["genre_id"] = title_info["genre"].get("data", {}).get("id")
        if title_info.get("keyword"):
            record["keyword_id"] = title_info["keyword"].get("data", {}).get("id")

        return record

    async def collect_all_titles(self) -> List[Dict[str, Any]]:
        """
        Collect all titles using optimized async I/O
        Uses Parallel Engine settings for max throughput
        """
        print("=" * 70)
        print("FLIXPATROL PARALLEL COLLECTOR")
        print(f"Optimized for: Ryzen 9 3950X + 128GB RAM")
        print(f"Async Semaphore: {SCRAPING_CONFIG.async_semaphore}")
        print(f"Connection Pool: {SCRAPING_CONFIG.connection_pool_size}")
        print("=" * 70)

        self.stats.start_time = datetime.now()

        # Create optimized connector
        connector = aiohttp.TCPConnector(
            limit=SCRAPING_CONFIG.connection_pool_size,
            limit_per_host=50,  # FlixPatrol specific
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
        )

        timeout = aiohttp.ClientTimeout(
            total=HTTP_TIMEOUT,
            connect=10,
        )

        # Semaphore for rate limiting
        semaphore = asyncio.Semaphore(SCRAPING_CONFIG.async_semaphore)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # First, get initial page to find pagination
            next_url = f"{self.base_url}/titles"
            page = 0

            while next_url:
                page += 1
                data = await self.fetch_page(session, next_url, semaphore)

                if not data:
                    break

                titles_data = data.get("data", [])
                if not titles_data:
                    break

                # Parse titles
                for item in titles_data:
                    record = self.parse_title(item)
                    self.all_titles.append(record)

                self.stats.total_fetched = len(self.all_titles)

                if page % 50 == 0:
                    elapsed = (datetime.now() - self.stats.start_time).total_seconds()
                    rate = self.stats.total_fetched / elapsed if elapsed > 0 else 0
                    print(f"Page {page}: {self.stats.total_fetched} titles ({rate:.0f}/sec)")

                # Get next page
                links = data.get("links", {})
                next_url = links.get("next")

                # Small delay to be respectful
                await asyncio.sleep(0.1)

        self.stats.end_time = datetime.now()
        return self.all_titles

    def save_data(self):
        """Save collected data"""
        if not self.all_titles:
            print("No data to save!")
            return

        df = pd.DataFrame(self.all_titles)
        df["source_api"] = "flixpatrol"
        df["collection_timestamp"] = datetime.now().isoformat()

        # Save Parquet
        parquet_path = self.output_dir / "flixpatrol_titles_parallel.parquet"
        df.to_parquet(parquet_path, index=False)
        print(f"\nSaved: {parquet_path}")

        # Save CSV
        csv_path = self.output_dir / "flixpatrol_titles_parallel.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

        # Calculate stats
        elapsed = (self.stats.end_time - self.stats.start_time).total_seconds()
        rate = self.stats.total_fetched / elapsed if elapsed > 0 else 0

        summary = {
            "total_records": int(len(df)),
            "collection_date": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "records_per_second": round(rate, 2),
            "movies": int(len(df[df["content_type"] == "movie"])),
            "tv_shows": int(len(df[df["content_type"] == "tv_show"])),
            "with_imdb_id": int(df["imdb_id"].notna().sum()),
            "with_tmdb_id": int(df["tmdb_id"].notna().sum()),
            "errors": self.stats.total_errors,
            "parallel_engine_config": {
                "async_semaphore": SCRAPING_CONFIG.async_semaphore,
                "connection_pool": SCRAPING_CONFIG.connection_pool_size,
            }
        }

        summary_path = self.output_dir / "flixpatrol_parallel_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 70)
        print("COLLECTION COMPLETE")
        print("=" * 70)
        print(f"Total Records: {summary['total_records']:,}")
        print(f"Elapsed Time: {elapsed:.1f} seconds")
        print(f"Speed: {rate:.1f} records/second")
        print(f"Movies: {summary['movies']:,}")
        print(f"TV Shows: {summary['tv_shows']:,}")
        print(f"Errors: {summary['errors']}")


async def main():
    collector = FlixPatrolParallelCollector()
    await collector.collect_all_titles()
    collector.save_data()


if __name__ == "__main__":
    # Windows asyncio fix
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
