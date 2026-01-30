#!/usr/bin/env python3
"""
FlixPatrol VIEWS Data Collector
Collects actual rankings/points data (NOT metadata)
Uses /v2/top10s and /v2/rankings endpoints

Required data columns:
- flixpatrol_id (title ID)
- flixpatrol_points (value field from API)
- flixpatrol_rank (ranking field from API)
- platform, country, date
"""

import asyncio
import aiohttp
import pandas as pd
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import base64

# Add Parallel Engine to path
sys.path.insert(0, str(Path(r"C:\Users\RoyT6\Downloads\Parallel Engine")))

from system_capability_engine.quick_config import (
    SCRAPING_LIMIT,
    IO_WORKERS,
    HTTP_TIMEOUT,
)
from system_capability_engine.optimizer import get_optimal_config, TaskType

# Configuration
API_KEY = "aku_4bXKmMWPSCaKxwn2tVzTXmcg"
BASE_URL = "https://api.flixpatrol.com/v2"
OUTPUT_DIR = Path(r"C:\Users\RoyT6\Downloads\Views TRaining Data")

# Get optimized config
SCRAPING_CONFIG = get_optimal_config(TaskType.SCRAPING)

# Date range: Jan 2023 to now
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime.now()

# Platforms to EXCLUDE
EXCLUDED_PLATFORMS = {"netflix"}

# 22 Target countries (ISO codes)
TARGET_COUNTRIES = [
    "US", "CN", "IN", "GB", "BR", "DE", "JP", "FR", "CA", "MX",
    "AU", "ES", "IT", "KR", "NL", "SE", "SG", "HK", "IE", "RU", "TR"
]

# Target platforms (from BFD Schema - excluding Netflix)
TARGET_PLATFORMS = {
    "amazon prime video", "disney+", "max", "hbo", "paramount+", "hulu",
    "peacock", "apple tv+", "curiositystream", "espn+", "dazn", "starz",
    "crunchyroll", "mubi", "amc+", "discovery+", "rtl+", "viaplay",
    "britbox", "crave", "movistar+", "starzplay", "now", "ocs", "joyn",
    "canal+", "shudder", "stan", "timvision", "sling tv", "fubotv",
    "skyshowtime", "mediaset infinity", "videoland", "binge", "kayo sports",
    "polsat box go", "itvx", "player", "tv 2 play", "tv4 play", "wow",
    "cda premium", "megogo", "atresplayer", "voyo", "go3", "kyivstar tv",
    "streamz", "prima+", "foxtel now"
}


@dataclass
class CollectionStats:
    total_records: int = 0
    total_errors: int = 0
    platforms_processed: int = 0
    countries_processed: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class FlixPatrolViewsCollector:
    """
    Collects actual VIEWS data (points and ranks) from FlixPatrol API
    NOT metadata - actual daily rankings with scores
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
        self.companies: Dict[str, str] = {}  # id -> name
        self.countries: Dict[str, str] = {}  # id -> code
        self.all_views_data: List[Dict[str, Any]] = []

    async def fetch_json(
        self,
        session: aiohttp.ClientSession,
        url: str,
        semaphore: asyncio.Semaphore,
    ) -> Optional[Dict]:
        """Fetch JSON from API with rate limiting"""
        async with semaphore:
            try:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:
                        print(f"Rate limited, waiting 60s...")
                        await asyncio.sleep(60)
                        return await self.fetch_json(session, url, semaphore)
                    else:
                        print(f"Error {response.status}: {url[:100]}")
                        self.stats.total_errors += 1
                        return None
            except Exception as e:
                print(f"Exception: {e}")
                self.stats.total_errors += 1
                return None

    async def get_all_companies(self, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore) -> Dict[str, Dict]:
        """Get all streaming platforms/companies - filtered to streaming services only"""
        print("\nFetching streaming platforms (SVOD/TVOD/AVOD only)...")
        companies = {}
        next_url = f"{self.base_url}/companies"

        # Type values for streaming services:
        # 1 = SVOD (Subscription), 2 = TVOD (Transactional), 3 = AVOD (Ad-supported)
        STREAMING_TYPES = {1, 2, 3}

        while next_url:
            data = await self.fetch_json(session, next_url, semaphore)
            if not data:
                break

            for item in data.get("data", []):
                company_data = item.get("data", {})
                if not company_data:
                    continue

                company_type = company_data.get("type")
                # Only include streaming services (SVOD, TVOD, AVOD)
                if company_type not in STREAMING_TYPES:
                    continue

                company_id = company_data.get("id")
                company_name = company_data.get("name", "")
                company_name_lower = company_name.lower() if company_name else ""

                # Skip Netflix
                if "netflix" in company_name_lower:
                    continue

                companies[company_id] = {
                    "name": company_name,
                    "type": company_type,
                }

            next_url = data.get("links", {}).get("next")
            await asyncio.sleep(0.1)

        # Safe print for console encoding
        try:
            print(f"\nFound {len(companies)} streaming platforms (excluding Netflix)")
            for cid, cdata in list(companies.items())[:10]:
                print(f"  - {cdata['name']}")
            if len(companies) > 10:
                print(f"  ... and {len(companies) - 10} more")
        except UnicodeEncodeError:
            print(f"\nFound {len(companies)} streaming platforms")

        return companies

    async def get_all_countries(self, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore) -> Dict[str, str]:
        """Get all countries with their IDs"""
        print("\nFetching all countries...")
        countries = {}
        next_url = f"{self.base_url}/countries"

        while next_url:
            data = await self.fetch_json(session, next_url, semaphore)
            if not data:
                break

            for item in data.get("data", []):
                country_data = item.get("data", {})
                if not country_data:
                    continue
                country_id = country_data.get("id")
                code_raw = country_data.get("code")
                if not code_raw:
                    continue
                country_code = code_raw.upper()

                # Only include target countries
                if country_code in TARGET_COUNTRIES:
                    countries[country_id] = country_code

            next_url = data.get("links", {}).get("next")
            await asyncio.sleep(0.1)

        print(f"Found {len(countries)} target countries")
        return countries

    async def collect_top10s_for_period(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        company_id: str,
        country_id: str,
        start_date: str,
        end_date: str,
    ) -> List[Dict]:
        """Collect TOP 10 data for a specific company/country/date range"""
        records = []

        # Use the top10s endpoint with date range
        url = (
            f"{self.base_url}/top10s?"
            f"company[eq]={company_id}&"
            f"country[eq]={country_id}&"
            f"date[type]=1&"  # Daily
            f"date[from][gte]={start_date}&"
            f"date[to][lte]={end_date}"
        )

        next_url = url
        while next_url:
            data = await self.fetch_json(session, next_url, semaphore)
            if not data:
                break

            for item in data.get("data", []):
                top10_data = item.get("data", {})

                # Extract title info
                movie_info = top10_data.get("movie", {}).get("data", {})

                record = {
                    "flixpatrol_id": movie_info.get("id"),
                    "title": movie_info.get("title"),
                    "flixpatrol_rank": top10_data.get("ranking"),
                    "flixpatrol_points": top10_data.get("value"),
                    "rank_previous": top10_data.get("rankingLast"),
                    "points_previous": top10_data.get("valueLast"),
                    "days_in_top10": top10_data.get("daysTotal"),
                    "content_type": "movie" if top10_data.get("type") == 1 else "tv_show",
                    "platform_id": company_id,
                    "platform_name": self.companies.get(company_id, {}).get("name"),
                    "country_id": country_id,
                    "country_code": self.countries.get(country_id),
                    "date_from": top10_data.get("date", {}).get("from"),
                    "date_to": top10_data.get("date", {}).get("to"),
                    "updated_at": top10_data.get("updatedAt"),
                }
                records.append(record)

            next_url = data.get("links", {}).get("next")
            await asyncio.sleep(0.05)  # Small delay between pages

        return records

    async def collect_rankings_data(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        company_id: str,
        country_id: str,
    ) -> List[Dict]:
        """Collect aggregated rankings data for a company/country"""
        records = []

        # Use rankings endpoint for aggregated data
        url = (
            f"{self.base_url}/rankings?"
            f"company[eq]={company_id}&"
            f"country[eq]={country_id}"
        )

        next_url = url
        page = 0
        while next_url:
            page += 1
            data = await self.fetch_json(session, next_url, semaphore)
            if not data:
                break

            for item in data.get("data", []):
                ranking_data = item.get("data", {})

                # Extract title info
                movie_info = ranking_data.get("movie", {}).get("data", {})

                record = {
                    "flixpatrol_id": movie_info.get("id"),
                    "title": movie_info.get("title"),
                    "flixpatrol_rank": ranking_data.get("ranking"),
                    "flixpatrol_points": ranking_data.get("value"),
                    "points_total": ranking_data.get("valueTotal"),
                    "days_tracked": ranking_data.get("days"),
                    "countries_count": ranking_data.get("countries"),
                    "content_type": "movie" if ranking_data.get("type") == 1 else "tv_show",
                    "audience": "kids" if ranking_data.get("audience") == 2 else "general",
                    "platform_id": company_id,
                    "platform_name": self.companies.get(company_id, {}).get("name"),
                    "country_id": country_id,
                    "country_code": self.countries.get(country_id),
                    "date_type": ranking_data.get("date", {}).get("type"),
                    "date_from": ranking_data.get("date", {}).get("from"),
                    "date_to": ranking_data.get("date", {}).get("to"),
                    "updated_at": ranking_data.get("updatedAt"),
                }
                records.append(record)

            next_url = data.get("links", {}).get("next")

            if page % 10 == 0:
                print(f"    Page {page}, {len(records)} records...")

            await asyncio.sleep(0.05)

        return records

    async def collect_all_views_data(self):
        """Main collection method - collects VIEWS data (points/ranks)"""
        print("=" * 70)
        print("FLIXPATROL VIEWS DATA COLLECTOR")
        print("Collecting rankings and points (NOT metadata)")
        print(f"Optimized for: Ryzen 9 3950X + 128GB RAM")
        print(f"Async Semaphore: {SCRAPING_CONFIG.async_semaphore}")
        print("=" * 70)

        self.stats.start_time = datetime.now()

        # Create optimized connector
        connector = aiohttp.TCPConnector(
            limit=SCRAPING_CONFIG.connection_pool_size,
            limit_per_host=30,  # Conservative for API
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
        )

        timeout = aiohttp.ClientTimeout(
            total=HTTP_TIMEOUT,
            connect=15,
        )

        semaphore = asyncio.Semaphore(min(SCRAPING_CONFIG.async_semaphore, 50))  # Conservative for API

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Step 1: Get all companies (platforms)
            self.companies = await self.get_all_companies(session, semaphore)

            # Step 2: Get all countries
            self.countries = await self.get_all_countries(session, semaphore)

            # Step 3: Collect rankings for each company/country combination
            print(f"\nCollecting views data for {len(self.companies)} platforms x {len(self.countries)} countries")
            print(f"Total combinations: {len(self.companies) * len(self.countries)}")
            print("-" * 70)

            total_combinations = len(self.companies) * len(self.countries)
            processed = 0

            for company_id, company_info in self.companies.items():
                company_name = company_info.get("name", "Unknown")
                print(f"\n[{processed // len(self.countries) + 1}/{len(self.companies)}] Processing: {company_name}")

                for country_id, country_code in self.countries.items():
                    processed += 1

                    # Collect rankings data
                    records = await self.collect_rankings_data(
                        session, semaphore, company_id, country_id
                    )

                    if records:
                        self.all_views_data.extend(records)
                        self.stats.total_records = len(self.all_views_data)
                        print(f"  {country_code}: {len(records)} records (Total: {self.stats.total_records:,})")

                    self.stats.countries_processed += 1

                self.stats.platforms_processed += 1

        self.stats.end_time = datetime.now()
        return self.all_views_data

    def save_data(self):
        """Save collected VIEWS data"""
        if not self.all_views_data:
            print("No views data to save!")
            return

        df = pd.DataFrame(self.all_views_data)
        df["source_api"] = "flixpatrol"
        df["data_type"] = "views_rankings"  # Mark as VIEWS data, not metadata
        df["collection_timestamp"] = datetime.now().isoformat()

        # Save Parquet
        parquet_path = self.output_dir / "flixpatrol_views_data.parquet"
        df.to_parquet(parquet_path, index=False)
        print(f"\nSaved: {parquet_path}")

        # Save CSV
        csv_path = self.output_dir / "flixpatrol_views_data.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

        # Calculate stats
        elapsed = (self.stats.end_time - self.stats.start_time).total_seconds()
        rate = self.stats.total_records / elapsed if elapsed > 0 else 0

        summary = {
            "data_type": "VIEWS DATA (rankings and points)",
            "total_records": int(len(df)),
            "collection_date": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "records_per_second": round(rate, 2),
            "platforms_processed": self.stats.platforms_processed,
            "countries_processed": self.stats.countries_processed,
            "unique_titles": int(df["flixpatrol_id"].nunique()) if "flixpatrol_id" in df.columns else 0,
            "movies": int(len(df[df["content_type"] == "movie"])) if "content_type" in df.columns else 0,
            "tv_shows": int(len(df[df["content_type"] == "tv_show"])) if "content_type" in df.columns else 0,
            "platforms": df["platform_name"].nunique() if "platform_name" in df.columns else 0,
            "countries": df["country_code"].nunique() if "country_code" in df.columns else 0,
            "date_range": {
                "min": str(df["date_from"].min()) if "date_from" in df.columns else None,
                "max": str(df["date_to"].max()) if "date_to" in df.columns else None,
            },
            "errors": self.stats.total_errors,
            "columns": list(df.columns),
            "key_columns": {
                "flixpatrol_id": "Title identifier",
                "flixpatrol_points": "Points/score value",
                "flixpatrol_rank": "Ranking position",
            }
        }

        summary_path = self.output_dir / "flixpatrol_views_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 70)
        print("VIEWS DATA COLLECTION COMPLETE")
        print("=" * 70)
        print(f"Data Type: VIEWS (rankings and points)")
        print(f"Total Records: {summary['total_records']:,}")
        print(f"Unique Titles: {summary['unique_titles']:,}")
        print(f"Platforms: {summary['platforms']}")
        print(f"Countries: {summary['countries']}")
        print(f"Elapsed Time: {elapsed:.1f} seconds")
        print(f"Speed: {rate:.1f} records/second")
        print(f"Errors: {summary['errors']}")
        if summary['date_range']['min']:
            print(f"Date Range: {summary['date_range']['min']} to {summary['date_range']['max']}")


async def main():
    collector = FlixPatrolViewsCollector()
    await collector.collect_all_views_data()
    collector.save_data()


if __name__ == "__main__":
    asyncio.run(main())
