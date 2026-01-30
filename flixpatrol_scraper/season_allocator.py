#!/usr/bin/env python3
"""
FlixPatrol Season Views Allocator
Fetches season data from FlixPatrol API and allocates aggregated TV show views per season.
Uses Parallel Engine for optimized async I/O.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import json
import sys
import base64
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging

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
INPUT_CSV = Path(r"C:\Users\RoyT6\Downloads\Views TRaining Data\FlixPatrol All Views Jan 25.csv")
OUTPUT_DIR = Path(r"C:\Users\RoyT6\Downloads\Views TRaining Data")

# Get optimized config for scraping
SCRAPING_CONFIG = get_optimal_config(TaskType.SCRAPING)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AllocationStats:
    total_shows: int = 0
    shows_with_seasons: int = 0
    shows_without_api_data: int = 0
    total_seasons_created: int = 0
    api_errors: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    season_cache: Dict[str, int] = field(default_factory=dict)


class SeasonAllocator:
    """
    Allocates aggregated TV show views to individual seasons.
    Fetches season counts from FlixPatrol API using async I/O.
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

        self.stats = AllocationStats()

        # View columns to allocate (country-specific)
        self.view_columns = [
            'total_views', 'views_us', 'views_cn', 'views_in', 'views_gb',
            'views_br', 'views_de', 'views_jp', 'views_fr', 'views_ca',
            'views_mx', 'views_au', 'views_es', 'views_it', 'views_kr',
            'views_nl', 'views_se', 'views_sg', 'views_hk', 'views_ie',
            'views_ru', 'views_tr', 'views_row'
        ]

    def load_data(self) -> pd.DataFrame:
        """Load the FlixPatrol views CSV"""
        logger.info(f"Loading data from {INPUT_CSV}")
        df = pd.read_csv(INPUT_CSV)
        logger.info(f"Loaded {len(df):,} rows")
        return df

    def identify_shows_needing_allocation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify TV shows without season allocation"""
        # Filter for TV shows
        tv_shows = df[df['title_type'] == 'tv_show'].copy()

        # Find those without season_number
        needs_allocation = tv_shows[
            tv_shows['season_number'].isna() |
            (tv_shows['season_number'] == '') |
            (tv_shows['season_number'] == 0)
        ].copy()

        self.stats.total_shows = len(needs_allocation)
        logger.info(f"Found {self.stats.total_shows:,} TV shows needing season allocation")

        return needs_allocation

    async def fetch_title_seasons(
        self,
        session: aiohttp.ClientSession,
        flixpatrol_id: str,
        semaphore: asyncio.Semaphore,
    ) -> Optional[int]:
        """Fetch season count for a title from FlixPatrol API"""
        if not flixpatrol_id or pd.isna(flixpatrol_id):
            return None

        # Check cache first
        if str(flixpatrol_id) in self.stats.season_cache:
            return self.stats.season_cache[str(flixpatrol_id)]

        async with semaphore:
            try:
                url = f"{self.base_url}/titles/{flixpatrol_id}"
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        title_data = data.get("data", {})
                        seasons = title_data.get("seasons", 1)
                        if seasons is None or seasons < 1:
                            seasons = 1
                        self.stats.season_cache[str(flixpatrol_id)] = seasons
                        return seasons
                    elif response.status == 429:
                        # Rate limited - wait and retry
                        logger.warning("Rate limited, waiting 30s...")
                        await asyncio.sleep(30)
                        return await self.fetch_title_seasons(session, flixpatrol_id, semaphore)
                    elif response.status == 404:
                        # Title not found - default to 1 season
                        self.stats.season_cache[str(flixpatrol_id)] = 1
                        return 1
                    else:
                        logger.debug(f"Error {response.status} for {flixpatrol_id}")
                        self.stats.api_errors += 1
                        return None
            except Exception as e:
                logger.debug(f"Exception fetching {flixpatrol_id}: {e}")
                self.stats.api_errors += 1
                return None

    async def fetch_all_seasons(
        self,
        shows_df: pd.DataFrame
    ) -> Dict[str, int]:
        """Fetch season counts for all TV shows"""
        logger.info("Fetching season data from FlixPatrol API...")
        logger.info(f"Using {SCRAPING_CONFIG.async_semaphore} concurrent connections")

        # Create optimized connector
        connector = aiohttp.TCPConnector(
            limit=SCRAPING_CONFIG.connection_pool_size,
            limit_per_host=50,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
        )

        timeout = aiohttp.ClientTimeout(
            total=HTTP_TIMEOUT,
            connect=10,
        )

        semaphore = asyncio.Semaphore(SCRAPING_CONFIG.async_semaphore)

        # Get unique FlixPatrol IDs
        flixpatrol_ids = shows_df['flixpatrol_id'].dropna().unique().tolist()
        logger.info(f"Fetching data for {len(flixpatrol_ids):,} unique titles")

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = []
            for fp_id in flixpatrol_ids:
                task = self.fetch_title_seasons(session, fp_id, semaphore)
                tasks.append((fp_id, task))

            # Process in batches for progress reporting
            batch_size = 500
            results = {}

            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i+batch_size]
                batch_results = await asyncio.gather(*[t[1] for t in batch])

                for (fp_id, _), seasons in zip(batch, batch_results):
                    if seasons is not None:
                        results[str(fp_id)] = seasons
                        self.stats.shows_with_seasons += 1
                    else:
                        self.stats.shows_without_api_data += 1

                progress = min(i + batch_size, len(tasks))
                logger.info(f"Progress: {progress:,}/{len(tasks):,} titles processed")

                # Small delay between batches to be respectful
                await asyncio.sleep(0.5)

        return results

    def calculate_season_decay_weights(self, num_seasons: int) -> List[float]:
        """
        Calculate view distribution weights per season using decay model.
        Later seasons typically have fewer views due to audience drop-off.
        Uses exponential decay: weight_i = 0.85^(i-1) normalized.
        """
        if num_seasons <= 1:
            return [1.0]

        decay_rate = 0.85
        raw_weights = [decay_rate ** (i) for i in range(num_seasons)]
        total = sum(raw_weights)
        normalized = [w / total for w in raw_weights]

        return normalized

    def allocate_views_to_seasons(
        self,
        row: pd.Series,
        num_seasons: int,
        view_columns: List[str]
    ) -> List[Dict[str, Any]]:
        """Create season-level rows with allocated views"""
        season_rows = []
        weights = self.calculate_season_decay_weights(num_seasons)

        for season_num in range(1, num_seasons + 1):
            season_weight = weights[season_num - 1]

            # Create new row for this season
            season_row = row.to_dict()

            # Update season-specific fields
            season_row['season_number'] = season_num
            season_row['max_seasons'] = num_seasons

            # Generate proper fc_uid with season suffix
            imdb_id = row.get('imdb_id', '')
            if imdb_id and not pd.isna(imdb_id):
                season_row['fc_uid'] = f"{imdb_id}_s{season_num}"
            else:
                # Fallback to title-based ID
                title_slug = str(row.get('title', 'unknown')).lower().replace(' ', '_')[:30]
                season_row['fc_uid'] = f"{title_slug}_s{season_num}"

            # Allocate views proportionally
            for col in view_columns:
                if col in row and not pd.isna(row[col]):
                    try:
                        # Handle string values with commas
                        val = row[col]
                        if isinstance(val, str):
                            val = val.replace(',', '').replace(' ', '')
                        original_views = float(val)
                        season_row[col] = int(original_views * season_weight)
                    except (ValueError, TypeError):
                        pass  # Keep original value if conversion fails

            season_rows.append(season_row)
            self.stats.total_seasons_created += 1

        return season_rows

    async def process_allocation(self) -> pd.DataFrame:
        """Main processing pipeline"""
        self.stats.start_time = datetime.now()

        print("=" * 70)
        print("FLIXPATROL SEASON VIEWS ALLOCATOR")
        print(f"Optimized for: Ryzen 9 3950X + 128GB RAM")
        print(f"Async Semaphore: {SCRAPING_CONFIG.async_semaphore}")
        print(f"Connection Pool: {SCRAPING_CONFIG.connection_pool_size}")
        print("=" * 70)

        # Load data
        df = self.load_data()

        # Separate movies and already-allocated shows (keep as-is)
        movies = df[df['title_type'] == 'movie'].copy()
        already_allocated = df[
            (df['title_type'] == 'tv_show') &
            (df['season_number'].notna()) &
            (df['season_number'] != '') &
            (df['season_number'] != 0)
        ].copy()

        logger.info(f"Movies (unchanged): {len(movies):,}")
        logger.info(f"Already allocated TV shows: {len(already_allocated):,}")

        # Get shows needing allocation
        needs_allocation = self.identify_shows_needing_allocation(df)

        if len(needs_allocation) == 0:
            logger.info("No TV shows need allocation!")
            return df

        # Fetch season data from API
        season_data = await self.fetch_all_seasons(needs_allocation)
        logger.info(f"Retrieved season data for {len(season_data):,} titles")

        # Allocate views to seasons
        logger.info("Allocating views to seasons...")

        # Get view columns that exist in the dataframe
        existing_view_cols = [c for c in self.view_columns if c in df.columns]

        new_rows = []
        for idx, row in needs_allocation.iterrows():
            fp_id = str(row.get('flixpatrol_id', ''))

            # Get season count (default to 1 if not found)
            num_seasons = season_data.get(fp_id, 1)

            # If we have max_seasons column with valid data, use that
            if 'max_seasons' in row and not pd.isna(row['max_seasons']) and row['max_seasons'] > 0:
                num_seasons = int(row['max_seasons'])

            # Allocate views to seasons
            season_rows = self.allocate_views_to_seasons(row, num_seasons, existing_view_cols)
            new_rows.extend(season_rows)

            if len(new_rows) % 10000 == 0:
                logger.info(f"Allocated {len(new_rows):,} season rows...")

        # Create new dataframe with allocated seasons
        allocated_df = pd.DataFrame(new_rows)

        # Combine all data
        final_df = pd.concat([movies, already_allocated, allocated_df], ignore_index=True)

        self.stats.end_time = datetime.now()

        return final_df

    def save_results(self, df: pd.DataFrame):
        """Save the allocated data"""
        # Save CSV
        csv_path = self.output_dir / "FlixPatrol_Views_Season_Allocated.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved: {csv_path}")

        # Save Parquet
        parquet_path = self.output_dir / "FlixPatrol_Views_Season_Allocated.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved: {parquet_path}")

        # Calculate stats
        elapsed = (self.stats.end_time - self.stats.start_time).total_seconds()

        # Verification counts
        tv_shows_final = df[df['title_type'] == 'tv_show']
        tv_with_seasons = tv_shows_final[
            tv_shows_final['season_number'].notna() &
            (tv_shows_final['season_number'] != '') &
            (tv_shows_final['season_number'] != 0)
        ]
        tv_without_seasons = tv_shows_final[
            tv_shows_final['season_number'].isna() |
            (tv_shows_final['season_number'] == '') |
            (tv_shows_final['season_number'] == 0)
        ]

        summary = {
            "input_file": str(INPUT_CSV),
            "output_file": str(csv_path),
            "collection_date": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "original_rows": self.stats.total_shows,
            "final_rows": int(len(df)),
            "total_seasons_created": self.stats.total_seasons_created,
            "tv_shows_with_seasons": int(len(tv_with_seasons)),
            "tv_shows_without_seasons": int(len(tv_without_seasons)),
            "movies_unchanged": int(len(df[df['title_type'] == 'movie'])),
            "api_titles_found": self.stats.shows_with_seasons,
            "api_errors": self.stats.api_errors,
            "parallel_engine_config": {
                "async_semaphore": SCRAPING_CONFIG.async_semaphore,
                "connection_pool": SCRAPING_CONFIG.connection_pool_size,
            }
        }

        summary_path = self.output_dir / "season_allocation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 70)
        print("SEASON ALLOCATION COMPLETE")
        print("=" * 70)
        print(f"Original TV Shows: {self.stats.total_shows:,}")
        print(f"Seasons Created: {self.stats.total_seasons_created:,}")
        print(f"Final Dataset Rows: {len(df):,}")
        print(f"TV Shows WITH season: {len(tv_with_seasons):,}")
        print(f"TV Shows WITHOUT season: {len(tv_without_seasons):,}")
        print(f"Elapsed Time: {elapsed:.1f} seconds")
        print(f"API Errors: {self.stats.api_errors}")
        print("=" * 70)

        # Verification
        if len(tv_without_seasons) == 0:
            print("\n*** SUCCESS: All TV shows now have per-season views! ***")
        else:
            print(f"\n*** WARNING: {len(tv_without_seasons)} TV shows still without seasons ***")


async def main():
    allocator = SeasonAllocator()
    result_df = await allocator.process_allocation()
    allocator.save_results(result_df)


if __name__ == "__main__":
    # Windows asyncio fix
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
