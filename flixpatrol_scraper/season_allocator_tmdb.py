#!/usr/bin/env python3
"""
FlixPatrol Season Views Allocator - TMDB Enhanced
Uses TMDB local data for season counts to allocate aggregated TV show views per season.
Uses Parallel Engine for optimized processing.
"""

import pandas as pd
import numpy as np
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

# Add Parallel Engine to path
sys.path.insert(0, str(Path(r"C:\Users\RoyT6\Downloads\Parallel Engine")))

from system_capability_engine.quick_config import CPU_WORKERS
from system_capability_engine.optimizer import get_optimal_config, TaskType

# Configuration
INPUT_CSV = Path(r"C:\Users\RoyT6\Downloads\Views TRaining Data\FlixPatrol All Views Jan 25.csv")
TMDB_TV_CSV = Path(r"C:\Users\RoyT6\Downloads\Freckles\00_ACTIVE_SOURCES\TMDB\TMDB_TV_22_MAY24.csv")
OUTPUT_DIR = Path(r"C:\Users\RoyT6\Downloads\Views TRaining Data")

# Get optimized config
DATA_CONFIG = get_optimal_config(TaskType.DATA_PIPELINE)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AllocationStats:
    total_shows: int = 0
    shows_with_tmdb_seasons: int = 0
    shows_without_tmdb_data: int = 0
    total_seasons_created: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class SeasonAllocatorTMDB:
    """
    Allocates aggregated TV show views to individual seasons.
    Uses TMDB local data for season counts.
    """

    def __init__(self):
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats = AllocationStats()

        # View columns to allocate (country-specific)
        self.view_columns = [
            'total_views', 'views_us', 'views_cn', 'views_in', 'views_gb',
            'views_br', 'views_de', 'views_jp', 'views_fr', 'views_ca',
            'views_mx', 'views_au', 'views_es', 'views_it', 'views_kr',
            'views_nl', 'views_se', 'views_sg', 'views_hk', 'views_ie',
            'views_ru', 'views_tr', 'views_row'
        ]

        # Load TMDB season data
        self.tmdb_seasons = self._load_tmdb_seasons()

    def _load_tmdb_seasons(self) -> Dict[int, int]:
        """Load TMDB TV show season counts"""
        logger.info(f"Loading TMDB TV data from {TMDB_TV_CSV}")
        tmdb = pd.read_csv(TMDB_TV_CSV, low_memory=False)

        # Create lookup dict: tmdb_id -> number_of_seasons
        seasons_lookup = {}
        for _, row in tmdb.iterrows():
            tmdb_id = row.get('id')
            num_seasons = row.get('number_of_seasons', 1)
            if pd.notna(tmdb_id) and pd.notna(num_seasons) and num_seasons > 0:
                seasons_lookup[int(tmdb_id)] = int(num_seasons)

        logger.info(f"Loaded season data for {len(seasons_lookup):,} TV shows from TMDB")
        return seasons_lookup

    def load_data(self) -> pd.DataFrame:
        """Load the FlixPatrol views CSV"""
        logger.info(f"Loading data from {INPUT_CSV}")
        df = pd.read_csv(INPUT_CSV, low_memory=False)
        logger.info(f"Loaded {len(df):,} rows")
        return df

    def identify_shows_needing_allocation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify TV shows without season allocation"""
        # Filter for TV shows
        tv_shows = df[df['title_type'] == 'tv_show'].copy()

        # Find those without season_number or with invalid season_number
        needs_allocation = tv_shows[
            tv_shows['season_number'].isna() |
            (tv_shows['season_number'] == '') |
            (tv_shows['season_number'] == 0)
        ].copy()

        self.stats.total_shows = len(needs_allocation)
        logger.info(f"Found {self.stats.total_shows:,} TV shows needing season allocation")

        return needs_allocation

    def get_season_count(self, row: pd.Series) -> int:
        """Get season count from TMDB data or existing max_seasons"""
        # First check if max_seasons is already populated
        if 'max_seasons' in row and pd.notna(row['max_seasons']) and row['max_seasons'] > 0:
            return int(row['max_seasons'])

        # Try TMDB lookup
        tmdb_id = row.get('tmdb_id')
        if pd.notna(tmdb_id):
            try:
                tmdb_id_int = int(float(tmdb_id))
                if tmdb_id_int in self.tmdb_seasons:
                    self.stats.shows_with_tmdb_seasons += 1
                    return self.tmdb_seasons[tmdb_id_int]
            except (ValueError, TypeError):
                pass

        # Default to 1 season
        self.stats.shows_without_tmdb_data += 1
        return 1

    def calculate_season_decay_weights(self, num_seasons: int) -> List[float]:
        """
        Calculate view distribution weights per season using decay model.
        Season 1 typically has highest views, later seasons decline.
        Uses exponential decay: weight_i = 0.80^(i-1) normalized.
        """
        if num_seasons <= 1:
            return [1.0]

        decay_rate = 0.80  # 20% decay per season
        raw_weights = [decay_rate ** i for i in range(num_seasons)]
        total = sum(raw_weights)
        normalized = [w / total for w in raw_weights]

        return normalized

    def parse_view_value(self, val) -> Optional[float]:
        """Parse view value handling comma-formatted strings"""
        if pd.isna(val):
            return None
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            # Remove commas and spaces
            val = val.replace(',', '').replace(' ', '').strip()
            if val == '' or val == '0':
                return 0.0
            try:
                return float(val)
            except ValueError:
                return None
        return None

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
            imdb_id = row.get('imdb_id')
            if imdb_id and not pd.isna(imdb_id):
                season_row['fc_uid'] = f"{imdb_id}_s{season_num}"
            else:
                # Fallback to title-based ID
                title = str(row.get('title', 'unknown'))
                title_slug = title.lower().replace(' ', '_').replace(':', '').replace("'", '')[:30]
                season_row['fc_uid'] = f"{title_slug}_s{season_num}"

            # Allocate views proportionally
            for col in view_columns:
                if col in row:
                    original_views = self.parse_view_value(row[col])
                    if original_views is not None and original_views > 0:
                        season_row[col] = int(original_views * season_weight)
                    else:
                        season_row[col] = 0

            season_rows.append(season_row)
            self.stats.total_seasons_created += 1

        return season_rows

    def process_allocation(self) -> pd.DataFrame:
        """Main processing pipeline"""
        self.stats.start_time = datetime.now()

        print("=" * 70)
        print("FLIXPATROL SEASON VIEWS ALLOCATOR - TMDB Enhanced")
        print(f"Optimized for: Ryzen 9 3950X + 128GB RAM")
        print(f"CPU Workers: {DATA_CONFIG.process_workers}")
        print("=" * 70)

        # Load data
        df = self.load_data()

        # Separate movies (keep as-is)
        movies = df[df['title_type'] == 'movie'].copy()
        logger.info(f"Movies (unchanged): {len(movies):,}")

        # Separate already-allocated shows (keep as-is)
        already_allocated = df[
            (df['title_type'] == 'tv_show') &
            (df['season_number'].notna()) &
            (df['season_number'] != '') &
            (df['season_number'] != 0)
        ].copy()
        logger.info(f"Already allocated TV shows: {len(already_allocated):,}")

        # Get shows needing allocation
        needs_allocation = self.identify_shows_needing_allocation(df)

        if len(needs_allocation) == 0:
            logger.info("No TV shows need allocation!")
            return df

        # Get view columns that exist in the dataframe
        existing_view_cols = [c for c in self.view_columns if c in df.columns]
        logger.info(f"View columns to allocate: {len(existing_view_cols)}")

        # Allocate views to seasons
        logger.info("Allocating views to seasons using TMDB data...")

        new_rows = []
        processed = 0

        for idx, row in needs_allocation.iterrows():
            # Get season count from TMDB
            num_seasons = self.get_season_count(row)

            # Allocate views to seasons
            season_rows = self.allocate_views_to_seasons(row, num_seasons, existing_view_cols)
            new_rows.extend(season_rows)

            processed += 1
            if processed % 5000 == 0:
                logger.info(f"Processed {processed:,}/{len(needs_allocation):,} shows, created {len(new_rows):,} season rows")

        # Create new dataframe with allocated seasons
        logger.info(f"Creating dataframe with {len(new_rows):,} allocated rows...")
        allocated_df = pd.DataFrame(new_rows)

        # Combine all data
        final_df = pd.concat([movies, already_allocated, allocated_df], ignore_index=True)

        self.stats.end_time = datetime.now()

        return final_df

    def save_results(self, df: pd.DataFrame):
        """Save the allocated data"""
        # Standardize column types to avoid parquet issues
        for col in self.view_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')

        if 'season_number' in df.columns:
            df['season_number'] = pd.to_numeric(df['season_number'], errors='coerce').fillna(0).astype('int64')
        if 'max_seasons' in df.columns:
            df['max_seasons'] = pd.to_numeric(df['max_seasons'], errors='coerce').fillna(0).astype('int64')

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
            "original_tv_shows": self.stats.total_shows,
            "final_rows": int(len(df)),
            "total_seasons_created": self.stats.total_seasons_created,
            "tv_shows_with_seasons": int(len(tv_with_seasons)),
            "tv_shows_without_seasons": int(len(tv_without_seasons)),
            "movies_unchanged": int(len(df[df['title_type'] == 'movie'])),
            "tmdb_matches": self.stats.shows_with_tmdb_seasons,
            "no_tmdb_data_defaulted_1": self.stats.shows_without_tmdb_data,
            "data_source": "TMDB + local max_seasons"
        }

        summary_path = self.output_dir / "season_allocation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 70)
        print("SEASON ALLOCATION COMPLETE")
        print("=" * 70)
        print(f"Original TV Shows: {self.stats.total_shows:,}")
        print(f"TMDB Season Matches: {self.stats.shows_with_tmdb_seasons:,}")
        print(f"Defaulted to 1 Season: {self.stats.shows_without_tmdb_data:,}")
        print(f"Seasons Created: {self.stats.total_seasons_created:,}")
        print(f"Final Dataset Rows: {len(df):,}")
        print(f"TV Shows WITH season: {len(tv_with_seasons):,}")
        print(f"TV Shows WITHOUT season: {len(tv_without_seasons):,}")
        print(f"Elapsed Time: {elapsed:.1f} seconds")
        print("=" * 70)

        # Verification
        if len(tv_without_seasons) == 0:
            print("\n*** SUCCESS: All TV shows now have per-season views! ***")
        else:
            print(f"\n*** WARNING: {len(tv_without_seasons)} TV shows still without seasons ***")

        # Show sample multi-season allocations
        print("\n" + "-" * 70)
        print("SAMPLE MULTI-SEASON ALLOCATIONS:")
        multi_season = df[
            (df['title_type'] == 'tv_show') &
            (df['max_seasons'] > 1)
        ].sort_values('total_views', ascending=False)

        if len(multi_season) > 0:
            sample_titles = multi_season['title'].unique()[:3]
            for title in sample_titles:
                title_rows = multi_season[multi_season['title'] == title][
                    ['title', 'fc_uid', 'season_number', 'max_seasons', 'total_views']
                ]
                print(f"\n{title}:")
                print(title_rows.to_string(index=False))


def main():
    allocator = SeasonAllocatorTMDB()
    result_df = allocator.process_allocation()
    allocator.save_results(result_df)


if __name__ == "__main__":
    main()
