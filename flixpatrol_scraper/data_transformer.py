#!/usr/bin/env python3
"""
FlixPatrol Data Transformer
Transforms scraped/API data to match BFD Schema V25.00
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import logging
import json

from config import (
    COUNTRIES,
    PLATFORM_CANONICAL_MAP,
    OUTPUT_DIR,
    FIELD_MAPPING,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlixPatrolDataTransformer:
    """
    Transforms FlixPatrol data to match BFD Schema V25.00
    Following _flixpatrol_ingestion_rules from the schema
    """

    def __init__(self):
        self.output_dir = Path(OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def transform_to_bfd_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform FlixPatrol data to BFD schema format

        FlixPatrol-specific columns (from schema):
        - flixpatrol_id (str)
        - flixpatrol_points (float64)
        - flixpatrol_rank (int32)

        All other fields merge into existing BFD columns
        """
        logger.info(f"Transforming {len(df)} records to BFD format...")

        # Create transformed DataFrame
        transformed = pd.DataFrame()

        # 1. FlixPatrol-specific columns
        transformed["flixpatrol_id"] = df.get("flixpatrol_id", pd.Series([None] * len(df)))
        transformed["flixpatrol_points"] = pd.to_numeric(
            df.get("flixpatrol_points", pd.Series([None] * len(df))),
            errors="coerce"
        ).astype("float64")
        transformed["flixpatrol_rank"] = pd.to_numeric(
            df.get("flixpatrol_rank", pd.Series([None] * len(df))),
            errors="coerce"
        ).astype("Int32")  # Nullable int

        # 2. ID columns (MERGE - use if BFD value is NULL)
        # Transform IMDB ID to proper format (prefix with 'tt')
        if "imdb_id" in df.columns or "imdbId" in df.columns:
            imdb_col = df.get("imdb_id", df.get("imdbId", pd.Series([None] * len(df))))
            transformed["imdb_id"] = imdb_col.apply(self._format_imdb_id)

        if "tmdb_id" in df.columns or "tmdbId" in df.columns:
            tmdb_col = df.get("tmdb_id", df.get("tmdbId", pd.Series([None] * len(df))))
            transformed["tmdb_id"] = pd.to_numeric(tmdb_col, errors="coerce").astype("Int64")

        # 3. Core metadata columns
        transformed["title"] = df.get("title", pd.Series([None] * len(df)))

        # Premiere date - parse to YYYY-MM-DD
        if "premiere_date" in df.columns or "premiere" in df.columns:
            premiere_col = df.get("premiere_date", df.get("premiere", pd.Series([None] * len(df))))
            transformed["premiere_date"] = premiere_col.apply(self._parse_date)

        transformed["overview"] = df.get("overview", df.get("description", pd.Series([None] * len(df))))

        # Runtime
        if "runtime_minutes" in df.columns or "length" in df.columns:
            runtime_col = df.get("runtime_minutes", df.get("length", pd.Series([None] * len(df))))
            transformed["runtime_minutes"] = pd.to_numeric(runtime_col, errors="coerce").astype("Int32")

        # Budget and revenue
        transformed["budget"] = pd.to_numeric(
            df.get("budget", pd.Series([None] * len(df))),
            errors="coerce"
        ).astype("float64")

        if "revenue" in df.columns or "boxOffice" in df.columns:
            revenue_col = df.get("revenue", df.get("boxOffice", pd.Series([None] * len(df))))
            transformed["revenue"] = pd.to_numeric(revenue_col, errors="coerce").astype("float64")

        # Production country
        transformed["production_country"] = df.get(
            "production_country",
            df.get("countries", pd.Series([None] * len(df)))
        )

        # Seasons (for TV)
        if "max_seasons" in df.columns or "seasons" in df.columns:
            seasons_col = df.get("max_seasons", df.get("seasons", pd.Series([None] * len(df))))
            transformed["max_seasons"] = pd.to_numeric(seasons_col, errors="coerce").astype("Int32")

        # 4. Collection metadata
        transformed["collection_date"] = df.get("_collection_date", pd.Series([None] * len(df)))
        transformed["collection_country"] = df.get("_country_code", pd.Series([None] * len(df)))
        transformed["collection_platform"] = df.get("_platform_canonical", pd.Series([None] * len(df)))
        transformed["content_type"] = df.get("content_type", df.get("_content_type", pd.Series([None] * len(df))))

        # 5. Data source tracking
        transformed["source_api"] = "flixpatrol"
        transformed["ingestion_timestamp"] = datetime.now().isoformat()

        # Clean up
        transformed = transformed.replace({np.nan: None, "": None})

        logger.info(f"Transformation complete. Columns: {list(transformed.columns)}")
        return transformed

    def _format_imdb_id(self, value) -> Optional[str]:
        """Format IMDB ID with 'tt' prefix"""
        if pd.isna(value) or value is None:
            return None

        value_str = str(value).strip()

        # Already formatted
        if value_str.startswith("tt"):
            return value_str

        # Numeric only - add prefix
        if value_str.isdigit():
            return f"tt{value_str.zfill(7)}"

        return value_str

    def _parse_date(self, value) -> Optional[str]:
        """Parse date to YYYY-MM-DD format"""
        if pd.isna(value) or value is None:
            return None

        value_str = str(value).strip()

        # Common date formats
        formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%B %d, %Y",
            "%Y",
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(value_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

        # Try to extract year at minimum
        import re
        year_match = re.search(r'(19|20)\d{2}', value_str)
        if year_match:
            return f"{year_match.group()}-01-01"

        return None

    def aggregate_by_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate daily data into quarterly/half-yearly periods
        Matches BFD temporal column structure
        """
        if "collection_date" not in df.columns:
            logger.warning("No collection_date column - cannot aggregate by period")
            return df

        df["collection_date"] = pd.to_datetime(df["collection_date"], errors="coerce")

        # Add period columns
        df["year"] = df["collection_date"].dt.year
        df["quarter"] = df["collection_date"].dt.quarter
        df["half"] = df["collection_date"].dt.month.apply(lambda m: 1 if m <= 6 else 2)

        # Create period identifiers
        df["period_quarter"] = df.apply(lambda r: f"q{r['quarter']}_{r['year']}", axis=1)
        df["period_half"] = df.apply(lambda r: f"h{r['half']}_{r['year']}", axis=1)

        return df

    def pivot_by_country(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot data to create country-specific columns
        Matches BFD views_{period}_{year}_{country} structure
        """
        if "collection_country" not in df.columns:
            logger.warning("No collection_country column")
            return df

        # Group by title + period, pivot by country
        # This creates the structure needed for BFD

        # First, ensure we have aggregated data
        df = self.aggregate_by_period(df)

        # Create pivot table for quarterly data
        pivot_cols = ["flixpatrol_id", "title", "period_quarter"]

        if all(col in df.columns for col in pivot_cols + ["flixpatrol_points"]):
            quarterly_pivot = df.pivot_table(
                index=["flixpatrol_id", "title", "period_quarter"],
                columns="collection_country",
                values="flixpatrol_points",
                aggfunc="sum"
            ).reset_index()

            # Rename columns to match BFD pattern
            quarterly_pivot.columns = [
                f"fp_points_{col}" if col in COUNTRIES else col
                for col in quarterly_pivot.columns
            ]

            return quarterly_pivot

        return df

    def deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicate data, keeping highest-quality records

        Priority (from schema):
        1. Records with IMDB ID
        2. Records with TMDB ID
        3. Records with most points
        """
        logger.info(f"Deduplicating {len(df)} records...")

        # Score each record
        df["_quality_score"] = 0
        df.loc[df["imdb_id"].notna(), "_quality_score"] += 10
        df.loc[df["tmdb_id"].notna(), "_quality_score"] += 5
        df.loc[df["flixpatrol_points"].notna(), "_quality_score"] += df["flixpatrol_points"].fillna(0) / 1000

        # Group by unique identifiers and keep best
        group_cols = ["flixpatrol_id", "collection_date", "collection_country", "collection_platform"]
        group_cols = [c for c in group_cols if c in df.columns]

        if group_cols:
            df = df.sort_values("_quality_score", ascending=False)
            df = df.drop_duplicates(subset=group_cols, keep="first")

        df = df.drop(columns=["_quality_score"], errors="ignore")

        logger.info(f"After deduplication: {len(df)} records")
        return df

    def validate(self, df: pd.DataFrame) -> Dict:
        """
        Validate transformed data against schema requirements

        Returns validation report
        """
        report = {
            "total_records": len(df),
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {}
        }

        # Check required FlixPatrol columns
        required_cols = ["flixpatrol_id", "flixpatrol_points", "flixpatrol_rank"]
        for col in required_cols:
            if col not in df.columns:
                report["errors"].append(f"Missing required column: {col}")
                report["valid"] = False

        # Check data types
        if "flixpatrol_points" in df.columns:
            if not pd.api.types.is_float_dtype(df["flixpatrol_points"]):
                report["warnings"].append("flixpatrol_points should be float64")

        # Statistics
        report["stats"]["unique_titles"] = df["flixpatrol_id"].nunique() if "flixpatrol_id" in df.columns else 0
        report["stats"]["with_imdb"] = df["imdb_id"].notna().sum() if "imdb_id" in df.columns else 0
        report["stats"]["with_tmdb"] = df["tmdb_id"].notna().sum() if "tmdb_id" in df.columns else 0

        if "collection_country" in df.columns:
            report["stats"]["countries"] = df["collection_country"].nunique()

        if "collection_platform" in df.columns:
            report["stats"]["platforms"] = df["collection_platform"].nunique()

        if "collection_date" in df.columns:
            report["stats"]["date_range"] = {
                "min": str(df["collection_date"].min()),
                "max": str(df["collection_date"].max())
            }

        return report

    def save(self, df: pd.DataFrame, filename: str):
        """Save transformed data"""
        # Parquet (primary)
        parquet_path = self.output_dir / f"{filename}.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved to {parquet_path}")

        # CSV (for inspection)
        csv_path = self.output_dir / f"{filename}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved to {csv_path}")

        # Validation report
        report = self.validate(df)
        report_path = self.output_dir / f"{filename}_validation.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Validation report: {report_path}")

        return report


def main():
    """Test transformer with sample data"""
    transformer = FlixPatrolDataTransformer()

    # Sample data
    sample_data = pd.DataFrame([
        {
            "flixpatrol_id": "test-123",
            "title": "Test Movie",
            "imdbId": "0123456",
            "tmdbId": 12345,
            "flixpatrol_points": 1500.0,
            "flixpatrol_rank": 1,
            "premiere": "2024-01-15",
            "length": 120,
            "_collection_date": "2024-01-20",
            "_country_code": "us",
            "_platform_canonical": "amazon_prime_video",
            "_content_type": "movie"
        }
    ])

    transformed = transformer.transform_to_bfd_format(sample_data)
    print("\nTransformed data:")
    print(transformed.to_string())

    report = transformer.validate(transformed)
    print("\nValidation report:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
