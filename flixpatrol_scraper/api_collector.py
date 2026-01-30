#!/usr/bin/env python3
"""
FlixPatrol API v2 Data Collector
Uses the paid FlixPatrol API with HTTP Basic Auth
Based on template from API Keys document
"""

import requests
from requests.auth import HTTPBasicAuth
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
from tenacity import retry, wait_exponential, stop_after_attempt

from config import (
    FLIXPATROL_API_KEY,
    FLIXPATROL_API_BASE,
    COUNTRIES,
    FLIXPATROL_COUNTRY_CODES,
    PLATFORMS_TO_TRACK,
    PLATFORM_CANONICAL_MAP,
    CONTENT_TYPES,
    REQUEST_DELAY_SECONDS,
    MAX_RETRIES,
    TIMEOUT_SECONDS,
    OUTPUT_DIR,
    FIELD_MAPPING,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{OUTPUT_DIR}/api_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FlixPatrolAPICollector:
    """
    Collects data from FlixPatrol API v2
    Authentication: HTTP Basic Auth with API key as username, empty password
    """

    def __init__(self):
        self.api_key = FLIXPATROL_API_KEY
        self.base_url = FLIXPATROL_API_BASE
        self.auth = HTTPBasicAuth(self.api_key, "")
        self.session = requests.Session()
        self.session.auth = self.auth
        self.collected_data = []

        # Create output directory
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    def _make_request(self, method: str, endpoint: str, params: dict = None) -> Optional[Dict]:
        """Make authenticated request to FlixPatrol API"""
        url = f"{self.base_url}/{endpoint}"

        try:
            if method.upper() == "GET":
                response = self.session.get(
                    url,
                    params=params,
                    timeout=TIMEOUT_SECONDS
                )
            elif method.upper() == "HEAD":
                response = self.session.head(
                    url,
                    timeout=TIMEOUT_SECONDS
                )
            elif method.upper() == "OPTIONS":
                response = self.session.options(
                    url,
                    timeout=TIMEOUT_SECONDS
                )
            else:
                logger.error(f"Unsupported method: {method}")
                return None

            if response.status_code == 200:
                if method.upper() == "HEAD":
                    return {"headers": dict(response.headers)}
                return response.json()
            elif response.status_code == 401:
                logger.error(f"Authentication failed for {url}")
                return None
            elif response.status_code == 429:
                logger.warning(f"Rate limited on {url}, waiting...")
                time.sleep(60)  # Wait 1 minute on rate limit
                return self._make_request(method, endpoint, params)
            else:
                logger.warning(f"Request failed: {response.status_code} - {response.text[:200]}")
                return None

        except requests.exceptions.Timeout:
            logger.error(f"Timeout on {url}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return None

    @retry(wait=wait_exponential(multiplier=1, min=2, max=60), stop=stop_after_attempt(MAX_RETRIES))
    def get_titles_list(self, page: int = 1, limit: int = 100) -> Optional[Dict]:
        """Get list of titles from API"""
        params = {
            "page": page,
            "limit": limit
        }
        return self._make_request("GET", "titles", params)

    @retry(wait=wait_exponential(multiplier=1, min=2, max=60), stop=stop_after_attempt(MAX_RETRIES))
    def get_title_detail(self, title_id: str) -> Optional[Dict]:
        """Get detailed information for a specific title"""
        return self._make_request("GET", f"titles/{title_id}")

    @retry(wait=wait_exponential(multiplier=1, min=2, max=60), stop=stop_after_attempt(MAX_RETRIES))
    def get_title_rankings(self, title_id: str) -> Optional[Dict]:
        """Get ranking history for a title"""
        return self._make_request("GET", f"titles/{title_id}/rankings")

    @retry(wait=wait_exponential(multiplier=1, min=2, max=60), stop=stop_after_attempt(MAX_RETRIES))
    def get_top_titles(self, platform: str, country: str, content_type: str = "movies",
                       date: str = None) -> Optional[Dict]:
        """Get top titles for a platform/country combination"""
        endpoint = f"top/{content_type}/{platform}/{country}"
        params = {}
        if date:
            params["date"] = date
        return self._make_request("GET", endpoint, params if params else None)

    def get_titles_count(self) -> int:
        """Get total count of titles via HEAD request"""
        result = self._make_request("HEAD", "titles")
        if result and "headers" in result:
            return int(result["headers"].get("x-total-count", 0))
        return 0

    def test_connection(self) -> bool:
        """Test API connection"""
        logger.info("Testing FlixPatrol API connection...")

        # Test OPTIONS (help)
        result = self._make_request("OPTIONS", "titles")
        if result:
            logger.info("OPTIONS /titles - OK")

        # Test HEAD (count)
        count = self.get_titles_count()
        logger.info(f"HEAD /titles - Total count: {count}")

        # Test GET titles
        result = self.get_titles_list(page=1, limit=5)
        if result:
            logger.info("GET /titles - OK")
            return True
        else:
            logger.error("GET /titles - FAILED")
            return False

    def collect_all_titles(self) -> List[Dict]:
        """Collect all titles from the API"""
        logger.info("Starting to collect all titles...")

        all_titles = []
        page = 1
        limit = 100

        # Get total count first
        total_count = self.get_titles_count()
        logger.info(f"Total titles to collect: {total_count}")

        while True:
            logger.info(f"Fetching page {page}...")
            result = self.get_titles_list(page=page, limit=limit)

            if not result:
                break

            data = result.get("data", [])
            if not data:
                break

            all_titles.extend(data)
            logger.info(f"Collected {len(all_titles)} / {total_count} titles")

            # Check if we've got all
            if len(data) < limit:
                break

            page += 1
            time.sleep(REQUEST_DELAY_SECONDS)

        logger.info(f"Total titles collected: {len(all_titles)}")
        return all_titles

    def collect_title_details(self, title_ids: List[str]) -> List[Dict]:
        """Collect detailed info for specific titles"""
        detailed_titles = []

        for i, title_id in enumerate(title_ids):
            logger.info(f"Fetching details for title {i+1}/{len(title_ids)}: {title_id}")

            detail = self.get_title_detail(title_id)
            if detail:
                detailed_titles.append(detail.get("data", detail))

            time.sleep(REQUEST_DELAY_SECONDS)

        return detailed_titles

    def collect_top_charts(self, start_date: str, end_date: str) -> List[Dict]:
        """
        Collect top charts for all platforms/countries/content types
        Date range: start_date to end_date (YYYY-MM-DD format)
        EXCLUDES Netflix
        """
        all_chart_data = []

        # Parse dates
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        final_date = datetime.strptime(end_date, "%Y-%m-%d")

        # Iterate through each date
        while current_date <= final_date:
            date_str = current_date.strftime("%Y-%m-%d")
            logger.info(f"Collecting charts for date: {date_str}")

            for country_code in COUNTRIES:
                fp_country = FLIXPATROL_COUNTRY_CODES.get(country_code, country_code)

                for platform in PLATFORMS_TO_TRACK:
                    # Skip Netflix (already excluded from PLATFORMS_TO_TRACK)
                    if "netflix" in platform.lower():
                        continue

                    for content_type in CONTENT_TYPES:
                        try:
                            result = self.get_top_titles(
                                platform=platform,
                                country=fp_country,
                                content_type=content_type,
                                date=date_str
                            )

                            if result and "data" in result:
                                for item in result["data"]:
                                    item["_collection_date"] = date_str
                                    item["_country_code"] = country_code
                                    item["_platform"] = platform
                                    item["_platform_canonical"] = PLATFORM_CANONICAL_MAP.get(platform, platform)
                                    item["_content_type"] = content_type
                                    all_chart_data.append(item)

                            time.sleep(REQUEST_DELAY_SECONDS)

                        except Exception as e:
                            logger.error(f"Error collecting {platform}/{fp_country}/{content_type}: {e}")

            # Move to next day
            current_date += timedelta(days=1)

            # Save checkpoint every week
            if current_date.day == 1:
                self._save_checkpoint(all_chart_data, f"checkpoint_{date_str}")

        return all_chart_data

    def _save_checkpoint(self, data: List[Dict], filename: str):
        """Save intermediate data checkpoint"""
        if not data:
            return

        df = pd.DataFrame(data)
        checkpoint_path = f"{OUTPUT_DIR}/{filename}.parquet"
        df.to_parquet(checkpoint_path, index=False)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def transform_to_bfd_schema(self, data: List[Dict]) -> pd.DataFrame:
        """Transform collected data to BFD schema format"""
        transformed = []

        for item in data:
            record = {}

            # Map fields according to schema
            for fp_field, bfd_field in FIELD_MAPPING.items():
                if fp_field in item:
                    value = item[fp_field]

                    # Transform imdbId to proper format
                    if bfd_field == "imdb_id" and value:
                        if str(value).isdigit():
                            value = f"tt{str(value).zfill(7)}"

                    record[bfd_field] = value

            # Add collection metadata
            record["collection_date"] = item.get("_collection_date")
            record["country_code"] = item.get("_country_code")
            record["platform"] = item.get("_platform_canonical")
            record["content_type"] = item.get("_content_type")

            transformed.append(record)

        return pd.DataFrame(transformed)

    def save_data(self, df: pd.DataFrame, filename: str):
        """Save DataFrame to output"""
        output_path = f"{OUTPUT_DIR}/{filename}.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"Data saved: {output_path} ({len(df)} rows)")

        # Also save CSV for inspection
        csv_path = f"{OUTPUT_DIR}/{filename}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"CSV saved: {csv_path}")


def main():
    """Main execution"""
    print("=" * 60)
    print("FLIXPATROL API v2 DATA COLLECTOR")
    print("=" * 60)

    collector = FlixPatrolAPICollector()

    # Test connection first
    if not collector.test_connection():
        print("\nAPI connection test failed. Falling back to web scraper...")
        return False

    print("\nAPI connection successful!")

    # Collect title metadata
    print("\n[1/3] Collecting title metadata...")
    titles = collector.collect_all_titles()

    if titles:
        df_titles = collector.transform_to_bfd_schema(titles)
        collector.save_data(df_titles, "flixpatrol_titles_metadata")

    # Collect top charts (this is where the views/rankings data is)
    print("\n[2/3] Collecting top charts data (Jan 2023 - Now)...")
    print("This will take significant time due to the date range and API rate limits...")

    from config import START_DATE, END_DATE
    charts = collector.collect_top_charts(START_DATE, END_DATE)

    if charts:
        df_charts = collector.transform_to_bfd_schema(charts)
        collector.save_data(df_charts, "flixpatrol_top_charts")

    print("\n[3/3] Collection complete!")
    print(f"Output directory: {OUTPUT_DIR}")

    return True


if __name__ == "__main__":
    main()
