#!/usr/bin/env python3
"""
FlixPatrol Data Collection Orchestrator
Main entry point for collecting streaming views data

Collects data for 22 countries, 51 platforms (excluding Netflix)
Date range: January 2023 to present
"""

import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd

from config import OUTPUT_DIR, START_DATE, END_DATE, COUNTRIES, PLATFORMS_TO_TRACK
from api_collector import FlixPatrolAPICollector
from web_scraper import FlixPatrolWebScraper
from data_transformer import FlixPatrolDataTransformer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{OUTPUT_DIR}/orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print startup banner"""
    print("=" * 70)
    print("  FLIXPATROL DATA COLLECTION SYSTEM")
    print("  BFD Schema V25.00 Compatible")
    print("=" * 70)
    print(f"\n  Date Range: {START_DATE} to {END_DATE}")
    print(f"  Countries: {len(COUNTRIES)}")
    print(f"  Platforms: {len(PLATFORMS_TO_TRACK)} (Netflix excluded)")
    print(f"  Output: {OUTPUT_DIR}")
    print("\n" + "=" * 70)


def run_api_collection(start_date: str = None, end_date: str = None) -> bool:
    """
    Run API-based collection (primary method)
    Returns True if successful and data collected
    """
    logger.info("=" * 50)
    logger.info("PHASE 1: API-BASED COLLECTION")
    logger.info("=" * 50)

    collector = FlixPatrolAPICollector()

    # Test connection
    logger.info("Testing API connection...")
    if not collector.test_connection():
        logger.warning("API connection failed or limited. Will use web scraper.")
        return False

    logger.info("API connection successful!")

    # Use provided dates or defaults
    start = start_date or START_DATE
    end = end_date or END_DATE

    # Collect title metadata
    logger.info("\n[API 1/2] Collecting title metadata...")
    titles = collector.collect_all_titles()

    if titles:
        df_titles = collector.transform_to_bfd_schema(titles)
        collector.save_data(df_titles, "api_titles_metadata")
        logger.info(f"Collected {len(df_titles)} title records via API")
    else:
        logger.warning("No title metadata collected via API")

    # Collect top charts
    logger.info(f"\n[API 2/2] Collecting top charts ({start} to {end})...")
    charts = collector.collect_top_charts(start, end)

    if charts:
        df_charts = collector.transform_to_bfd_schema(charts)
        collector.save_data(df_charts, "api_top_charts")
        logger.info(f"Collected {len(df_charts)} chart records via API")
        return True
    else:
        logger.warning("No chart data collected via API")
        return False


def run_web_scraping(start_date: str = None, end_date: str = None) -> bool:
    """
    Run web scraping (fallback/supplement method)
    Returns True if successful
    """
    logger.info("=" * 50)
    logger.info("PHASE 2: WEB SCRAPING")
    logger.info("=" * 50)

    scraper = FlixPatrolWebScraper()

    # Use provided dates or defaults
    start = start_date or START_DATE
    end = end_date or END_DATE

    # Scrape top charts for all platforms/countries
    logger.info(f"\n[SCRAPE 1/2] Scraping top charts ({start} to {end})...")
    logger.info("Note: This process will take significant time due to date range...")

    chart_data = scraper.scrape_all_top_charts(start, end)

    if chart_data:
        df = pd.DataFrame(chart_data)
        scraper.save_data(df, "scraped_top_charts")
        logger.info(f"Scraped {len(df)} chart records")

        # Get title IDs for detail scraping
        title_ids = df['flixpatrol_id'].dropna().unique().tolist()

        # Scrape title details
        if title_ids:
            logger.info(f"\n[SCRAPE 2/2] Scraping details for {len(title_ids)} titles...")
            details = scraper.scrape_title_details(title_ids[:1000])  # Limit to first 1000

            if details:
                df_details = pd.DataFrame(details)
                scraper.save_data(df_details, "scraped_title_details")
                logger.info(f"Scraped {len(df_details)} title detail records")

        return True
    else:
        logger.error("Web scraping failed to collect data")
        return False


def transform_and_merge():
    """
    Transform and merge all collected data to BFD format
    """
    logger.info("=" * 50)
    logger.info("PHASE 3: TRANSFORM AND MERGE")
    logger.info("=" * 50)

    transformer = FlixPatrolDataTransformer()
    output_path = Path(OUTPUT_DIR)

    all_data = []

    # Load API data if exists
    api_charts_path = output_path / "api_top_charts.parquet"
    if api_charts_path.exists():
        logger.info("Loading API chart data...")
        df_api = pd.read_parquet(api_charts_path)
        all_data.append(df_api)
        logger.info(f"  Loaded {len(df_api)} records from API")

    # Load scraped data if exists
    scraped_charts_path = output_path / "scraped_top_charts.parquet"
    if scraped_charts_path.exists():
        logger.info("Loading scraped chart data...")
        df_scraped = pd.read_parquet(scraped_charts_path)
        all_data.append(df_scraped)
        logger.info(f"  Loaded {len(df_scraped)} records from scraper")

    if not all_data:
        logger.error("No data files found to transform!")
        return None

    # Merge all data
    logger.info("\nMerging datasets...")
    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined dataset: {len(combined)} records")

    # Transform to BFD format
    logger.info("\nTransforming to BFD Schema V25.00 format...")
    transformed = transformer.transform_to_bfd_format(combined)

    # Deduplicate
    logger.info("\nDeduplicating records...")
    deduplicated = transformer.deduplicate(transformed)

    # Save final output
    logger.info("\nSaving final output...")
    report = transformer.save(deduplicated, "flixpatrol_bfd_final")

    # Print summary
    print("\n" + "=" * 70)
    print("  COLLECTION COMPLETE")
    print("=" * 70)
    print(f"\n  Total Records: {report['total_records']}")
    print(f"  Unique Titles: {report['stats'].get('unique_titles', 'N/A')}")
    print(f"  With IMDB ID: {report['stats'].get('with_imdb', 'N/A')}")
    print(f"  With TMDB ID: {report['stats'].get('with_tmdb', 'N/A')}")
    print(f"  Countries: {report['stats'].get('countries', 'N/A')}")
    print(f"  Platforms: {report['stats'].get('platforms', 'N/A')}")

    if 'date_range' in report['stats']:
        print(f"  Date Range: {report['stats']['date_range']['min']} to {report['stats']['date_range']['max']}")

    if report['errors']:
        print(f"\n  Errors: {len(report['errors'])}")
        for err in report['errors']:
            print(f"    - {err}")

    if report['warnings']:
        print(f"\n  Warnings: {len(report['warnings'])}")
        for warn in report['warnings']:
            print(f"    - {warn}")

    print(f"\n  Output Files:")
    print(f"    - {OUTPUT_DIR}/flixpatrol_bfd_final.parquet")
    print(f"    - {OUTPUT_DIR}/flixpatrol_bfd_final.csv")
    print(f"    - {OUTPUT_DIR}/flixpatrol_bfd_final_validation.json")
    print("\n" + "=" * 70)

    return report


def run_quick_test():
    """Run a quick test with limited data"""
    logger.info("Running quick test mode...")

    # Test with just 1 day, 2 countries, 2 platforms
    from datetime import datetime, timedelta

    test_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    print(f"\nQuick test: Scraping {test_date} for US and GB, Amazon and Disney+")

    scraper = FlixPatrolWebScraper()

    # Test single scrape
    data = scraper.scrape_top_chart(
        platform="amazon-prime-video",
        country="united-states",
        date=test_date,
        content_type="movies"
    )

    if data:
        print(f"Success! Retrieved {len(data)} records")
        print("\nSample record:")
        import json
        print(json.dumps(data[0], indent=2, default=str))
        return True
    else:
        print("Test failed - no data retrieved")
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="FlixPatrol Data Collection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run full collection (API + scraper)
  python main.py --api-only         # Only use API
  python main.py --scrape-only      # Only use web scraper
  python main.py --transform-only   # Only transform existing data
  python main.py --test             # Run quick test
  python main.py --start 2024-01-01 --end 2024-03-31  # Custom date range
        """
    )

    parser.add_argument('--api-only', action='store_true',
                        help='Only run API-based collection')
    parser.add_argument('--scrape-only', action='store_true',
                        help='Only run web scraping')
    parser.add_argument('--transform-only', action='store_true',
                        help='Only transform existing data')
    parser.add_argument('--test', action='store_true',
                        help='Run quick test with limited data')
    parser.add_argument('--start', type=str, default=None,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                        help='End date (YYYY-MM-DD)')

    args = parser.parse_args()

    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print_banner()

    # Quick test mode
    if args.test:
        success = run_quick_test()
        sys.exit(0 if success else 1)

    # Transform only mode
    if args.transform_only:
        report = transform_and_merge()
        sys.exit(0 if report else 1)

    # Collection modes
    api_success = False
    scrape_success = False

    if args.api_only or not args.scrape_only:
        # Try API first
        api_success = run_api_collection(args.start, args.end)

    if args.scrape_only or (not args.api_only and not api_success):
        # Use scraper as fallback or primary
        scrape_success = run_web_scraping(args.start, args.end)

    # Transform if we got any data
    if api_success or scrape_success:
        transform_and_merge()
    else:
        logger.error("No data collected from any source!")
        sys.exit(1)


if __name__ == "__main__":
    main()
