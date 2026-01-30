#!/usr/bin/env python3
"""
FlixPatrol Data Collection - Run Now
Collects title metadata from FlixPatrol API and saves to Views Training Data
"""

import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
import json
import time
from datetime import datetime
from pathlib import Path

# Configuration
API_KEY = "aku_4bXKmMWPSCaKxwn2tVzTXmcg"
BASE_URL = "https://api.flixpatrol.com/v2"
OUTPUT_DIR = Path(r"C:\Users\RoyT6\Downloads\Views TRaining Data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def collect_all_titles():
    """Collect all titles from FlixPatrol API"""
    print("=" * 60)
    print("FLIXPATROL DATA COLLECTION")
    print("=" * 60)

    session = requests.Session()
    session.auth = HTTPBasicAuth(API_KEY, "")

    all_titles = []
    next_url = f"{BASE_URL}/titles"
    page = 0

    while next_url:
        page += 1
        print(f"\nFetching page {page}...")

        try:
            response = session.get(next_url, timeout=30)

            if response.status_code == 200:
                data = response.json()

                # Extract titles from response
                titles_data = data.get("data", [])
                if not titles_data:
                    print("No more data")
                    break

                for item in titles_data:
                    title_info = item.get("data", {})

                    # Extract relevant fields
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
                        "facebook_link": title_info.get("linkFacebook"),
                        "twitter_link": title_info.get("linkTwitter"),
                        "instagram_link": title_info.get("linkInstagram"),
                        "wikipedia_link": title_info.get("linkWikipedia"),
                        "reddit_link": title_info.get("linkReddit"),
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

                    all_titles.append(record)

                print(f"  Collected {len(all_titles)} titles so far...")

                # Get next page URL
                links = data.get("links", {})
                next_url = links.get("next")

                # Rate limiting
                time.sleep(0.5)

            elif response.status_code == 401:
                print("Authentication failed!")
                break
            elif response.status_code == 429:
                print("Rate limited, waiting 60 seconds...")
                time.sleep(60)
                continue
            else:
                print(f"Error: {response.status_code}")
                print(response.text[:500])
                break

        except Exception as e:
            print(f"Error: {e}")
            break

    return all_titles

def save_data(titles):
    """Save collected data to files"""
    if not titles:
        print("No data to save!")
        return

    df = pd.DataFrame(titles)

    # Add collection metadata
    df["source_api"] = "flixpatrol"
    df["collection_timestamp"] = datetime.now().isoformat()

    # Save as Parquet
    parquet_path = OUTPUT_DIR / "flixpatrol_titles.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"\nSaved: {parquet_path}")

    # Save as CSV
    csv_path = OUTPUT_DIR / "flixpatrol_titles.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Save summary
    summary = {
        "total_records": len(df),
        "collection_date": datetime.now().isoformat(),
        "columns": list(df.columns),
        "movies": len(df[df["content_type"] == "movie"]),
        "tv_shows": len(df[df["content_type"] == "tv_show"]),
        "with_imdb_id": df["imdb_id"].notna().sum(),
        "with_tmdb_id": df["tmdb_id"].notna().sum(),
    }

    summary_path = OUTPUT_DIR / "flixpatrol_collection_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Total Records: {summary['total_records']}")
    print(f"Movies: {summary['movies']}")
    print(f"TV Shows: {summary['tv_shows']}")
    print(f"With IMDB ID: {summary['with_imdb_id']}")
    print(f"With TMDB ID: {summary['with_tmdb_id']}")
    print(f"\nOutput Directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    titles = collect_all_titles()
    save_data(titles)
