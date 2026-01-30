"""
FlixPatrol Data Collection Configuration
Based on SCHEMA_V25.00 requirements
Excludes Netflix - focuses on all other platforms
"""

# FlixPatrol API Configuration
FLIXPATROL_API_KEY = "aku_4bXKmMWPSCaKxwn2tVzTXmcg"
FLIXPATROL_API_BASE = "https://api.flixpatrol.com/v2"

# Date Range
START_DATE = "2023-01-01"
END_DATE = "2026-01-21"

# 22 Countries from Schema V25.00
COUNTRIES = [
    "us",   # United States - 37%
    "cn",   # China - 15%
    "in",   # India - 8%
    "gb",   # United Kingdom - 6.5%
    "br",   # Brazil - 5%
    "de",   # Germany - 4%
    "jp",   # Japan - 4%
    "fr",   # France - 3.5%
    "ca",   # Canada - 3%
    "mx",   # Mexico - 2.5%
    "au",   # Australia - 2%
    "es",   # Spain - 2%
    "it",   # Italy - 1.5%
    "kr",   # South Korea - 1.5%
    "nl",   # Netherlands - 1%
    "se",   # Sweden - 0.5%
    "sg",   # Singapore - 0.5%
    "hk",   # Hong Kong - 0.5%
    "ie",   # Ireland - 0.3%
    "ru",   # Russia - 0.5%
    "tr",   # Turkey - 0.5%
    "row",  # Rest of World - 2%
]

# FlixPatrol country code mapping (FlixPatrol uses different codes)
FLIXPATROL_COUNTRY_CODES = {
    "us": "united-states",
    "cn": "china",
    "in": "india",
    "gb": "united-kingdom",
    "br": "brazil",
    "de": "germany",
    "jp": "japan",
    "fr": "france",
    "ca": "canada",
    "mx": "mexico",
    "au": "australia",
    "es": "spain",
    "it": "italy",
    "kr": "south-korea",
    "nl": "netherlands",
    "se": "sweden",
    "sg": "singapore",
    "hk": "hong-kong",
    "ie": "ireland",
    "ru": "russia",
    "tr": "turkey",
    "row": "world",  # Global/World for ROW
}

# 50 Platforms to track (bfd_tracked=true, EXCLUDING Netflix)
PLATFORMS_TO_TRACK = [
    "amazon-prime-video",
    "disney",  # Disney+
    "hbo",     # Max/HBO
    "paramount-plus",
    "hulu",
    "peacock",
    "apple-tv",
    "curiosity-stream",
    "espn-plus",
    "dazn",
    "starz",
    "crunchyroll",
    "mubi",
    "amc-plus",
    "discovery-plus",
    "rtl-plus",
    "viaplay",
    "britbox",
    "crave",
    "movistar-plus",
    "starzplay",
    "now-tv",
    "ocs",
    "joyn",
    "canal-plus",
    "shudder",
    "stan",
    "timvision",
    "sling-tv",
    "fubo-tv",
    "skyshowtime",
    "mediaset-infinity",
    "videoland",
    "binge",
    "kayo-sports",
    "polsat-box-go",
    "itvx",  # ITV
    "player",
    "tv2-play-denmark",
    "tv4-play",
    "wow",
    "cda-premium",
    "megogo",
    "atresplayer",
    "voyo",
    "tv2-play-norway",
    "go3",
    "kyivstar-tv",
    "streamz",
    "prima-plus",
    "foxtel-now",
]

# Platform canonical name mapping (FlixPatrol URL -> BFD canonical)
PLATFORM_CANONICAL_MAP = {
    "amazon-prime-video": "amazon_prime_video",
    "disney": "disney_plus",
    "hbo": "max_hbo",
    "paramount-plus": "paramount_plus",
    "hulu": "hulu",
    "peacock": "peacock",
    "apple-tv": "apple_tv_plus",
    "curiosity-stream": "curiositystream",
    "espn-plus": "espn_plus",
    "dazn": "dazn",
    "starz": "starz",
    "crunchyroll": "crunchyroll",
    "mubi": "mubi",
    "amc-plus": "amc_plus",
    "discovery-plus": "discovery_plus",
    "rtl-plus": "rtl_plus",
    "viaplay": "viaplay",
    "britbox": "britbox",
    "crave": "crave",
    "movistar-plus": "movistar_plus",
    "starzplay": "starzplay",
    "now-tv": "now_tv",
    "ocs": "ocs",
    "joyn": "joyn",
    "canal-plus": "canal_plus",
    "shudder": "shudder",
    "stan": "stan",
    "timvision": "timvision",
    "sling-tv": "sling_tv",
    "fubo-tv": "fubotv",
    "skyshowtime": "skyshowtime",
    "mediaset-infinity": "mediaset",
    "videoland": "videoland",
    "binge": "binge",
    "kayo-sports": "kayo_sports",
    "polsat-box-go": "polsat",
    "itvx": "itv",
    "player": "player",
    "tv2-play-denmark": "tv_2_denmark",
    "tv4-play": "tv4_sweden",
    "wow": "wow_germany",
    "cda-premium": "cda_poland",
    "megogo": "megogo_ukraine",
    "atresplayer": "atresmedia_spain",
    "voyo": "voyo_czech/slovakia",
    "tv2-play-norway": "tv2_norge",
    "go3": "go3_baltics",
    "kyivstar-tv": "kyivstar_ukraine",
    "streamz": "streamz_belgium",
    "prima-plus": "iprima_czech_republic",
    "foxtel-now": "foxtel_australia",
}

# Content types to scrape
CONTENT_TYPES = ["movies", "tv-shows"]

# Scraping settings
REQUEST_DELAY_SECONDS = 1.5  # Delay between requests to avoid rate limiting
MAX_RETRIES = 3
TIMEOUT_SECONDS = 30

# Output configuration
OUTPUT_DIR = "C:\\Users\\RoyT6\\Downloads\\Views TRaining Data"
OUTPUT_FORMAT = "parquet"  # parquet or csv

# Headers for web scraping
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# BFD Schema field mapping (FlixPatrol -> BFD)
FIELD_MAPPING = {
    "id": "flixpatrol_id",
    "imdbId": "imdb_id",
    "tmdbId": "tmdb_id",
    "title": "title",
    "premiere": "premiere_date",
    "description": "overview",
    "length": "runtime_minutes",
    "budget": "budget",
    "boxOffice": "revenue",
    "countries": "production_country",
    "seasons": "max_seasons",
    "points": "flixpatrol_points",
    "rank": "flixpatrol_rank",
}
