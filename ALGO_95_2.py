#!/usr/bin/env python3
"""
================================================================================
                     ALGO-95.2 VIEWERSHIP PREDICTION ENGINE
                 Context-Aware Streaming Viewership Prediction
                 with Abstract Environmental Weighting
================================================================================

ALGO-95.2 Whitepaper Implementation
Status: Production-Grade Mathematical Extension
Compatibility: 100% backward-compatible with ALGO-65.2
Target Accuracy: ≤ 2.0% MAPE (≥98% accuracy)

MASTER EQUATIONS:
-----------------
1. ALGO-65.2 Base (PRESERVED):
   V̂_i,p,t = C_i × P_p,i × T_i,t × E_t × M_i

2. ALGO-95.2 Extension (ADDITIVE):
   V̂^95_i,p,t = V̂_i,p,t × R_i × G_d,t × Q_d,t × A_i,p,t

Where:
   R_i     = Completion/Retention Quality Index [0.25, 4.00]
   G_d,t   = Geopolitical Risk Suppression [0.58, 1.00]
   Q_d,t   = Quality of Experience (QoE)
   A_i,p,t = Platform Availability & Licensing Factor

Author: Roy Taylor / Framecore
Date: January 2026
================================================================================
"""

from __future__ import annotations

import os
import sys
import json
import math
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum, auto
import warnings

# Try to import optional dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    warnings.warn("NumPy not found - using pure Python calculations")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    warnings.warn("Pandas not found - limited DataFrame support")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base paths - all paths flow from here
BASE_PATH = Path(r"C:\Users\RoyT6\Downloads")
DEFAULT_CONFIG_PATH = Path(__file__).parent / "ALGO-95.2 Full File Load.json"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "REPORTS"

# Directory paths (mapped from ALGO_95_2_DIRECTORY_MAPPING.md)
DIRECTORY_PATHS = {
    "schema": BASE_PATH / "Schema",
    "components": BASE_PATH / "Components",
    "abstract_data": BASE_PATH / "Abstract Data",
    "vbus": BASE_PATH / "VBUS",
    "mapie": BASE_PATH / "MAPIE",
    "orchestrator": BASE_PATH / "Orchestrator",
    "studios": BASE_PATH / "Studios",
    "talent": BASE_PATH / "Talent",
    "daily_top_10s": BASE_PATH / "Daily Top 10s",
    "credibility": BASE_PATH / "Credibility",
    "money_engine": BASE_PATH / "Money Engine",
    "streamers_by_country": BASE_PATH / "Components" / "Streamers By Country",  # Updated path
    "algo_engine": BASE_PATH / "ALGO Engine",
    "views_training_data": BASE_PATH / "Views TRaining Data",
}

# Component file paths
COMPONENT_FILES = {
    "genre_decay": "cranberry genre decay table.json",
    "platform_weights": "platform_allocation_weights.json",
    "platform_exclusivity": "platform_exclusivity_patterns.json",
    "country_weights": "country_viewership_weights_2025.json",
    "watchtime_config": "component_WatchTime_to_Views_Config.json",  # v2.0 content-aware
}

# Platform subscriber counts (Q4 2025) for sanity checks
PLATFORM_SUBSCRIBERS_Q4_2025 = {
    "netflix": {"global_M": 325, "us_M": 92},
    "disney_plus": {"global_M": 125, "us_M": 57},
    "prime_video": {"global_M": 200, "us_M": 110},
    "max": {"global_M": 128, "us_M": 74},
    "hbo_max": {"global_M": 128, "us_M": 74},
    "paramount_plus": {"global_M": 78, "us_M": 51},
    "hulu": {"global_M": 52, "us_M": 52},
    "peacock": {"global_M": 41, "us_M": 41},
    "apple_tv_plus": {"global_M": 50, "us_M": 25},
}

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("ALGO-95.2")


# =============================================================================
# GENRE DECAY PARAMETERS (from whitepaper Section 2.3)
# =============================================================================

@dataclass
class GenreDecayParams:
    """Genre-specific decay parameters for temporal modeling"""
    lambda_decay: float  # Decay rate coefficient (higher = faster decay)
    baseline: float      # Baseline retention factor B_g (long-tail viewing)

    @property
    def half_life_weeks(self) -> float:
        """Calculate half-life in weeks"""
        return math.log(2) / self.lambda_decay if self.lambda_decay > 0 else float('inf')


# Genre decay parameters from Table 11 in whitepaper
GENRE_DECAY_PARAMS: Dict[str, GenreDecayParams] = {
    "reality_tv": GenreDecayParams(lambda_decay=0.08, baseline=0.15),
    "documentary": GenreDecayParams(lambda_decay=0.12, baseline=0.10),
    "drama": GenreDecayParams(lambda_decay=0.15, baseline=0.05),
    "action": GenreDecayParams(lambda_decay=0.18, baseline=0.04),
    "comedy": GenreDecayParams(lambda_decay=0.22, baseline=0.03),
    "horror": GenreDecayParams(lambda_decay=0.25, baseline=0.02),
    # Extended genres with interpolated values
    "thriller": GenreDecayParams(lambda_decay=0.20, baseline=0.035),
    "sci_fi": GenreDecayParams(lambda_decay=0.17, baseline=0.045),
    "romance": GenreDecayParams(lambda_decay=0.19, baseline=0.04),
    "animation": GenreDecayParams(lambda_decay=0.14, baseline=0.08),
    "family": GenreDecayParams(lambda_decay=0.10, baseline=0.12),
    "crime": GenreDecayParams(lambda_decay=0.16, baseline=0.05),
    "mystery": GenreDecayParams(lambda_decay=0.18, baseline=0.04),
    "fantasy": GenreDecayParams(lambda_decay=0.16, baseline=0.06),
    "default": GenreDecayParams(lambda_decay=0.15, baseline=0.05),
}


# =============================================================================
# EVENT INTERFERENCE COEFFICIENTS (from Table 1 in whitepaper)
# =============================================================================

@dataclass
class EventParams:
    """Event interference parameters"""
    beta: float           # Interference coefficient (0 to 1)
    duration_hours: float # Duration of event
    platform_modifier: Optional[Dict[str, float]] = None  # Platform-specific adjustments


EVENT_INTERFERENCE: Dict[str, EventParams] = {
    "super_bowl": EventParams(beta=0.45, duration_hours=5.0,
                              platform_modifier={"sports": -0.20}),
    "fifa_world_cup_match": EventParams(beta=0.52, duration_hours=2.0),
    "fifa_world_cup_final": EventParams(beta=0.52, duration_hours=2.5),
    "olympics_opening": EventParams(beta=0.38, duration_hours=4.0,
                                   platform_modifier={"nbc": -0.15, "peacock": -0.15}),
    "presidential_election": EventParams(beta=0.28, duration_hours=6.0,
                                         platform_modifier={"news": 0.40}),
    "royal_wedding_uk": EventParams(beta=0.22, duration_hours=3.0),
    "academy_awards": EventParams(beta=0.18, duration_hours=4.0,
                                  platform_modifier={"film": -0.10}),
    "uefa_champions_final": EventParams(beta=0.41, duration_hours=2.5),
    "march_madness_final": EventParams(beta=0.33, duration_hours=3.0,
                                       platform_modifier={"sports": -0.18}),
    "new_years_eve": EventParams(beta=0.35, duration_hours=8.0),
    "christmas_day": EventParams(beta=0.29, duration_hours=24.0,
                                platform_modifier={"family": 0.15}),
}


# =============================================================================
# PLATFORM PARAMETERS
# =============================================================================

@dataclass
class PlatformParams:
    """Platform-specific parameters"""
    subscribers_millions: float
    mau_ratio: float  # Monthly Active Users / Subscribers
    elasticity_beta: float  # Promotion elasticity coefficient
    content_boost: float  # Original content boost factor


PLATFORM_PARAMS: Dict[str, PlatformParams] = {
    "netflix": PlatformParams(subscribers_millions=83.2, mau_ratio=0.65,
                              elasticity_beta=0.42, content_boost=1.40),
    "hulu": PlatformParams(subscribers_millions=51.1, mau_ratio=0.58,
                          elasticity_beta=0.55, content_boost=1.65),
    "disney_plus": PlatformParams(subscribers_millions=46.5, mau_ratio=0.71,
                                  elasticity_beta=0.68, content_boost=1.55),
    "max": PlatformParams(subscribers_millions=52.7, mau_ratio=0.52,
                         elasticity_beta=0.58, content_boost=1.35),
    "peacock": PlatformParams(subscribers_millions=33.0, mau_ratio=0.42,
                              elasticity_beta=0.72, content_boost=1.30),
    "paramount_plus": PlatformParams(subscribers_millions=27.9, mau_ratio=0.48,
                                     elasticity_beta=0.65, content_boost=1.28),
    "apple_tv_plus": PlatformParams(subscribers_millions=25.0, mau_ratio=0.55,
                                    elasticity_beta=1.23, content_boost=1.50),
    "prime_video": PlatformParams(subscribers_millions=200.0, mau_ratio=0.35,
                                  elasticity_beta=0.38, content_boost=1.25),
}

# Reference subscriber base for platform reach calculation
SUBSCRIBER_REFERENCE = 100.0  # millions
REACH_EXPONENT = 0.73  # Sub-linear scaling exponent


# =============================================================================
# ABSTRACT WEIGHTING COEFFICIENTS
# =============================================================================

@dataclass
class WeatherCoefficients:
    """Weather impact coefficients"""
    alpha_hdd: float = 0.00023    # Heating degree days (cold weather boost)
    alpha_cdd: float = -0.00018   # Cooling degree days (hot weather penalty)
    alpha_precip: float = 0.35     # Precipitation (rainy day boost)
    alpha_daylight: float = -0.04  # Daylight hours (longer days = less viewing)


@dataclass
class EconomicCoefficients:
    """Economic impact coefficients"""
    delta_cci: float = -0.0015     # Consumer Confidence Index (per point drop)
    delta_unemployment: float = 0.008  # Unemployment rate (per percentage point)
    delta_inflation: float = -0.0045   # Inflation (per percentage point CPI)


@dataclass
class HealthCoefficients:
    """Health crisis impact coefficients"""
    gamma_covid: float = 0.34      # COVID-19 epidemic boost
    gamma_flu: float = 0.12        # Seasonal flu boost


# =============================================================================
# DIRECTORY INTEGRATION
# =============================================================================

class DirectoryIntegration:
    """
    Loads configuration from the 14 ALGO ecosystem directories.

    Directory Mapping (from ALGO_95_2_DIRECTORY_MAPPING.md):
    - Schema: Data structure definitions (SCHEMA_V24.00.json)
    - Components: Lookup tables, weights, decay parameters
    - Abstract Data: 9 engines, 56 signals
    - VBUS: System bus, path resolution
    - MAPIE: Model accuracy, prediction calibration
    - Streamers By Country: Platform availability data
    """

    _instance = None
    _loaded = False

    def __init__(self):
        self.genre_decay_data: Dict[str, Any] = {}
        self.platform_weights_data: Dict[str, Any] = {}
        self.platform_market_share: Dict[str, Dict[str, float]] = {}
        self.genre_platform_affinity: Dict[str, Dict[str, float]] = {}
        self.country_weights: Dict[str, float] = {}
        self.platform_availability: Dict[str, List[str]] = {}
        self._vbus = None
        # v2.0: Content-aware view duration config
        self.watchtime_config: Dict[str, Any] = {}
        self.content_type_durations: Dict[str, Dict[str, Any]] = {}
        self.genre_durations: Dict[str, Dict[str, Any]] = {}
        self.title_overrides: Dict[str, Dict[str, Any]] = {}
        self.subscriber_sanity_checks: Dict[str, Any] = {}

    @classmethod
    def get_instance(cls) -> 'DirectoryIntegration':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_all(self) -> bool:
        """Load all configuration from ecosystem directories"""
        if self._loaded:
            return True

        success = True

        # Load genre decay from Components
        if not self._load_genre_decay():
            logger.warning("Could not load genre decay - using defaults")
            success = False

        # Load platform weights from Components
        if not self._load_platform_weights():
            logger.warning("Could not load platform weights - using defaults")
            success = False

        # Load streaming availability from Streamers By Country
        if not self._load_streaming_availability():
            logger.warning("Could not load streaming availability")
            success = False

        # v2.0: Load content-aware watchtime configuration
        if not self._load_watchtime_config():
            logger.warning("Could not load watchtime config - using default 15min")
            success = False

        self._loaded = True
        logger.info(f"Directory integration loaded: {success}")
        return success

    def _load_genre_decay(self) -> bool:
        """Load genre decay parameters from cranberry genre decay table.json"""
        try:
            filepath = DIRECTORY_PATHS["components"] / COMPONENT_FILES["genre_decay"]
            if not filepath.exists():
                return False

            with open(filepath, 'r', encoding='utf-8') as f:
                self.genre_decay_data = json.load(f)

            # Update global GENRE_DECAY_PARAMS with loaded data
            genres = self.genre_decay_data.get("genres", {})
            for genre_name, params in genres.items():
                # Convert daily lambda to weekly (multiply by 7)
                lambda_daily = params.get("lambda_daily", 0.015)
                lambda_weekly = lambda_daily * 7
                baseline = params.get("baseline_B", 0.05)

                # Normalize genre name
                genre_key = genre_name.lower().replace(" ", "_").replace("-", "_")

                GENRE_DECAY_PARAMS[genre_key] = GenreDecayParams(
                    lambda_decay=lambda_weekly,
                    baseline=baseline
                )

            logger.info(f"Loaded {len(genres)} genre decay parameters from {filepath.name}")
            return True

        except Exception as e:
            logger.error(f"Error loading genre decay: {e}")
            return False

    def _load_platform_weights(self) -> bool:
        """Load platform allocation weights from Components"""
        try:
            filepath = DIRECTORY_PATHS["components"] / COMPONENT_FILES["platform_weights"]
            if not filepath.exists():
                return False

            with open(filepath, 'r', encoding='utf-8') as f:
                self.platform_weights_data = json.load(f)

            # Extract platform market share by country
            market_share = self.platform_weights_data.get("platform_market_share_by_country", {})
            for country, data in market_share.items():
                if country.startswith("_"):
                    continue
                platforms = data.get("platforms", {})
                self.platform_market_share[country] = {}
                for platform, pdata in platforms.items():
                    if isinstance(pdata, dict):
                        self.platform_market_share[country][platform] = pdata.get("normalized_percent", 0) / 100.0
                    else:
                        self.platform_market_share[country][platform] = pdata / 100.0

            # Extract genre-platform affinity
            affinity = self.platform_weights_data.get("genre_platform_affinity", {})
            for genre, platforms in affinity.items():
                if genre.startswith("_"):
                    continue
                self.genre_platform_affinity[genre] = {}
                for platform, value in platforms.items():
                    if platform.startswith("_"):
                        continue
                    if isinstance(value, (int, float)):
                        self.genre_platform_affinity[genre][platform] = value

            logger.info(f"Loaded platform weights for {len(self.platform_market_share)} countries")
            return True

        except Exception as e:
            logger.error(f"Error loading platform weights: {e}")
            return False

    def _load_streaming_availability(self) -> bool:
        """Load streaming platform availability by country"""
        try:
            streamers_dir = DIRECTORY_PATHS["streamers_by_country"]
            if not streamers_dir.exists():
                return False

            # Look for streaming_lookup_*.json files
            lookup_files = list(streamers_dir.glob("streaming_lookup_*.json"))

            for filepath in lookup_files:
                country_code = filepath.stem.replace("streaming_lookup_", "").upper()
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Extract platform list from the 'platforms' key
                    if isinstance(data, dict):
                        # Structure: {"metadata": {...}, "platforms": {"netflix": {...}, ...}}
                        platforms_data = data.get("platforms", data)
                        platforms = [k for k in platforms_data.keys() if k != "metadata"]
                        self.platform_availability[country_code] = platforms

                except Exception as e:
                    logger.warning(f"Could not load {filepath.name}: {e}")

            logger.info(f"Loaded streaming availability for {len(self.platform_availability)} countries")
            return len(self.platform_availability) > 0

        except Exception as e:
            logger.error(f"Error loading streaming availability: {e}")
            return False

    def _load_watchtime_config(self) -> bool:
        """
        Load content-aware watchtime-to-views configuration (v2.0).

        This loads the component_WatchTime_to_Views_Config.json which contains:
        - content_type_durations: Duration by content type (film, tv_show, etc.)
        - genre_durations: Duration by genre (prestige_drama, kids_shows, etc.)
        - title_specific_overrides: Known titles with specific durations
        - subscriber_sanity_checks: Validation rules for impossible view counts
        """
        try:
            filepath = DIRECTORY_PATHS["components"] / "Weighters" / COMPONENT_FILES["watchtime_config"]
            if not filepath.exists():
                logger.warning(f"Watchtime config not found: {filepath}")
                return False

            with open(filepath, 'r', encoding='utf-8') as f:
                self.watchtime_config = json.load(f)

            # Extract content type durations
            self.content_type_durations = self.watchtime_config.get("content_type_durations", {})

            # Extract genre durations
            self.genre_durations = self.watchtime_config.get("genre_durations", {})

            # Extract title-specific overrides
            self.title_overrides = self.watchtime_config.get("title_specific_overrides", {})

            # Extract subscriber sanity checks
            self.subscriber_sanity_checks = self.watchtime_config.get("subscriber_sanity_checks", {})

            logger.info(f"Loaded watchtime config v2.0: {len(self.content_type_durations)} content types, "
                       f"{len(self.genre_durations)} genres, {len(self.title_overrides)} title overrides")
            return True

        except Exception as e:
            logger.error(f"Error loading watchtime config: {e}")
            return False

    def get_content_duration(self, title: str, content_type: str = "tv_show",
                            genre: Optional[str] = None) -> float:
        """
        Get content-aware view duration in minutes.

        Priority order:
        1. title_specific_overrides[title] -> exact title match
        2. genre_durations[genre] -> if genre provided
        3. content_type_durations[content_type] -> by content type
        4. Default: 34 minutes (weighted average)

        Args:
            title: Content title name
            content_type: Type of content (film, tv_show, etc.)
            genre: Optional genre for more specific lookup

        Returns:
            Effective view duration in minutes
        """
        # 1. Check title-specific override (exact match)
        if title in self.title_overrides:
            override = self.title_overrides[title]
            if isinstance(override, dict):
                return override.get("runtime_min", 34)
            return override

        # 1b. Check partial title match
        for override_title, override_data in self.title_overrides.items():
            if override_title in title:
                if isinstance(override_data, dict):
                    return override_data.get("runtime_min", 34)
                return override_data

        # 2. Check genre durations
        if genre:
            genre_key = genre.lower().replace(" ", "_").replace("-", "_")
            if genre_key in self.genre_durations:
                genre_data = self.genre_durations[genre_key]
                return genre_data.get("effective_view_duration_min",
                                     genre_data.get("avg_runtime_min", 34))

        # 3. Check content type durations
        content_key = content_type.lower().replace(" ", "_")
        if content_key in self.content_type_durations:
            ct_data = self.content_type_durations[content_key]
            return ct_data.get("effective_view_duration_min",
                              ct_data.get("avg_runtime_min", 34))

        # 4. Default fallback
        default_data = self.content_type_durations.get("default", {})
        return default_data.get("effective_view_duration_min", 34)

    def validate_subscriber_ratio(self, views: float, platform: str,
                                  region: str = "us", period: str = "weekly") -> Tuple[str, float]:
        """
        Validate view count against subscriber sanity checks.

        Args:
            views: Estimated view count
            platform: Platform name (netflix, disney_plus, etc.)
            region: Region code (us, global)
            period: Time period (weekly, monthly)

        Returns:
            Tuple of (status, ratio) where status is 'PASS', 'WARNING', or 'FAIL_IMPOSSIBLE'
        """
        platform_key = platform.lower().replace(" ", "_").replace("+", "_plus")

        # Get platform subscribers
        platform_subs = PLATFORM_SUBSCRIBERS_Q4_2025.get(platform_key, {})
        if not platform_subs:
            return ("UNKNOWN", 0.0)

        # Get subscriber count for region
        if region.lower() == "us":
            subs_millions = platform_subs.get("us_M", platform_subs.get("global_M", 100))
        else:
            subs_millions = platform_subs.get("global_M", 100)

        subs = subs_millions * 1_000_000

        # Calculate ratio
        ratio = views / subs if subs > 0 else 0.0

        # Get thresholds from config or use defaults
        sanity_config = self.subscriber_sanity_checks.get(f"max_views_per_subscriber_{period}", {})
        warning_threshold = sanity_config.get("warning_threshold", 0.8)
        max_threshold = sanity_config.get("absolute_max", 2.0)

        # Determine status
        if ratio <= warning_threshold:
            return ("PASS", ratio)
        elif ratio <= max_threshold:
            return ("WARNING", ratio)
        else:
            return ("FAIL_IMPOSSIBLE", ratio)

    def get_genre_decay(self, genre: str) -> GenreDecayParams:
        """Get genre decay parameters (from loaded data or defaults)"""
        genre_key = genre.lower().replace(" ", "_").replace("-", "_")
        return GENRE_DECAY_PARAMS.get(genre_key, GENRE_DECAY_PARAMS.get("default"))

    def get_platform_share(self, country: str, platform: str) -> float:
        """Get platform market share for country"""
        country_upper = country.upper()
        platform_lower = platform.lower()

        if country_upper in self.platform_market_share:
            return self.platform_market_share[country_upper].get(platform_lower, 0.10)

        # Fall back to ROW (Rest of World)
        if "ROW" in self.platform_market_share:
            return self.platform_market_share["ROW"].get(platform_lower, 0.10)

        return 0.10  # Default 10% share

    def get_genre_affinity(self, genre: str, platform: str) -> float:
        """Get genre-platform affinity multiplier"""
        genre_key = genre.lower().replace(" ", "_").replace("-", "_")
        platform_lower = platform.lower()

        if genre_key in self.genre_platform_affinity:
            affinity = self.genre_platform_affinity[genre_key]
            return affinity.get(platform_lower, affinity.get("_default", 1.0))

        return 1.0  # Neutral affinity

    def is_platform_available(self, country: str, platform: str) -> bool:
        """Check if platform is available in country"""
        country_upper = country.upper()
        platform_lower = platform.lower()

        if country_upper in self.platform_availability:
            return platform_lower in [p.lower() for p in self.platform_availability[country_upper]]

        # Default to available if no data
        return True


# Global directory integration instance
def get_directory_integration() -> DirectoryIntegration:
    """Get the directory integration singleton and ensure it's loaded"""
    di = DirectoryIntegration.get_instance()
    if not di._loaded:
        di.load_all()
    return di


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ContentTitle:
    """Content title with all features for prediction"""
    title_id: str
    title: str
    content_type: str  # "movie" or "series"
    genre: str
    platform: str
    release_date: Optional[datetime] = None
    runtime_minutes: Optional[float] = None
    imdb_rating: Optional[float] = None
    imdb_votes: Optional[int] = None
    is_original: bool = False
    is_exclusive: bool = False
    season_number: Optional[int] = None

    # Computed signals (filled by engine)
    content_quality_score: float = 1.0
    marketing_awareness: float = 1.0
    platform_boost: float = 1.0
    temporal_decay: float = 1.0


@dataclass
class EnvironmentalContext:
    """Environmental and contextual data for a region and time"""
    region: str
    date: datetime

    # Weather data
    heating_degree_days: float = 0.0
    cooling_degree_days: float = 0.0
    precipitation_ratio: float = 0.0  # Days with precip / total days
    avg_daylight_hours: float = 12.0

    # Events
    active_events: List[str] = field(default_factory=list)

    # Health
    epidemic_severity: float = 0.0  # 0-1 scale
    quarantine_stringency: float = 0.0  # 0-1 scale
    flu_severity: float = 0.0

    # Economic
    cci_change: float = 0.0  # Consumer Confidence Index delta
    unemployment_change: float = 0.0  # Percentage point change
    inflation_rate: float = 0.0  # CPI percentage

    # Geopolitical
    geopolitical_risk_index: float = 0.0  # 0-100 scale

    # Quality of Experience
    four_k_share: float = 0.35  # Fraction of streams at 4K+


@dataclass
class PredictionResult:
    """Result of viewership prediction"""
    title_id: str
    title: str
    platform: str
    predicted_views: float
    predicted_hours: float

    # Component multipliers
    base_views: float
    completion_quality: float
    geopolitical_factor: float
    qoe_factor: float
    availability_factor: float
    abstract_adjustment: float

    # Confidence
    confidence_interval_low: float = 0.0
    confidence_interval_high: float = 0.0

    # Metadata
    prediction_date: str = ""
    target_quarter: str = ""


# =============================================================================
# ALGO-95.2 CORE EQUATIONS
# =============================================================================

class ALGO952Equations:
    """
    Core mathematical equations for ALGO-95.2.

    All equations are implemented according to the whitepaper specification.
    """

    @staticmethod
    def temporal_decay(weeks_since_release: float, genre: str) -> float:
        """
        Equation 8: Genre-specific temporal decay

        D(t,g) = B_g + (1 - B_g) × exp(-λ_g × t)

        Args:
            weeks_since_release: Time since release in weeks
            genre: Genre category (normalized to lowercase with underscores)

        Returns:
            Decay factor [B_g, 1.0]
        """
        genre_key = genre.lower().replace(" ", "_").replace("-", "_")
        params = GENRE_DECAY_PARAMS.get(genre_key, GENRE_DECAY_PARAMS["default"])

        decay = params.baseline + (1 - params.baseline) * math.exp(
            -params.lambda_decay * weeks_since_release
        )
        return decay

    @staticmethod
    def completion_quality_index(cr90: float, session_ratio: float) -> float:
        """
        Section 2: Completion Quality Index R_i

        R_i = 1 + 0.6 × (CR90,i - 0.50) + 0.4 × (SR_i - 0.45)

        Bounded: R_i^95 = clip[0.25, 4.00](R_i)

        Args:
            cr90: Fraction of viewers reaching ≥90% runtime
            session_ratio: Average session duration / runtime

        Returns:
            Completion quality factor [0.25, 4.00]
        """
        r_i = 1.0 + 0.6 * (cr90 - 0.50) + 0.4 * (session_ratio - 0.45)
        return max(0.25, min(4.00, r_i))

    @staticmethod
    def geopolitical_risk(gpri: float) -> float:
        """
        Section 3: Geopolitical Risk Multiplier G_d,t

        G_d,t = 1 - 0.42 × min(1, GPRI_d,t / 60)
        Floor: G_d,t ≥ 0.58

        Args:
            gpri: Geopolitical Risk Index (0-100)

        Returns:
            Geopolitical factor [0.58, 1.00]
        """
        g_d = 1.0 - 0.42 * min(1.0, gpri / 60.0)
        return max(0.58, g_d)

    @staticmethod
    def quality_of_experience(four_k_share: float,
                             bitrate_mbps: Optional[float] = None,
                             buffer_ratio: Optional[float] = None) -> float:
        """
        Section 4: Quality of Experience Q_d,t

        Primary: Q_d,t = 1 + 0.25 × (4K_share,d,t - 0.35)

        Extended (if telemetry available):
        Q_d,t^95 = 1 + 0.25(4K - 0.35) + 0.10(bitrate/15 - 0.5) - 0.15(buffer - 0.01)

        Args:
            four_k_share: Fraction of streams at 4K+
            bitrate_mbps: Average bitrate in Mbps (optional)
            buffer_ratio: Buffer ratio (optional)

        Returns:
            QoE factor
        """
        q_d = 1.0 + 0.25 * (four_k_share - 0.35)

        if bitrate_mbps is not None and buffer_ratio is not None:
            q_d += 0.10 * (bitrate_mbps / 15.0 - 0.5)
            q_d -= 0.15 * (buffer_ratio - 0.01)

        return q_d

    @staticmethod
    def platform_reach(subscribers_millions: float) -> float:
        """
        Platform reach factor with sub-linear scaling.

        P_reach = (S / S_ref)^β

        Where:
            S_ref = 100M (reference baseline)
            β = 0.73 (sub-linear scaling)

        Args:
            subscribers_millions: Platform subscribers in millions

        Returns:
            Platform reach factor
        """
        return math.pow(subscribers_millions / SUBSCRIBER_REFERENCE, REACH_EXPONENT)

    @staticmethod
    def exclusivity_premium(is_exclusive: bool,
                           is_primary_region: bool = True,
                           platforms_available: int = 1) -> float:
        """
        Exclusivity premium factor E_exclusive.

        1.45 if exclusive to platform in all regions
        1.28 if exclusive in primary region
        1.12 if exclusive in some regions
        1.00 if available on 2+ platforms

        Args:
            is_exclusive: Whether content is exclusive
            is_primary_region: Whether exclusive in primary region
            platforms_available: Number of platforms content is on

        Returns:
            Exclusivity factor [1.00, 1.45]
        """
        if is_exclusive and platforms_available == 1:
            return 1.45
        elif is_exclusive and is_primary_region:
            return 1.28
        elif is_exclusive:
            return 1.12
        return 1.00

    @staticmethod
    def weather_weight(hdd: float, cdd: float, precip_ratio: float,
                       daylight_hours: float,
                       coefficients: Optional[WeatherCoefficients] = None) -> float:
        """
        Equation 3: Weather Impact Model W_weather

        W_weather = 1 + α_HDD×HDD + α_CDD×CDD + α_precip×P + α_daylight×D

        Args:
            hdd: Heating degree days for quarter
            cdd: Cooling degree days for quarter
            precip_ratio: Days with precipitation / total days
            daylight_hours: Average daily daylight hours
            coefficients: Optional custom coefficients

        Returns:
            Weather adjustment factor
        """
        c = coefficients or WeatherCoefficients()

        w = 1.0
        w += c.alpha_hdd * hdd
        w += c.alpha_cdd * cdd
        w += c.alpha_precip * precip_ratio
        w += c.alpha_daylight * (daylight_hours - 12.0)  # Normalized to 12 hours

        return w

    @staticmethod
    def event_weight(active_events: List[str],
                    platform: Optional[str] = None,
                    overlap_attenuation: float = 0.85) -> float:
        """
        Equation 4: Major Event Impact W_events

        W_events = Π[1 - β_e × I_e(t) × C_overlap(t)]

        Args:
            active_events: List of active event names
            platform: Platform name for platform-specific adjustments
            overlap_attenuation: Concurrent event attenuation factor

        Returns:
            Event adjustment factor
        """
        if not active_events:
            return 1.0

        w = 1.0

        for event in active_events:
            event_key = event.lower().replace(" ", "_").replace("-", "_")
            params = EVENT_INTERFERENCE.get(event_key)

            if params:
                beta = params.beta

                # Apply platform-specific modifier if available
                if platform and params.platform_modifier:
                    platform_key = platform.lower().replace("_", "")
                    for p_key, modifier in params.platform_modifier.items():
                        if p_key in platform_key:
                            beta = beta * (1 + modifier)
                            break

                w *= (1 - beta)

        # Apply overlap attenuation for multiple events
        if len(active_events) > 1:
            w *= overlap_attenuation

        return w

    @staticmethod
    def health_weight(epidemic_severity: float, quarantine_stringency: float,
                     flu_severity: float = 0.0,
                     coefficients: Optional[HealthCoefficients] = None) -> float:
        """
        Equation 5: Epidemic Response Function W_health

        W_health = 1 + γ_epidemic × S(r,t) × Q(r,t)

        Args:
            epidemic_severity: Epidemic severity index (0-1)
            quarantine_stringency: Quarantine/lockdown stringency (0-1)
            flu_severity: Seasonal flu severity ratio
            coefficients: Optional custom coefficients

        Returns:
            Health adjustment factor
        """
        c = coefficients or HealthCoefficients()

        w = 1.0

        # COVID/pandemic response
        w += c.gamma_covid * epidemic_severity * quarantine_stringency

        # Seasonal flu
        if flu_severity > 0:
            w += c.gamma_flu * math.log(1 + flu_severity)

        return w

    @staticmethod
    def economic_weight(cci_change: float, unemployment_change: float,
                       inflation_rate: float,
                       coefficients: Optional[EconomicCoefficients] = None) -> float:
        """
        Equation 6: Economic Adjustment W_econ

        W_econ = 1 + δ_CCI×ΔCCI + δ_unemp×ΔUnemployment + δ_inflation×ΔInflation

        Args:
            cci_change: Consumer Confidence Index change
            unemployment_change: Unemployment rate change (percentage points)
            inflation_rate: Inflation rate (CPI percentage)
            coefficients: Optional custom coefficients

        Returns:
            Economic adjustment factor
        """
        c = coefficients or EconomicCoefficients()

        w = 1.0
        w += c.delta_cci * cci_change
        w += c.delta_unemployment * unemployment_change
        w += c.delta_inflation * inflation_rate

        return w

    @staticmethod
    def hours_to_views(total_hours: float, runtime_minutes: float) -> float:
        """
        Equation 9: Hours-to-Views Conversion

        Views = Total_Hours_Viewed / Runtime(hours)

        Args:
            total_hours: Total hours viewed
            runtime_minutes: Content runtime in minutes

        Returns:
            Equivalent view count
        """
        if runtime_minutes <= 0:
            return 0.0
        runtime_hours = runtime_minutes / 60.0
        return total_hours / runtime_hours

    @staticmethod
    def minutes_to_views_content_aware(total_minutes: float, effective_duration: float) -> float:
        """
        Equation 9b (v2.0): Content-Aware Minutes-to-Views Conversion

        Views = Total_Minutes_Viewed / Effective_View_Duration

        This replaces the legacy flat 15-minute assumption with content-aware
        durations based on actual episode lengths.

        Args:
            total_minutes: Total minutes viewed (often in millions from Nielsen)
            effective_duration: Content-specific view duration in minutes
                              (from get_content_duration())

        Returns:
            Equivalent view count

        Example:
            Stranger Things: 2375M minutes / 60 min = 39.6M views
            Bluey: 762M minutes / 7 min = 108.9M views
            Movie: 1100M minutes / 79 min = 13.9M views
        """
        if effective_duration <= 0:
            effective_duration = 34.0  # Default fallback
        return total_minutes / effective_duration

    @staticmethod
    def minutes_to_views_legacy(total_minutes: float) -> float:
        """
        DEPRECATED: Legacy flat 15-minute assumption.

        This method is preserved for backward compatibility but should not
        be used for new calculations. Use minutes_to_views_content_aware instead.

        WARNING: This causes systematic inflation for prestige TV (4x) and
        movies (7x), and under-counts kids content (0.5x).
        """
        return total_minutes / 15.0


# =============================================================================
# ALGO-95.2 PREDICTION ENGINE
# =============================================================================

class ALGO952Engine:
    """
    ALGO-95.2 Viewership Prediction Engine.

    Implements the full ALGO-95.2 prediction pipeline:
    1. Base prediction (ALGO-65.2 compatible)
    2. Extended modifiers (R_i, G_d, Q_d, A_i,p,t)
    3. Abstract environmental weighting

    Directory Integration:
    - Loads genre decay from Components/cranberry genre decay table.json
    - Loads platform weights from Components/platform_allocation_weights.json
    - Loads streaming availability from Streamers By Country/
    """

    VERSION = "95.2.1"  # v2.0 content-aware duration support added 2026-01-21
    TARGET_MAPE = 2.0  # Target: ≤2% Mean Absolute Percentage Error

    def __init__(self, config_path: Optional[Path] = None, load_directories: bool = True):
        """
        Initialize the ALGO-95.2 engine.

        Args:
            config_path: Path to configuration JSON file
            load_directories: Whether to load configuration from ecosystem directories
        """
        self.equations = ALGO952Equations()
        self.config = self._load_config(config_path)
        self.run_timestamp = datetime.now(timezone.utc)

        # Directory integration (loads from Components, Streamers By Country, etc.)
        self.directory_integration: Optional[DirectoryIntegration] = None
        if load_directories:
            try:
                self.directory_integration = get_directory_integration()
                logger.info("Directory integration loaded successfully")
            except Exception as e:
                logger.warning(f"Directory integration failed: {e} - using defaults")

        logger.info(f"ALGO-95.2 Engine v{self.VERSION} initialized")

    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load config from {config_path}: {e}")
        return {}

    def calculate_abstract_adjustment(self, context: EnvironmentalContext) -> float:
        """
        Calculate the abstract environmental adjustment factor A(r,t).

        A(r,t) = W_weather × W_events × W_health × W_econ

        Args:
            context: Environmental context data

        Returns:
            Combined abstract adjustment factor
        """
        # Weather weight
        w_weather = self.equations.weather_weight(
            hdd=context.heating_degree_days,
            cdd=context.cooling_degree_days,
            precip_ratio=context.precipitation_ratio,
            daylight_hours=context.avg_daylight_hours
        )

        # Event weight
        w_events = self.equations.event_weight(
            active_events=context.active_events
        )

        # Health weight
        w_health = self.equations.health_weight(
            epidemic_severity=context.epidemic_severity,
            quarantine_stringency=context.quarantine_stringency,
            flu_severity=context.flu_severity
        )

        # Economic weight
        w_econ = self.equations.economic_weight(
            cci_change=context.cci_change,
            unemployment_change=context.unemployment_change,
            inflation_rate=context.inflation_rate
        )

        return w_weather * w_events * w_health * w_econ

    def estimate_views_from_minutes(self, total_minutes: float, title: str,
                                     content_type: str = "tv_show",
                                     genre: Optional[str] = None,
                                     platform: Optional[str] = None,
                                     validate: bool = True) -> Dict[str, Any]:
        """
        Convert watch time minutes to view count using content-aware methodology (v2.0).

        This method replaces the legacy flat 15-minute assumption with intelligent
        duration lookup based on title, content type, and genre.

        Args:
            total_minutes: Total minutes viewed (e.g., 2375000000 for 2375M)
            title: Content title name (for title-specific override lookup)
            content_type: Type of content (film, tv_show, kids_animation, etc.)
            genre: Optional genre for more specific duration lookup
            platform: Optional platform for subscriber ratio validation

        Returns:
            Dict containing:
            - views: Estimated view count
            - duration_used: The effective duration used (minutes)
            - method: 'title_override', 'genre', 'content_type', or 'default'
            - legacy_views: What the old 15-min method would have calculated
            - inflation_factor: legacy_views / views (>1 means legacy was inflated)
            - sanity_check: (status, ratio) if platform provided

        Example:
            >>> engine.estimate_views_from_minutes(
            ...     total_minutes=2375000000,  # 2375M for Stranger Things
            ...     title="Stranger Things",
            ...     content_type="tv_show",
            ...     genre="prestige_drama",
            ...     platform="netflix"
            ... )
            {'views': 39583333, 'duration_used': 60, 'method': 'title_override',
             'legacy_views': 158333333, 'inflation_factor': 4.0,
             'sanity_check': ('PASS', 0.43)}
        """
        # Get content-aware duration
        duration = 34.0  # Default
        method = "default"

        if self.directory_integration:
            # Check title override first
            if title in self.directory_integration.title_overrides:
                override = self.directory_integration.title_overrides[title]
                duration = override.get("runtime_min", 34) if isinstance(override, dict) else override
                method = "title_override"
            else:
                # Check partial title match
                for override_title, override_data in self.directory_integration.title_overrides.items():
                    if override_title in title:
                        duration = override_data.get("runtime_min", 34) if isinstance(override_data, dict) else override_data
                        method = "title_override"
                        break

                # If no title match, try genre
                if method == "default" and genre:
                    genre_key = genre.lower().replace(" ", "_").replace("-", "_")
                    if genre_key in self.directory_integration.genre_durations:
                        gd = self.directory_integration.genre_durations[genre_key]
                        duration = gd.get("effective_view_duration_min", gd.get("avg_runtime_min", 34))
                        method = "genre"

                # If still default, try content type
                if method == "default":
                    ct_key = content_type.lower().replace(" ", "_")
                    if ct_key in self.directory_integration.content_type_durations:
                        ctd = self.directory_integration.content_type_durations[ct_key]
                        duration = ctd.get("effective_view_duration_min", ctd.get("avg_runtime_min", 34))
                        method = "content_type"

        # Calculate views
        views = self.equations.minutes_to_views_content_aware(total_minutes, duration)
        legacy_views = self.equations.minutes_to_views_legacy(total_minutes)

        result = {
            "views": int(views),
            "duration_used": duration,
            "method": method,
            "legacy_views": int(legacy_views),
            "inflation_factor": round(legacy_views / views, 2) if views > 0 else 0,
        }

        # Validate against subscriber counts if platform provided
        if validate and platform and self.directory_integration:
            sanity_status, sanity_ratio = self.directory_integration.validate_subscriber_ratio(
                views, platform, region="us", period="weekly"
            )
            result["sanity_check"] = (sanity_status, round(sanity_ratio, 2))

        return result

    def calculate_platform_availability(self,
                                        title: ContentTitle,
                                        platform: str,
                                        country: str = "US") -> float:
        """
        Calculate platform availability factor A_i,p,t.

        A_i,p,t = P_reach^α × E_exclusive × T_tenure × D_competition × Genre_Affinity

        Integrates with:
        - Components/platform_allocation_weights.json for market share
        - Components/cranberry genre decay table.json for genre parameters
        - Streamers By Country/ for platform availability

        Args:
            title: Content title data
            platform: Platform name
            country: Target country code (default: US)

        Returns:
            Platform availability factor
        """
        platform_key = platform.lower().replace(" ", "_").replace("+", "_plus")
        params = PLATFORM_PARAMS.get(platform_key)

        if not params:
            logger.warning(f"Unknown platform: {platform}, using default")
            return 1.0

        # Platform reach
        p_reach = self.equations.platform_reach(params.subscribers_millions)

        # Exclusivity premium
        e_exclusive = self.equations.exclusivity_premium(
            is_exclusive=title.is_exclusive,
            platforms_available=1 if title.is_exclusive else 2
        )

        # Original content boost
        content_boost = params.content_boost if title.is_original else 1.0

        # Genre-platform affinity (from directory integration)
        genre_affinity = 1.0
        if self.directory_integration:
            genre_affinity = self.directory_integration.get_genre_affinity(
                title.genre, platform
            )

            # Check platform availability in country
            if not self.directory_integration.is_platform_available(country, platform):
                logger.info(f"Platform {platform} not available in {country}")
                return 0.0  # Not available

        return p_reach * e_exclusive * content_boost * genre_affinity

    def predict_views(self,
                     title: ContentTitle,
                     context: EnvironmentalContext,
                     base_views: float,
                     cr90: float = 0.50,
                     session_ratio: float = 0.45) -> PredictionResult:
        """
        Generate viewership prediction using ALGO-95.2.

        V̂^95 = V̂_base × R_i × G_d × Q_d × A_i,p,t × A(r,t)

        Integrates with ecosystem directories:
        - Components/cranberry genre decay table.json for T_i,d temporal decay
        - Components/platform_allocation_weights.json for platform factors
        - Streamers By Country/ for availability

        Args:
            title: Content title data
            context: Environmental context
            base_views: Base view prediction from ALGO-65.2 or prior model
            cr90: Completion rate (fraction reaching 90% runtime)
            session_ratio: Session duration / runtime ratio

        Returns:
            PredictionResult with predicted views and component factors
        """
        # 1. Completion Quality Index (R_i)
        r_i = self.equations.completion_quality_index(cr90, session_ratio)

        # 2. Geopolitical Risk (G_d)
        g_d = self.equations.geopolitical_risk(context.geopolitical_risk_index)

        # 3. Quality of Experience (Q_d)
        q_d = self.equations.quality_of_experience(context.four_k_share)

        # 4. Platform Availability (A_i,p,t) - now with country/region support
        a_ipt = self.calculate_platform_availability(
            title, title.platform, country=context.region
        )

        # 5. Abstract Adjustment (A(r,t))
        a_rt = self.calculate_abstract_adjustment(context)

        # 6. Calculate weeks since release for temporal decay
        # Uses genre decay parameters loaded from Components directory
        if title.release_date:
            weeks_since = (context.date - title.release_date).days / 7.0
            temporal_factor = self.equations.temporal_decay(weeks_since, title.genre)
        else:
            temporal_factor = 1.0

        # Final prediction
        predicted_views = base_views * r_i * g_d * q_d * a_ipt * a_rt * temporal_factor

        # Calculate hours from views (inverse of hours_to_views)
        runtime = title.runtime_minutes or 90.0  # Default 90 min
        predicted_hours = predicted_views * (runtime / 60.0)

        # Simple confidence interval (±5% for normal conditions)
        ci_margin = 0.05 * predicted_views

        return PredictionResult(
            title_id=title.title_id,
            title=title.title,
            platform=title.platform,
            predicted_views=predicted_views,
            predicted_hours=predicted_hours,
            base_views=base_views,
            completion_quality=r_i,
            geopolitical_factor=g_d,
            qoe_factor=q_d,
            availability_factor=a_ipt,
            abstract_adjustment=a_rt * temporal_factor,
            confidence_interval_low=predicted_views - ci_margin,
            confidence_interval_high=predicted_views + ci_margin,
            prediction_date=self.run_timestamp.isoformat(),
            target_quarter=f"Q{(context.date.month - 1) // 3 + 1} {context.date.year}"
        )

    def batch_predict(self,
                     titles: List[ContentTitle],
                     context: EnvironmentalContext,
                     base_views_map: Dict[str, float]) -> List[PredictionResult]:
        """
        Generate batch predictions for multiple titles.

        Args:
            titles: List of content titles
            context: Environmental context (same for all)
            base_views_map: Map of title_id -> base views

        Returns:
            List of PredictionResults
        """
        results = []

        for title in titles:
            base_views = base_views_map.get(title.title_id, 0.0)
            if base_views > 0:
                result = self.predict_views(title, context, base_views)
                results.append(result)

        return results

    def generate_report(self,
                       results: List[PredictionResult],
                       output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate prediction report.

        Args:
            results: List of prediction results
            output_path: Optional path to save report

        Returns:
            Report dictionary
        """
        total_views = sum(r.predicted_views for r in results)
        total_hours = sum(r.predicted_hours for r in results)

        report = {
            "algo_version": self.VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_titles": len(results),
                "total_predicted_views": total_views,
                "total_predicted_hours": total_hours,
                "average_views_per_title": total_views / len(results) if results else 0,
            },
            "predictions": [asdict(r) for r in results],
            "model_parameters": {
                "target_mape": self.TARGET_MAPE,
                "genres_supported": list(GENRE_DECAY_PARAMS.keys()),
                "platforms_supported": list(PLATFORM_PARAMS.keys()),
            }
        }

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to {output_path}")

        return report


# =============================================================================
# CLI INTERFACE
# =============================================================================

def print_banner():
    """Print ALGO-95.2 banner"""
    banner = """
================================================================================
       _    _     ____  ___         ___  ____   ____
      / \\  | |   / ___|/ _ \\       / _ \\| ___| |___ \\
     / _ \\ | |  | |  _| | | |_____| (_) |___ \\   __) |
    / ___ \\| |__| |_| | |_| |_____|  _  |___) | / __/
   /_/   \\_\\_____\\____|\\___/       |_| |_|____(_)_____|

                 VIEWERSHIP PREDICTION ENGINE v95.2
          Context-Aware Streaming Prediction with Abstract Weighting
================================================================================
    Target Accuracy: >=98% (<=2.0% MAPE)
    Author: Roy Taylor / Framecore
    Status: Production-Grade
================================================================================
"""
    print(banner)


def run_demo():
    """Run demonstration with sample data"""
    print_banner()

    # Initialize engine with directory integration
    print("\n" + "="*60)
    print("DIRECTORY INTEGRATION")
    print("="*60)

    engine = ALGO952Engine()

    # Show loaded configuration
    if engine.directory_integration:
        di = engine.directory_integration
        print(f"\n  Genre Decay Parameters:    {len(GENRE_DECAY_PARAMS)} genres loaded")
        print(f"  Platform Market Share:     {len(di.platform_market_share)} countries")
        print(f"  Genre-Platform Affinity:   {len(di.genre_platform_affinity)} genres")
        print(f"  Streaming Availability:    {len(di.platform_availability)} country files")

        # Show source files
        print("\n  Source Files:")
        print(f"    - Components/{COMPONENT_FILES['genre_decay']}")
        print(f"    - Components/{COMPONENT_FILES['platform_weights']}")
        print(f"    - Streamers By Country/streaming_lookup_*.json")
    else:
        print("\n  Using default configuration (directory integration not loaded)")

    # Create sample title
    title = ContentTitle(
        title_id="demo_001",
        title="Demo Series",
        content_type="series",
        genre="drama",
        platform="netflix",
        release_date=datetime(2025, 10, 1),
        runtime_minutes=50,
        imdb_rating=8.2,
        imdb_votes=125000,
        is_original=True,
        is_exclusive=True
    )

    # Create sample context (Q1 2026)
    context = EnvironmentalContext(
        region="US",
        date=datetime(2026, 1, 15),
        heating_degree_days=800,
        cooling_degree_days=0,
        precipitation_ratio=0.35,
        avg_daylight_hours=10.0,
        active_events=[],
        epidemic_severity=0.0,
        quarantine_stringency=0.0,
        flu_severity=1.2,
        cci_change=-5,
        unemployment_change=0.2,
        inflation_rate=3.5,
        geopolitical_risk_index=25,
        four_k_share=0.42
    )

    # Generate prediction
    base_views = 5_000_000  # Base prediction of 5M views
    result = engine.predict_views(title, context, base_views)

    # Print results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"\n  Title:              {result.title}")
    print(f"  Platform:           {result.platform}")
    print(f"  Target Quarter:     {result.target_quarter}")
    print(f"\n  Base Views:         {result.base_views:,.0f}")
    print(f"  Predicted Views:    {result.predicted_views:,.0f}")
    print(f"  Predicted Hours:    {result.predicted_hours:,.0f}")

    print(f"\n  Component Factors:")
    print(f"    Completion Quality (R_i):     {result.completion_quality:.4f}")
    print(f"    Geopolitical Risk (G_d):      {result.geopolitical_factor:.4f}")
    print(f"    Quality of Experience (Q_d):  {result.qoe_factor:.4f}")
    print(f"    Platform Availability (A):    {result.availability_factor:.4f}")
    print(f"    Abstract Adjustment:          {result.abstract_adjustment:.4f}")

    print(f"\n  Confidence Interval:")
    print(f"    Low:  {result.confidence_interval_low:,.0f}")
    print(f"    High: {result.confidence_interval_high:,.0f}")

    print("\n" + "="*60)

    # Save report
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"ALGO_95_2_DEMO_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    report = engine.generate_report([result], report_path)
    print(f"\n  Report saved to: {report_path}")

    return result


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="ALGO-95.2 Viewership Prediction Engine"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration with sample data"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for reports"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.demo:
        run_demo()
    else:
        print_banner()
        print("\nUsage: python ALGO_95_2.py --demo")
        print("\nFor full pipeline integration, import and use ALGO952Engine class.")
        print("\nExample:")
        print("  from ALGO_95_2 import ALGO952Engine, ContentTitle, EnvironmentalContext")
        print("  engine = ALGO952Engine()")
        print("  result = engine.predict_views(title, context, base_views)")


if __name__ == "__main__":
    main()
