"""
Dynamic filter helpers for extracting driver, constructor, and race names
from local CSV data. Designed to keep query parsing aligned with the dataset.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
import logging
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import chardet
import pandas as pd


LOGGER = logging.getLogger(__name__)
_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

_CIRCUIT_PREFIXES = [
    ("circuit", "de"),
    ("circuit", "of", "the"),
    ("circuit", "of"),
    ("autodromo", "de"),
    ("autodromo", "di"),
    ("autodromo",),
]
_CIRCUIT_SUFFIXES = [
    ("grand", "prix", "circuit"),
    ("international", "circuit"),
    ("city", "circuit"),
    ("motor", "speedway"),
    ("motorsport", "park"),
    ("raceway",),
    ("speedway",),
    ("circuit",),
    ("track",),
    ("ring",),
]


@dataclass(frozen=True)
class FilterData:
    driver_full_names: List[str]
    driver_surnames: List[str]
    constructor_names: List[str]
    race_names: List[str]
    circuit_names: List[str]


@dataclass(frozen=True)
class FilterLookups:
    driver_full_names: Dict[str, str]
    driver_surnames: Dict[str, str]
    constructor_names: Dict[str, str]
    race_names: Dict[str, str]
    circuit_names: Dict[str, str]


def _normalize(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"\bgp\b", "grand prix", text)
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _contains_phrase(haystack: str, needle: str) -> bool:
    """Check whole-phrase containment with simple space boundaries."""
    return f" {needle} " in f" {haystack} "


def read_csv_with_encoding(file_path: str) -> pd.DataFrame:
    """Read CSV with automatic encoding detection."""
    with open(file_path, "rb") as handle:
        result = chardet.detect(handle.read())
    encoding = result.get("encoding") or "utf-8"
    confidence = result.get("confidence", 0.0)
    LOGGER.debug(
        "Reading %s as %s (%.2f confidence)",
        os.path.basename(file_path),
        encoding,
        confidence,
    )
    return pd.read_csv(file_path, encoding=encoding)


def _read_csv_dicts(path: str) -> Iterator[Dict[str, object]]:
    """Read CSV rows with automatic encoding detection."""
    df = read_csv_with_encoding(path)
    df.columns = df.columns.str.strip()
    for row in df.to_dict(orient="records"):
        yield {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}


@lru_cache(maxsize=1)
def load_filter_data(data_dir: str = _DEFAULT_DATA_DIR) -> FilterData:
    """Load driver, constructor, race, and circuit names from CSVs."""
    base_dir = Path(data_dir)
    drivers_path = base_dir / "drivers.csv"
    constructors_path = base_dir / "constructors.csv"
    races_path = base_dir / "races.csv"
    circuits_path = base_dir / "circuits.csv"

    driver_full_names = []
    driver_surnames = []
    for row in _read_csv_dicts(str(drivers_path)):
        given = row.get("forename", "")
        family = row.get("surname", "")
        if given and family:
            full = f"{given} {family}"
            driver_full_names.append(full)
            driver_surnames.append(family)

    constructor_names = []
    for row in _read_csv_dicts(str(constructors_path)):
        name = row.get("name", "")
        if name:
            constructor_names.append(name)

    race_names = []
    for row in _read_csv_dicts(str(races_path)):
        name = row.get("name", "")
        if name:
            race_names.append(name)

    circuit_names = []
    for row in _read_csv_dicts(str(circuits_path)):
        name = row.get("name", "")
        if name:
            circuit_names.append(name)

    return FilterData(
        driver_full_names=driver_full_names,
        driver_surnames=driver_surnames,
        constructor_names=constructor_names,
        race_names=race_names,
        circuit_names=circuit_names,
    )


@lru_cache(maxsize=1)
def _load_filter_lookups(data_dir: str = _DEFAULT_DATA_DIR) -> FilterLookups:
    """Load and normalize lookups for fast matching."""
    data = load_filter_data(data_dir)
    return FilterLookups(
        driver_full_names=_build_lookup(data.driver_full_names),
        driver_surnames=_build_lookup(data.driver_surnames),
        constructor_names=_build_lookup(data.constructor_names),
        race_names=_build_lookup(data.race_names),
        circuit_names=_build_circuit_lookup(data.circuit_names),
    )


def _build_lookup(names: Iterable[str]) -> Dict[str, str]:
    """Map normalized name to canonical name."""
    lookup: Dict[str, str] = {}
    for name in names:
        norm = _normalize(name)
        if norm:
            lookup[norm] = name
    return lookup


def _strip_phrase(
    tokens: List[str],
    phrase: Tuple[str, ...],
    from_start: bool,
) -> List[str]:
    if from_start:
        if tokens[: len(phrase)] == list(phrase):
            return tokens[len(phrase) :]
        return tokens
    if tokens[-len(phrase) :] == list(phrase):
        return tokens[: -len(phrase)]
    return tokens


def _generate_circuit_aliases(name: str) -> List[str]:
    normalized = _normalize(name)
    tokens = normalized.split()
    if not tokens:
        return []

    stripped = tokens
    for prefix in _CIRCUIT_PREFIXES:
        stripped = _strip_phrase(stripped, prefix, from_start=True)
    for suffix in _CIRCUIT_SUFFIXES:
        stripped = _strip_phrase(stripped, suffix, from_start=False)

    alias = " ".join(stripped).strip()
    if alias and alias != normalized:
        return [alias]
    return []


def _build_circuit_lookup(names: Iterable[str]) -> Dict[str, str]:
    lookup = _build_lookup(names)
    for name in names:
        for alias in _generate_circuit_aliases(name):
            lookup.setdefault(alias, name)
    return lookup


def _match_longest(question: str, lookup: Dict[str, str]) -> Optional[str]:
    """Return the longest matching name in the question."""
    normalized_question = _normalize(question)
    candidates = sorted(lookup.keys(), key=len, reverse=True)
    for key in candidates:
        if _contains_phrase(normalized_question, key):
            return lookup[key]
    return None


def extract_year(question: str) -> Optional[int]:
    """Extract 4-digit year from question."""
    match = re.search(r"(19|20)\d{2}", question)
    return int(match.group()) if match else None


def extract_driver_name(question: str, data_dir: str = _DEFAULT_DATA_DIR) -> Optional[str]:
    """Extract driver name or surname from question using CSV data."""
    lookups = _load_filter_lookups(data_dir)
    match = _match_longest(question, lookups.driver_full_names)
    if match:
        return match
    return _match_longest(question, lookups.driver_surnames)


def extract_constructor_name(question: str, data_dir: str = _DEFAULT_DATA_DIR) -> Optional[str]:
    """Extract constructor name from question using CSV data."""
    lookups = _load_filter_lookups(data_dir)
    return _match_longest(question, lookups.constructor_names)


def extract_race_name(question: str, data_dir: str = _DEFAULT_DATA_DIR) -> Optional[str]:
    """Extract race name from question using CSV data."""
    lookups = _load_filter_lookups(data_dir)
    return _match_longest(question, lookups.race_names)


def extract_circuit_name(question: str, data_dir: str = _DEFAULT_DATA_DIR) -> Optional[str]:
    """Extract circuit name from question using CSV data."""
    lookups = _load_filter_lookups(data_dir)
    return _match_longest(question, lookups.circuit_names)
