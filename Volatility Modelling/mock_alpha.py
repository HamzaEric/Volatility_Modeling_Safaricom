"""Mock helper for simulating AlphaVantage API responses.

This module patches ``requests.get`` so that any call
returns a JSON payload loaded from a local file, matching
the AlphaVantage "Time Series Daily" endpoint format.
No network request is made.

The mock routes requests to the correct local JSON file
based on the ``symbol`` query parameter:

- ``symbol=AMBUJACEM.BSE``
  -> ``data/alpha_ambujacem_bse_full.json``
- ``symbol=SUZLON.BSE``
  -> ``data/alpha_suzlon_bse_full.json``

It also respects the ``outputsize`` query parameter:

- ``outputsize=full``    -> returns all observations.
- ``outputsize=compact`` -> returns the latest 100.

If the mock cannot find or load the requested data, it
emits a warning and returns an empty dictionary instead
of raising an error.

Usage inside a notebook::

    from mock_alpha import activate_mock, deactivate_mock

    activate_mock()
    response = requests.get(url)   # intercepted locally
    # ... work with response.json() or response.text ...
    deactivate_mock()
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse


# --- Configuration -------------------------------------------------

_DATA_DIR = Path("data")

# Map ticker symbols (lowercased) to their local JSON filenames
_TICKER_FILES: dict[str, str] = {
    "ambujacem.bse": "alpha_ambujacem_bse_full.json",
    "suzlon.bse": "alpha_suzlon_bse_full.json",
}

_COMPACT_LIMIT: int = 100


# --- Helpers -------------------------------------------------------


def _load_payload(filepath: Path) -> dict[str, Any]:
    """Read an AlphaVantage JSON file from disk.

    Parameters
    ----------
    filepath : Path
        Path to the JSON file.

    Returns
    -------
    dict[str, Any]
        Parsed JSON content.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _trim_to_compact(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *payload* with only the latest days.

    Parameters
    ----------
    payload : dict[str, Any]
        Full AlphaVantage JSON payload containing a
        ``"Time Series (Daily)"`` key.

    Returns
    -------
    dict[str, Any]
        A shallow copy of *payload* whose time-series
        section has at most ``_COMPACT_LIMIT`` entries
        containing the most recent dates.
    """
    ts_key = "Time Series (Daily)"

    all_dates = sorted(payload[ts_key].keys(), reverse=True)
    latest = all_dates[:_COMPACT_LIMIT]
    trimmed_ts = {date: payload[ts_key][date] for date in latest}

    result = dict(payload)
    result[ts_key] = trimmed_ts

    meta = dict(result.get("Meta Data", {}))
    meta["4. Output Size"] = "Compact"
    result["Meta Data"] = meta

    return result


def _build_response(url: str, data: dict[str, Any]) -> Mock:
    """Create a response-like mock object.

    Parameters
    ----------
    url : str
        Original request URL.

    data : dict[str, Any]
        Data to expose through ``json()``, ``text``, and
        ``content``.

    Returns
    -------
    Mock
        A response-like object compatible with common
        ``requests.Response`` usage.
    """
    response_text = json.dumps(data)

    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.ok = True
    mock_resp.reason = "OK"
    mock_resp.url = url
    mock_resp.text = response_text
    mock_resp.content = response_text.encode("utf-8")
    mock_resp.encoding = "utf-8"
    mock_resp.json.return_value = data
    mock_resp.raise_for_status.return_value = None

    return mock_resp


# --- Internal state ------------------------------------------------

_patcher: Any = None
_payload_cache: dict[str, dict[str, Any]] = {}


def _resolve_filepath(symbol: str) -> Path:
    """Return the Path for a given ticker symbol.

    Parameters
    ----------
    symbol : str
        Ticker symbol as it appears in the API URL,
        for example ``"AMBUJACEM.BSE"``.

    Returns
    -------
    Path
        Resolved path to the local JSON file.

    Raises
    ------
    FileNotFoundError
        If the ticker is not recognised or the JSON
        file is missing on disk.
    """
    key = symbol.lower()
    filename = _TICKER_FILES.get(key)

    if filename is None:
        available = ", ".join(t.upper() for t in sorted(_TICKER_FILES))
        raise FileNotFoundError(
            f"No local data file mapped for symbol '{symbol}'. "
            f"Available tickers: {available}"
        )

    filepath = _DATA_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}. "
            f"Make sure the file exists in the '{_DATA_DIR}' directory."
        )

    return filepath


def _mock_get(url: str, **kwargs: Any) -> Mock:
    """Replacement for ``requests.get`` reading local JSON.

    Parses the ``symbol`` and ``outputsize`` query
    parameters from *url* to decide which file to load
    and whether to return the full or compact payload.

    If the mock cannot resolve or load the requested data,
    it emits a warning and returns an empty dictionary
    instead of raising an error.

    Parameters
    ----------
    url : str
        The URL that was passed to ``requests.get``.

    **kwargs : Any
        Additional keyword arguments accepted for compatibility
        with ``requests.get``. They are ignored by the mock.

    Returns
    -------
    Mock
        A mock response object with ``status_code``, ``json()``,
        ``text``, ``content``, and ``raise_for_status()``.
    """
    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    try:
        # Determine which ticker was requested
        symbol = params.get("symbol", [""])[0]

        if not symbol:
            raise ValueError(
                "The mock requires a 'symbol' query parameter in the URL."
            )

        # Load and cache the payload for this ticker
        cache_key = symbol.lower()

        if cache_key not in _payload_cache:
            filepath = _resolve_filepath(symbol)
            _payload_cache[cache_key] = _load_payload(filepath)

        full_data = _payload_cache[cache_key]

        # Respect outputsize parameter
        output_size = params.get("outputsize", ["full"])[0].lower()

        if output_size == "compact":
            data = _trim_to_compact(full_data)
        else:
            data = full_data

    except Exception as exc:
        warnings.warn(
            f"AlphaVantage mock failed for URL '{url}'. "
            f"Returning an empty dictionary instead. "
            f"Original error: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        data = {}

    return _build_response(url=url, data=data)


# --- Public API ----------------------------------------------------


def activate_mock() -> None:
    """Start intercepting ``requests.get``.

    After calling this function, every ``requests.get(...)``
    call will return the JSON content of the appropriate
    local data file based on the ``symbol`` query parameter.

    The ``outputsize`` parameter controls whether the full
    dataset or only the latest 100 observations are returned.

    If the mock fails to find or load the local data, it will
    emit a warning and return an empty dictionary instead of
    stopping the notebook.

    Calling this function when the mock is already active
    is a no-op.
    """
    global _patcher

    if _patcher is not None:
        return  # already active

    _patcher = patch(
        "requests.get",
        side_effect=_mock_get,
    )
    _patcher.start()


def deactivate_mock() -> None:
    """Restore the original ``requests.get`` behaviour.

    Also clears the internal payload cache. Calling this
    function when the mock is not active is a no-op.
    """
    global _patcher, _payload_cache

    if _patcher is None:
        return  # nothing to stop

    _patcher.stop()
    _patcher = None
    _payload_cache = {}
