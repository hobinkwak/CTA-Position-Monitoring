"""
Read and validate price/benchmark data from the data/ folder.

Expected files:
    data/raw/futures_rolled.{csv,parquet,feather,xlsx}   main input (ratio-adjusted continuous futures)
    data/raw/futures_c1.*, futures_c2.*, roll_dates.*    (when rolling in-house)
    data/raw/benchmark.*                                  SG Trend Index level
    data/raw/fx.*                                         (optional) for contract-count conversion
All are wide format: first column = date index, remaining columns = values.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_CONFIG, PATHS, DataConfig, Paths
from .universe import UNIVERSE, Universe

#: Search order: earlier extensions are tried first.
SUPPORTED_SUFFIXES: tuple[str, ...] = (".parquet", ".feather", ".csv", ".xlsx", ".xls")


class DataNotFoundError(FileNotFoundError):
    """Raised when a required file is not found in data/."""


@dataclass
class ValidationReport:
    """Result of the price data quality check."""

    n_rows: int = 0
    n_cols: int = 0
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    missing_markets: list[str] = field(default_factory=list)
    nonpositive: dict[str, int] = field(default_factory=dict)
    long_gaps: dict[str, int] = field(default_factory=dict)
    stale_runs: dict[str, int] = field(default_factory=dict)
    extreme_moves: dict[str, int] = field(default_factory=dict)
    first_valid: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not (self.missing_markets or self.nonpositive)

    def to_frame(self) -> pd.DataFrame:
        """Per-market summary table."""
        codes = sorted(
            set(self.first_valid)
            | set(self.nonpositive)
            | set(self.long_gaps)
            | set(self.stale_runs)
            | set(self.extreme_moves)
        )
        return pd.DataFrame(
            {
                "first_valid": pd.Series(self.first_valid),
                "nonpositive": pd.Series(self.nonpositive),
                "max_gap_days": pd.Series(self.long_gaps),
                "max_stale_days": pd.Series(self.stale_runs),
                "extreme_moves": pd.Series(self.extreme_moves),
            },
            index=codes,
        ).fillna(0)

    def summary(self) -> str:
        lines = [
            f"rows={self.n_rows}  cols={self.n_cols}  "
            f"range={_fmt(self.start)}..{_fmt(self.end)}",
        ]
        if self.missing_markets:
            lines.append(f"  [MISSING] markets in the universe but not in the data: {self.missing_markets}")
        if self.nonpositive:
            lines.append(f"  [ERROR]   prices <= 0: {self.nonpositive}")
        if self.long_gaps:
            lines.append(f"  [WARN]    long gaps (business days): {self.long_gaps}")
        if self.stale_runs:
            lines.append(f"  [WARN]    stale runs (business days): {self.stale_runs}")
        if self.extreme_moves:
            lines.append(
                f"  [WARN]    extreme daily moves (|z|>{DATA_CONFIG.roll_jump_zscore:g}): "
                f"{self.extreme_moves} <- roll adjustment may be missing"
            )
        if len(lines) == 1:
            lines.append("  [OK] no issues")
        return "\n".join(lines)


def _fmt(ts: pd.Timestamp | None) -> str:
    return "?" if ts is None else ts.strftime("%Y-%m-%d")


class DataLoader:
    """Access layer for the `data/` directory.

    Auto-detects the file extension, normalizes the index to a DatetimeIndex, and
    keeps only universe columns in universe order.

    Examples
    --------
    >>> loader = DataLoader()
    >>> px = loader.load_rolled_prices()        # doctest: +SKIP
    >>> bm = loader.load_benchmark()            # doctest: +SKIP
    """

    def __init__(
        self,
        paths: Paths = PATHS,
        universe: Universe = UNIVERSE,
        data_config: DataConfig = DATA_CONFIG,
    ) -> None:
        self.paths = paths
        self.universe = universe
        self.cfg = data_config

    # ------------------------------------------------------------------ #
    # File lookup / reading
    # ------------------------------------------------------------------ #

    def resolve(self, stem: str, required: bool = True) -> Path | None:
        """Find `data/raw/<stem>.<ext>` in extension priority order."""
        for directory in (self.paths.raw, self.paths.data, self.paths.processed):
            for suffix in SUPPORTED_SUFFIXES:
                candidate = directory / f"{stem}{suffix}"
                if candidate.exists():
                    return candidate
        if required:
            raise DataNotFoundError(
                f"File '{stem}' not found. "
                f"Place e.g. {stem}{SUPPORTED_SUFFIXES[2]} in {self.paths.raw}."
            )
        return None

    @staticmethod
    def read_table(path: Path) -> pd.DataFrame:
        """Read according to the extension and set the first column as a DatetimeIndex."""
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            df = pd.read_parquet(path)
        elif suffix == ".feather":
            df = pd.read_feather(path)
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        elif suffix == ".csv":
            df = _read_csv_tolerant(path)
        else:
            raise ValueError(f"Unsupported file extension: {path}")

        # Keep the index if it is already dates; otherwise use the first column.
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index(df.columns[0])
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "date"
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df.apply(pd.to_numeric, errors="coerce")

    # ------------------------------------------------------------------ #
    # Loaders
    # ------------------------------------------------------------------ #

    def _load_price_file(self, stem: str, required: bool = True) -> pd.DataFrame | None:
        path = self.resolve(stem, required=required)
        if path is None:
            return None
        df = self.read_table(path)
        return self._align_universe(df, source=path.name)

    def _align_universe(self, df: pd.DataFrame, source: str = "") -> pd.DataFrame:
        """Keep only universe columns, in definition order."""
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        known = [c for c in self.universe.codes if c in df.columns]
        unknown = [c for c in df.columns if c not in self.universe.codes]
        if unknown:
            warnings.warn(
                f"{source}: ignoring columns not in the universe -> {unknown}", stacklevel=3
            )
        if not known:
            raise ValueError(
                f"{source}: no column matches a universe market code. "
                f"Expected e.g.: {self.universe.codes[:5]} ... / found: {list(df.columns)[:8]}"
            )
        return df[known]

    def load_rolled_prices(self, required: bool = True) -> pd.DataFrame | None:
        """Ratio-adjusted continuous futures daily closes (the main input)."""
        return self._load_price_file(self.cfg.rolled_file, required=required)

    def load_front_prices(self, required: bool = False) -> pd.DataFrame | None:
        """Raw front-month (C1) prices, for contract-count conversion and in-house rolling."""
        return self._load_price_file(self.cfg.front_file, required=required)

    def load_second_prices(self, required: bool = False) -> pd.DataFrame | None:
        """Raw second-month (C2) prices, for in-house rolling."""
        return self._load_price_file(self.cfg.second_file, required=required)

    def load_roll_dates(self, required: bool = False) -> dict[str, pd.DatetimeIndex] | None:
        """Roll dates as {market_code: DatetimeIndex}.

        File format: each column is a market code and cell values are roll date strings.
        (Column lengths may differ by market; blanks are ignored.)
        """
        path = self.resolve(self.cfg.roll_dates_file, required=required)
        if path is None:
            return None
        suffix = path.suffix.lower()
        if suffix == ".csv":
            raw = pd.read_csv(path)
        elif suffix in (".xlsx", ".xls"):
            raw = pd.read_excel(path)
        elif suffix == ".parquet":
            raw = pd.read_parquet(path)
        else:
            raw = pd.read_feather(path)

        out: dict[str, pd.DatetimeIndex] = {}
        for col in raw.columns:
            code = str(col).strip()
            if code not in self.universe:
                continue
            dates = pd.to_datetime(raw[col].dropna(), errors="coerce").dropna()
            out[code] = pd.DatetimeIndex(sorted(set(dates))).tz_localize(None)
        if not out:
            raise ValueError(f"{path.name}: no column matches a universe code.")
        return out

    def load_benchmark(self, required: bool = True) -> pd.Series | None:
        """SG Trend Index level series."""
        path = self.resolve(self.cfg.benchmark_file, required=required)
        if path is None:
            return None
        df = self.read_table(path)
        if self.cfg.benchmark_column in df.columns:
            series = df[self.cfg.benchmark_column]
        else:
            if df.shape[1] > 1:
                warnings.warn(
                    f"{path.name}: column '{self.cfg.benchmark_column}' not found, "
                    f"using first column '{df.columns[0]}'.",
                    stacklevel=2,
                )
            series = df.iloc[:, 0]
        return series.dropna().rename(self.cfg.benchmark_column)

    def load_fx(self, required: bool = False) -> pd.DataFrame | None:
        """FX for USD conversion. Example columns: EURUSD, GBPUSD, USDJPY, AUDUSD."""
        path = self.resolve(self.cfg.fx_file, required=required)
        if path is None:
            return None
        df = self.read_table(path)
        df.columns = [str(c).strip().upper() for c in df.columns]
        return df

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self, prices: pd.DataFrame, label: str = "prices") -> ValidationReport:
        """Quality check of a price panel; returns a ValidationReport.

        Only fatal items (price <= 0, missing market) set ok=False; no exception is raised.
        The caller prints report.summary() to show the result.
        """
        rep = ValidationReport(
            n_rows=len(prices),
            n_cols=prices.shape[1],
            start=prices.index.min() if len(prices) else None,
            end=prices.index.max() if len(prices) else None,
        )
        rep.missing_markets = [c for c in self.universe.codes if c not in prices.columns]

        rtn = prices.pct_change()
        for code in prices.columns:
            s = prices[code]
            valid = s.dropna()
            if valid.empty:
                rep.missing_markets.append(code)
                continue

            rep.first_valid[code] = valid.index[0].strftime("%Y-%m-%d")

            n_bad = int((valid <= 0).sum())
            if n_bad:
                rep.nonpositive[code] = n_bad

            # Check gaps/stale runs only after the first valid value
            body = s.loc[valid.index[0] :]
            gap = _max_consecutive(body.isna().to_numpy())
            if gap > self.cfg.max_stale_days:
                rep.long_gaps[code] = gap

            stale = _max_consecutive((body.ffill().diff() == 0).to_numpy())
            if stale > self.cfg.max_stale_days:
                rep.stale_runs[code] = stale

            r = rtn[code].dropna()
            if len(r) > 30 and r.std() > 0:
                z = (r - r.mean()).abs() / r.std()
                n_extreme = int((z > self.cfg.roll_jump_zscore).sum())
                if n_extreme:
                    rep.extreme_moves[code] = n_extreme

        rep.missing_markets = sorted(set(rep.missing_markets))
        return rep

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #

    def available(self) -> pd.DataFrame:
        """Overview of what exists in data/."""
        stems = {
            "rolled prices": self.cfg.rolled_file,
            "front (C1)": self.cfg.front_file,
            "second (C2)": self.cfg.second_file,
            "roll dates": self.cfg.roll_dates_file,
            "benchmark": self.cfg.benchmark_file,
            "fx": self.cfg.fx_file,
        }
        rows = []
        for label, stem in stems.items():
            path = self.resolve(stem, required=False)
            rows.append(
                {
                    "dataset": label,
                    "stem": stem,
                    "found": path is not None,
                    "path": str(path.relative_to(self.paths.root)) if path else "-",
                }
            )
        return pd.DataFrame(rows).set_index("dataset")

    def save_processed(self, df: pd.DataFrame | pd.Series, name: str) -> Path:
        """Save an intermediate result as parquet in data/processed."""
        self.paths.ensure()
        path = self.paths.processed / f"{name}.parquet"
        frame = df.to_frame() if isinstance(df, pd.Series) else df
        frame.to_parquet(path)
        return path


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _read_csv_tolerant(path: Path) -> pd.DataFrame:
    """Tolerate encoding variants of exported CSV files."""
    for encoding in ("utf-8-sig", "cp949", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", errors="replace")


def _max_consecutive(mask: np.ndarray) -> int:
    """Longest run of consecutive True values in a boolean array."""
    if mask.size == 0 or not mask.any():
        return 0
    best = run = 0
    for flag in mask:
        run = run + 1 if flag else 0
        if run > best:
            best = run
    return best


__all__ = ["DataLoader", "DataNotFoundError", "ValidationReport", "SUPPORTED_SUFFIXES"]
