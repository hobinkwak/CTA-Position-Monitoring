"""
Ensemble replication model: the final model.

Trend followers' strategies vary across trend speeds. Combining a range of parameters
(here 3 members with momentum lookbacks of 16, 32 and 52 weeks, normalised momentum cap
of 1.0 and volatility lookback of 90 days) matches how actual managers trade across
speeds and raises the R-squared to the benchmark.

Returns are linear in positions, so **averaging positions == averaging returns**:
    mean_L( sum_i w^L_i r_i ) = sum_i mean_L(w^L_i) * r_i
Positions are the final output (exposure reporting), so positions are averaged.

Diversification makes the ensemble position less volatile than each member, so to read
exposure as "per $1 of benchmark" it is rescaled back to benchmark volatility
(`rescale=True`, the default). R-squared is unaffected.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import WEEKS_PER_YEAR, ModelParams, ensemble_params
from .portfolio import FitResult, ReplicationModel
from .volatility import RealizedVolatility, VolatilityPanel


class EnsembleReplicator:
    """Combines several ReplicationModels and averages their positions.

    Examples
    --------
    >>> from ctarep.config import ensemble_params
    >>> ens = EnsembleReplicator(ensemble_params())
    >>> [m.label for m in ens.models]
    ['16/1/90', '32/1/90', '52/1/90']
    """

    def __init__(
        self,
        params_list: list[ModelParams] | None = None,
        weights: dict[str, float] | None = None,
        sector_map: dict[str, str] | None = None,
    ) -> None:
        params_list = params_list or ensemble_params()
        if not params_list:
            raise ValueError("ensemble requires at least one ModelParams")
        self.models: list[ReplicationModel] = [
            ReplicationModel(p, sector_map=sector_map) for p in params_list
        ]
        self.weights = self._normalize_weights(weights)

        # fit outputs
        self.positions: pd.DataFrame | None = None
        self.returns: pd.Series | None = None
        self.member_returns: pd.DataFrame | None = None
        self.member_positions: dict[str, pd.DataFrame] = {}
        self.fit_result: FitResult | None = None
        self.rescale_factor: float = 1.0

    def __repr__(self) -> str:
        state = "fitted" if self.positions is not None else "unfitted"
        return f"<EnsembleReplicator members={self.labels} [{state}]>"

    def _normalize_weights(self, weights: dict[str, float] | None) -> dict[str, float]:
        labels = [m.label for m in self.models]
        if weights is None:
            return {lab: 1.0 / len(labels) for lab in labels}
        missing = set(labels) - set(weights)
        if missing:
            raise ValueError(f"weights missing for members: {sorted(missing)}")
        total = sum(weights[lab] for lab in labels)
        if total <= 0:
            raise ValueError("ensemble weights must sum to a positive number")
        return {lab: weights[lab] / total for lab in labels}

    @property
    def labels(self) -> list[str]:
        return [m.label for m in self.models]

    # ------------------------------------------------------------------ #

    def fit(
        self,
        weekly_returns: pd.DataFrame,
        daily_returns: pd.DataFrame,
        benchmark_returns: pd.Series,
        method: str = "in_sample",
        expanding_min_weeks: int = 104,
        rescale: bool = True,
    ) -> "EnsembleReplicator":
        """Fit each member, then take the weighted average of positions.

        Parameters
        ----------
        rescale : if True, rescale the ensemble position back to benchmark volatility
                  using **a single full-sample constant**.

        Note: rolling volatility matching was tried and **rejected**. Benchmark
        volatility has declined structurally (18-23% in 2000-02 to 8-12% recently), so
        the constant k does run hot in the recent period, but switching to a rolling
        match widens the cumulative gap from +2.7% to +21.6%.
        """
        # Members with the same volatility lookback share one volatility panel
        vol_cache: dict[tuple[int, str], VolatilityPanel] = {}

        member_rtn: dict[str, pd.Series] = {}
        for model in self.models:
            key = (model.params.volatility_lookback, model.params.vol_method)
            if key not in vol_cache:
                vol_cache[key] = RealizedVolatility.from_params(model.params).compute(
                    daily_returns
                )
            model.fit(
                weekly_returns,
                daily_returns,
                benchmark_returns,
                method=method,
                expanding_min_weeks=expanding_min_weeks,
                vol_panel=vol_cache[key],
            )
            self.member_positions[model.label] = model.positions
            member_rtn[model.label] = model.returns

        self.member_returns = pd.DataFrame(member_rtn)

        # --- weighted average of positions ----------------------------------
        stacked = pd.concat(
            [self.member_positions[m.label] * self.weights[m.label] for m in self.models],
            axis=0,
        )
        blended = stacked.groupby(level=0).sum(min_count=1)
        blended = blended.reindex(weekly_returns.index)

        # --- re-match benchmark volatility ----------------------------------
        rep_r = ReplicationModel.portfolio_returns(blended, weekly_returns)
        common = rep_r.dropna().index.intersection(benchmark_returns.dropna().index)
        target_vol = float(benchmark_returns.loc[common].std(ddof=1)) * np.sqrt(WEEKS_PER_YEAR)
        raw_vol = float(rep_r.loc[common].std(ddof=1)) * np.sqrt(WEEKS_PER_YEAR)

        self.rescale_factor = (
            (target_vol / raw_vol) if (rescale and raw_vol > 0) else 1.0
        )
        mode = f"{method}+ensemble_rescale" if rescale else f"{method}+no_rescale"

        self.positions = blended * self.rescale_factor
        self.returns = ReplicationModel.portfolio_returns(
            self.positions, weekly_returns
        ).rename("ensemble")

        self.fit_result = FitResult(
            leverage=self.rescale_factor,
            method=mode,
            target_vol=target_vol,
            unlevered_vol=raw_vol,
            n_weeks=len(common),
        )
        return self

    # ------------------------------------------------------------------ #

    def require_fitted(self) -> None:
        if self.positions is None:
            raise RuntimeError("EnsembleReplicator: call fit() first.")

    def latest_positions(self) -> pd.Series:
        self.require_fitted()
        valid = self.positions.dropna(how="all")
        return valid.iloc[-1].rename(valid.index[-1])

    def gross_exposure(self) -> pd.Series:
        self.require_fitted()
        return self.positions.abs().sum(axis=1)

    def net_exposure(self) -> pd.Series:
        self.require_fitted()
        return self.positions.sum(axis=1)

    def turnover(self) -> pd.Series:
        self.require_fitted()
        return self.positions.diff().abs().sum(axis=1)

    def member_dispersion(self) -> pd.DataFrame:
        """Cross-member position standard deviation: disagreement across trend speeds."""
        self.require_fitted()
        stacked = pd.concat(
            [self.member_positions[lab] for lab in self.labels],
            keys=self.labels,
        )
        return stacked.groupby(level=1).std(ddof=0)

    def all_returns(self) -> pd.DataFrame:
        """Member and ensemble returns in one frame."""
        self.require_fitted()
        return pd.concat([self.member_returns, self.returns], axis=1)


__all__ = ["EnsembleReplicator"]
