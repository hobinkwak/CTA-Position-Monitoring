"""
ctarep - Python implementation of "Replicating CTA Positioning" (Kestner, 2020).

Paper: https://ssrn.com/abstract=3674828

Quick start
-----------
>>> from ctarep import CTAReplicationPipeline          # doctest: +SKIP
>>> result = CTAReplicationPipeline().run()            # doctest: +SKIP
>>> result.snapshot.summary()                          # doctest: +SKIP
"""
from .benchmark import Benchmark
from .config import (
    PARAM_GRID,
    PATHS,
    RUN_CONFIG,
    ModelParams,
    ParameterGrid,
    Paths,
    RunConfig,
    ensemble_params,
)
from .dataloader import DataLoader, DataNotFoundError, ValidationReport
from .ensemble import EnsembleReplicator
from .evaluation import ReplicationEvaluator, RollingRegression, r_squared
from .exposure import ExposureEstimator, ExposureSnapshot
from .pipeline import CTAReplicationPipeline, PipelineResult
from .portfolio import ReplicationModel
from .rolling import FuturesRoller, RollDiagnostics
from .scenario import (
    DEFAULT_SCENARIOS,
    Scenario,
    ScenarioEngine,
    ScenarioProjection,
    TriggerLevels,
)
from .signals import NormalizedMomentum, SignalPanel
from .timegrid import WeeklyGrid
from .universe import UNIVERSE, Market, Sector, Universe
from .volatility import RealizedVolatility, VolatilityPanel

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # config
    "PATHS",
    "Paths",
    "RUN_CONFIG",
    "RunConfig",
    "PARAM_GRID",
    "ParameterGrid",
    "ModelParams",
    "ensemble_params",
    # universe
    "UNIVERSE",
    "Universe",
    "Market",
    "Sector",
    # data
    "DataLoader",
    "DataNotFoundError",
    "ValidationReport",
    "FuturesRoller",
    "RollDiagnostics",
    "WeeklyGrid",
    # model
    "RealizedVolatility",
    "VolatilityPanel",
    "NormalizedMomentum",
    "SignalPanel",
    "ReplicationModel",
    "EnsembleReplicator",
    "Benchmark",
    # evaluation
    "ReplicationEvaluator",
    "RollingRegression",
    "r_squared",
    # exposure
    "ExposureEstimator",
    "ExposureSnapshot",
    # scenario
    "Scenario",
    "DEFAULT_SCENARIOS",
    "ScenarioEngine",
    "ScenarioProjection",
    "TriggerLevels",
    # pipeline
    "CTAReplicationPipeline",
    "PipelineResult",
]
