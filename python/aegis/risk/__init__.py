"""Risk: factor histories, scenarios, VaR and Expected Shortfall."""

from aegis.risk.backtest import (
    BacktestResult,
    BaselZone,
    backtest_series,
    basel_zone,
    christoffersen_independence,
    conditional_coverage,
    kupiec_pof,
    rolling_backtest,
)
from aegis.risk.factors import FactorHistory, FactorMapping, build_factor_history
from aegis.risk.report import (
    RiskReport,
    build_market,
    default_factor_mappings,
    factor_history_for,
    run_report,
)
from aegis.risk.scenarios import Scenario, ScenarioError, apply_shocks, load_scenarios
from aegis.risk.var import (
    VarMethod,
    VarResult,
    contribution_report,
    expected_shortfall,
    factor_exposures,
    historical_pnl,
    historical_var,
    monte_carlo_var,
    parametric_var,
    value_at_risk,
)

__all__ = [
    "BacktestResult",
    "BaselZone",
    "FactorHistory",
    "FactorMapping",
    "RiskReport",
    "Scenario",
    "ScenarioError",
    "VarMethod",
    "VarResult",
    "apply_shocks",
    "backtest_series",
    "basel_zone",
    "build_factor_history",
    "build_market",
    "christoffersen_independence",
    "conditional_coverage",
    "contribution_report",
    "default_factor_mappings",
    "expected_shortfall",
    "factor_exposures",
    "factor_history_for",
    "historical_pnl",
    "historical_var",
    "kupiec_pof",
    "load_scenarios",
    "monte_carlo_var",
    "parametric_var",
    "rolling_backtest",
    "run_report",
    "value_at_risk",
]
