from dataclasses import dataclass, field
from functools import total_ordering
from typing import Any, Union, Optional

import numpy as np
from dataclasses_json import DataClassJsonMixin

# When the primary metric difference is below this threshold, compare using
# secondary metrics instead. The threshold only applies to the primary metric.
# Override via set_tiebreak_threshold() during initialization (reads from config).
TIEBREAK_THRESHOLD: float = 0.005


def set_tiebreak_threshold(value: float) -> None:
    """Set the module-level tiebreak threshold (called once from config during init)."""
    global TIEBREAK_THRESHOLD
    TIEBREAK_THRESHOLD = value


@dataclass
@total_ordering
class MetricValue_old(DataClassJsonMixin):
    """
    Represents the value of a metric to be optimized, which can be compared to other metric values.
    Comparisons (and max, min) are based on which value is better, not which is larger.
    """

    value: Union[float, int, np.number, np.floating, np.ndarray, dict, None]
    maximize: Optional[bool] = field(default=None)
    name: Optional[str] = field(
        default=None
    )  # e.g., "accuracy", "loss", "f1_score"
    description: Optional[str] = field(
        default=None
    )  # e.g., "Classification accuracy on validation set"

    def __post_init__(self):
        if self.value is not None:
            if isinstance(self.value, dict):
                self.value = {k: float(v) for k, v in self.value.items()}
            else:
                assert isinstance(self.value, (float, int, np.number, np.floating))
                self.value = float(self.value)

    def __gt__(self, other) -> bool:
        """True if self is a _better_ (not necessarily larger) metric value than other"""
        if self.value is None:
            return False
        if other.value is None:
            return True

        assert type(self) is type(other) and (self.maximize == other.maximize)

        # For multi-dataset metrics, use mean for comparison
        self_val = (
            np.mean(list(self.value.values()))
            if isinstance(self.value, dict)
            else self.value
        )
        other_val = (
            np.mean(list(other.value.values()))
            if isinstance(other.value, dict)
            else other.value
        )

        if self_val == other_val:
            return False

        comp = self_val > other_val
        return comp if self.maximize else not comp  # type: ignore

    def __eq__(self, other: Any) -> bool:
        return self.value == other.value

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.maximize is None:
            opt_dir = "?"
        elif self.maximize:
            opt_dir = "↑"
        else:
            opt_dir = "↓"
        metric_name = f"({self.name})" if self.name else ""
        if isinstance(self.value_npsafe, dict):
            values_str = ", ".join(f"{k}:{v:.4f}" for k, v in self.value_npsafe.items())
            mean_val = np.mean(list(self.value_npsafe.values()))
            return f"Metric{opt_dir}{metric_name}[{values_str}](mean={mean_val:.4f})"
        else:
            return f"Metric{opt_dir}{metric_name}({self.value_npsafe:.4f})"

    @property
    def is_worst(self):
        """True if the metric value is the worst possible value."""
        return self.value is None

    @property
    def value_npsafe(self):
        if self.value is None:
            return float("nan")
        if isinstance(self.value, dict):
            return {
                k: v if v is not None else float("nan") for k, v in self.value.items()
            }
        return self.value

    def get_dataset_value(self, dataset_name: str) -> Optional[float]:
        """Get the metric value for a specific dataset"""
        if isinstance(self.value, dict):
            return self.value.get(dataset_name)
        return None

    def get_mean_value(self) -> float:
        """Get the mean value across all datasets (or single value if not multi-dataset)"""
        if self.value is None:
            return float("nan")
        if isinstance(self.value, dict):
            return float(np.mean(list(self.value.values())))
        return float(self.value)


@dataclass
@total_ordering
class MetricValue(DataClassJsonMixin):
    """
    Represents one or more metric values to be optimized, which can be compared to other metric values.
    Comparisons (and max, min) are based on which value is better, not which is larger.

    The value can be:
    - A single number (float/int)
    - A dictionary in the format:
      {
        "metric_names": [
          {
            "metric_name": str,
            "lower_is_better": bool,
            "description": str,
            "data": [
                {"dataset_name": str, "final_value": float, "best_value": float},
                {"dataset_name": str, "final_value": float, "best_value": float},
                ...
            ]
          },
          ...
        ]
      }
    """

    value: Union[float, int, np.number, np.floating, dict, None]
    maximize: Optional[bool] = field(default=None)
    name: Optional[str] = field(default=None)
    description: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.value is not None:
            if isinstance(self.value, dict):
                # Check if it's the new format with metric_names list
                if "metric_names" in self.value:
                    # New format - validate and convert values to float
                    for metric in self.value["metric_names"]:
                        for data_point in metric["data"]:
                            if data_point["final_value"] is not None:
                                data_point["final_value"] = float(
                                    data_point["final_value"]
                                )
                            if data_point["best_value"] is not None:
                                data_point["best_value"] = float(
                                    data_point["best_value"]
                                )
                else:
                    # Old format - convert to float
                    self.value = {
                        k: float(v) if v is not None else None
                        for k, v in self.value.items()
                    }
            else:
                # Single value case
                assert isinstance(self.value, (float, int, np.number, np.floating))
                self.value = float(self.value)

    def __gt__(self, other) -> bool:
        if self.value is None:
            return False
        if other.value is None:
            return True

        assert type(self) is type(other)

        # For multiple metrics: use primary metric first, then tie-breaker
        if isinstance(self.value, dict) and "metric_names" in self.value:
            return self._compare_multiple_metrics(other)
        
        # For single metric or old format: use mean values
        self_val = self.get_mean_value()
        other_val = other.get_mean_value()

        if self_val == other_val:
            return False

        # Determine if we should maximize or minimize
        should_maximize = self._should_maximize()
        comp = self_val > other_val
        return comp if should_maximize else not comp

    def _compare_multiple_metrics(self, other) -> bool:
        """
        Compare metrics using primary metric (Dice) as main criterion,
        and secondary metrics (HD95) as tie-breakers.

        Strategy:
        1. Compare primary metric (first metric, typically Dice)
        2. If the primary metric is close (difference < TIEBREAK_THRESHOLD),
           use secondary metrics as tie-breakers without any extra threshold

        Returns:
            True if self is better than other
        """
        self_metrics = self.value["metric_names"]
        other_metrics = other.value["metric_names"]
        
        if len(self_metrics) != len(other_metrics):
            return self.get_mean_value() > other.get_mean_value()
        
        for idx, (self_metric, other_metric) in enumerate(zip(self_metrics, other_metrics)):
            self_values = [d["final_value"] for d in self_metric["data"] 
                          if d["final_value"] is not None]
            other_values = [d["final_value"] for d in other_metric["data"] 
                           if d["final_value"] is not None]
            
            if not self_values or not other_values:
                continue
            
            self_val = float(np.mean(self_values))
            other_val = float(np.mean(other_values))

            # Only the primary metric uses the tiebreak threshold. Once we
            # fall back to secondary metrics, any strict improvement counts.
            if idx == 0 and abs(self_val - other_val) < TIEBREAK_THRESHOLD:
                continue

            if self_val == other_val:
                continue

            lower_is_better = self_metric.get("lower_is_better", False)
            if lower_is_better:
                return self_val < other_val
            else:
                return self_val > other_val
        
        # All metrics are tied
        return False

    def _should_maximize(self) -> bool:
        """Determine if we should maximize based on the metric format"""
        if isinstance(self.value, dict):
            # New format
            if "metric_names" in self.value:
                # Use the first metric's lower_is_better value
                try:
                    return not self.value["metric_names"][0]["lower_is_better"]
                except Exception as e:
                    print(f"error during metric value: {e}")
            # Old format
            return bool(self.maximize)
        # Single value case
        return bool(self.maximize)

    def __str__(self) -> str:
        if isinstance(self.value, dict):
            # New format with metric_names list
            if "metric_names" in self.value:
                parts = []
                for idx, metric in enumerate(self.value["metric_names"]):
                    opt_dir = (
                        "↓"
                        if "lower_is_better" in metric and metric["lower_is_better"]
                        else "↑"
                    )
                    try:
                        # Calculate mean value for this metric
                        values = [d['final_value'] for d in metric["data"] if d['final_value'] is not None]
                        mean_val = np.mean(values) if values else float('nan')
                        
                        # Mark primary metric
                        primary_marker = "★" if idx == 0 else ""
                        parts.append(f"{metric['metric_name']}{primary_marker}{opt_dir}={mean_val:.4f}")
                    except Exception as e:
                        print(f"error during metric value: {e}")
                        parts.append(f"{metric['metric_name']}=Error")
                return "Metrics(" + ", ".join(parts) + ")"
            # Old format
            opt_dir = "↓" if not self.maximize else "↑"
            values_str = ", ".join(f"{k}:{v:.4f}" for k, v in self.value.items())
            mean_val = np.mean([v for v in self.value.values() if v is not None])
            return f"Metric{opt_dir}({self.name})[{values_str}](mean={mean_val:.4f})"
        # Single value case
        opt_dir = "?" if self.maximize is None else ("↑" if self.maximize else "↓")
        metric_name = f"({self.name})" if self.name else ""
        return f"Metric{opt_dir}{metric_name}({self.value_npsafe:.4f})"

    def __eq__(self, other: Any) -> bool:
        """Compare equality of metric values"""
        if not isinstance(other, MetricValue):
            raise NotImplementedError
        if self.value is None and other.value is None:
            return True
        if self.value is None or other.value is None:
            return False

        # For new format, compare entire dictionaries
        if isinstance(self.value, dict) and isinstance(other.value, dict):
            # If both are new format with metric_names
            if "metric_names" in self.value and "metric_names" in other.value:
                return self.value == other.value
            # If both are old format (no metric_names)
            elif "metric_names" not in self.value and "metric_names" not in other.value:
                return self.value == other.value
            # Mixed formats should not be equal
            return False
        # Single values
        return self.value == other.value

    def __repr__(self) -> str:
        """Return string representation"""
        return str(self)

    @property
    def value_npsafe(self):
        """Return a NaN-safe version of the value"""
        if self.value is None:
            return float("nan")
        if isinstance(self.value, dict):
            # New format with metric_names list
            if "metric_names" in self.value:
                return {
                    "metric_names": [
                        {
                            **metric,
                            "data": [
                                {
                                    **data_point,
                                    "final_value": (
                                        data_point["final_value"]
                                        if data_point["final_value"] is not None
                                        else float("nan")
                                    ),
                                    "best_value": (
                                        data_point["best_value"]
                                        if data_point["best_value"] is not None
                                        else float("nan")
                                    ),
                                }
                                for data_point in metric["data"]
                            ],
                        }
                        for metric in self.value["metric_names"]
                    ]
                }
            # Old format
            return {
                k: v if v is not None else float("nan") for k, v in self.value.items()
            }
        # Single value case
        return self.value if self.value is not None else float("nan")

    def get_mean_value(self) -> float:
        """
        Get the representative value for this metric.
        
        For multiple metrics: returns the PRIMARY metric's mean value (first metric, typically Dice)
        For single metric: returns that metric's value
        
        This is used for logging and display purposes.
        """
        if self.value is None:
            return float("nan")
        if isinstance(self.value, dict):
            # New format with multiple metrics
            if "metric_names" in self.value:
                # Return primary metric (first one, typically Dice) for better interpretability
                if self.value["metric_names"]:
                    primary_metric = self.value["metric_names"][0]
                    values = [
                        d["final_value"]
                        for d in primary_metric["data"]
                        if d["final_value"] is not None
                    ]
                    return float(np.mean(values)) if values else float("nan")
                return float("nan")
            # Old format
            values = [v for v in self.value.values() if v is not None]
            return float(np.mean(values)) if values else float("nan")
        # Single value case
        return float(self.value)


@dataclass
class WorstMetricValue(MetricValue):
    """
    Represents an invalid metric value, e.g. when the agent creates a buggy solution.
    Always compares worse than any valid metric value.
    """

    value: None = None

    def __repr__(self):
        return super().__repr__()

    def __str__(self):
        return super().__str__()
