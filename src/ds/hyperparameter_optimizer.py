"""Hyperparameter optimization system for ML experiments."""

import logging
from typing import Callable, Dict, Any, List, Optional, Union
from datetime import datetime
from uuid import UUID
import numpy as np
from pydantic import BaseModel

try:
    import optuna
    from optuna.trial import Trial
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None
    Trial = None

from src.ds.experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)


class OptimizationResult(BaseModel):
    """Results from hyperparameter optimization."""
    
    best_params: Dict[str, Any]
    best_score: float
    all_trials: List[Dict[str, Any]]
    optimization_history: List[float]
    param_importance: Optional[Dict[str, float]] = None
    convergence_info: Optional[Dict[str, Any]] = None
    n_trials: int
    optimization_time: float
    strategy: str


class HyperparameterOptimizer:
    """Hyperparameter optimization engine."""

    def __init__(
        self,
        experiment_tracker: ExperimentTracker,
        strategy: str = "bayesian"
    ):
        """
        Initialize HyperparameterOptimizer.

        Args:
            experiment_tracker: ExperimentTracker instance for logging
            strategy: Optimization strategy ('bayesian', 'grid', 'random')

        Raises:
            ImportError: If optuna is not installed and strategy is 'bayesian'
        """
        if strategy == "bayesian" and not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for Bayesian optimization. "
                "Install it with: pip install optuna"
            )
        
        self.tracker = experiment_tracker
        self.strategy = strategy.lower()
        self.study: Optional[optuna.Study] = None
        self.optimization_result: Optional[OptimizationResult] = None
        
        if self.strategy not in ["bayesian", "grid", "random"]:
            raise ValueError(
                f"Invalid strategy '{strategy}'. "
                "Must be one of: 'bayesian', 'grid', 'random'"
            )
        
        logger.info(f"HyperparameterOptimizer initialized with strategy: {self.strategy}")

    def optimize(
        self,
        objective_function: Callable,
        param_space: Dict[str, Any],
        n_trials: int = 100,
        n_jobs: int = 1,
        experiment_name: str = "hyperparameter_optimization",
        direction: str = "maximize",
        early_stopping_patience: Optional[int] = None,
        early_stopping_threshold: float = 0.001
    ) -> OptimizationResult:
        """
        Run hyperparameter optimization.

        Args:
            objective_function: Function to optimize. Should accept params dict
                               and return a score (float)
            param_space: Dictionary defining parameter search space
            n_trials: Number of trials to run
            n_jobs: Number of parallel jobs (-1 for all cores)
            experiment_name: Name for the optimization experiment
            direction: 'maximize' or 'minimize'
            early_stopping_patience: Number of trials without improvement before stopping
            early_stopping_threshold: Minimum improvement threshold for early stopping

        Returns:
            OptimizationResult with optimization results

        Raises:
            ValueError: If invalid parameters provided
        """
        start_time = datetime.utcnow()
        
        if self.strategy == "bayesian":
            result = self._optimize_bayesian(
                objective_function,
                param_space,
                n_trials,
                n_jobs,
                experiment_name,
                direction,
                early_stopping_patience,
                early_stopping_threshold
            )
        elif self.strategy == "grid":
            result = self._optimize_grid(
                objective_function,
                param_space,
                experiment_name,
                direction
            )
        elif self.strategy == "random":
            result = self._optimize_random(
                objective_function,
                param_space,
                n_trials,
                experiment_name,
                direction
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        end_time = datetime.utcnow()
        result.optimization_time = (end_time - start_time).total_seconds()
        
        self.optimization_result = result
        
        logger.info(
            f"Optimization complete. Best score: {result.best_score:.4f}, "
            f"Best params: {result.best_params}"
        )
        
        return result

    def _optimize_bayesian(
        self,
        objective_function: Callable,
        param_space: Dict[str, Any],
        n_trials: int,
        n_jobs: int,
        experiment_name: str,
        direction: str,
        early_stopping_patience: Optional[int],
        early_stopping_threshold: float
    ) -> OptimizationResult:
        """Run Bayesian optimization using Optuna."""
        
        # Create Optuna study
        self.study = optuna.create_study(
            direction=direction,
            study_name=experiment_name
        )
        
        # Track trials for early stopping
        best_value = None
        trials_without_improvement = 0
        
        def objective_wrapper(trial: Trial) -> float:
            nonlocal best_value, trials_without_improvement
            
            # Sample parameters from the space
            params = self._sample_params_optuna(trial, param_space)
            
            # Start a run in experiment tracker
            run = self.tracker.start_run(
                experiment_name=experiment_name,
                run_name=f"trial_{trial.number}",
                tags={"optimization": "bayesian", "trial": str(trial.number)}
            )
            
            try:
                # Log parameters
                self.tracker.log_params(params)
                
                # Evaluate objective function
                score = objective_function(params)
                
                # Log score as metric
                self.tracker.log_metrics({"score": score})
                
                # End run successfully
                self.tracker.end_run(status="FINISHED")
                
                # Check for early stopping
                if early_stopping_patience is not None:
                    if best_value is None:
                        best_value = score
                        trials_without_improvement = 0
                    else:
                        improvement = score - best_value if direction == "maximize" else best_value - score
                        if improvement > early_stopping_threshold:
                            best_value = score
                            trials_without_improvement = 0
                        else:
                            trials_without_improvement += 1
                            
                        if trials_without_improvement >= early_stopping_patience:
                            logger.info(
                                f"Early stopping triggered after {trial.number + 1} trials "
                                f"({trials_without_improvement} trials without improvement)"
                            )
                            self.study.stop()
                
                return score
                
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                self.tracker.end_run(status="FAILED")
                raise
        
        # Run optimization
        self.study.optimize(
            objective_wrapper,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=False
        )
        
        # Extract results
        best_trial = self.study.best_trial
        all_trials = [
            {
                "number": t.number,
                "params": t.params,
                "value": t.value,
                "state": str(t.state)
            }
            for t in self.study.trials
        ]
        
        optimization_history = [
            t.value for t in self.study.trials
            if t.value is not None
        ]
        
        # Calculate parameter importance
        try:
            param_importance = optuna.importance.get_param_importances(self.study)
        except Exception as e:
            logger.warning(f"Could not calculate parameter importance: {e}")
            param_importance = None
        
        # Convergence info
        convergence_info = {
            "converged": trials_without_improvement >= (early_stopping_patience or 0),
            "trials_without_improvement": trials_without_improvement,
            "total_trials": len(self.study.trials)
        }
        
        return OptimizationResult(
            best_params=best_trial.params,
            best_score=best_trial.value,
            all_trials=all_trials,
            optimization_history=optimization_history,
            param_importance=param_importance,
            convergence_info=convergence_info,
            n_trials=len(self.study.trials),
            optimization_time=0.0,  # Will be set by caller
            strategy="bayesian"
        )

    def _optimize_grid(
        self,
        objective_function: Callable,
        param_space: Dict[str, Any],
        experiment_name: str,
        direction: str
    ) -> OptimizationResult:
        """Run grid search optimization."""
        
        # Generate all parameter combinations
        param_combinations = self._generate_grid_combinations(param_space)
        
        all_trials = []
        optimization_history = []
        best_score = None
        best_params = None
        
        for trial_num, params in enumerate(param_combinations):
            # Start a run in experiment tracker
            run = self.tracker.start_run(
                experiment_name=experiment_name,
                run_name=f"trial_{trial_num}",
                tags={"optimization": "grid", "trial": str(trial_num)}
            )
            
            try:
                # Log parameters
                self.tracker.log_params(params)
                
                # Evaluate objective function
                score = objective_function(params)
                
                # Log score as metric
                self.tracker.log_metrics({"score": score})
                
                # End run successfully
                self.tracker.end_run(status="FINISHED")
                
                # Track results
                all_trials.append({
                    "number": trial_num,
                    "params": params,
                    "value": score,
                    "state": "COMPLETE"
                })
                optimization_history.append(score)
                
                # Update best
                if best_score is None:
                    best_score = score
                    best_params = params
                else:
                    is_better = (
                        score > best_score if direction == "maximize"
                        else score < best_score
                    )
                    if is_better:
                        best_score = score
                        best_params = params
                
            except Exception as e:
                logger.error(f"Trial {trial_num} failed: {e}")
                self.tracker.end_run(status="FAILED")
                all_trials.append({
                    "number": trial_num,
                    "params": params,
                    "value": None,
                    "state": "FAILED"
                })
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_trials=all_trials,
            optimization_history=optimization_history,
            param_importance=None,
            convergence_info={"total_trials": len(param_combinations)},
            n_trials=len(param_combinations),
            optimization_time=0.0,  # Will be set by caller
            strategy="grid"
        )

    def _optimize_random(
        self,
        objective_function: Callable,
        param_space: Dict[str, Any],
        n_trials: int,
        experiment_name: str,
        direction: str
    ) -> OptimizationResult:
        """Run random search optimization."""
        
        all_trials = []
        optimization_history = []
        best_score = None
        best_params = None
        
        for trial_num in range(n_trials):
            # Sample random parameters
            params = self._sample_params_random(param_space)
            
            # Start a run in experiment tracker
            run = self.tracker.start_run(
                experiment_name=experiment_name,
                run_name=f"trial_{trial_num}",
                tags={"optimization": "random", "trial": str(trial_num)}
            )
            
            try:
                # Log parameters
                self.tracker.log_params(params)
                
                # Evaluate objective function
                score = objective_function(params)
                
                # Log score as metric
                self.tracker.log_metrics({"score": score})
                
                # End run successfully
                self.tracker.end_run(status="FINISHED")
                
                # Track results
                all_trials.append({
                    "number": trial_num,
                    "params": params,
                    "value": score,
                    "state": "COMPLETE"
                })
                optimization_history.append(score)
                
                # Update best
                if best_score is None:
                    best_score = score
                    best_params = params
                else:
                    is_better = (
                        score > best_score if direction == "maximize"
                        else score < best_score
                    )
                    if is_better:
                        best_score = score
                        best_params = params
                
            except Exception as e:
                logger.error(f"Trial {trial_num} failed: {e}")
                self.tracker.end_run(status="FAILED")
                all_trials.append({
                    "number": trial_num,
                    "params": params,
                    "value": None,
                    "state": "FAILED"
                })
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_trials=all_trials,
            optimization_history=optimization_history,
            param_importance=None,
            convergence_info={"total_trials": n_trials},
            n_trials=n_trials,
            optimization_time=0.0,  # Will be set by caller
            strategy="random"
        )

    def _sample_params_optuna(
        self,
        trial: Trial,
        param_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sample parameters using Optuna trial."""
        
        params = {}
        
        for param_name, param_config in param_space.items():
            param_type = param_config.get("type")
            
            if param_type == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False)
                )
            elif param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False)
                )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config["choices"]
                )
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
        
        return params

    def _sample_params_random(
        self,
        param_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sample parameters randomly."""
        
        params = {}
        
        for param_name, param_config in param_space.items():
            param_type = param_config.get("type")
            
            if param_type == "float":
                low = param_config["low"]
                high = param_config["high"]
                log = param_config.get("log", False)
                
                if log:
                    params[param_name] = np.exp(
                        np.random.uniform(np.log(low), np.log(high))
                    )
                else:
                    params[param_name] = np.random.uniform(low, high)
                    
            elif param_type == "int":
                low = param_config["low"]
                high = param_config["high"]
                log = param_config.get("log", False)
                
                if log:
                    params[param_name] = int(np.exp(
                        np.random.uniform(np.log(low), np.log(high + 1))
                    ))
                else:
                    params[param_name] = np.random.randint(low, high + 1)
                    
            elif param_type == "categorical":
                params[param_name] = np.random.choice(param_config["choices"])
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
        
        return params

    def _generate_grid_combinations(
        self,
        param_space: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search."""
        
        import itertools
        
        param_names = []
        param_values = []
        
        for param_name, param_config in param_space.items():
            param_type = param_config.get("type")
            
            if param_type == "categorical":
                values = param_config["choices"]
            elif param_type in ["float", "int"]:
                # For grid search, expect 'values' key with explicit grid points
                if "values" in param_config:
                    values = param_config["values"]
                else:
                    raise ValueError(
                        f"Grid search requires 'values' key for parameter '{param_name}'. "
                        f"Example: {{'type': 'float', 'values': [0.1, 0.5, 1.0]}}"
                    )
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
            
            param_names.append(param_name)
            param_values.append(values)
        
        # Generate all combinations
        combinations = []
        for values in itertools.product(*param_values):
            combinations.append(dict(zip(param_names, values)))
        
        return combinations

    def get_best_params(
        self,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Get best parameters with confidence intervals.

        Args:
            confidence_level: Confidence level for intervals (0-1)

        Returns:
            Dictionary with best parameters and confidence info

        Raises:
            RuntimeError: If no optimization has been run
        """
        if self.optimization_result is None:
            raise RuntimeError("No optimization has been run yet.")
        
        result = {
            "best_params": self.optimization_result.best_params,
            "best_score": self.optimization_result.best_score,
            "n_trials": self.optimization_result.n_trials,
            "strategy": self.optimization_result.strategy
        }
        
        # Add confidence intervals for Bayesian optimization
        if self.strategy == "bayesian" and self.study is not None:
            try:
                # Calculate confidence intervals from trial distribution
                scores = [
                    t.value for t in self.study.trials
                    if t.value is not None
                ]
                
                if len(scores) > 1:
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    
                    # Calculate confidence interval
                    from scipy import stats
                    confidence_interval = stats.t.interval(
                        confidence_level,
                        len(scores) - 1,
                        loc=mean_score,
                        scale=stats.sem(scores)
                    )
                    
                    result["confidence_interval"] = {
                        "lower": confidence_interval[0],
                        "upper": confidence_interval[1],
                        "confidence_level": confidence_level
                    }
                    result["score_statistics"] = {
                        "mean": mean_score,
                        "std": std_score,
                        "min": min(scores),
                        "max": max(scores)
                    }
            except Exception as e:
                logger.warning(f"Could not calculate confidence intervals: {e}")
        
        return result

    def visualize_optimization(
        self,
        output_path: str = "optimization_report.html"
    ) -> str:
        """
        Generate optimization visualizations.

        Args:
            output_path: Path to save visualization report

        Returns:
            Path to generated report

        Raises:
            RuntimeError: If no optimization has been run
        """
        if self.optimization_result is None:
            raise RuntimeError("No optimization has been run yet.")
        
        import matplotlib.pyplot as plt
        from pathlib import Path
        
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            f"Hyperparameter Optimization Report ({self.optimization_result.strategy})",
            fontsize=16
        )
        
        # 1. Optimization history (convergence plot)
        ax = axes[0, 0]
        history = self.optimization_result.optimization_history
        ax.plot(history, marker='o', linestyle='-', alpha=0.7)
        ax.axhline(
            y=self.optimization_result.best_score,
            color='r',
            linestyle='--',
            label=f'Best: {self.optimization_result.best_score:.4f}'
        )
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Score')
        ax.set_title('Optimization Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Parameter importance (if available)
        ax = axes[0, 1]
        if self.optimization_result.param_importance:
            params = list(self.optimization_result.param_importance.keys())
            importances = list(self.optimization_result.param_importance.values())
            
            ax.barh(params, importances)
            ax.set_xlabel('Importance')
            ax.set_title('Parameter Importance')
            ax.grid(True, alpha=0.3, axis='x')
        else:
            ax.text(
                0.5, 0.5,
                'Parameter importance\nnot available',
                ha='center', va='center',
                transform=ax.transAxes
            )
            ax.set_title('Parameter Importance')
        
        # 3. Score distribution
        ax = axes[1, 0]
        scores = [t["value"] for t in self.optimization_result.all_trials if t["value"] is not None]
        ax.hist(scores, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(
            x=self.optimization_result.best_score,
            color='r',
            linestyle='--',
            label=f'Best: {self.optimization_result.best_score:.4f}'
        )
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = f"""
        Optimization Summary
        {'=' * 40}
        
        Strategy: {self.optimization_result.strategy}
        Total Trials: {self.optimization_result.n_trials}
        Optimization Time: {self.optimization_result.optimization_time:.2f}s
        
        Best Score: {self.optimization_result.best_score:.4f}
        
        Best Parameters:
        """
        
        for param, value in self.optimization_result.best_params.items():
            if isinstance(value, float):
                summary_text += f"\n  {param}: {value:.6f}"
            else:
                summary_text += f"\n  {param}: {value}"
        
        if self.optimization_result.convergence_info:
            summary_text += f"\n\nConvergence Info:"
            for key, value in self.optimization_result.convergence_info.items():
                summary_text += f"\n  {key}: {value}"
        
        ax.text(
            0.1, 0.9,
            summary_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace'
        )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Optimization visualization saved to {output_path}")
        
        return output_path

    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate optimization summary report.

        Returns:
            Dictionary with summary information

        Raises:
            RuntimeError: If no optimization has been run
        """
        if self.optimization_result is None:
            raise RuntimeError("No optimization has been run yet.")
        
        result = self.optimization_result
        
        # Calculate statistics
        scores = [t["value"] for t in result.all_trials if t["value"] is not None]
        
        summary = {
            "strategy": result.strategy,
            "n_trials": result.n_trials,
            "optimization_time": result.optimization_time,
            "best_score": result.best_score,
            "best_params": result.best_params,
            "score_statistics": {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "median": float(np.median(scores))
            },
            "param_importance": result.param_importance,
            "convergence_info": result.convergence_info,
            "recommendations": self._generate_recommendations()
        }
        
        return summary

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for further tuning."""
        
        if self.optimization_result is None:
            return []
        
        recommendations = []
        result = self.optimization_result
        
        # Check convergence
        if result.convergence_info and result.convergence_info.get("converged"):
            recommendations.append(
                "Optimization converged. Consider expanding the search space "
                "or trying different parameter ranges."
            )
        elif result.n_trials < 50:
            recommendations.append(
                f"Only {result.n_trials} trials were run. "
                "Consider running more trials for better results."
            )
        
        # Check parameter importance
        if result.param_importance:
            # Find parameters with low importance
            low_importance_params = [
                param for param, importance in result.param_importance.items()
                if importance < 0.1
            ]
            if low_importance_params:
                recommendations.append(
                    f"Parameters with low importance: {', '.join(low_importance_params)}. "
                    "Consider fixing these to their best values."
                )
            
            # Find most important parameter
            most_important = max(
                result.param_importance.items(),
                key=lambda x: x[1]
            )
            recommendations.append(
                f"Most important parameter: {most_important[0]} "
                f"(importance: {most_important[1]:.3f}). "
                "Focus on fine-tuning this parameter."
            )
        
        # Strategy-specific recommendations
        if result.strategy == "random":
            recommendations.append(
                "Consider using Bayesian optimization for more efficient search."
            )
        elif result.strategy == "grid":
            recommendations.append(
                "Grid search explored all combinations. "
                "Consider using Bayesian optimization to explore a larger space efficiently."
            )
        
        return recommendations
