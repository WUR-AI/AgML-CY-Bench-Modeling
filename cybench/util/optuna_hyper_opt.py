from typing import Any, Dict, List
import optuna
from omegaconf import DictConfig


#class OptunaOptimizer:

def extract_search_space(
    cfg: DictConfig,
    trial: optuna.Trial,
    prefix: str = "",
    config_group_map: Dict[str, List[str]] = None
) -> Dict[str, Any]:
    """
    Extract search spaces from config, including config group choices.

    Args:
        cfg: Hydra config
        trial: Optuna trial
        prefix: Prefix for parameter names
        config_group_map: Map of config keys to available config file choices
                         e.g., {"processor": ["cnn", "lstm", "transformer"]}
    """
    suggestions = {}

    def _recursive_extract(node, current_prefix):
        if isinstance(node, DictConfig):
            for key, value in node.items():
                if key == "_search_":
                    # Found search space definition
                    for param_name, search_range in value.items():
                        full_name = f"{current_prefix}.{param_name}" if current_prefix else param_name

                        # Check if this is a config group choice
                        if config_group_map and param_name in config_group_map:
                            # This is a choice between different config files
                            suggestions[param_name] = trial.suggest_categorical(
                                full_name,
                                config_group_map[param_name]
                            )
                        elif isinstance(search_range, list):
                            if len(search_range) == 2 and all(isinstance(x, (int, float)) for x in search_range):
                                # Numeric range
                                if all(isinstance(x, int) for x in search_range):
                                    suggestions[param_name] = trial.suggest_int(
                                        full_name, search_range[0], search_range[1]
                                    )
                                else:
                                    suggestions[param_name] = trial.suggest_float(
                                        full_name, search_range[0], search_range[1]
                                    )
                            else:
                                # Categorical (including config names)
                                suggestions[param_name] = trial.suggest_categorical(
                                    full_name, search_range
                                )
                elif key != "_target_" and isinstance(value, DictConfig):
                    # Recurse into nested configs
                    new_prefix = f"{current_prefix}.{key}" if current_prefix else key
                    nested_suggestions = _recursive_extract(value, new_prefix)
                    if nested_suggestions:
                        suggestions[key] = nested_suggestions

        return suggestions

    return _recursive_extract(cfg, prefix)
