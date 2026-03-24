import copy
import os
from typing import Any, Dict, Optional

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML config file from the given path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(
    base: Dict[str, Any], override: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Recursively merge two config dicts. Values in override take precedence.
    """
    if override is None:
        return base
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = merge_configs(out[k], v)
        else:
            out[k] = v
    return out


def validate_config(config: Dict[str, Any], required_keys: Optional[list] = None):
    """
    Optionally validate that required keys exist in the config.
    """
    if required_keys:
        missing = [k for k in required_keys if k not in config]
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")


def deep_get(config: dict, dotted_key: str) -> Any:
    """Get value from nested dict using dotted key notation."""
    cur = config
    parts = dotted_key.split(".")
    for part in parts:
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit():
            idx = int(part)
            if 0 <= idx < len(cur):
                cur = cur[idx]
            else:
                raise KeyError(f"List index {idx} out of range for key {dotted_key}")
        else:
            raise KeyError(f"Key {dotted_key} not found in config")
    return cur


def deep_set(config: dict, dotted_key: str, value: Any) -> None:
    """Set value in nested dict using dotted key notation."""
    cur = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part.isdigit():
            # Handle list indices
            idx = int(part)
            if isinstance(cur, list):
                while len(cur) <= idx:
                    cur.append({})
                if not isinstance(cur[idx], dict):
                    cur[idx] = {}
                cur = cur[idx]
            else:
                raise ValueError(f"Cannot index non-list with {part}")
        else:
            # Handle dict keys
            if part not in cur:
                cur[part] = {}
            elif not isinstance(cur[part], (dict, list)):
                cur[part] = {}
            cur = cur[part]

    final_key = parts[-1]
    if final_key.isdigit() and isinstance(cur, list):
        idx = int(final_key)
        while len(cur) <= idx:
            cur.append(None)
        cur[idx] = value
    else:
        # If both existing value and new value are dicts, merge them
        if (
            final_key in cur
            and isinstance(cur[final_key], dict)
            and isinstance(value, dict)
        ):
            cur[final_key] = deep_update(cur[final_key], value)
        else:
            cur[final_key] = value


def deep_update(base: dict, upd: dict) -> dict:
    """Deep merge two dictionaries."""
    # Handle None/empty base case
    if base is None:
        base = {}

    # Handle None/empty update case
    if upd is None:
        upd = {}

    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out
