# rules_prechecks.py
from __future__ import annotations
from typing import Any

def _to_float(x: Any):
    try:
        if x is None:
            return None
        if isinstance(x, str):
            s = x.strip().replace("€", "").replace(",", ".")
            if s == "":
                return None
            return float(s)
        return float(x)
    except Exception:
        return None

def precheck_row_minimal(row: dict) -> list[dict]:
    issues: list[dict] = []

    # Beispiel: Preis
    price_col_candidates = ["Preis", "price", "Price", "Produktpreis"]
    price_val = None
    used_col = None
    for c in price_col_candidates:
        if c in row:
            used_col = c
            price_val = _to_float(row.get(c))
            break

    if used_col is not None:
        if price_val is None:
            issues.append({
                "type": "missing_or_invalid_price",
                "field": used_col,
                "message": "Preis fehlt oder ist nicht numerisch parsebar."
            })
        elif price_val <= 0:
            issues.append({
                "type": "non_positive_price",
                "field": used_col,
                "message": f"Preis ist <= 0 (value={price_val})."
            })

    return issues