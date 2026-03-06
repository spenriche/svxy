from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import yfinance as yf


SVXY_SYMBOL = "SVXY"
VIX_SYMBOL = "^VIX"
DEFAULT_TOP_N = 5
DEFAULT_RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.04"))
NEGATIVE_EXTRINSIC_TOLERANCE = -0.05


class ScanError(RuntimeError):
    pass


@dataclass(slots=True)
class RankedOption:
    expiration: str
    strike: float
    mid_price: float
    delta: float
    implied_volatility: float
    intrinsic_value: float
    extrinsic_value: float
    extrinsic_per_delta: float
    delta_exposure: float
    capital_efficiency: float


def _norm_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _black_scholes_call_delta(
    spot: float,
    strike: float,
    years_to_expiry: float,
    implied_volatility: float,
    risk_free_rate: float,
) -> float | None:
    if (
        spot <= 0
        or strike <= 0
        or years_to_expiry <= 0
        or implied_volatility <= 0
    ):
        return None

    d1 = (
        math.log(spot / strike)
        + (risk_free_rate + 0.5 * implied_volatility**2) * years_to_expiry
    ) / (implied_volatility * math.sqrt(years_to_expiry))
    return _norm_cdf(d1)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None

    if math.isnan(converted):
        return None
    return converted


def _round(value: float) -> float:
    return round(float(value), 4)


def _within_expiration_window(expiration: datetime, now: datetime) -> bool:
    return now + timedelta(days=182) <= expiration <= now + timedelta(days=548)


def _fetch_quote_price(ticker: yf.Ticker, symbol: str) -> float:
    fast_info = getattr(ticker, "fast_info", {}) or {}
    fast_price = _coerce_float(fast_info.get("lastPrice"))
    if fast_price:
        return fast_price

    history = ticker.history(period="5d", interval="1d", auto_adjust=False)
    if history.empty:
        raise ScanError(f"Market price missing for {symbol}.")

    close = _coerce_float(history["Close"].dropna().iloc[-1])
    if close is None:
        raise ScanError(f"Market price missing for {symbol}.")
    return close


def _fetch_vix_regime() -> dict[str, Any]:
    vix_history = yf.Ticker(VIX_SYMBOL).history(
        period="10d",
        interval="1d",
        auto_adjust=False,
    )
    if vix_history.empty or "Close" not in vix_history:
        return {
            "enabled": True,
            "ideal_entry": False,
            "reason": "VIX data unavailable.",
        }

    valid_closes = [
        float(close)
        for close in vix_history["Close"].dropna().tolist()
        if _coerce_float(close) is not None
    ]
    if len(valid_closes) < 2:
        return {
            "enabled": True,
            "ideal_entry": False,
            "reason": "Not enough VIX history.",
        }

    latest = valid_closes[-1]
    previous = valid_closes[-2]
    ideal_entry = latest > 30 and latest < previous
    return {
        "enabled": True,
        "ideal_entry": ideal_entry,
        "latest_vix": round(latest, 4),
        "previous_vix": round(previous, 4),
        "reason": (
            "VIX is above 30 and has started declining."
            if ideal_entry
            else "VIX filter not satisfied."
        ),
    }


def _format_option(
    *,
    spot_price: float,
    expiration: datetime,
    raw_option: dict[str, Any],
    now: datetime,
    risk_free_rate: float,
) -> RankedOption | None:
    strike = _coerce_float(raw_option.get("strike"))
    bid = _coerce_float(raw_option.get("bid"))
    ask = _coerce_float(raw_option.get("ask"))
    implied_volatility = _coerce_float(raw_option.get("impliedVolatility"))

    if strike is None or bid is None or ask is None:
        return None
    if bid < 0 or ask < 0:
        return None

    mid_price = (bid + ask) / 2.0
    if mid_price <= 0:
        return None

    years_to_expiry = max(
        (expiration - now).total_seconds() / (365.25 * 24 * 60 * 60),
        0.0,
    )
    delta = _coerce_float(raw_option.get("delta"))
    if delta is None:
        delta = _black_scholes_call_delta(
            spot=spot_price,
            strike=strike,
            years_to_expiry=years_to_expiry,
            implied_volatility=implied_volatility or 0.0,
            risk_free_rate=risk_free_rate,
        )

    if delta is None or delta <= 0:
        return None

    intrinsic_value = max(0.0, spot_price - strike)
    extrinsic_value = mid_price - intrinsic_value
    if extrinsic_value < NEGATIVE_EXTRINSIC_TOLERANCE:
        return None

    extrinsic_value = max(extrinsic_value, 0.0)
    delta_exposure = delta * spot_price
    capital_efficiency = delta_exposure / mid_price
    extrinsic_per_delta = extrinsic_value / delta

    return RankedOption(
        expiration=expiration.date().isoformat(),
        strike=_round(strike),
        mid_price=_round(mid_price),
        delta=_round(delta),
        implied_volatility=_round(implied_volatility or 0.0),
        intrinsic_value=_round(intrinsic_value),
        extrinsic_value=_round(extrinsic_value),
        extrinsic_per_delta=_round(extrinsic_per_delta),
        delta_exposure=_round(delta_exposure),
        capital_efficiency=_round(capital_efficiency),
    )


def scan_svxy_deep_itm_calls(
    *,
    top_n: int = DEFAULT_TOP_N,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> dict[str, Any]:
    now = datetime.now(UTC)
    ticker = yf.Ticker(SVXY_SYMBOL)
    spot_price = _fetch_quote_price(ticker, SVXY_SYMBOL)

    lower_bound_strike = spot_price * (1 - 0.60)
    upper_bound_strike = spot_price * (1 - 0.40)
    expirations = getattr(ticker, "options", ()) or ()

    ranked_options: list[RankedOption] = []
    for expiration_text in expirations:
        expiration = datetime.strptime(expiration_text, "%Y-%m-%d").replace(tzinfo=UTC)
        if not _within_expiration_window(expiration, now):
            continue

        option_chain = ticker.option_chain(expiration_text)
        calls_frame = option_chain.calls
        for raw_option in calls_frame.to_dict("records"):
            strike = _coerce_float(raw_option.get("strike"))
            if strike is None or not (lower_bound_strike <= strike <= upper_bound_strike):
                continue

            ranked = _format_option(
                spot_price=spot_price,
                expiration=expiration,
                raw_option=raw_option,
                now=now,
                risk_free_rate=risk_free_rate,
            )
            if ranked is not None:
                ranked_options.append(ranked)

    ranked_options.sort(
        key=lambda option: (
            option.extrinsic_per_delta,
            -option.capital_efficiency,
            -option.delta,
        )
    )

    return {
        "symbol": SVXY_SYMBOL,
        "generated_at": now.isoformat(),
        "svxy_price": _round(spot_price),
        "filters": {
            "expiration_window_months": [6, 18],
            "strike_itm_percent": [40, 60],
            "lower_bound_strike": _round(lower_bound_strike),
            "upper_bound_strike": _round(upper_bound_strike),
        },
        "market_regime": _fetch_vix_regime(),
        "candidate_count": len(ranked_options),
        "top_options": [asdict(option) for option in ranked_options[:top_n]],
    }


def format_report(result: dict[str, Any]) -> str:
    lines = [
        f"SVXY Price: {result['svxy_price']}",
        f"Generated At: {result['generated_at']}",
        "",
        "Top Options:",
        "",
    ]

    if not result["top_options"]:
        lines.append("No options matched the filter criteria.")
        return "\n".join(lines)

    for index, option in enumerate(result["top_options"], start=1):
        lines.extend(
            [
                f"Option {index}",
                f"Expiration: {option['expiration']}",
                f"Strike: {option['strike']}",
                f"Mid Price: {option['mid_price']}",
                f"Delta: {option['delta']}",
                f"Intrinsic: {option['intrinsic_value']}",
                f"Extrinsic: {option['extrinsic_value']}",
                f"Extrinsic/Delta: {option['extrinsic_per_delta']}",
                f"Capital Efficiency: {option['capital_efficiency']}",
                "",
            ]
        )

    regime = result.get("market_regime", {})
    if regime.get("enabled"):
        lines.extend(
            [
                "Market Regime Filter:",
                f"Ideal Entry: {regime.get('ideal_entry', False)}",
                f"Reason: {regime.get('reason', 'Unavailable')}",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    result = scan_svxy_deep_itm_calls()
    output_format = os.getenv("OUTPUT_FORMAT", "text").strip().lower()

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return

    print(format_report(result))


if __name__ == "__main__":
    main()
