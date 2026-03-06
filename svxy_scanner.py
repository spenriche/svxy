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
VIX3M_SYMBOL = "^VIX3M"
VVIX_SYMBOL = "^VVIX"
DEFAULT_TOP_N = 5
DEFAULT_RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.04"))
NEGATIVE_EXTRINSIC_TOLERANCE = -0.05
MIN_OPEN_INTEREST = int(os.getenv("MIN_OPEN_INTEREST", "100"))
MIN_VOLUME = int(os.getenv("MIN_VOLUME", "5"))
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "0.25"))
SECONDARY_ITM_PERCENT = [5, 30]
PRIMARY_DELTA_RANGE = [0.90, 0.98]
SECONDARY_DELTA_RANGE = [0.70, 0.90]


class ScanError(RuntimeError):
    pass


@dataclass(slots=True)
class RankedOption:
    expiration: str
    strike: float
    bid: float
    ask: float
    mid_price: float
    delta: float
    implied_volatility: float
    volume: int
    open_interest: int
    bid_ask_spread: float
    spread_pct: float
    days_to_expiry: int
    intrinsic_value: float
    extrinsic_value: float
    extrinsic_per_delta: float
    extrinsic_per_day: float
    delta_exposure: float
    capital_efficiency: float
    break_even: float
    replacement_discount: float


@dataclass(slots=True)
class BestCallSpread:
    expiration: str
    buy_strike: float
    sell_strike: float
    buy_mid_price: float
    sell_mid_price: float
    spread_cost: float
    max_value: float
    max_profit: float
    max_return: float
    candidate_count: int


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


def _coerce_int(value: Any) -> int:
    converted = _coerce_float(value)
    if converted is None:
        return 0
    return int(converted)


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


def _fetch_close_history(symbol: str, *, period: str = "3mo") -> list[dict[str, Any]]:
    history = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=False)
    if history.empty or "Close" not in history:
        return []

    points: list[dict[str, Any]] = []
    for index, row in history.iterrows():
        close = _coerce_float(row.get("Close"))
        if close is None:
            continue
        points.append(
            {
                "date": index.date().isoformat(),
                "close": _round(close),
            }
        )
    return points


def _latest_and_previous(points: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    if not points:
        return None, None
    latest = _coerce_float(points[-1].get("close"))
    previous = _coerce_float(points[-2].get("close")) if len(points) > 1 else None
    return latest, previous


def _build_market_regime() -> dict[str, Any]:
    vix_history = _fetch_close_history(VIX_SYMBOL)
    vix3m_history = _fetch_close_history(VIX3M_SYMBOL)
    vvix_history = _fetch_close_history(VVIX_SYMBOL)

    latest_vix, previous_vix = _latest_and_previous(vix_history)
    latest_vix3m, previous_vix3m = _latest_and_previous(vix3m_history)
    latest_vvix, previous_vvix = _latest_and_previous(vvix_history)

    if latest_vix is None:
        return {
            "enabled": True,
            "ideal_entry": False,
            "entry_score": 0,
            "reason": "VIX data unavailable.",
            "history": [],
        }

    term_spread = (
        _round(latest_vix3m - latest_vix)
        if latest_vix3m is not None
        else None
    )
    term_ratio = (
        _round(latest_vix / latest_vix3m)
        if latest_vix3m not in (None, 0)
        else None
    )
    curve_state = (
        "backwardation"
        if latest_vix3m is not None and latest_vix > latest_vix3m
        else "contango"
        if latest_vix3m is not None and latest_vix < latest_vix3m
        else "flat"
    )

    score = 0
    if latest_vix > 30:
        score += 25
    if previous_vix is not None and latest_vix < previous_vix:
        score += 20
    if latest_vix3m is not None and latest_vix < latest_vix3m:
        score += 20
    if previous_vvix is not None and latest_vvix is not None and latest_vvix < previous_vvix:
        score += 15

    ideal_entry = score >= 65
    reason_parts = []
    reason_parts.append(
        "VIX above 30" if latest_vix > 30 else "VIX below 30"
    )
    if previous_vix is not None:
        reason_parts.append("VIX falling" if latest_vix < previous_vix else "VIX rising")
    if latest_vix3m is not None:
        reason_parts.append(
            "curve in contango" if latest_vix < latest_vix3m else "curve in backwardation"
        )
    if latest_vvix is not None and previous_vvix is not None:
        reason_parts.append(
            "VVIX falling" if latest_vvix < previous_vvix else "VVIX elevated"
        )

    history_length = min(len(vix_history), len(vix3m_history), len(vvix_history))
    history = []
    if history_length:
        vix_slice = vix_history[-history_length:]
        vix3m_slice = vix3m_history[-history_length:]
        vvix_slice = vvix_history[-history_length:]
        for vix_point, vix3m_point, vvix_point in zip(vix_slice, vix3m_slice, vvix_slice):
            history.append(
                {
                    "date": vix_point["date"],
                    "vix": vix_point["close"],
                    "vix3m": vix3m_point["close"],
                    "vvix": vvix_point["close"],
                }
            )

    return {
        "enabled": True,
        "ideal_entry": ideal_entry,
        "entry_score": score,
        "curve_state": curve_state,
        "latest_vix": _round(latest_vix),
        "previous_vix": _round(previous_vix) if previous_vix is not None else None,
        "latest_vix3m": _round(latest_vix3m) if latest_vix3m is not None else None,
        "previous_vix3m": _round(previous_vix3m) if previous_vix3m is not None else None,
        "latest_vvix": _round(latest_vvix) if latest_vvix is not None else None,
        "previous_vvix": _round(previous_vvix) if previous_vvix is not None else None,
        "term_spread": term_spread,
        "term_ratio": term_ratio,
        "reason": ", ".join(reason_parts),
        "history": history[-30:],
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
    volume = _coerce_int(raw_option.get("volume"))
    open_interest = _coerce_int(raw_option.get("openInterest"))

    if strike is None or bid is None or ask is None:
        return None
    if bid < 0 or ask < 0:
        return None

    mid_price = (bid + ask) / 2.0
    if mid_price <= 0:
        return None

    bid_ask_spread = ask - bid
    spread_pct = bid_ask_spread / mid_price if mid_price > 0 else math.inf

    years_to_expiry = max(
        (expiration - now).total_seconds() / (365.25 * 24 * 60 * 60),
        0.0,
    )
    days_to_expiry = max((expiration.date() - now.date()).days, 1)

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
    break_even = strike + mid_price
    replacement_discount = delta_exposure - mid_price
    extrinsic_per_day = extrinsic_value / days_to_expiry

    return RankedOption(
        expiration=expiration.date().isoformat(),
        strike=_round(strike),
        bid=_round(bid),
        ask=_round(ask),
        mid_price=_round(mid_price),
        delta=_round(delta),
        implied_volatility=_round(implied_volatility or 0.0),
        volume=volume,
        open_interest=open_interest,
        bid_ask_spread=_round(bid_ask_spread),
        spread_pct=_round(spread_pct),
        days_to_expiry=days_to_expiry,
        intrinsic_value=_round(intrinsic_value),
        extrinsic_value=_round(extrinsic_value),
        extrinsic_per_delta=_round(extrinsic_per_delta),
        extrinsic_per_day=_round(extrinsic_per_day),
        delta_exposure=_round(delta_exposure),
        capital_efficiency=_round(capital_efficiency),
        break_even=_round(break_even),
        replacement_discount=_round(replacement_discount),
    )


def _option_sort_key(option: RankedOption) -> tuple[float, float, float]:
    return (
        option.extrinsic_per_delta,
        -option.capital_efficiency,
        -option.delta,
    )


def _build_exception_reasons(
    *,
    option: RankedOption,
    strict_lower_bound_strike: float,
    strict_upper_bound_strike: float,
    primary_delta_min: float,
    primary_delta_max: float,
) -> list[str]:
    reasons: list[str] = []
    if not (strict_lower_bound_strike <= option.strike <= strict_upper_bound_strike):
        reasons.append("outside primary ITM band")
    if not (primary_delta_min <= option.delta <= primary_delta_max):
        reasons.append(
            f"outside deep ITM delta band {primary_delta_min:.2f}-{primary_delta_max:.2f}"
        )
    if option.open_interest < MIN_OPEN_INTEREST:
        reasons.append(f"open interest below {MIN_OPEN_INTEREST}")
    if option.volume < MIN_VOLUME:
        reasons.append(f"volume below {MIN_VOLUME}")
    if option.spread_pct > MAX_SPREAD_PCT:
        reasons.append(f"spread above {round(MAX_SPREAD_PCT * 100, 2)}%")
    return reasons


def _build_best_call_spread(
    *,
    ticker: yf.Ticker,
    spot_price: float,
    best_option: RankedOption | None,
) -> BestCallSpread | None:
    if best_option is None:
        return None

    option_chain = ticker.option_chain(best_option.expiration)
    calls_frame = option_chain.calls
    candidates: list[BestCallSpread] = []

    for raw_option in calls_frame.to_dict("records"):
        strike = _coerce_float(raw_option.get("strike"))
        bid = _coerce_float(raw_option.get("bid"))
        ask = _coerce_float(raw_option.get("ask"))
        if strike is None or bid is None or ask is None:
            continue
        if strike <= spot_price:
            continue

        sell_mid_price = (bid + ask) / 2.0
        if sell_mid_price <= 0:
            continue

        spread_cost = best_option.mid_price - sell_mid_price
        max_value = strike - best_option.strike
        if spread_cost <= 0 or max_value <= 0:
            continue

        max_profit = max_value - spread_cost
        if max_profit <= 0:
            continue

        max_return = max_profit / spread_cost
        candidates.append(
            BestCallSpread(
                expiration=best_option.expiration,
                buy_strike=best_option.strike,
                sell_strike=_round(strike),
                buy_mid_price=best_option.mid_price,
                sell_mid_price=_round(sell_mid_price),
                spread_cost=_round(spread_cost),
                max_value=_round(max_value),
                max_profit=_round(max_profit),
                max_return=_round(max_return),
                candidate_count=0,
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda spread: (
            -spread.max_return,
            -spread.max_profit,
            spread.spread_cost,
        )
    )
    best_spread = candidates[0]
    best_spread.candidate_count = len(candidates)
    return best_spread


def _calculate_entry_score(
    *,
    market_regime: dict[str, Any],
    top_option: RankedOption | None,
) -> int:
    score = int(market_regime.get("entry_score", 0) or 0)
    if top_option is None:
        return score
    if top_option.spread_pct <= 0.05:
        score += 10
    if top_option.capital_efficiency >= 1.75:
        score += 10
    return min(score, 100)


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
    secondary_lower_bound_strike = spot_price * (1 - 0.30)
    secondary_upper_bound_strike = spot_price * (1 - 0.05)
    primary_delta_min, primary_delta_max = PRIMARY_DELTA_RANGE
    secondary_delta_min, secondary_delta_max = SECONDARY_DELTA_RANGE
    expirations = getattr(ticker, "options", ()) or ()

    qualified_options: list[RankedOption] = []
    exception_options: list[tuple[RankedOption, list[str]]] = []
    for expiration_text in expirations:
        expiration = datetime.strptime(expiration_text, "%Y-%m-%d").replace(tzinfo=UTC)
        if not _within_expiration_window(expiration, now):
            continue

        option_chain = ticker.option_chain(expiration_text)
        calls_frame = option_chain.calls
        for raw_option in calls_frame.to_dict("records"):
            strike = _coerce_float(raw_option.get("strike"))
            if strike is None:
                continue

            in_primary_band = lower_bound_strike <= strike <= upper_bound_strike
            in_secondary_band = (
                secondary_lower_bound_strike <= strike <= secondary_upper_bound_strike
            )
            if not in_primary_band and not in_secondary_band:
                continue

            ranked = _format_option(
                spot_price=spot_price,
                expiration=expiration,
                raw_option=raw_option,
                now=now,
                risk_free_rate=risk_free_rate,
            )
            if ranked is not None:
                in_primary_delta_band = primary_delta_min <= ranked.delta <= primary_delta_max
                in_secondary_delta_band = (
                    secondary_delta_min <= ranked.delta < secondary_delta_max
                )
                exception_reasons = _build_exception_reasons(
                    option=ranked,
                    strict_lower_bound_strike=lower_bound_strike,
                    strict_upper_bound_strike=upper_bound_strike,
                    primary_delta_min=primary_delta_min,
                    primary_delta_max=primary_delta_max,
                )
                if in_primary_band and in_primary_delta_band and not exception_reasons:
                    qualified_options.append(ranked)
                elif in_secondary_band and in_secondary_delta_band:
                    exception_options.append((ranked, exception_reasons))

    qualified_options.sort(key=_option_sort_key)
    exception_options.sort(key=lambda item: _option_sort_key(item[0]))

    market_regime = _build_market_regime()
    top_option = qualified_options[0] if qualified_options else (
        exception_options[0][0] if exception_options else None
    )
    best_call_spread = _build_best_call_spread(
        ticker=ticker,
        spot_price=spot_price,
        best_option=top_option,
    )
    entry_score = _calculate_entry_score(market_regime=market_regime, top_option=top_option)
    market_regime["entry_score"] = entry_score
    market_regime["ideal_entry"] = entry_score >= 75

    top_options: list[dict[str, Any]] = []
    for option in qualified_options:
        payload = asdict(option)
        payload["filter_status"] = "qualified"
        payload["exception_reasons"] = []
        top_options.append(payload)
        if len(top_options) >= top_n:
            break

    if len(top_options) < top_n:
        for option, exception_reasons in exception_options:
            payload = asdict(option)
            payload["filter_status"] = "exception"
            payload["exception_reasons"] = exception_reasons
            top_options.append(payload)
            if len(top_options) >= top_n:
                break

    return {
        "symbol": SVXY_SYMBOL,
        "generated_at": now.isoformat(),
        "svxy_price": _round(spot_price),
        "filters": {
            "expiration_window_months": [6, 18],
            "strike_itm_percent": [40, 60],
            "deep_itm_delta_range": PRIMARY_DELTA_RANGE,
            "lower_bound_strike": _round(lower_bound_strike),
            "upper_bound_strike": _round(upper_bound_strike),
            "secondary_itm_percent": SECONDARY_ITM_PERCENT,
            "secondary_delta_range": SECONDARY_DELTA_RANGE,
            "secondary_lower_bound_strike": _round(secondary_lower_bound_strike),
            "secondary_upper_bound_strike": _round(secondary_upper_bound_strike),
            "min_open_interest": MIN_OPEN_INTEREST,
            "min_volume": MIN_VOLUME,
            "max_spread_pct": MAX_SPREAD_PCT,
        },
        "market_regime": market_regime,
        "candidate_count": len(qualified_options),
        "exception_candidate_count": len(exception_options),
        "top_options": top_options,
        "best_call_spread": asdict(best_call_spread) if best_call_spread else None,
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
    else:
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
                    f"Spread %: {option['spread_pct']}",
                    f"Open Interest: {option['open_interest']}",
                    f"Volume: {option['volume']}",
                    f"Filter Status: {option['filter_status']}",
                    "",
                ]
            )

    best_call_spread = result.get("best_call_spread")
    if best_call_spread:
        lines.extend(
            [
                "Best Call Spread:",
                f"Buy {best_call_spread['expiration']} {best_call_spread['buy_strike']} Call",
                f"Sell {best_call_spread['expiration']} {best_call_spread['sell_strike']} Call",
                f"Spread Cost: {best_call_spread['spread_cost']}",
                f"Max Value: {best_call_spread['max_value']}",
                f"Max Profit: {best_call_spread['max_profit']}",
                f"Max Return: {best_call_spread['max_return']}",
                "",
            ]
        )

    regime = result.get("market_regime", {})
    if regime.get("enabled"):
        lines.extend(
            [
                "Market Regime Filter:",
                f"Ideal Entry: {regime.get('ideal_entry', False)}",
                f"Entry Score: {regime.get('entry_score', 0)}",
                f"Curve State: {regime.get('curve_state', 'Unavailable')}",
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
