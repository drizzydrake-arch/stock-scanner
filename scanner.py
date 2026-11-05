#!/usr/bin/env python3
"""
MEP Stock Scanner - finds MACD + EMA pullback trades across the whole US market,
explains each one in plain English and posts them to Discord.

Same rules as the "MEP Signals" Pine Script:
  BUY : price above a rising 50-day average, pulled back to the 9-day average,
        MACD turned up from below zero, price closed back above the 9-day average,
        volume above normal, RSI healthy (40-65, rising, dipped during the pullback),
        liquid enough to trade ($5M+ a day), no warning signs (chop, gaps, thin volume, earnings).

Extras:
  * Scans every common stock on NASDAQ / NYSE / AMEX (about 5,000+), free data.
  * Checks how this exact setup performed on the same stock over the last 2 years.
  * Gives each trade a 0-100 score and an A/B/C grade so you can focus on the best.
  * Lists "almost ready" stocks to watch, and explains every blocked setup.

Run after the US close. Education only - not financial advice.
"""

import datetime as dt
import io
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ------------------------------- SETTINGS -------------------------------
CFG = dict(
    # strategy (same as the Pine Script)
    ema_fast=9, ema_trend=50,
    macd_fast=12, macd_slow=26, macd_sig=9,
    atr_len=14, vol_len=20,
    confirm_win=3, pb_bars=8, pb_tol=0.5,
    vol_mult=1.0, slope_len=10, flat_thr=1.0,
    swing_len=5, stop_buf=0.2, rr=2.0, cooldown=3,
    weave_win=10, weave_max=3, tangle_thr=0.05,
    thin_vol=0.5, gap_mult=1.5, earn_days=3,
    near_thr=0.2,
    # RSI filter (momentum health check)
    use_rsi=True,
    rsi_len=14,
    rsi_min=40,                # BUY: RSI must be at least this (not still collapsing)
    rsi_max=65,                # BUY: RSI must be at most this (not already overheated)
    rsi_dip=50,                # BUY: RSI dipped to this or lower during the pullback
    # liquidity filter (how easily you can get in and out)
    use_liquidity=True,
    min_dollar_volume=5_000_000,   # average $ traded per day (price x shares)
    max_order_pct_adv=1.0,         # your order must be under this % of a normal day's $ volume
    max_zero_vol_days=0,           # allowed days with zero volume in the last 20
    # beginner-friendly switches
    allow_shorts=False,        # shorting is advanced; keep False until you are experienced
    use_spy_filter=True,       # only buy when the S&P 500 is also in an uptrend
    # which stocks to include
    min_price=5.0,
    min_avg_volume=500_000,    # average shares traded per day
    # your account (used for the "how many shares" line)
    account=10000.0,
    risk_pct=1.0,
    # output
    max_trades_shown=8,
    max_watch_shown=10,
    max_blocked_shown=10,
    chunk=150,
    history_period="2y",
)

NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"
BAD_NAME = (r"\bwarrants?\b|\brights?\b|\bunits?\b|\bpreferred\b|notes due|"
            r"depositary|subordinated|acquisition corp")

FALLBACK = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "AVGO", "NFLX",
    "JPM", "BAC", "V", "MA", "XOM", "CVX", "UNH", "LLY", "COST", "WMT",
    "HD", "CRM", "ORCL", "ADBE", "QCOM", "INTC", "DIS", "PYPL", "UBER", "SHOP",
]

GREEN, RED, YELLOW, BLUE = 0x2ECC71, 0xE74C3C, 0xF1C40F, 0x3498DB


# ------------------------------ INDICATORS ------------------------------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def barssince(cond):
    grp = cond.cumsum()
    since = (~cond).astype(int).groupby(grp).cumsum().astype(float)
    since[grp == 0] = np.nan
    return since


def build(df):
    """Compute every indicator and setup flag for one stock. Returns a DataFrame."""
    c = CFG
    d = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    d = d.dropna(subset=["Open", "High", "Low", "Close"])
    o, h, l, cl, v = d["Open"], d["High"], d["Low"], d["Close"], d["Volume"]

    ema_f, ema_t = ema(cl, c["ema_fast"]), ema(cl, c["ema_trend"])
    macd = ema(cl, c["macd_fast"]) - ema(cl, c["macd_slow"])
    sig = ema(macd, c["macd_sig"])
    hist = macd - sig

    tr = pd.concat([h - l, (h - cl.shift()).abs(), (l - cl.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / c["atr_len"], adjust=False).mean()
    vol_avg = v.rolling(c["vol_len"]).mean()
    vol_ratio = v / vol_avg

    slope = (ema_t - ema_t.shift(c["slope_len"])) / ema_t.shift(c["slope_len"]) * 100
    is_flat = slope.abs() < c["flat_thr"]

    # RSI (Wilder), 0-100 momentum gauge
    delta = cl.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / c["rsi_len"], adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / c["rsi_len"], adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).fillna(100.0).where(avg_gain + avg_loss > 0, 50.0)

    # liquidity: average dollars traded per day, and days with no trading
    dollar_vol = (cl * v).rolling(c["vol_len"]).mean()
    zero_days = (v <= 0).astype(int).rolling(c["vol_len"]).sum()

    # warning signs
    crossed = (cl > ema_t) != (cl.shift() > ema_t.shift())
    chop = is_flat & (crossed.astype(int).rolling(c["weave_win"]).sum() >= c["weave_max"])
    tangled = hist.abs().rolling(5).max() < atr * c["tangle_thr"]
    thin = v < vol_avg * c["thin_vol"]
    gap = (o - cl.shift()).abs() > atr * c["gap_mult"]
    illiquid = (pd.Series(c["use_liquidity"], index=d.index)
                & ((dollar_vol < c["min_dollar_volume"]) | (zero_days > c["max_zero_vol_days"])))
    anomaly = chop | tangled | thin | gap | illiquid

    # setup rules
    trend_up = (cl > ema_t) & (slope > -c["flat_thr"])
    trend_dn = (cl < ema_t) & (slope < c["flat_thr"])
    cross_up = (macd > sig) & (macd.shift() <= sig.shift()) & (macd < 0)
    cross_dn = (macd < sig) & (macd.shift() >= sig.shift()) & (macd > 0)
    recent_up = barssince(cross_up) <= c["confirm_win"]
    recent_dn = barssince(cross_dn) <= c["confirm_win"]
    touch_l = (l <= ema_f + atr * c["pb_tol"]).astype(int).rolling(c["pb_bars"]).sum() > 0
    touch_s = (h >= ema_f - atr * c["pb_tol"]).astype(int).rolling(c["pb_bars"]).sum() > 0
    vol_ok = vol_ratio >= c["vol_mult"]

    # RSI rules: BUY needs a healthy, rising RSI that dipped during the pullback
    rsi_up = rsi > rsi.shift()
    rsi_ok_l = ((rsi >= c["rsi_min"]) & (rsi <= c["rsi_max"]) & rsi_up
                & (rsi.rolling(c["pb_bars"]).min() <= c["rsi_dip"]))
    rsi_ok_s = ((rsi <= 100 - c["rsi_min"]) & (rsi >= 100 - c["rsi_max"]) & ~rsi_up
                & (rsi.rolling(c["pb_bars"]).max() >= 100 - c["rsi_dip"]))
    if not c["use_rsi"]:
        rsi_ok_l = pd.Series(True, index=d.index)
        rsi_ok_s = pd.Series(True, index=d.index)

    long_raw = trend_up & recent_up & touch_l & (cl > ema_f) & (hist > 0) & vol_ok & rsi_ok_l
    short_raw = trend_dn & recent_dn & touch_s & (cl < ema_f) & (hist < 0) & vol_ok & rsi_ok_s

    rising = (hist > hist.shift()) & (hist.shift() > hist.shift(2))
    near_long = (trend_up & touch_l & (macd < 0) & (hist < 0) & rising
                 & ((hist.abs() / atr) < c["near_thr"]) & ~long_raw)

    x = pd.DataFrame({
        "o": o, "h": h, "l": l, "c": cl, "v": v,
        "ema_f": ema_f, "ema_t": ema_t, "macd": macd, "sig": sig, "hist": hist,
        "atr": atr, "vol_avg": vol_avg, "vol_ratio": vol_ratio, "slope": slope,
        "rsi": rsi, "dollar_vol": dollar_vol, "zero_days": zero_days, "rsi_ok_l": rsi_ok_l,
        "chop": chop, "tangled": tangled, "thin": thin, "gap": gap, "illiquid": illiquid,
        "anomaly": anomaly,
        "long_raw": long_raw, "short_raw": short_raw, "near_long": near_long,
        "stop_long": l.rolling(c["swing_len"]).min() - atr * c["stop_buf"],
        "stop_short": h.rolling(c["swing_len"]).max() + atr * c["stop_buf"],
        "hi60": h.rolling(60).max(), "lo60": l.rolling(60).min(),
    })
    x["long_setup"] = x["long_raw"] & ~x["anomaly"]
    x["short_setup"] = x["short_raw"] & ~x["anomaly"]
    return x


def backtest(x, allow_shorts=False):
    """Replay the same rules over the stock's history. Returns simple stats in R units."""
    c = CFG
    n = len(x)
    cl, h, l = x["c"].values, x["h"].values, x["l"].values
    ema_t, macd, sig = x["ema_t"].values, x["macd"].values, x["sig"].values
    ls, ss = x["long_setup"].values, x["short_setup"].values
    sl_, ss_ = x["stop_long"].values, x["stop_short"].values

    pos, entry, stop, tgt, risk, last_exit = 0, 0.0, 0.0, 0.0, 0.0, -999
    results = []
    for i in range(60, n):
        if pos != 0:
            r = None
            if pos == 1:
                if l[i] <= stop:
                    r = -1.0
                elif h[i] >= tgt:
                    r = c["rr"]
                elif (macd[i] < sig[i] and macd[i - 1] >= sig[i - 1]) or cl[i] < ema_t[i]:
                    r = (cl[i] - entry) / risk
            else:
                if h[i] >= stop:
                    r = -1.0
                elif l[i] <= tgt:
                    r = c["rr"]
                elif (macd[i] > sig[i] and macd[i - 1] <= sig[i - 1]) or cl[i] > ema_t[i]:
                    r = (entry - cl[i]) / risk
            if r is not None:
                results.append(r)
                pos, last_exit = 0, i
                continue
        if pos == 0 and i - last_exit > c["cooldown"]:
            if ls[i] and sl_[i] < cl[i]:
                pos, entry, stop = 1, cl[i], sl_[i]
                risk = entry - stop
                tgt = entry + c["rr"] * risk
            elif allow_shorts and ss[i] and ss_[i] > cl[i]:
                pos, entry, stop = -1, cl[i], ss_[i]
                risk = stop - entry
                tgt = entry - c["rr"] * risk
    m = len(results)
    wins = sum(1 for r in results if r > 0)
    return dict(n=m, wins=wins, avg_r=float(np.mean(results)) if m else 0.0)


def score_trade(last, direction, stats, spy):
    c = CFG
    stop = last["stop_long"] if direction == "BUY" else last["stop_short"]
    risk = abs(last["c"] - stop)
    pts = {}  # adds up to 100
    pts["Trend strength"] = float(min(15, abs(last["slope"]) * 3))
    pts["Volume"] = float(np.clip((last["vol_ratio"] - 0.8) / 1.2 * 15, 0, 15))
    if spy is None:
        pts["Market"] = 7.5
    else:
        aligned = (direction == "BUY" and spy == "UP") or (direction == "SELL" and spy == "DOWN")
        pts["Market"] = 15.0 if aligned else 0.0
    room = ((last["hi60"] - last["c"]) if direction == "BUY" else (last["c"] - last["lo60"])) / risk
    pts["Room to run"] = float(np.clip(room / c["rr"], 0, 1) * 15)
    if stats["n"] >= 3:
        pts["Track record"] = float(np.clip((stats["avg_r"] + 0.5) / 1.5, 0, 1) * 20)
    else:
        pts["Track record"] = 8.0
    # RSI: best when it is recovering from the dip but has plenty of room before overheated (50 to 60)
    rsi = float(last["rsi"])
    rsi_mid = 55.0 if direction == "BUY" else 45.0
    pts["RSI"] = float(np.clip(1 - abs(rsi - rsi_mid) / 20, 0, 1) * 10)
    # Liquidity: 5M dollars/day scores 0, 50M+ scores full marks
    dv = max(float(last["dollar_vol"]), 1.0)
    pts["Liquidity"] = float(np.clip(math.log10(dv / c["min_dollar_volume"]) / 1.0, 0, 1) * 10)
    total = float(sum(pts.values()))
    grade = "A" if total >= 75 else "B" if total >= 60 else "C"
    return total, grade, pts


# ---------------------------- DATA / UNIVERSE ----------------------------
def _load_list(url, sym_col):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), sep="|", dtype=str)
    df = df[df[sym_col].notna() & ~df[sym_col].str.startswith("File Creation")]
    if "Test Issue" in df:
        df = df[df["Test Issue"] == "N"]
    if "ETF" in df:
        df = df[df["ETF"] == "N"]
    if "Financial Status" in df:
        df = df[df["Financial Status"] == "N"]
    name = df["Security Name"].fillna("")
    df = df[~name.str.contains(BAD_NAME, case=False, regex=True)]
    return [s.strip() for s in df[sym_col] if s.strip().isalpha() and len(s.strip()) <= 5]


def get_universe():
    syms = []
    for url, col in ((NASDAQ_URL, "Symbol"), (OTHER_URL, "ACT Symbol")):
        try:
            syms += _load_list(url, col)
        except Exception as e:
            print(f"(Could not load {url}: {e})")
    syms = sorted(set(syms))
    if len(syms) < 500:
        print("Using the small fallback list instead of the full market.")
        return FALLBACK
    return syms


def download_chunk(tickers):
    for attempt in range(3):
        try:
            return yf.download(tickers, period=CFG["history_period"], interval="1d",
                               group_by="ticker", auto_adjust=True, threads=True,
                               progress=False)
        except Exception:
            time.sleep(5 * (attempt + 1))
    return None


def split_frames(data, tickers):
    out = {}
    if data is None or data.empty:
        return out
    if not isinstance(data.columns, pd.MultiIndex):
        if len(tickers) == 1 and len(data) >= 120:
            out[tickers[0]] = data
        return out
    for t in tickers:
        try:
            d = data[t].dropna(how="all")
            if len(d) >= 120:
                out[t] = d
        except KeyError:
            pass
    return out


def days_to_earnings(ticker):
    try:
        cal = yf.Ticker(ticker).calendar
        if isinstance(cal, dict):
            d = cal.get("Earnings Date")
            if d:
                first = d[0] if isinstance(d, (list, tuple)) else d
                return (pd.Timestamp(first).date() - dt.date.today()).days
    except Exception:
        pass
    return None


def spy_trend():
    try:
        df = yf.Ticker("SPY").history(period="1y", interval="1d", auto_adjust=True)
        cl = df["Close"]
        trend = "UP" if cl.iloc[-1] > ema(cl, CFG["ema_trend"]).iloc[-1] else "DOWN"
        return trend, df.index[-1].date()
    except Exception:
        return None, None


# ------------------------------- DISCORD --------------------------------
def post(content=None, embeds=None):
    payload = {}
    if content:
        payload["content"] = content[:1990]
        print("\n" + content)
    if embeds:
        payload["embeds"] = embeds
        for e in embeds:
            print("\n" + e.get("title", "") + "\n" + e.get("description", ""))
    url = os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        return
    for _ in range(4):
        try:
            r = requests.post(url, json=payload, timeout=20)
            if r.status_code == 429:
                time.sleep(float(r.json().get("retry_after", 2)) + 0.5)
                continue
            if r.status_code >= 400:
                print(f"(Discord error {r.status_code}: {r.text[:200]})")
            break
        except Exception as e:
            print(f"(Discord send failed: {e})")
            time.sleep(2)
    time.sleep(1.2)


def money(v):
    return f"${v:,.2f}"


def money_short(v):
    if v >= 1e9:
        return f"${v / 1e9:.1f}B"
    if v >= 1e6:
        return f"${v / 1e6:.0f}M"
    return f"${v / 1e3:.0f}K"


def reasons(last):
    r = []
    if last["chop"]:
        r.append("price is zig-zagging sideways (choppy)")
    if last["tangled"]:
        r.append("MACD lines are tangled, so there is no clear signal")
    if last["thin"]:
        r.append("unusually low trading volume")
    if last["gap"]:
        r.append("a big price gap, which makes the stop unreliable")
    if last["illiquid"]:
        r.append("not enough daily trading in dollars, so it can be hard to buy or sell at a fair price")
    return r


def track_record_text(t, stats):
    if stats["n"] < 3:
        return (f"Only {stats['n']} past trade(s) of this kind on {t} in 2 years, "
                f"too few to judge. Treat it as untested.")
    losses = stats["n"] - stats["wins"]
    sign = "+" if stats["avg_r"] >= 0 else ""
    return (f"In the last 2 years this setup on {t} triggered {stats['n']} times: "
            f"{stats['wins']} ended in profit, {losses} in a loss. "
            f"Average result: {sign}{stats['avg_r']:.1f}R "
            f"(1R = the amount you risk on one trade). Past results do not guarantee the future.")


def trade_embed(r):
    c = CFG
    buy = r["direction"] == "BUY"
    stars = {"A": "A (strongest)", "B": "B (good)", "C": "C (weaker)"}[r["grade"]]
    risk_dollars = c["account"] * c["risk_pct"] / 100
    lines = [
        "**THE PLAN**",
        f"• {'Buy' if buy else 'Short-sell'} around **{money(r['price'])}** (today's closing price)",
        f"• **Safety exit (stop): {money(r['stop'])}**. If the price reaches it, "
        f"{'sell' if buy else 'buy back'} and accept a small loss ({money(r['risk'])} per share)",
        f"• **Profit target: {money(r['target'])}**, about {c['rr']:.0f}x the amount you risk",
        f"• With a {money(c['account'])} account risking {c['risk_pct']:g}% ({money(risk_dollars)}): "
        f"about **{r['shares']} shares** (position {money(r['shares'] * r['price'])})",
        "",
        "**WHY IT WAS PICKED**",
    ]
    if buy:
        lines += [
            f"• Uptrend: price is above its 50-day average, which is rising ({r['slope']:+.1f}%)",
            "• Pullback: it dipped back to its 9-day average, then closed back above it",
            "• Momentum: MACD turned up from below zero (a fresh buy signal)",
            f"• RSI is {r['rsi']:.0f} and rising: it cooled off during the dip and is recovering, "
            f"with room to climb before it is overheated (70+)",
            f"• Volume: {r['vol_ratio']:.1f}x normal, so more traders than usual joined in",
            f"• Liquidity: about {money_short(r['dollar_vol'])} traded per day. Your order is only "
            f"{r['order_pct']:.3f}% of that, so you can get in and out easily",
            "• Market: the S&P 500 is also in an uptrend" if r["spy"] == "UP"
            else "• Market: S&P 500 trend could not be checked",
        ]
    else:
        lines += [
            f"• Downtrend: price is below its falling 50-day average ({r['slope']:+.1f}%)",
            "• Bounce failed at the 9-day average",
            "• MACD turned down from above zero",
            f"• RSI is {r['rsi']:.0f} and falling",
            f"• Volume: {r['vol_ratio']:.1f}x normal",
            f"• Liquidity: about {money_short(r['dollar_vol'])} traded per day "
            f"(your order is {r['order_pct']:.3f}% of that)",
            "• ADVANCED: short selling needs a margin account and can lose more than you put in",
        ]
    lines += ["", "**HOW IT DID BEFORE ON THIS STOCK**", track_record_text(r["ticker"], r["stats"]), ""]
    ed = r["earn_days"]
    lines.append(f"**Earnings:** next report in about {ed} days (outside the 3-day danger zone)."
                 if ed is not None else "**Earnings:** date unknown, check it before buying.")
    lines.append("\nPlace the stop order when you enter. Not financial advice.")
    return {
        "title": f"{'🟢 BUY' if buy else '🔴 SELL'} {r['ticker']}  |  Grade {stars}  |  Score {r['score']:.0f}/100",
        "description": "\n".join(lines)[:4000],
        "color": GREEN if buy else RED,
    }


def watch_embed(items):
    lines = ["These are **not** buy signals yet. Momentum is turning up but has not crossed. "
             "Look at them again tomorrow.\n"]
    for t, last in items:
        lines.append(f"• **{t}** {money(last['c'])}: in an uptrend, pulled back to its 9-day line, "
                     f"MACD is close to turning up")
    return {"title": "👀 ALMOST READY (watch list)", "description": "\n".join(lines)[:4000], "color": BLUE}


def blocked_embed(items):
    lines = ["The pattern appeared, but a safety check said skip it. This protects you from bad trades.\n"]
    for t, d, why in items:
        lines.append(f"• **{t}** ({d}): {why}")
    return {"title": "⛔ BLOCKED (and why)", "description": "\n".join(lines)[:4000], "color": YELLOW}


GLOSSARY = (
    "**9-day / 50-day average (EMA):** the average price over recent days. The 50-day shows the big "
    "trend, the 9-day shows the short-term swings.\n\n"
    "**MACD:** a momentum gauge. When it crosses up from low levels, buyers are taking over.\n\n"
    "**Pullback:** a small dip inside an uptrend. Buying dips is safer than chasing price highs.\n\n"
    "**Stop (safety exit):** the price where you sell to limit your loss. Always set it when you buy.\n\n"
    "**Target:** the price where you take profit.\n\n"
    "**R:** the amount you risk on a trade. A +2R win earns twice what you risked.\n\n"
    "**Volume:** how many shares traded. High volume means more people agree with the move.\n\n"
    "**Chop:** price going sideways in a zig-zag. Signals are unreliable here, so the bot skips it.\n\n"
    "**Gap:** price jumping overnight. Your stop can be skipped over, so the bot avoids it.\n\n"
    "**Earnings:** company reports can move the price 10%+ in a day, so the bot skips stocks "
    "reporting within 3 days.\n\n"
    "**Position size:** how many shares to buy so a stopped-out trade only costs about 1% of your "
    "account.\n\n"
    "**RSI:** a 0 to 100 gauge of recent buying vs selling. Under 30 is 'oversold', over 70 is "
    "'overheated'. The bot wants it between 40 and 65 and rising: recovered from the dip but not "
    "stretched.\n\n"
    "**Liquidity:** how easily you can buy or sell without moving the price. The bot only uses "
    "stocks that trade at least $5 million a day, and checks your order is a tiny slice of that.\n\n"
    "**Grade A/B/C:** a ranking, not a promise. It rewards strong trends, high volume, room to run, "
    "and a good past record."
)


# --------------------------------- MAIN ---------------------------------
def main():
    c = CFG
    if "--glossary" in sys.argv:
        post(embeds=[{"title": "📘 Beginner's glossary", "description": GLOSSARY, "color": BLUE}])
        return

    now_et = pd.Timestamp.now(tz="America/New_York")
    manual = os.getenv("GITHUB_EVENT_NAME") == "workflow_dispatch" or "--force" in sys.argv
    spy, spy_date = spy_trend()
    if (not manual) and spy_date is not None and now_et.weekday() < 5 and spy_date < now_et.date():
        print("US market was closed today (holiday?). Nothing to scan.")
        return
    partial = (manual and spy_date == now_et.date() and now_et.weekday() < 5
               and (now_et.hour < 16 or (now_et.hour == 16 and now_et.minute < 10)))

    tickers = get_universe()
    print(f"Scanning {len(tickers)} tickers...")

    cands, near, blocked = [], [], []
    scanned = passed = 0
    for k in range(0, len(tickers), c["chunk"]):
        chunk = tickers[k:k + c["chunk"]]
        frames = split_frames(download_chunk(chunk), chunk)
        scanned += len(frames)
        for t, df in frames.items():
            try:
                x = build(df)
                last = x.iloc[-1]
                if not (last["c"] >= c["min_price"] and last["vol_avg"] >= c["min_avg_volume"]):
                    continue
                passed += 1
                if last["long_setup"] and last["stop_long"] < last["c"]:
                    cands.append((t, "BUY", x))
                elif c["allow_shorts"] and last["short_setup"] and last["stop_short"] > last["c"]:
                    cands.append((t, "SELL", x))
                elif last["long_raw"] and last["anomaly"]:
                    blocked.append((t, "BUY", "; ".join(reasons(last))))
                elif last["near_long"] and not last["anomaly"]:
                    near.append((t, last))
            except Exception:
                continue
        if (k // c["chunk"]) % 10 == 0:
            print(f"  ...{min(k + c['chunk'], len(tickers))}/{len(tickers)}")

    # ---- second pass: earnings, market filter, history check, score
    cands.sort(key=lambda z: float(z[2].iloc[-1]["vol_ratio"]), reverse=True)
    results = []
    for t, direction, x in cands[:40]:
        last = x.iloc[-1]
        if c["use_spy_filter"] and spy is not None:
            if direction == "BUY" and spy == "DOWN":
                blocked.append((t, "BUY", "the S&P 500 is in a downtrend, so buying is riskier today"))
                continue
            if direction == "SELL" and spy == "UP":
                blocked.append((t, "SELL", "the S&P 500 is in an uptrend"))
                continue
        ed = days_to_earnings(t)
        if ed is not None and 0 <= ed <= c["earn_days"]:
            blocked.append((t, direction, f"earnings report in {ed} day(s), which can cause a big price jump"))
            continue
        stats = backtest(x, c["allow_shorts"])
        total, grade, pts = score_trade(last, direction, stats, spy)
        stop = float(last["stop_long"] if direction == "BUY" else last["stop_short"])
        price = float(last["c"])
        risk = abs(price - stop)
        shares = int(math.floor(c["account"] * c["risk_pct"] / 100 / risk)) if risk > 0 else 0
        shares = max(0, min(shares, int(c["account"] // price)))
        order_pct = shares * price / max(float(last["dollar_vol"]), 1.0) * 100
        if c["use_liquidity"] and order_pct > c["max_order_pct_adv"]:
            blocked.append((t, direction, f"your order would be {order_pct:.1f}% of a normal day's "
                                          f"trading, which could move the price against you"))
            continue
        target = price + c["rr"] * risk if direction == "BUY" else price - c["rr"] * risk
        results.append(dict(
            ticker=t, direction=direction, price=price, stop=stop, target=target, risk=risk,
            shares=shares, score=total, grade=grade, stats=stats, slope=float(last["slope"]),
            vol_ratio=float(last["vol_ratio"]), spy=spy, earn_days=ed,
            rsi=float(last["rsi"]), dollar_vol=float(last["dollar_vol"]), order_pct=order_pct,
        ))
    results.sort(key=lambda r: r["score"], reverse=True)

    # near setups: filter out earnings week for the few we show
    near.sort(key=lambda z: abs(float(z[1]["hist"]) / float(z[1]["atr"])))
    near_show = []
    for t, last in near[:25]:
        ed = days_to_earnings(t)
        if ed is not None and 0 <= ed <= c["earn_days"]:
            continue
        near_show.append((t, last))
        if len(near_show) >= c["max_watch_shown"]:
            break

    # ---- report
    mood = ("🟢 Uptrend. Conditions are good for buying." if spy == "UP"
            else "🔴 Downtrend. Buy signals are blocked to protect you." if spy == "DOWN"
            else "⚪ Unknown")
    header = (f"📊 **Daily scan: {now_et.strftime('%a %d %b %Y')}**\n"
              f"Market mood (S&P 500): {mood}\n"
              f"Checked **{scanned:,}** US stocks → **{passed:,}** liquid enough → "
              f"**{len(results)}** trade setup(s), **{len(near_show)}** almost ready, "
              f"**{len(blocked)}** blocked by safety checks.")
    if partial:
        header += "\n⚠️ The market is still open, so today's candle is not finished. Results may change."
    if not results:
        header += "\n\nNo trades today. Waiting for good setups is part of the strategy."
    post(content=header)

    for r in results[:c["max_trades_shown"]]:
        post(embeds=[trade_embed(r)])
    if len(results) > c["max_trades_shown"]:
        post(content=f"+{len(results) - c['max_trades_shown']} more lower-ranked setups not shown.")
    if near_show:
        post(embeds=[watch_embed(near_show)])
    if blocked:
        post(embeds=[blocked_embed(blocked[:c["max_blocked_shown"]])])

    if results:
        os.makedirs("results", exist_ok=True)
        out = pd.DataFrame([{k: v for k, v in r.items() if k != "stats"} for r in results])
        out.to_csv(f"results/signals_{dt.date.today()}.csv", index=False)


if __name__ == "__main__":
    main()
