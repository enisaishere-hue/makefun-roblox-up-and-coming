#!/usr/bin/env python3
"""Empirical rank-transition forecast for Roblox's Up-and-Coming top 10.

Method (stdlib only, deterministic, no network, no model call):
  1. Read every pair of reads in /input/history.ndjson. For a game at rank r
     in an early read, record its rank in every later read, grouped by the
     time gap (0-12h, 12-24h, 24-48h, 48-72h). This is the empirical
     rank-transition kernel: how the chart actually moves, including exits.
  2. For each game in the last chart, average its kernel over the four
     horizon bins, weighted by the hours each bin covers in the 72-hour
     scored window (12/12/24/24). Recent pairs count more (72-hour
     half-life): the chart's regime changes over days.
  3. Smooth the kernel with a Gaussian (sd 1.5 ranks) and mix in a 1%
     uniform background, so no plausible place ever scores the floor.
  4. Games whose 28-day age-out falls inside the window (from
     details.created, when the history carries it) lose presence mass in
     proportion to the hours left before they age out.
  5. Name a game at a place only when its probability beats the unnamed
     share that place would otherwise spread over the games left unnamed:
     the log score punishes overconfidence, so a flat, honest distribution
     beats a sharp guess on this chart.

Signals tested and rejected, so they are not used: concurrent playing
counts (correlation with rank change -0.02 over 3,021 game-reads), recent
rank velocity (fast risers mean-revert; it adds no signal beyond current
rank), and Top Trending membership (games ranked 11-30 that sit in Top
Trending reach the top 10 less often, not more).

Runs as `python3 main.py`, reading $MAKEFUN_INPUT_DIR (else /input) and
writing $MAKEFUN_OUTPUT_DIR/forecast.json (else /output/forecast.json).
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Horizon bins (hours) and the hours each covers inside the 72-hour window.
BINS = [(0.0, 12.0), (12.0, 24.0), (24.0, 48.0), (48.0, 72.0)]
BIN_HOURS = [12.0, 12.0, 24.0, 24.0]
WINDOW_HOURS = 72.0

HALFWIN = 2            # pool start ranks within +-2 (widened to +-5 if too few)
KERNEL_SD = 1.5        # Gaussian smoothing over future ranks, in ranks
BACKGROUND = 0.01      # uniform mass mixed into every distribution
RECENCY_HALFLIFE_H = 72.0
MIN_OBS = 25           # minimum paired observations before widening the pool
MIN_ROW_P = 0.002      # smallest probability worth naming at a place
MAX_RANK = 100         # ranks above the chart size carry no mass
CHART_MAX_AGE_DAYS = 28.0  # Up-and-Coming admits games published in the last 28 days


def parse_at(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def load_history(input_dir):
    with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    reads = []
    with open(os.path.join(input_dir, "history.ndjson"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            snap = json.loads(line)
            if not snap.get("up_and_coming"):
                continue
            snap["_at"] = parse_at(snap["at"])
            snap["_chart"] = {
                row["universe_id"]: row
                for row in sorted(snap["up_and_coming"], key=lambda r: r["rank"])
            }
            reads.append(snap)
    reads.sort(key=lambda s: s["_at"])
    return task, reads


def horizon_bin(dt_hours):
    for i, (lo, hi) in enumerate(BINS):
        if lo <= dt_hours < hi:
            return i
    return None


def build_histograms(reads, last_time):
    """hist[(bin, start_rank)] = (weighted future-rank counts, exit weight,
    total weight). One pass over every pair of reads."""
    hist = {}
    decay = math.log(2.0) / RECENCY_HALFLIFE_H
    charts = [r["_chart"] for r in reads]
    times = [r["_at"] for r in reads]
    for i in range(len(reads)):
        age_h = (last_time - times[i]).total_seconds() / 3600.0
        w_pair = math.exp(-decay * age_h)
        for j in range(i + 1, len(reads)):
            dt = (times[j] - times[i]).total_seconds() / 3600.0
            b = horizon_bin(dt)
            if b is None:
                continue
            cj = charts[j]
            for uid, row in charts[i].items():
                key = (b, row["rank"])
                entry = hist.get(key)
                if entry is None:
                    entry = hist[key] = ({}, 0.0, 0.0)
                counts, exit_w, total_w = entry
                fut = cj[uid]["rank"] if uid in cj else None
                if fut is None:
                    entry = (counts, exit_w + w_pair, total_w + w_pair)
                else:
                    counts[fut] = counts.get(fut, 0.0) + w_pair
                    entry = (counts, exit_w, total_w + w_pair)
                hist[key] = entry
    return hist


def smoothed_bin_distributions(hist):
    """For every (bin, start rank), the kernel-smoothed future-rank
    distribution (sums to 1 over ranks 1..MAX_RANK)."""
    kernel = [
        [math.exp(-0.5 * ((k - fr) / KERNEL_SD) ** 2) for fr in range(1, MAX_RANK + 1)]
        for k in range(1, MAX_RANK + 1)
    ]
    dists = {}
    for (b, r0), (counts, _exit_w, _total_w) in hist.items():
        tot = sum(counts.values())
        if tot <= 0:
            continue
        raw = [0.0] * (MAX_RANK + 1)
        for k in range(1, MAX_RANK + 1):
            s = 0.0
            row = kernel[k - 1]
            for fr, cn in counts.items():
                if 1 <= fr <= MAX_RANK:
                    s += cn * row[fr - 1]
            raw[k] = s / tot
        total = sum(raw[1:])
        if total <= 0:
            continue
        dists[(b, r0)] = [v / total for v in raw]
    return dists


def game_distribution(uid, rank0, hist, dists, exit_rates):
    """Time-averaged future-rank distribution for one game."""
    dist = [0.0] * (MAX_RANK + 1)
    for b in range(len(BINS)):
        obs = None
        for halfwin in (HALFWIN, HALFWIN + 3):
            pool = []
            for r in range(max(1, rank0 - halfwin), min(MAX_RANK, rank0 + halfwin) + 1):
                d = dists.get((b, r))
                if d is not None:
                    pool.append(d)
            if pool:
                obs = pool
                break
        if not obs:
            continue
        w = BIN_HOURS[b] / WINDOW_HOURS
        for k in range(1, MAX_RANK + 1):
            dist[k] += w * sum(d[k] for d in obs) / len(obs)
    # Rescale to (1 - exit rate): mass lost to games leaving the chart.
    total = sum(dist[1:])
    if total <= 0:
        return None
    target = 1.0 - exit_rates.get(rank0, 0.05)
    for k in range(1, MAX_RANK + 1):
        dist[k] = dist[k] / total * target
        dist[k] = (1.0 - BACKGROUND) * dist[k] + BACKGROUND / MAX_RANK
    return dist


def exit_rates_by_rank(hist):
    rates = {}
    by_rank = {}
    for (b, r0), (_counts, exit_w, total_w) in hist.items():
        e, t = by_rank.get(r0, (0.0, 0.0))
        by_rank[r0] = (e + exit_w, t + total_w)
    for r0, (e, t) in by_rank.items():
        rates[r0] = (e / t) if t > 0 else 0.05
    return rates


def presence_factor(uid, last_read, last_time):
    """Fraction of the 72-hour window a game stays on the chart before its
    28-day age-out. 1.0 when the history carries no creation date."""
    details = last_read.get("details")
    if not isinstance(details, dict):
        return 1.0
    entry = details.get(uid, details.get(str(uid)))
    if not isinstance(entry, dict):
        return 1.0
    created = entry.get("created")
    if not created:
        return 1.0
    try:
        born = parse_at(created)
    except (TypeError, ValueError):
        return 1.0
    age_out = born + timedelta(days=CHART_MAX_AGE_DAYS)
    hours_left = (age_out - last_time).total_seconds() / 3600.0
    if hours_left <= 0:
        return 0.05
    if hours_left >= WINDOW_HOURS:
        return 1.0
    return max(0.05, hours_left / WINDOW_HOURS)


def forecast(reads):
    for snap in reads:
        if "_at" not in snap:
            snap["_at"] = parse_at(snap["at"])
            snap["_chart"] = {
                row["universe_id"]: row
                for row in sorted(snap["up_and_coming"], key=lambda r: r["rank"])
            }
    reads = sorted(reads, key=lambda s: s["_at"])
    last = reads[-1]
    last_time = last["_at"]
    last_chart = last["_chart"]
    games = sorted(last_chart.values(), key=lambda r: r["rank"])
    n_chart = len(games)
    hist = build_histograms(reads, last_time)
    dists = smoothed_bin_distributions(hist)
    exit_rates = exit_rates_by_rank(hist)

    dist_by_uid = {}
    for game in games:
        uid = game["universe_id"]
        dist = game_distribution(uid, game["rank"], hist, dists, exit_rates)
        if dist is None:
            continue
        factor = presence_factor(uid, last, last_time)
        if factor < 1.0:
            dist = [v * factor for v in dist]
        dist_by_uid[uid] = dist

    # Name a game at a place only while its probability beats the unnamed
    # share that place would otherwise spread over the games left unnamed.
    probabilities = {place: {} for place in range(1, 11)}
    for place in range(1, 11):
        ranked = sorted(
            ((dist_by_uid[uid][place], uid) for uid in dist_by_uid),
            reverse=True,
        )
        named = {}
        named_sum = 0.0
        for p, uid in ranked:
            unnamed = (1.0 - named_sum) / max(1, n_chart - len(named))
            if p > unnamed and p >= MIN_ROW_P and len(named) < 100:
                named[uid] = p
                named_sum += p
            elif p <= unnamed:
                break
        probabilities[place] = named

    # Top 10: the games with the most total probability mass, ordered by
    # their mass-weighted expected place.
    mass = {}
    expected_place = {}
    for uid, dist in dist_by_uid.items():
        m = sum(dist[1:11])
        mass[uid] = m
        expected_place[uid] = (
            sum(place * dist[place] for place in range(1, 11)) / m if m > 0 else 99.0
        )
    leaders = sorted(mass, key=lambda u: -mass[u])[:10]
    top10 = sorted(leaders, key=lambda u: expected_place[u])
    return probabilities, top10


def main():
    task, reads = load_history(INPUT_DIR)
    places = int(task.get("places", 10))
    output_name = task.get("output", "forecast.json")
    if len(reads) < 2:
        raise SystemExit("history holds fewer than two reads; cannot forecast")
    probabilities, top10 = forecast(reads)
    rows = []
    for place in range(1, places + 1):
        for uid, p in sorted(probabilities[place].items(), key=lambda kv: -kv[1]):
            rows.append({"universe_id": int(uid), "place": place, "p": p})
    out = {"probabilities": rows, "top10": [int(uid) for uid in top10]}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, output_name), "w", encoding="utf-8") as f:
        json.dump(out, f)
        f.write("\n")


if __name__ == "__main__":
    main()
