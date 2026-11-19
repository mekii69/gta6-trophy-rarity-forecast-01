"""gta6-trophy-rarity-forecast-01: forecast each GTA VI PS5 trophy's Exophase
earn rate at the case target, as five quantiles, scored by pinball loss on
the logit scale.

Model (standard library only, deterministic, offline).
------------------------------------------------------
A trophy's earn rate is a ratio: rate = 100 * E / N, where N is the players
Exophase tracks and E the tracked players who earned that trophy.  The
practice data shows why forecasting the ratio directly fails: on launch
night Exophase's page sat unchanged (638 players, every rate flat) and then
refreshed once to 3,868 players, so every rate but the 100% one fell -- a
median logit drop of about -1.6 -- because 3,230 newly tracked players held
almost no trophies yet (only 103 new earns outside the 100% trophy).  New
players dilute rates; playing time earns trophies back.  Which force wins
over the final's 4.5 days is what the history up to the deadline has to say,
so the program models the two counts separately and ensembles five views
of the same counts, all fitted on the case's own history (recent 48 h when
there is enough of it):

  A.  Counts power law: log E_i and log N fit against x = ln(1 + hours
      since release); both extrapolated, rate = 100 * E / N.
  A2. Counts momentum: log E_i and log N fit against days since release
      (daily growth rates, capped), same ratio.  A decelerates, A2 does not.
  B.  Logit trend: least-squares line of logit(rate_i) on x, read at the
      target (the house curve baseline's shape, over the whole recent
      history instead of 24 h).
  C.  Cohort model: at each refresh step the marginal cohort's earn
      probability p_i = dE_i / dN, recency-weighted; its log-trend is
      projected to the target (a maturing player base earns more per sync),
      then E_target = E_last + p_target * (N_target - N_last) plus the
      existing players' earning per day.
  D.  Lag model: E_i(t) ~= q_i * N(t - d_i): players earn trophy i about
      d_i after they are first tracked.  d_i is picked per trophy from
      0..72 h by fit quality; q_i is the median ratio.

Slopes shrink toward their tier mean (bronze / silver / gold / platinum
earn at different speeds).  N_target is the geometric mean of A and A2,
floored at modest launch-week growth and capped at 10x the last read;
E never falls and never exceeds N_target.  The median is the mean of the
models' logits, capped to move at most 5.5 logit units from the last read.

Spread (logit units) combines the logit fit's residual scale, the observed
refresh-to-refresh jump volatility scaled by the refreshes expected before
the target, a random-walk floor, and the disagreement between the five
models -- so when the models split between dilution-collapse and
progression-recovery the band widens instead of pretending certainty.
Floored at 0.30, capped at 1.60.  Quantiles are
inv_logit(logit(median) + z_tau * s), rounded to 4 decimals half-up like
the starter.

Only distinct reads are fitted: Exophase refreshes hours apart, so long
runs of identical reads carry no new information and would overweight
flat stretches.

Source of every number: the case file itself (the house's reads of
https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/).
No network, no hardcoded rates, no randomness.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

SPREAD_FLOOR = 0.30
SPREAD_CAP = 1.60
RW_PER_SQRT_DAY = 0.25
MAX_LOGIT_MOVE = 5.5
N_GROWTH_FLOOR_PER_DAY = 0.15
N_GROWTH_CAP = 10.0
RESID_DEFAULT = 0.30
JUMP_DEFAULT = 0.30
LAG_GRID_HOURS = (0, 6, 12, 24, 36, 48, 72)


def clamp(x, lo, hi):
    return min(hi, max(lo, x))


def logit(pct):
    p = clamp(pct, 0.01, 99.99)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    lg = clamp(lg, -9.21, 9.21)
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def ols(xs, ys):
    """Least squares y = a + b*x. Returns (a, b, resid_std, n) or None."""
    n = len(xs)
    if n == 0:
        return None
    if n == 1:
        return (ys[0], 0.0, 0.0, 1)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return (my, 0.0, 0.0, n)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    if n > 2:
        var = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
        resid = math.sqrt(max(var, 0.0))
    else:
        resid = 0.0
    return (a, b, resid, n)


def wmedian(pairs):
    """Weighted median of values with weights. pairs = [(value, weight)]."""
    pairs = sorted((v, w) for v, w in pairs if w > 0)
    total = sum(w for _, w in pairs)
    if total <= 0 or not pairs:
        return None
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= total / 2.0:
            return v
    return pairs[-1][0]


def distinct_reads(history):
    """Collapse runs of identical reads (same players + same rates)."""
    out = []
    prev_key = None
    for h in history:
        key = (h.get("players_tracked"), tuple(h.get("rates") or []))
        if key != prev_key:
            out.append(h)
            prev_key = key
    return out


def forecast_case(case):
    trophies = case["trophies"]
    history = distinct_reads(case.get("history") or [])
    if not history:
        raise ValueError("case has no history")
    horizon_days = case["horizon_minutes"] / 1440.0

    release = None
    game = case.get("game") or {}
    if game.get("release_date"):
        try:
            release = datetime.fromisoformat(game["release_date"]).replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            release = None
    if release is None:
        release = parse_time(history[0]["at"])

    def hours_since(at_s):
        return (parse_time(at_s) - release).total_seconds() / 3600.0

    def x_of(at_s):
        return math.log1p(max(0.0, hours_since(at_s)))

    def days_since(at_s):
        return hours_since(at_s) / 24.0

    t_last = parse_time(history[-1]["at"])
    t_target = parse_time(case["target"])
    x_last = x_of(history[-1]["at"])
    x_target = x_of(case["target"])
    d_target = days_since(case["target"])
    span_x = max(x_last - x_of(history[0]["at"]), 0.05)
    n_last = history[-1].get("players_tracked") or 0

    # Recent window: last 48 h of distinct reads when there are >= 4.
    cutoff = t_last.timestamp() - 48 * 3600
    recent = [h for h in history
              if parse_time(h["at"]).timestamp() >= cutoff]
    fit_hist = recent if len(recent) >= 4 else history

    # ---- N projections: power law in x, and momentum in days. ----
    xs, ds, ys_n = [], [], []
    for h in fit_hist:
        n = h.get("players_tracked")
        if n and n > 0:
            xs.append(x_of(h["at"]))
            ds.append(days_since(h["at"]))
            ys_n.append(math.log(n))
    fit_nx = ols(xs, ys_n)
    fit_nd = ols(ds, ys_n)
    n_cands = []
    if fit_nx is not None:
        b = clamp(fit_nx[1], 0.0, 3.0)
        n_cands.append(math.exp(fit_nx[0] + b * x_target))
    if fit_nd is not None:
        b = clamp(fit_nd[1], 0.0, 1.0)
        n_cands.append(math.exp(fit_nd[0] + b * d_target))
    if n_cands:
        n_target = math.exp(sum(math.log(c) for c in n_cands) / len(n_cands))
    else:
        n_target = float(n_last) if n_last else 1.0
    n_target = clamp(
        n_target,
        (n_last or 1) * (1.0 + N_GROWTH_FLOOR_PER_DAY * horizon_days),
        (n_last or 1) * N_GROWTH_CAP,
    )
    if n_last:
        n_target = max(n_target, float(n_last))

    def n_at_x(x):
        """Fitted N at log-time x (model A curve), for the lag model."""
        if fit_nx is None:
            return float(n_last or 1)
        b = clamp(fit_nx[1], 0.0, 3.0)
        return math.exp(fit_nx[0] + b * x)

    # ---- Per-trophy series over the fit window. ----
    series = []
    for i in range(len(trophies)):
        s = []
        for h in fit_hist:
            rates = h.get("rates") or []
            if i >= len(rates) or rates[i] is None:
                continue
            n = h.get("players_tracked") or 0
            if n <= 0:
                continue
            s.append((x_of(h["at"]), days_since(h["at"]),
                      parse_time(h["at"]), float(rates[i]), n,
                      float(rates[i]) * n / 100.0))
        series.append(s)

    def tier_slopes(fits):
        by_tier = {}
        for t, f in zip(trophies, fits):
            if f is not None:
                by_tier.setdefault(t["tier"], []).append(f[1])
        means = {k: sum(v) / len(v) for k, v in by_tier.items()}
        all_b = [f[1] for f in fits if f is not None]
        overall = sum(all_b) / len(all_b) if all_b else 0.0
        return means, overall

    fits_e = []   # log E on x
    fits_e2 = []  # log E on days
    fits_l = []   # logit(rate) on x
    for s in series:
        fits_e.append(ols([p[0] for p in s],
                          [math.log(max(p[5], 0.5)) for p in s])
                      if len(s) >= 2 else None)
        fits_e2.append(ols([p[1] for p in s],
                           [math.log(max(p[5], 0.5)) for p in s])
                       if len(s) >= 2 else None)
        fits_l.append(ols([p[0] for p in s], [logit(p[3]) for p in s])
                      if len(s) >= 2 else None)
    tier_e, overall_e = tier_slopes(fits_e)
    tier_e2, overall_e2 = tier_slopes(fits_e2)
    tier_l, overall_l = tier_slopes(fits_l)

    resids = [f[2] for f in fits_l if f is not None and f[3] >= 3]
    resid_global = sorted(resids)[len(resids) // 2] if resids else RESID_DEFAULT
    resid_global = clamp(resid_global, 0.05, 1.0)

    jumps = []
    jump_by_trophy = []
    for s in series:
        js = [logit(p2[3]) - logit(p1[3]) for p1, p2 in zip(s, s[1:])]
        jump_by_trophy.append(js)
        jumps.extend(js)
    if jumps:
        mean_j = sum(jumps) / len(jumps)
        jump_global = math.sqrt(
            sum((j - mean_j) ** 2 for j in jumps) / len(jumps))
    else:
        jump_global = JUMP_DEFAULT
    jump_global = clamp(jump_global, 0.05, 1.0)

    gaps = []
    for h1, h2 in zip(history, history[1:]):
        gaps.append((parse_time(h2["at"]) - parse_time(h1["at"]))
                    .total_seconds() / 60.0)
    gaps = [g for g in gaps if g > 0]
    med_gap = sorted(gaps)[len(gaps) // 2] if gaps else 240.0
    n_refresh = clamp(case["horizon_minutes"] / max(med_gap, 1.0), 0.0, 12.0)

    # N lookup on the step curve (for the lag model's historical ratios).
    step_times = [parse_time(h["at"]) for h in history]
    step_ns = [h.get("players_tracked") or 0 for h in history]

    def n_step_at(t):
        best = None
        for st, sn in zip(step_times, step_ns):
            if st <= t:
                best = sn
            else:
                break
        return best

    last_rates = history[-1].get("rates") or []
    rows = []
    for i, trophy in enumerate(trophies):
        s = series[i]
        last_rate = (float(last_rates[i]) if i < len(last_rates)
                     and last_rates[i] is not None else None)
        if last_rate is None:
            last_rate = s[-1][3] if s else 0.0
        last_e = s[-1][5] if s else 0.0
        lg_last = logit(last_rate)

        if last_rate >= 99.995:
            rows.append({"id": trophy["id"],
                         "q": [round4(v) for v in
                               (99.9, 99.97, 99.99, 99.99, 99.99)]})
            continue

        logits = []

        # A: counts power law in x.
        f_e = fits_e[i]
        if f_e is not None and s:
            _, b_e, _, n_e = f_e
            w = n_e / (n_e + 4.0)
            b_e = w * b_e + (1.0 - w) * tier_e.get(trophy["tier"], overall_e)
            b_e = clamp(b_e, -1.0, 4.0)
            e_tgt = clamp(math.exp(f_e[0] + b_e * x_target), last_e, n_target)
            logits.append(logit(100.0 * e_tgt / n_target))

        # A2: counts momentum in days (daily growth capped).
        f_e2 = fits_e2[i]
        if f_e2 is not None and s:
            _, b_e2, _, n_e2 = f_e2
            w = n_e2 / (n_e2 + 4.0)
            b_e2 = w * b_e2 + (1.0 - w) * tier_e2.get(trophy["tier"],
                                                     overall_e2)
            b_e2 = clamp(b_e2, -0.5, 0.5)
            e_tgt = clamp(math.exp(f_e2[0] + b_e2 * d_target),
                          last_e, n_target)
            logits.append(logit(100.0 * e_tgt / n_target))

        # B: logit trend in x.
        f_l = fits_l[i]
        if f_l is not None and s:
            _, b_l, _, n_l = f_l
            w = n_l / (n_l + 4.0)
            b_l = w * b_l + (1.0 - w) * tier_l.get(trophy["tier"], overall_l)
            b_l = clamp(b_l, -2.0, 3.0)
            logits.append(clamp(f_l[0] + b_l * x_target, -9.21, 9.21))

        # C: cohort marginal earn rate, trend-projected.
        if len(s) >= 2:
            steps = []
            for p1, p2 in zip(s, s[1:]):
                if p2[4] > p1[4]:
                    age = (t_last - p2[2]).total_seconds() / 86400.0
                    steps.append((clamp((p2[5] - p1[5]) / (p2[4] - p1[4]),
                                         0.0, 1.0),
                                  float(p2[4] - p1[4])
                                  * math.exp(-max(age, 0.0)),
                                  p2[1]))
            p_med = wmedian([(v, w) for v, w, _ in steps]) if steps else None
            if p_med is not None:
                p_tgt = p_med
                trend_pts = [(d, math.log(max(v, 1e-6)))
                             for v, _, d in steps if v > 0]
                if len(trend_pts) >= 3:
                    f_p = ols([p[0] for p in trend_pts],
                              [p[1] for p in trend_pts])
                    if f_p is not None:
                        b_p = clamp(f_p[1], -1.0, 1.0)
                        p_tgt = math.exp(f_p[0] + b_p * d_target)
                p_tgt = clamp(p_tgt, 0.0, min(0.95, 3.0 * max(p_med, 1e-4)))
                # Existing players' earning per day.
                g_i = 0.0
                f_g = ols([p[0] for p in s], [p[5] for p in s])
                if f_g is not None:
                    hours_last = max(hours_since(history[-1]["at"]), 0.0)
                    g_i = max(0.0, f_g[1] / (24.0 * (1.0 + hours_last)))
                    span_days = max(
                        (s[-1][2] - s[0][2]).total_seconds() / 86400.0, 0.1)
                    g_cap = max(0.0, (s[-1][5] - s[0][5]) / span_days) * 2.0
                    g_i = min(g_i, g_cap)
                e_tgt = clamp(last_e + p_tgt * max(0.0, n_target - (n_last or 0))
                              + g_i * horizon_days, last_e, n_target)
                logits.append(logit(100.0 * e_tgt / n_target))

        # D: lag model E(t) ~= q * N(t - d).
        if len(s) >= 3:
            best = None  # (score, d, q)
            from datetime import timedelta
            for d_h in LAG_GRID_HOURS:
                # ratios from the step curve: N at (read time - d)
                ratios = []
                for p in s:
                    n_lag = n_step_at(p[2] - timedelta(hours=d_h))
                    if n_lag and n_lag > 0 and p[5] > 0:
                        ratios.append(p[5] / n_lag)
                if len(ratios) >= 2:
                    lr = [math.log(r) for r in ratios]
                    m = sum(lr) / len(lr)
                    sc = math.sqrt(sum((v - m) ** 2 for v in lr) / len(lr))
                    q = math.exp(m)
                    # Grid ascends in d, so the first (smallest d) wins ties.
                    if best is None or sc < best[0] - 1e-9:
                        best = (sc, d_h, q)
            if best is not None:
                _, d_h, q = best
                from datetime import timedelta
                x_lag = x_of((t_target - timedelta(hours=d_h)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"))
                e_tgt = clamp(q * n_at_x(x_lag), last_e, n_target)
                logits.append(logit(100.0 * e_tgt / n_target))

        if logits:
            med_logit = sum(logits) / len(logits)
        else:
            med_logit = lg_last
        med_logit = clamp(med_logit, lg_last - MAX_LOGIT_MOVE,
                          lg_last + MAX_LOGIT_MOVE)
        median = inv_logit(med_logit)

        # ---- Spread. ----
        resid_i = f_l[2] if (f_l is not None and f_l[3] >= 3) else resid_global
        resid_i = clamp(resid_i, 0.05, 1.0)
        js = jump_by_trophy[i]
        if len(js) >= 2:
            mj = sum(js) / len(js)
            jump_i = math.sqrt(sum((j - mj) ** 2 for j in js) / len(js))
        else:
            jump_i = jump_global
        jump_i = clamp(jump_i, 0.05, 1.0)
        if len(logits) >= 2:
            ml = sum(logits) / len(logits)
            disagree = math.sqrt(sum((v - ml) ** 2 for v in logits)
                                 / len(logits))
        else:
            disagree = 0.0
        spread = math.sqrt(
            (resid_i ** 2) * (1.0 + horizon_days / max(span_x / 0.693, 0.25))
            + (0.5 * jump_i * math.sqrt(n_refresh)) ** 2
            + (RW_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2
            + min(disagree, 0.8) ** 2
        )
        spread = clamp(spread, SPREAD_FLOOR, SPREAD_CAP)

        qs = []
        prev = 0.01
        for z in Z:
            q = clamp(inv_logit(med_logit + z * spread), 0.01, 99.99)
            q = round4(max(q, prev))
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})

    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
