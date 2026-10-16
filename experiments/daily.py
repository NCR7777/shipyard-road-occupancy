"""Daily readings under continuous release and the 30-day criterion K_stable (the fleet without growing backlog). Scenario
parameter daily_stats (experiments/runner.compute calls daily() after the simulation); without it this module is not used.

Daily readings (day d = [d day_s, (d+1) day_s), d from 0):
  released[d]   tasks released that day (release time before the end of the day);
  completed[d]  tasks completed that day (one completed exactly at the end of a day counts to the next day, on the last day to
                that day: as thr_by_day of Sim.metrics);
  backlog[d]    tasks not completed at the end of the day = released up to the end of the day - sum of completed[0..d]
                (queued for a vehicle and under way);
  flow_mean[d], flow_p95[d]   flow time of the tasks completed that day (release -> completion, min);
  qwait_p95[d]  p95 of the time the tasks completed that day queued for a vehicle (release -> dispatch, min);
  wait_p95[d]   p95 of the waiting of the tasks completed that day (queueing for a vehicle + waiting within the rule: under
                reservation at the stop for the departure, under stepwise requests for resources; min).
p95 as in Sim.metrics (the floor(0.95 n)-th in ascending order). The window WIN gets the mean and p95 of the flow time and the
p95 of the waiting of the tasks completed in it.

K_stable (the smallest integer fleet whose backlog has no positive trend):
  for each seed the end-of-day backlog of the days WIN (days 6-30, from 1) is fitted by least squares, giving a slope (tasks a
  day); the ten slopes give a mean and a t interval (95 %, 9 degrees of freedom).
  Main criterion: lower end of the interval <= 0 (no significant positive trend of the backlog: the interval contains 0 or lies
  below it);
  tolerance criterion: mean slope <= TOL x daily demand (0.1 % a day: less than 3 % of one day's demand over 30 days).
  A fleet is K_stable only if both K and K + 1 meet the criterion (a single fleet meeting it by chance does not count); the
  search starts at K_req - 2 (experiments/stable_fleet.py).
Window days 6-30: day 1 is the warm-up (as for K_req); on days 2-5 the backlog starts from an empty system and is still rising,
so they are left out of the trend.

Run:   imported by experiments/runner.py (daily) and experiments/stable_fleet.py (slope, trend).
"""
WIN = (6, 30)          # regression window (days, from 1, both ends included)
TOL = 0.001            # tolerance criterion: slope <= 0.1 % x daily demand a day
T975 = {9: 2.262}      # 0.975 quantile of the t distribution (ten seeds)


def q95(xs):
    return sorted(xs)[min(len(xs) - 1, int(0.95 * len(xs)))] if xs else None


def daily(sim, p, tasks):
    """Daily readings (see the module docstring). tasks: the tasks released in this point (lifts without a vehicle excluded)."""
    day, D = p["day_s"], p["days"]
    rel = sorted(t["release"] for t in tasks if not t.get("lift"))
    by_day = [[] for _ in range(D)]
    for r in sim.done:
        by_day[min(int(r["done"] // day), D - 1)].append(r)
    out = dict(released=[], completed=[], backlog=[], flow_mean=[], flow_p95=[], qwait_p95=[], wait_p95=[])
    n_done = 0
    for d in range(D):
        end = (d + 1) * day
        n_rel = sum(1 for x in rel if x < end)
        rs = by_day[d]
        n_done += len(rs)
        fl = [(r["done"] - r["release"]) / 60 for r in rs]
        out["released"].append(sum(1 for x in rel if d * day <= x < end))
        out["completed"].append(len(rs))
        out["backlog"].append(n_rel - n_done)
        out["flow_mean"].append(sum(fl) / len(fl) if fl else None)
        out["flow_p95"].append(q95(fl))
        out["qwait_p95"].append(q95([(r["dispatch"] - r["release"]) / 60 for r in rs]))
        out["wait_p95"].append(q95([(r["dispatch"] - r["release"] + r["wait"]) / 60 for r in rs]))
    win = [r for d in range(WIN[0] - 1, min(WIN[1], D)) for r in by_day[d]]
    fl = [(r["done"] - r["release"]) / 60 for r in win]
    out["window"] = dict(days=list(WIN), n=len(win), flow_mean=sum(fl) / len(fl) if fl else None, flow_p95=q95(fl),
                         qwait_p95=q95([(r["dispatch"] - r["release"]) / 60 for r in win]),
                         wait_p95=q95([(r["dispatch"] - r["release"] + r["wait"]) / 60 for r in win]))
    return out


def slope(backlog, win=WIN):
    """Least-squares slope (tasks a day) of the end-of-day backlog over the days win."""
    ys = backlog[win[0] - 1:win[1]]
    xs = range(len(ys))
    mx, my = (len(ys) - 1) / 2, sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def trend(slopes, n):
    """Slopes of the ten seeds -> mean, t interval and the two criteria (see the module docstring); n is the daily demand (the
    vehicle releases a day of the target line)."""
    k = len(slopes)
    m = sum(slopes) / k
    sd = (sum((s - m) ** 2 for s in slopes) / (k - 1)) ** 0.5
    h = T975[k - 1] * sd / k ** 0.5
    return dict(slope=m, lo=m - h, hi=m + h, rel=m / n, stable=m - h <= 0, stable_tol=m <= TOL * n)
