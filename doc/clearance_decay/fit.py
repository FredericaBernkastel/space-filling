"""Least-squares fit of log d_k against log k, per series, plus windowed slopes."""
import csv, math, io, collections

PITCH = {'frac4': 1.0 / 4096, 'frac1': 1.0 / 8192}   # Argmax2D grid pitch
FLOOR = {s: 20 * p for s, p in PITCH.items()}        # fit only well above it

rows = collections.defaultdict(list)
with io.open('decay.csv', encoding='utf-8') as fh:
    for r in csv.DictReader(fh):
        rows[r['series']].append((int(r['k']), float(r['d'])))


def fit(pts, k_lo, k_hi, d_min=0.0):
    xs = [(math.log(k), math.log(d)) for k, d in pts
          if k_lo <= k <= k_hi and d > max(d_min, 0.0)]
    n = len(xs)
    if n < 8:
        return None
    mx = sum(x for x, _ in xs) / n
    my = sum(y for _, y in xs) / n
    sxy = sum((x - mx) * (y - my) for x, y in xs)
    sxx = sum((x - mx) ** 2 for x, _ in xs)
    syy = sum((y - my) ** 2 for _, y in xs)
    slope = sxy / sxx
    return dict(slope=slope, icpt=my - slope * mx,
                r2=(sxy ** 2) / (sxx * syy) if syy else float('nan'), n=n)


PRED = {'ff2': -0.5, 'ff3': -1 / 3, 'ff4': -0.25, 'ff4hi': -0.25}
SPEC = {                     # series -> (k_lo, k_hi, d_min)
    'ff2':    (100, 3000, 0.0),
    'ff3':    (100, 1200, 0.0),
    'ff4':    (100, 800, 0.0),
    'ff4hi':  (100, 400, 0.0),
    'frac4':  (100, 3000, FLOOR['frac4']),
    'frac1':  (100, 3000, FLOOR['frac1']),
}

print('%-7s %5s %8s %10s %7s %8s %10s' %
      ('series', 'n', 'slope', 'predicted', 'R^2', 'N_eff', 'd_last'))
print('-' * 62)
for s, (lo, hi, dmin) in SPEC.items():
    pts = sorted(rows[s])
    f = fit(pts, lo, hi, dmin)
    p = PRED.get(s)
    print('%-7s %5d %8.4f %10s %7.4f %8.2f %10.2e' %
          (s, f['n'], f['slope'], ('%.4f' % p) if p else '     --',
           f['r2'], -1 / f['slope'], pts[-1][1]))

print('\nwindowed slopes (fit over [k/3, k], to see whether the exponent is still drifting)')
for s in ('ff2', 'ff3', 'ff4', 'frac4', 'frac1'):
    pts = sorted(rows[s])
    kmax = pts[-1][0]
    cells = []
    for k in (200, 400, 800, 1500, 3000):
        if k > kmax:
            cells.append('     -')
            continue
        f = fit(pts, max(30, k // 3), k, SPEC[s][2])
        cells.append('%6.3f' % f['slope'] if f else '     -')
    print('  %-7s ' % s + ' '.join(cells) + '     (k = 200, 400, 800, 1500, 3000)')

print('\nquantization check')
for s in ('frac4', 'frac1'):
    pts = sorted(rows[s])
    print('  %-6s pitch %.3e; last clearance = %6.1f px; drops below 20 px at k = %s'
          % (s, PITCH[s], pts[-1][1] / PITCH[s],
             next((k for k, d in pts if d < FLOOR[s]), 'never')))
