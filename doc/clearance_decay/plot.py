"""Builds doc/figures/fig5-decay.svg from decay.csv.

Log-log plot of clearance against insertion count, with least-squares fits, the
theoretical -1/D references, and a results table. Run after `cargo run --release`.
"""
import csv, io, math, collections

CSV = 'decay.csv'
OUT = '../figures/fig5-decay.svg'
PITCH = {'frac4': 1.0 / 4096, 'frac1': 1.0 / 8192}

# ---------------------------------------------------------------- data + fits
rows = collections.defaultdict(list)
with io.open(CSV, encoding='utf-8') as fh:
    for r in csv.DictReader(fh):
        rows[r['series']].append((int(r['k']), float(r['d'])))
for s in rows:
    rows[s].sort()


def fit(pts, k_lo, k_hi, d_min=0.0):
    used = [(k, d) for k, d in pts if k_lo <= k <= k_hi and d > d_min]
    xs = [(math.log10(k), math.log10(d)) for k, d in used]
    n = len(xs)
    mx = sum(x for x, _ in xs) / n
    my = sum(y for _, y in xs) / n
    sxy = sum((x - mx) * (y - my) for x, y in xs)
    sxx = sum((x - mx) ** 2 for x, _ in xs)
    syy = sum((y - my) ** 2 for _, y in xs)
    a = sxy / sxx
    # The drawn line spans the points actually fitted — never beyond, so a series
    # truncated by the grid pitch is not silently extrapolated.
    return dict(slope=a, icpt=my - a * mx, r2=sxy ** 2 / (sxx * syy), n=n,
                k_lo=used[0][0], k_hi=used[-1][0])


SPEC = {
    'ff2':   dict(lo=100, hi=3000, dmin=0.0, pred=-0.5,      col='#2c5aa0', lab='D = 2'),
    'ff3':   dict(lo=100, hi=1200, dmin=0.0, pred=-1 / 3,    col='#17796b', lab='D = 3'),
    'ff4':   dict(lo=100, hi=800,  dmin=0.0, pred=-0.25,     col='#b26a00', lab='D = 4'),
    'ff4hi': dict(lo=100, hi=400,  dmin=0.0, pred=-0.25,     col='#b3261e', lab='D = 4, 4x restarts'),
    'frac4': dict(lo=100, hi=3000, dmin=20 * PITCH['frac4'], pred=None, col='#6a3d9a', lab='radius = d/4'),
    'frac1': dict(lo=100, hi=3000, dmin=20 * PITCH['frac1'], pred=None, col='#b3261e', lab='radius = d'),
}
F = {s: fit(rows[s], v['lo'], v['hi'], v['dmin']) for s, v in SPEC.items()}

for s, f in F.items():
    print('%-7s slope %+.4f  R2 %.4f  n %4d  N_eff %5.2f' %
          (s, f['slope'], f['r2'], f['n'], -1 / f['slope']))


def decimate(pts, m=190):
    """Log-spaced subsample, so the polyline is compact but keeps its shape."""
    if len(pts) <= m:
        return pts
    n = len(pts)
    idx = sorted({min(n - 1, int(round((n - 1) ** (i / (m - 1))))) for i in range(m)})
    return [pts[i] for i in idx]


# ---------------------------------------------------------------- svg helpers
P = []                                        # output buffer
def e(s):
    P.append(s)


class Ax:
    """Log-log axes mapped onto a pixel box."""
    def __init__(self, x0, y0, x1, y1, xr, yr):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.xr, self.yr = xr, yr

    def px(self, k):
        t = (math.log10(k) - self.xr[0]) / (self.xr[1] - self.xr[0])
        return self.x0 + t * (self.x1 - self.x0)

    def py(self, d):
        t = (math.log10(d) - self.yr[0]) / (self.yr[1] - self.yr[0])
        return self.y1 - t * (self.y1 - self.y0)

    def poly(self, pts, colour, w=1.1, op=1.0):
        s = ' '.join('%.1f,%.1f' % (self.px(k), self.py(d)) for k, d in pts if d > 0)
        e('<polyline points="%s" fill="none" stroke="%s" stroke-width="%s" '
          'opacity="%s" stroke-linejoin="round"/>' % (s, colour, w, op))

    def fitline(self, f, colour, w=2.6, dash=None):
        a, b = f['slope'], f['icpt']
        k0, k1 = f['k_lo'], f['k_hi']
        y0, y1 = 10 ** (a * math.log10(k0) + b), 10 ** (a * math.log10(k1) + b)
        e('<path d="M%.1f %.1f L%.1f %.1f" fill="none" stroke="%s" stroke-width="%s"%s/>'
          % (self.px(k0), self.py(y0), self.px(k1), self.py(y1), colour, w,
             ' stroke-dasharray="%s"' % dash if dash else ''))

    def refline(self, f, slope, colour):
        """Theory slope, anchored at the fit's left end."""
        a, b = f['slope'], f['icpt']
        k0, k1 = f['k_lo'], f['k_hi']
        y0 = 10 ** (a * math.log10(k0) + b)
        y1 = y0 * (k1 / k0) ** slope
        e('<path d="M%.1f %.1f L%.1f %.1f" fill="none" stroke="%s" stroke-width="1.5" '
          'stroke-dasharray="6 4" opacity=".9"/>'
          % (self.px(k0), self.py(y0), self.px(k1), self.py(y1), colour))

    def frame(self, xt, yt, ylab):
        for k in xt:
            x = self.px(k)
            e('<path d="M%.1f %d V%d" stroke="#e4eaef" stroke-width="1"/>' % (x, self.y0, self.y1))
            e('<text class="sm" x="%.1f" y="%d" text-anchor="middle">%s</text>'
              % (x, self.y1 + 20, '{:,}'.format(k).replace(',', ' ')))
        for d in yt:
            y = self.py(d)
            e('<path d="M%d %.1f H%d" stroke="#e4eaef" stroke-width="1"/>' % (self.x0, y, self.x1))
            e('<text class="sm" x="%d" y="%.1f" text-anchor="end">%s</text>'
              % (self.x0 - 8, y + 4, ylab(d)))
        e('<path d="M%d %d H%d M%d %d V%d" stroke="#8a9aa8" stroke-width="1.1" fill="none"/>'
          % (self.x0, self.y1, self.x1, self.x0, self.y0, self.y1))


def tick(d):
    if d >= 0.01:
        return ('%g' % d)
    return '%g' % d


# ---------------------------------------------------------------- figure
W, H = 1480, 1000
e('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" '
  'font-family="Helvetica Neue, Helvetica, Arial, sans-serif" fill="#16202b">' % (W, H, W, H))
e('<title>Clearance decay as a measurement of effective dimension</title>')
e('''<defs><style>
  .ttl { font-size:19px; font-weight:700; letter-spacing:1.5px; }
  .sub { font-size:13.4px; fill:#5a6b7a; }
  .pn  { font-size:14.4px; font-weight:700; }
  .pl  { font-size:12.2px; fill:#3d4d5c; }
  .sm  { font-size:11.4px; fill:#5a6b7a; }
  .smb { font-size:11.4px; font-weight:700; }
  .th, .th text { font-size:12.2px; font-weight:700; letter-spacing:.8px; fill:#ffffff; }
  .rl  { font-size:12.4px; font-weight:700; }
  .td  { font-size:12.2px; }
  .mt  { font-family:Georgia,"Times New Roman",serif; font-style:italic; }
  .cap { font-family:Georgia,"Times New Roman",serif; font-size:13.2px; fill:#4b5b6a; }
</style></defs>''')
e('<rect width="%d" height="%d" fill="#ffffff"/>' % (W, H))

e('<text class="ttl" x="32" y="36">THE SLOPE OF THE CLEARANCE SEQUENCE MEASURES EFFECTIVE DIMENSION</text>')
e('<text class="sub" x="32" y="58">11 400 insertions through this crate’s own solvers. '
  'Ground truth on the left, an unknown answer on the right</text>')
e('<path d="M32 72 H1448" stroke="#16202b" stroke-width="1.6"/>')

# ---- panel (a)
e('<rect x="32" y="90" width="768" height="520" rx="7" fill="#ffffff" stroke="#d5dee6"/>')
e('<rect x="32" y="90" width="768" height="34" rx="7" fill="#0d1721"/>')
e('<rect x="32" y="112" width="768" height="12" fill="#0d1721"/>')
e('<text class="pn" x="48" y="113" fill="#ffffff">(a)  Validation — farthest-first traversal, where the answer is known</text>')

A = Ax(112, 146, 776, 500, (0, 3.5), (-2.25, -0.15))
A.frame([1, 3, 10, 30, 100, 300, 1000, 3000], [0.01, 0.03, 0.1, 0.3], tick)
for s in ('ff2', 'ff3', 'ff4'):
    v, f = SPEC[s], F[s]
    A.poly(decimate(rows[s]), v['col'], w=1.0, op=.42)
    A.refline(f, v['pred'], '#16202b')
    A.fitline(f, v['col'])
e('<text class="sm" x="444" y="536" text-anchor="middle">insertions k</text>')
e('<text class="sm" x="66" y="323" text-anchor="middle" transform="rotate(-90 66 323)">clearance dₖ</text>')

lg = [('ff4', 'D = 4', -0.25), ('ff3', 'D = 3', -1 / 3), ('ff2', 'D = 2', -0.5)]
ly = 170
for s, lab, pred in lg:
    f = F[s]
    e('<path d="M%d %d H%d" stroke="%s" stroke-width="2.6"/>' % (600, ly, 626, SPEC[s]['col']))
    e('<text class="smb" x="634" y="%d" fill="%s">%s</text>' % (ly + 4, SPEC[s]['col'], lab))
    e('<text class="sm" x="682" y="%d">fitted %+.3f</text>' % (ly + 4, f['slope']))
    ly += 19
e('<path d="M600 %d H626" stroke="#16202b" stroke-width="1.5" stroke-dasharray="6 4"/>' % ly)
e('<text class="sm" x="634" y="%d">theory −1/D</text>' % (ly + 4))

e('<text class="pl" x="48" y="562">Each series inserts a zero-radius point at the deepest spot, so dₖ is exactly the covering radius. '
  'The fits recover</text>')
e('<text class="pl" x="48" y="580">2.10 for D = 2, but 3.53 for D = 3 and 4.93 for D = 4 — the estimator is itself '
  'short of its asymptotic regime.</text>')

# ---- panel (b)
e('<rect x="820" y="90" width="628" height="520" rx="7" fill="#ffffff" stroke="#d5dee6"/>')
e('<rect x="820" y="90" width="628" height="34" rx="7" fill="#6a3d9a"/>')
e('<rect x="820" y="112" width="628" height="12" fill="#6a3d9a"/>')
e('<text class="pn" x="836" y="113" fill="#ffffff">(b)  Application — the fractal distribution of example 01</text>')

B = Ax(900, 146, 1424, 470, (0, 3.5), (-3.7, -0.15))
floor = 20 * PITCH['frac1']
e('<rect x="900" y="%.1f" width="524" height="%.1f" fill="#fbe9e7" opacity=".7"/>'
  % (B.py(floor), 470 - B.py(floor)))
e('<text class="sm" x="1416" y="%.1f" text-anchor="end" fill="#8f1d17">under 20 grid cells: quantization-limited</text>'
  % (B.py(floor) + 15))
B.frame([1, 3, 10, 30, 100, 300, 1000, 3000], [0.001, 0.01, 0.1], tick)
for s in ('frac4', 'frac1'):
    v, f = SPEC[s], F[s]
    B.poly(decimate(rows[s]), v['col'], w=1.0, op=.45)
    B.fitline(f, v['col'])
B.refline(F['frac4'], -0.5, '#16202b')
e('<text class="sm" x="1162" y="506" text-anchor="middle">insertions k</text>')
e('<text class="sm" x="854" y="308" text-anchor="middle" transform="rotate(-90 854 308)">clearance dₖ</text>')

ly = 172
for s, lab in (('frac1', 'radius = d'), ('frac4', 'radius = d / 4')):
    f = F[s]
    e('<path d="M1150 %d H1176" stroke="%s" stroke-width="2.6"/>' % (ly, SPEC[s]['col']))
    e('<text class="smb" x="1184" y="%d" fill="%s">%s</text>' % (ly + 4, SPEC[s]['col'], lab))
    e('<text class="sm" x="1272" y="%d">%+.3f → N = %.2f</text>' % (ly + 4, f['slope'], -1 / f['slope']))
    ly += 19
e('<path d="M1150 %d H1176" stroke="#16202b" stroke-width="1.5" stroke-dasharray="6 4"/>' % ly)
e('<text class="sm" x="1184" y="%d">uniform 2D reference −0.5</text>' % (ly + 4))

e('<text class="pl" x="836" y="532">Both series take the <tspan class="mt">exact</tspan> global maximum, so no search error enters. '
  'Both are clean power</text>')
e('<text class="pl" x="836" y="550">laws (R² = 0.999 and 0.993) with exponents steeper than −0.5: the distributions '
  'occupy</text>')
e('<text class="pl" x="836" y="568">less than the plane they are drawn in, and the two rules are cleanly distinguished.</text>')
e('<text class="pl" x="836" y="592">A slope steeper than −1/2 in the plane is the signature of a fractal support.</text>')

# ---- panel (c): results table
e('<rect x="32" y="630" width="1416" height="34" fill="#0d1721"/>')
cols = [44, 300, 520, 690, 850, 1010, 1170, 1320]
heads = ['SERIES', 'RULE', 'AMBIENT D', 'POINTS FITTED', 'FITTED SLOPE', 'PREDICTED', 'R²', 'N MEASURED']
e('<g class="th">')
for x, h in zip(cols, heads):
    e('<text x="%d" y="652">%s</text>' % (x, h))
e('</g>')

TAB = [
    ('ff2',   'point at the deepest spot',        '2', -0.5),
    ('ff3',   'point at the deepest spot',        '3', -1 / 3),
    ('ff4',   'point at the deepest spot',        '4', -0.25),
    ('ff4hi', 'as ff4, four times the restarts',  '4', -0.25),
    ('frac4', 'ball of radius d / 4 (example 01)', '2', None),
    ('frac1', 'ball of radius d (maximal)',        '2', None),
]
y = 664
for i, (s, rule, dim, pred) in enumerate(TAB):
    f = F[s]
    e('<rect x="32" y="%d" width="1416" height="30" fill="%s"/>'
      % (y, '#f6f9fb' if i % 2 else '#ffffff'))
    e('<rect x="%d" y="%d" width="4" height="30" fill="%s"/>' % (32, y, SPEC[s]['col']))
    vals = [s, rule, dim, '%d' % f['n'], '%+.4f' % f['slope'],
            ('%+.4f' % pred) if pred else '—',
            '%.4f' % f['r2'], '%.2f' % (-1 / f['slope'])]
    for x, v, cls in zip(cols, vals, ['rl', 'td', 'td', 'td', 'rl', 'td', 'td', 'rl']):
        e('<text class="%s" x="%d" y="%d">%s</text>' % (cls, x, y + 20, v))
    y += 30
e('<rect x="32" y="664" width="1416" height="%d" fill="none" stroke="#c3cfd9" stroke-width="1.2"/>' % (y - 664))

# ---- caption
cap = [
    'Figure 5 | The diagnostic of §2.1, carried out. (a) Farthest-first traversal, where the covering radius must scale as k^(−1/D): the fit recovers',
    '2.10 against a true 2, but only 3.53 against 3 and 4.93 against 4. The control row ff4hi isolates the cause — quadrupling the restarts moves the slope',
    'further from −1/4, not closer, so the shortfall is not search error but the pre-asymptotic regime. Reaching k ≫ 2^D insertions is itself subject to the',
    'curse of dimensionality, so the estimator inherits the very limitation it measures. (b) Applied where no ground truth exists, the same fit is far cleaner',
    '(R² = 0.9993) and returns 1.89 for the distribution of example 01 and 1.21 for maximal balls — non-integer dimensions below the ambient 2, which is',
    'what a fractal support should give. Shaded: clearances within 20 cells of the discrete solver’s grid pitch, excluded from every fit.',
]
for i, line in enumerate(cap):
    e('<text class="cap" x="32" y="%d">%s</text>' % (876 + 20 * i, line))

e('</svg>')

with io.open(OUT, 'w', encoding='utf-8', newline='\n') as fh:
    fh.write('\n'.join(P) + '\n')
print('\nwrote %s (%d elements)' % (OUT, len(P)))
