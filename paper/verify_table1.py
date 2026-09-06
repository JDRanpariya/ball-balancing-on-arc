import json, numpy as np, re, sys, os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, 'ram'))
sys.path.insert(0, os.path.join(_here, 'scripts'))
from bench13 import BENCH13, NAME_MAP
from compute_statistics import wilson_ci  # 95% Wilson score interval
d = json.load(open(os.path.join(_here, 'data/exp1_hardware/consolidated.json')))['controllers']
tex = open(os.path.join(_here, 'ram/sections/05_results.tex')).read()
rows = []
in_tab = False; past_header = False
for line in tex.splitlines():
    if 'tab:results' in line: in_tab = True
    if in_tab and 'data source / mechanism' in line.lower():
        past_header = True; continue
    first = line.split('&')[0].strip()
    if (in_tab and past_header and '&' in line and r'\\' in line
            and 'midrule' not in line and 'toprule' not in line
            and 'bottomrule' not in line and 'Data source' not in line
            and 'Controller' not in line and first
            and r'\textbf{' not in first):
        rows.append(line.strip())
    if in_tab and 'bottomrule' in line: break
print(f"Found {len(rows)} data rows\n")
print(f"{'ctrl':<14}{'SR p/d':<10}{'mean p/d':<16}{'med p/d':<16}{'eff p/d':<16}{'status'}")
mismatch = 0
for r in rows:
    parts = [p.strip() for p in r.split('&')]
    name = parts[0].replace(r'\textbf{', '').replace('}', '').strip()
    sr_p = re.search(r'(\d+)', parts[2]).group(1)
    ci_p = re.findall(r'(\d+)', parts[3])  # ['lo', 'hi'] from "[93, 100]"
    mean_p = re.search(r'(\d+\.\d+)', parts[4]).group(1)
    m = re.search(r'(\d+\.\d+)', parts[5]); med_p = m.group(1) if m else '?'
    e = re.search(r'(\d+\.\d+)', parts[6]); eff_p = e.group(1) if e else '?'
    name_norm = name.replace(' ', '')
    # paper table uses 'TD3 (DR)' but NAME_MAP drops the (DR) suffix for these
    name_nodr = name_norm.replace('(DR)', '')
    key = next((k for k in BENCH13
                if NAME_MAP.get(k, k).replace(' ', '') in (name_norm, name_nodr)), None)
    if key is None:
        print(f"{name}: KEY NOT FOUND"); continue
    tr = d[key]; n = len(tr); ok = sum(1 for t in tr if t.get('success'))
    st = np.array([t['settling_time'] for t in tr if t.get('success')])
    eff = [np.sqrt(np.mean(np.clip(np.array(t['actions']), -0.9, 0.9) ** 2)) for t in tr if t.get('actions')]
    sr_d = f"{100 * ok // n}"; mean_d = f"{st.mean():.2f}"; med_d = f"{np.median(st):.2f}"; eff_d = f"{np.mean(eff):.3f}"
    lo, hi = wilson_ci(ok, n); ci_d = [str(round(lo)), str(round(hi))]
    ci_ok = (ci_p == ci_d)
    ok_ = (sr_p == sr_d and mean_p == mean_d and med_p == med_d and eff_p == eff_d and ci_ok)
    if not ok_: mismatch += 1
    ci_str = f"[{','.join(ci_p)}]/[{','.join(ci_d)}]"
    print(f"{name:<14}{sr_p + ' / ' + sr_d:<8}{ci_str:<18}{mean_p + ' / ' + mean_d:<14}{med_p + ' / ' + med_d:<14}{eff_p + ' / ' + eff_d:<14}{'OK' if ok_ else '*** MISMATCH ***'}")
print(f"\n{'PASSED - results table (Table II in the paper) matches data exactly' if mismatch == 0 else str(mismatch) + ' ROWS MISMATCH'}")
