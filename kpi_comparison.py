import os
import json
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class AdvancedSPCEngine:
    d2 = 1.128

    @staticmethod
    def calculate_wilson_score_limits(p_hat, n, z=3.0):
        denominator = 1 + (z**2) / n
        center = p_hat + (z**2) / (2 * n)
        spread = z * np.sqrt((p_hat * (1 - p_hat)) / n + (z**2) / (4 * (n**2)))
        ucl = (center + spread) / denominator
        lcl = (center - spread) / denominator
        return np.maximum(lcl, 0), np.minimum(ucl, 1)

    @staticmethod
    def calculate_laney_p_chart(df):
        n = df['Number of Discharges'].values
        x = df['Number of Readmissions'].values
        n = np.where(n == 0, 1, n)
        p = x / n
        p_bar = np.sum(x) / np.sum(n)
        sigma_p = np.sqrt(p_bar * (1 - p_bar) / n)
        sigma_p = np.where(sigma_p == 0, 1e-9, sigma_p)
        z_scores = (p - p_bar) / sigma_p
        mr = np.abs(np.diff(z_scores))
        mr_bar = np.mean(mr) if len(mr) > 0 else 0
        sigma_z = mr_bar / AdvancedSPCEngine.d2 if mr_bar > 0 else 1.0
        ucl = p_bar + 3 * sigma_z * sigma_p
        lcl = p_bar - 3 * sigma_z * sigma_p
        lcl = np.maximum(lcl, 0)
        small_n_mask = n < 10
        if np.any(small_n_mask):
            lcl_small, ucl_small = AdvancedSPCEngine.calculate_wilson_score_limits(
                p[small_n_mask], n[small_n_mask], z=3.0
            )
            ucl[small_n_mask] = ucl_small
            lcl[small_n_mask] = lcl_small
        out_of_control = (p > ucl) | (p < lcl)
        return {
            'outliers': out_of_control,
            'ucl': ucl,
            'lcl': lcl,
            'cl': p_bar
        }


class HospitalSPCAnalyzer:
    @staticmethod
    def classify_hrrp(err):
        if err <= 1.0:
            return 'Safe'
        elif err <= 1.05:
            return 'Mild'
        elif err <= 1.20:
            return 'Moderate'
        else:
            return 'Severe'


def build_json_data(csv_path):
    df = pd.read_csv(csv_path)

    required_cols = ['Excess Readmission Ratio', 'Number of Discharges',
                     'Number of Readmissions', 'State', 'Measure Name']
    hosp_col = 'Facility Name' if 'Facility Name' in df.columns else None
    if hosp_col is None:
        for candidate in ['Hospital Name', 'Provider ID', 'Hospital']:
            if candidate in df.columns:
                hosp_col = candidate
                break
        else:
            raise ValueError("No hospital identifier column found.")
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        if col in ['Excess Readmission Ratio', 'Number of Discharges', 'Number of Readmissions']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=required_cols + [hosp_col])
    df = df[df['Number of Discharges'] > 0]

    hospital_weighted_err = {}
    hospital_state = {}
    for hosp in df[hosp_col].unique():
        sub = df[df[hosp_col] == hosp]
        err_vals = sub['Excess Readmission Ratio'].values
        vol_vals = sub['Number of Discharges'].values
        weighted_err = np.sum(err_vals * vol_vals) / np.sum(vol_vals) if np.sum(vol_vals) > 0 else 0
        hospital_weighted_err[hosp] = weighted_err
        hospital_state[hosp] = sub['State'].iloc[0]

    measures = df['Measure Name'].unique()
    laney_outlier_count = {hosp: 0 for hosp in hospital_weighted_err.keys()}
    trad_outlier_count = {hosp: 0 for hosp in hospital_weighted_err.keys()}
    total_measures_per_hospital = {hosp: 0 for hosp in hospital_weighted_err.keys()}

    for measure in measures:
        measure_df = df[df['Measure Name'] == measure].copy()
        if len(measure_df) < 2:
            continue
        laney_res = AdvancedSPCEngine.calculate_laney_p_chart(measure_df)
        laney_outliers = laney_res['outliers']
        n_vals = measure_df['Number of Discharges'].values
        x_vals = measure_df['Number of Readmissions'].values
        p_vals = x_vals / n_vals
        p_bar_trad = np.sum(x_vals) / np.sum(n_vals)
        sigma_trad = np.sqrt(p_bar_trad * (1 - p_bar_trad) / n_vals)
        ucl_trad = p_bar_trad + 3 * sigma_trad
        lcl_trad = np.maximum(p_bar_trad - 3 * sigma_trad, 0)
        trad_outliers = (p_vals > ucl_trad) | (p_vals < lcl_trad)

        for idx, hosp in enumerate(measure_df[hosp_col].values):
            total_measures_per_hospital[hosp] += 1
            if laney_outliers[idx]:
                laney_outlier_count[hosp] += 1
            if trad_outliers[idx]:
                trad_outlier_count[hosp] += 1

    hospital_data = []
    for hosp in hospital_weighted_err.keys():
        hospital_data.append({
            'Hospital': hosp,
            'State': hospital_state[hosp],
            'Weighted_ERR': round(float(hospital_weighted_err[hosp]), 4),
            'Laney_Outlier_Count': int(laney_outlier_count[hosp]),
            'Trad_Outlier_Count': int(trad_outlier_count[hosp]),
            'Total_Measures': int(total_measures_per_hospital[hosp]),
            'Laney_Outlier_Flag': bool(laney_outlier_count[hosp] > 0),
            'Trad_Outlier_Flag': bool(trad_outlier_count[hosp] > 0),
            'Risk': HospitalSPCAnalyzer.classify_hrrp(hospital_weighted_err[hosp])
        })
    hosp_df = pd.DataFrame(hospital_data)

    laney_outliers = hosp_df[hosp_df['Laney_Outlier_Flag'] == True]['Weighted_ERR'].tolist()
    laney_non_outliers = hosp_df[hosp_df['Laney_Outlier_Flag'] == False]['Weighted_ERR'].tolist()
    trad_outliers = hosp_df[hosp_df['Trad_Outlier_Flag'] == True]['Weighted_ERR'].tolist()
    trad_non_outliers = hosp_df[hosp_df['Trad_Outlier_Flag'] == False]['Weighted_ERR'].tolist()

    def mann_whitney_test(a, b):
        if len(a) == 0 or len(b) == 0:
            return None, None
        stat, p_val = stats.mannwhitneyu(a, b, alternative='two-sided')
        return stat, float(p_val)

    laney_mw_stat, laney_mw_p = mann_whitney_test(laney_outliers, laney_non_outliers)
    trad_mw_stat, trad_mw_p = mann_whitney_test(trad_outliers, trad_non_outliers)

    laney_x = hosp_df['Laney_Outlier_Count'].values.tolist()
    laney_y = hosp_df['Weighted_ERR'].values.tolist()
    mask_laney = ~np.isnan(laney_x) & ~np.isnan(laney_y)
    if np.sum(mask_laney) > 1:
        slope_l, intercept_l, r_l, p_l, _ = stats.linregress(
            np.array(laney_x)[mask_laney], np.array(laney_y)[mask_laney])
        laney_trend = {'slope': float(slope_l), 'intercept': float(intercept_l), 'r2': float(r_l**2)}
    else:
        laney_trend = None

    trad_x = hosp_df['Trad_Outlier_Count'].values.tolist()
    trad_y = hosp_df['Weighted_ERR'].values.tolist()
    mask_trad = ~np.isnan(trad_x) & ~np.isnan(trad_y)
    if np.sum(mask_trad) > 1:
        slope_t, intercept_t, r_t, p_t, _ = stats.linregress(
            np.array(trad_x)[mask_trad], np.array(trad_y)[mask_trad])
        trad_trend = {'slope': float(slope_t), 'intercept': float(intercept_t), 'r2': float(r_t**2)}
    else:
        trad_trend = None

    states_data = {}
    for state in hosp_df['State'].unique():
        sub = hosp_df[hosp_df['State'] == state]
        states_data[state] = {
            'avg_err': round(float(sub['Weighted_ERR'].mean()), 4),
            'hospital_count': int(len(sub)),
            'hospitals': sub[['Hospital', 'Weighted_ERR', 'Laney_Outlier_Count',
                               'Trad_Outlier_Count', 'Total_Measures', 'Risk']].to_dict('records')
        }

    hospital_measures = {}
    for _, row in hosp_df.iterrows():
        hosp = row['Hospital']
        hosp_original = df[df[hosp_col] == hosp]
        meas_list = []
        for measure in measures:
            measure_sub = hosp_original[hosp_original['Measure Name'] == measure]
            if measure_sub.empty:
                continue
            meas_list.append({
                'measure': measure,
                'err': round(float(measure_sub['Excess Readmission Ratio'].iloc[0]), 4),
                'discharges': int(measure_sub['Number of Discharges'].iloc[0]),
                'readmissions': int(measure_sub['Number of Readmissions'].iloc[0]),
                'risk': HospitalSPCAnalyzer.classify_hrrp(measure_sub['Excess Readmission Ratio'].iloc[0])
            })
        hospital_measures[hosp] = meas_list

    risk_counts = {
        'Safe': int((hosp_df['Weighted_ERR'] <= 1.0).sum()),
        'Mild': int(((hosp_df['Weighted_ERR'] > 1.0) & (hosp_df['Weighted_ERR'] <= 1.05)).sum()),
        'Moderate': int(((hosp_df['Weighted_ERR'] > 1.05) & (hosp_df['Weighted_ERR'] <= 1.20)).sum()),
        'Severe': int((hosp_df['Weighted_ERR'] > 1.20).sum())
    }

    data_json = {
        'summary': {
            'total_hospitals': int(len(hosp_df)),
            'laney_outlier_count': int(len(laney_outliers)),
            'trad_outlier_count': int(len(trad_outliers)),
            'mean_weighted_err': round(float(hosp_df['Weighted_ERR'].mean()), 4),
            'risk_counts': risk_counts
        },
        'comparison': {
            'laney': {
                'outliers': [round(v, 4) for v in laney_outliers],
                'non_outliers': [round(v, 4) for v in laney_non_outliers],
                'outlier_mean': round(float(np.mean(laney_outliers)), 4) if laney_outliers else 0,
                'non_outlier_mean': round(float(np.mean(laney_non_outliers)), 4) if laney_non_outliers else 0,
                'outlier_median': round(float(np.median(laney_outliers)), 4) if laney_outliers else 0,
                'non_outlier_median': round(float(np.median(laney_non_outliers)), 4) if laney_non_outliers else 0,
                'mw_p': laney_mw_p
            },
            'trad': {
                'outliers': [round(v, 4) for v in trad_outliers],
                'non_outliers': [round(v, 4) for v in trad_non_outliers],
                'outlier_mean': round(float(np.mean(trad_outliers)), 4) if trad_outliers else 0,
                'non_outlier_mean': round(float(np.mean(trad_non_outliers)), 4) if trad_non_outliers else 0,
                'outlier_median': round(float(np.median(trad_outliers)), 4) if trad_outliers else 0,
                'non_outlier_median': round(float(np.median(trad_non_outliers)), 4) if trad_non_outliers else 0,
                'mw_p': trad_mw_p
            }
        },
        'scatter': {
            'laney': {
                'x': [int(v) for v in laney_x],
                'y': [round(v, 4) for v in laney_y],
                'trend': laney_trend
            },
            'trad': {
                'x': [int(v) for v in trad_x],
                'y': [round(v, 4) for v in trad_y],
                'trend': trad_trend
            }
        },
        'states': states_data,
        'hospital_measures': hospital_measures,
        'hospitals_list': sorted([h for h in hospital_data], key=lambda x: x['Weighted_ERR'], reverse=True)
    }

    return data_json


CSS_TEMPLATE = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg: #0b1121;
    --bg-card: #111827;
    --bg-card-hover: #1a2332;
    --border: #1e2d3d;
    --text: #e2e8f0;
    --text-muted: #64748b;
    --text-dim: #475569;
    --accent: #6366f1;
    --accent-glow: rgba(99,102,241,0.3);
    --green: #10b981;
    --green-bg: rgba(16,185,129,0.12);
    --amber: #f59e0b;
    --amber-bg: rgba(245,158,11,0.12);
    --orange: #f97316;
    --orange-bg: rgba(249,115,22,0.12);
    --red: #ef4444;
    --red-bg: rgba(239,68,68,0.12);
    --purple: #8b5cf6;
    --purple-bg: rgba(139,92,246,0.12);
    --radius: 12px;
    --radius-sm: 8px;
}

* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    min-height: 100vh;
}

.header {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #111827 100%);
    border-bottom: 1px solid var(--border);
    padding: 2.5rem 2rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(99,102,241,0.06) 0%, transparent 50%),
                radial-gradient(circle at 70% 50%, rgba(139,92,246,0.06) 0%, transparent 50%);
    pointer-events: none;
}
.header h1 {
    font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em;
    background: linear-gradient(135deg, #e2e8f0, #a5b4fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    position: relative; z-index: 1;
}
.header p {
    color: var(--text-muted); margin-top: 0.75rem; font-size: 1rem; position: relative; z-index: 1;
    max-width: 700px; margin-left: auto; margin-right: auto;
}

.container { max-width: 1440px; margin: 0 auto; padding: 0 2rem; }

.tabs {
    display: flex; gap: 0.25rem; background: var(--bg-card); border-radius: var(--radius);
    padding: 0.375rem; margin: -1.5rem auto 2rem; position: relative; z-index: 10;
    border: 1px solid var(--border); max-width: 600px; justify-content: center;
}
.tab-btn {
    padding: 0.625rem 1.5rem; border-radius: var(--radius-sm); font-weight: 600;
    font-size: 0.9rem; cursor: pointer; transition: all 0.2s; border: none;
    background: transparent; color: var(--text-muted); font-family: inherit;
}
.tab-btn:hover { color: var(--text); }
.tab-btn.active { background: var(--accent); color: white; }

.tab-panel { display: none; animation: fadeIn 0.3s ease; }
.tab-panel.active { display: block; }

@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1.25rem;
    margin-bottom: 2rem;
}

.kpi-card {
    background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 1.5rem; position: relative; overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
}
.kpi-card:hover { border-color: var(--text-dim); transform: translateY(-2px); }
.kpi-card .kpi-value {
    font-size: 2.2rem; font-weight: 800; line-height: 1; margin-bottom: 0.375rem;
}
.kpi-card .kpi-label {
    font-size: 0.85rem; color: var(--text-muted); font-weight: 500;
}
.kpi-card .kpi-icon {
    position: absolute; top: 1rem; right: 1rem; font-size: 1.5rem; opacity: 0.3;
}
.kpi-accent .kpi-value { color: var(--accent); }
.kpi-purple .kpi-value { color: var(--purple); }
.kpi-orange .kpi-value { color: var(--orange); }
.kpi-green .kpi-value { color: var(--green); }
.kpi-amber .kpi-value { color: var(--amber); }

.risk-bar {
    display: flex; height: 8px; border-radius: 4px; overflow: hidden; margin-top: 1rem;
    background: var(--border);
}
.risk-bar .r-safe { background: var(--green); }
.risk-bar .r-mild { background: var(--amber); }
.risk-bar .r-moderate { background: var(--orange); }
.risk-bar .r-severe { background: var(--red); }

.section-title {
    font-size: 1.25rem; font-weight: 700; margin-bottom: 1.25rem; display: flex; align-items: center; gap: 0.5rem;
    color: var(--text);
}

.card {
    background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 1.5rem; margin-bottom: 1.5rem;
}

.chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }
.chart-box {
    background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 1.5rem;
}
.chart-box canvas { max-height: 380px; }
.chart-box h3 { font-size: 1rem; font-weight: 600; margin-bottom: 1rem; }

.scatter-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }

.search-bar {
    width: 100%; padding: 0.75rem 1rem; background: var(--bg); border: 1px solid var(--border);
    border-radius: var(--radius-sm); color: var(--text); font-size: 0.95rem; font-family: inherit;
    margin-bottom: 1rem; outline: none; transition: border-color 0.2s;
}
.search-bar:focus { border-color: var(--accent); }

select.filter-select {
    padding: 0.5rem 1rem; background: var(--bg); border: 1px solid var(--border);
    border-radius: var(--radius-sm); color: var(--text); font-size: 0.9rem; font-family: inherit;
    outline: none; margin-bottom: 1rem; margin-right: 0.5rem; cursor: pointer;
}
select.filter-select:focus { border-color: var(--accent); }

.table-wrapper { overflow-x: auto; max-height: 600px; overflow-y: auto; }
.compare-table { width: 100%; margin-top: 0.5rem; }
.compare-table th { padding: 0.6rem 1rem; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.03em; }
.compare-table td { padding: 0.55rem 1rem; font-size: 0.9rem; }
.val-purple { color: var(--purple); font-weight: 700; font-family: monospace; font-size: 0.95rem; }
.val-orange { color: var(--orange); font-weight: 700; font-family: monospace; font-size: 0.95rem; }
table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
thead { position: sticky; top: 0; z-index: 2; }
th {
    background: var(--bg); padding: 0.875rem 1rem; text-align: left; font-weight: 600;
    color: var(--text-muted); border-bottom: 2px solid var(--border); cursor: pointer;
    white-space: nowrap; user-select: none; font-size: 0.8rem; text-transform: uppercase;
    letter-spacing: 0.03em;
}
th:hover { color: var(--text); }
th .sort-arrow { margin-left: 0.25rem; font-size: 0.7rem; }
td {
    padding: 0.75rem 1rem; border-bottom: 1px solid var(--border); color: var(--text);
    vertical-align: middle;
}
tr:hover td { background: var(--bg-card-hover); }

.badge {
    display: inline-block; padding: 0.25rem 0.75rem; border-radius: 9999px;
    font-weight: 600; font-size: 0.8rem;
}
.badge-safe { background: var(--green-bg); color: var(--green); }
.badge-mild { background: var(--amber-bg); color: var(--amber); }
.badge-moderate { background: var(--orange-bg); color: var(--orange); }
.badge-severe { background: var(--red-bg); color: var(--red); }
.badge-laney { background: var(--purple-bg); color: var(--purple); }
.badge-trad { background: rgba(249,115,22,0.12); color: var(--orange); }

.state-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 1rem; }
.state-card {
    background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 1.25rem; cursor: pointer; transition: all 0.2s;
}
.state-card:hover { border-color: var(--accent); transform: translateY(-3px); }
.state-card .st-name { font-size: 1.1rem; font-weight: 700; color: var(--text); }
.state-card .st-err { font-size: 1.5rem; font-weight: 800; margin: 0.5rem 0; }
.state-card .st-meta { font-size: 0.85rem; color: var(--text-muted); }

.modal-overlay {
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.7);
    z-index: 100; justify-content: center; align-items: center; padding: 2rem;
}
.modal-overlay.show { display: flex; }
.modal {
    background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius);
    max-height: 85vh; overflow-y: auto; width: 100%; max-width: 900px; padding: 2rem;
    animation: fadeIn 0.2s ease;
}
.modal h2 { font-size: 1.4rem; font-weight: 700; margin-bottom: 1.5rem; }
.modal-close {
    position: sticky; float: right; top: 0; background: none; border: none;
    color: var(--text-muted); font-size: 1.5rem; cursor: pointer; z-index: 1;
}
.modal-close:hover { color: var(--text); }

.stat-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
.stat-pill {
    padding: 0.75rem 1.25rem; border-radius: var(--radius-sm); font-weight: 600;
    background: var(--bg); border: 1px solid var(--border);
}
.stat-pill .pill-val { font-size: 1.5rem; font-weight: 800; line-height: 1; }
.stat-pill .pill-lbl { font-size: 0.8rem; color: var(--text-muted); margin-top: 0.25rem; }

@media (max-width: 768px) {
    .chart-grid, .scatter-grid { grid-template-columns: 1fr; }
    .kpi-grid { grid-template-columns: repeat(2, 1fr); }
    .header h1 { font-size: 1.5rem; }
    .tabs { margin: -1rem 1rem 1.5rem; }
    .container { padding: 0 1rem; }
}
"""

JS_TEMPLATE = r"""
// ── Data injection ──
const DATA = __DATA_PLACEHOLDER__;

// ── Helpers ──
function riskColor(risk) {
    const m = { Safe: 'var(--green)', Mild: 'var(--amber)', Moderate: 'var(--orange)', Severe: 'var(--red)' };
    return m[risk] || 'var(--text-muted)';
}

// ── Tab switching ──
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('panel-' + btn.dataset.tab).classList.add('active');
        if (btn.dataset.tab === 'overview') { window.dispatchEvent(new Event('resize')); }
    });
});

// ── KPIs ──
(function renderKPIs() {
    const s = DATA.summary;
    document.getElementById('kpi-total').textContent = s.total_hospitals.toLocaleString();
    document.getElementById('kpi-laney').textContent = s.laney_outlier_count.toLocaleString();
    document.getElementById('kpi-trad').textContent = s.trad_outlier_count.toLocaleString();
    document.getElementById('kpi-mean').textContent = s.mean_weighted_err.toFixed(4);
    const rc = s.risk_counts;
    const total = rc.Safe + rc.Mild + rc.Moderate + rc.Severe || 1;
    document.getElementById('bar-safe').style.width = (rc.Safe/total*100) + '%';
    document.getElementById('bar-mild').style.width = (rc.Mild/total*100) + '%';
    document.getElementById('bar-moderate').style.width = (rc.Moderate/total*100) + '%';
    document.getElementById('bar-severe').style.width = (rc.Severe/total*100) + '%';
    document.getElementById('rc-safe').textContent = 'Safe: ' + rc.Safe;
    document.getElementById('rc-mild').textContent = 'Mild: ' + rc.Mild;
    document.getElementById('rc-moderate').textContent = 'Moderate: ' + rc.Moderate;
    document.getElementById('rc-severe').textContent = 'Severe: ' + rc.Severe;

    const laney = DATA.comparison.laney;
    const trad = DATA.comparison.trad;
    document.getElementById('cmp-laney-count').textContent = laney.outliers.length;
    document.getElementById('cmp-trad-count').textContent = trad.outliers.length;
    document.getElementById('cmp-laney-mean').textContent = laney.outlier_mean.toFixed(4);
    document.getElementById('cmp-trad-mean').textContent = trad.outlier_mean.toFixed(4);
    document.getElementById('cmp-laney-nonmean').textContent = laney.non_outlier_mean.toFixed(4);
    document.getElementById('cmp-trad-nonmean').textContent = trad.non_outlier_mean.toFixed(4);
    document.getElementById('cmp-laney-med').textContent = laney.outlier_median.toFixed(4);
    document.getElementById('cmp-trad-med').textContent = trad.outlier_median.toFixed(4);
    document.getElementById('cmp-laney-nonmed').textContent = laney.non_outlier_median.toFixed(4);
    document.getElementById('cmp-trad-nonmed').textContent = trad.non_outlier_median.toFixed(4);
    document.getElementById('cmp-laney-p').textContent = laney.mw_p !== null ? laney.mw_p.toExponential(3) : 'N/A';
    document.getElementById('cmp-trad-p').textContent = trad.mw_p !== null ? trad.mw_p.toExponential(3) : 'N/A';
})();

// ── Overview Charts ──
function buildComparisonChart(ctxId, data, label, color) {
    const ctx = document.getElementById(ctxId).getContext('2d');
    const all = [...data.outliers, ...data.non_outliers];
    if (all.length === 0) return;
    const N = Math.min(all.length, 3000);
    const lo = Math.min(...all), hi = Math.max(...all);
    const span = hi - lo;
    if (span === 0) {
        // single value: show as a note
        ctx.canvas.parentNode.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:3rem;">All values identical: ' + lo.toFixed(4) + '</p>';
        return;
    }
    const nBins = Math.max(8, Math.min(50, Math.ceil(Math.sqrt(N))));
    const step = span / nBins;
    const histOut = new Array(nBins).fill(0);
    const histNon = new Array(nBins).fill(0);
    data.outliers.forEach(v => { const i = Math.min(nBins-1, Math.floor((v-lo)/step)); histOut[i]++; });
    data.non_outliers.forEach(v => { const i = Math.min(nBins-1, Math.floor((v-lo)/step)); histNon[i]++; });

    const labels = [];
    for (let i = 0; i < nBins; i++) labels.push((lo + i*step).toFixed(3));

    const pval = data.mw_p !== null ? 'MW p=' + data.mw_p.toExponential(2) : 'MW p=N/A';
    const statsStr = 'Out μ=' + data.outlier_mean.toFixed(3) + ' m=' + data.outlier_median.toFixed(3) +
        ' | Non-Out μ=' + data.non_outlier_mean.toFixed(3) + ' m=' + data.non_outlier_median.toFixed(3);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [
                {
                    label: 'Outliers (' + data.outliers.length + ')',
                    data: histOut,
                    backgroundColor: color + 'AA',
                    borderColor: color,
                    borderWidth: 1,
                    borderRadius: 3
                },
                {
                    label: 'Non-Outliers (' + data.non_outliers.length + ')',
                    data: histNon,
                    backgroundColor: 'rgba(148,163,184,0.3)',
                    borderColor: 'rgba(148,163,184,0.55)',
                    borderWidth: 1,
                    borderRadius: 3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                title: { display: true, text: label + ' — ' + pval, color: 'var(--text)', font: { size: 14 } },
                subtitle: { display: true, text: statsStr, color: 'var(--text-muted)', font: { size: 10 }, padding: { bottom: 10 } },
                legend: { labels: { color: 'var(--text-muted)', usePointStyle: true, padding: 20 } },
                tooltip: {
                    callbacks: {
                        title: function(items) { return 'ERR range: ' + items[0].label; }
                    }
                }
            },
            scales: {
                x: { stacked: false, ticks: { color: 'var(--text-dim)', maxTicksLimit: 14, font: { size: 10 } }, grid: { color: 'rgba(30,45,61,0.4)' } },
                y: { beginAtZero: true, ticks: { color: 'var(--text-dim)' }, grid: { color: 'rgba(30,45,61,0.4)' } }
            }
        }
    });
}

function buildScatterChart(ctxId, data, label, color) {
    const ctx = document.getElementById(ctxId).getContext('2d');
    const points = data.x.map((x, i) => ({ x: x, y: data.y[i] }));
    const datasets = [{
        label: 'Hospitals',
        data: points,
        backgroundColor: color + '88',
        borderColor: color,
        pointRadius: 5,
        pointHoverRadius: 7
    }];
    if (data.trend) {
        const xMin = Math.min(...data.x), xMax = Math.max(...data.x);
        const t = data.trend;
        datasets.push({
            label: 'Trend (R²=' + t.r2.toFixed(3) + ')',
            data: [{ x: xMin, y: t.slope * xMin + t.intercept }, { x: xMax, y: t.slope * xMax + t.intercept }],
            type: 'line',
            borderColor: '#f1f5f9',
            borderWidth: 2,
            borderDash: [6, 4],
            pointRadius: 0,
            fill: false
        });
    }
    new Chart(ctx, {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                title: { display: true, text: label + ' — Outlier Count vs ERR', color: 'var(--text)', font: { size: 14 } },
                legend: { labels: { color: 'var(--text-muted)', usePointStyle: true, padding: 20 } }
            },
            scales: {
                x: { title: { display: true, text: 'Outlier Measures Count', color: 'var(--text-muted)' }, ticks: { color: 'var(--text-dim)', stepSize: 1 }, grid: { color: 'rgba(30,45,61,0.3)' } },
                y: { title: { display: true, text: 'Weighted ERR', color: 'var(--text-muted)' }, ticks: { color: 'var(--text-dim)' }, grid: { color: 'rgba(30,45,61,0.3)' } }
            }
        }
    });
}

window.addEventListener('load', () => {
    buildComparisonChart('chart-laney', DATA.comparison.laney, 'Laney p\' Chart', 'var(--purple)');
    buildComparisonChart('chart-trad', DATA.comparison.trad, 'Traditional p Chart', 'var(--orange)');
    buildScatterChart('scatter-laney', DATA.scatter.laney, 'Laney p\' Chart', 'var(--purple)');
    buildScatterChart('scatter-trad', DATA.scatter.trad, 'Traditional p Chart', 'var(--orange)');
});

// ── State Grid ──
(function renderStates() {
    const container = document.getElementById('state-grid');
    const states = DATA.states;
    const sortedKeys = Object.keys(states).sort((a, b) => states[b].avg_err - states[a].avg_err);
    for (const st of sortedKeys) {
        const s = states[st];
        const color = s.avg_err <= 1.0 ? 'var(--green)' : (s.avg_err <= 1.05 ? 'var(--amber)' : (s.avg_err <= 1.2 ? 'var(--orange)' : 'var(--red)'));
        const card = document.createElement('div');
        card.className = 'state-card';
        card.style.borderTop = '3px solid ' + color;
        card.innerHTML = '<div class="st-name">' + st + '</div>' +
            '<div class="st-err" style="color:' + color + '">' + s.avg_err.toFixed(4) + '</div>' +
            '<div class="st-meta">Hospitals: ' + s.hospital_count + '</div>';
        card.addEventListener('click', () => showStateModal(st, s));
        container.appendChild(card);
    }
})();

// ── Hospital Table ──
(function buildHospitalTable() {
    const items = DATA.hospitals_list;
    let filtered = [...items];
    let sortKey = 'Weighted_ERR';
    let sortDir = -1;

    function render() {
        const tbody = document.getElementById('hosp-tbody');
        tbody.innerHTML = '';
        for (const h of filtered) {
            const risk = h.Risk.toLowerCase();
            const row = document.createElement('tr');
            row.innerHTML =
                '<td style="font-weight:600;cursor:pointer;color:var(--accent)" data-hosp="' + h.Hospital.replace(/"/g, '&quot;') + '">' + h.Hospital + '</td>' +
                '<td style="font-family:monospace;font-weight:600">' + h.Weighted_ERR.toFixed(4) + '</td>' +
                '<td>' + h.State + '</td>' +
                '<td><span class="badge badge-' + risk + '">' + h.Risk + '</span></td>' +
                '<td style="text-align:center"><span class="badge badge-laney">' + h.Laney_Outlier_Count + '</span></td>' +
                '<td style="text-align:center"><span class="badge badge-trad">' + h.Trad_Outlier_Count + '</span></td>' +
                '<td style="text-align:center;color:var(--text-muted)">' + h.Total_Measures + '</td>';
            row.querySelector('td[data-hosp]').addEventListener('click', () => showHospitalModal(h.Hospital));
            tbody.appendChild(row);
        }
    }

    function sort(key) {
        if (sortKey === key) { sortDir *= -1; } else { sortKey = key; sortDir = -1; }
        filtered.sort((a, b) => {
            const va = a[key], vb = b[key];
            if (typeof va === 'number') return (va - vb) * sortDir;
            return String(va).localeCompare(String(vb)) * sortDir;
        });
        document.querySelectorAll('th').forEach(th => th.querySelector('.sort-arrow').textContent = '');
        const arrow = document.querySelector('th[data-sort="' + key + '"] .sort-arrow');
        if (arrow) arrow.textContent = sortDir === 1 ? ' ▲' : ' ▼';
        render();
    }

    document.querySelectorAll('th[data-sort]').forEach(th => {
        th.addEventListener('click', () => sort(th.dataset.sort));
    });

    document.getElementById('hosp-search').addEventListener('input', e => {
        const q = e.target.value.toLowerCase();
        filtered = items.filter(h => h.Hospital.toLowerCase().includes(q) || h.State.toLowerCase().includes(q));
        render();
    });

    document.getElementById('risk-filter').addEventListener('change', e => {
        const val = e.target.value;
        const q = document.getElementById('hosp-search').value.toLowerCase();
        filtered = items.filter(h => {
            const rMatch = val === 'all' || h.Risk === val;
            const sMatch = !q || h.Hospital.toLowerCase().includes(q) || h.State.toLowerCase().includes(q);
            return rMatch && sMatch;
        });
        render();
    });

    sort('Weighted_ERR');
})();

// ── Modals ──
function showStateModal(stateCode, stateData) {
    const overlay = document.getElementById('modal-overlay');
    const body = document.getElementById('modal-body');
    const color = stateData.avg_err <= 1.0 ? 'var(--green)' : (stateData.avg_err <= 1.05 ? 'var(--amber)' : (stateData.avg_err <= 1.2 ? 'var(--orange)' : 'var(--red)'));
    const riskLabel = stateData.avg_err <= 1.0 ? 'Safe' : (stateData.avg_err <= 1.05 ? 'Mild' : (stateData.avg_err <= 1.2 ? 'Moderate' : 'Severe'));
    let html = '<h2>' + stateCode + ' <span class="badge" style="background:' + color + '22;color:' + color + '">' + riskLabel + '</span></h2>';
    html += '<div class="stat-row">';
    html += '<div class="stat-pill"><div class="pill-val" style="color:' + color + '">' + stateData.avg_err.toFixed(4) + '</div><div class="pill-lbl">Avg Weighted ERR</div></div>';
    html += '<div class="stat-pill"><div class="pill-val">' + stateData.hospital_count + '</div><div class="pill-lbl">Hospitals</div></div>';
    html += '</div>';
    html += '<table><thead><tr><th>Hospital</th><th>ERR</th><th>Risk</th><th>Laney</th><th>Trad</th><th>Measures</th></tr></thead><tbody>';
    const sortedHosp = [...stateData.hospitals].sort((a,b) => b.Weighted_ERR - a.Weighted_ERR);
    for (const h of sortedHosp) {
        const r = h.Risk.toLowerCase();
        html += '<tr>' +
            '<td style="font-weight:600;cursor:pointer;color:var(--accent)" class="hosp-link" data-hosp="' + h.Hospital.replace(/"/g, '&quot;') + '">' + h.Hospital + '</td>' +
            '<td style="font-family:monospace">' + h.Weighted_ERR.toFixed(4) + '</td>' +
            '<td><span class="badge badge-' + r + '">' + h.Risk + '</span></td>' +
            '<td style="text-align:center"><span class="badge badge-laney">' + h.Laney_Outlier_Count + '</span></td>' +
            '<td style="text-align:center"><span class="badge badge-trad">' + h.Trad_Outlier_Count + '</span></td>' +
            '<td style="text-align:center;color:var(--text-muted)">' + h.Total_Measures + '</td>' +
            '</tr>';
    }
    html += '</tbody></table>';
    body.innerHTML = html;
    body.querySelectorAll('.hosp-link').forEach(el => {
        el.addEventListener('click', () => {
            showHospitalModal(el.dataset.hosp);
        });
    });
    overlay.classList.add('show');
}

function showHospitalModal(hospName) {
    const overlay = document.getElementById('modal-overlay');
    const body = document.getElementById('modal-body');
    const hosp = DATA.hospitals_list.find(h => h.Hospital === hospName);
    const measures = DATA.hospital_measures[hospName] || [];
    let html = '<h2>' + hospName + '</h2>';
    html += '<div class="stat-row">';
    const risk = hosp.Risk.toLowerCase();
    const color = riskColor(hosp.Risk);
    html += '<div class="stat-pill"><div class="pill-val" style="color:' + color + '">' + hosp.Weighted_ERR.toFixed(4) + '</div><div class="pill-lbl">Weighted ERR</div></div>';
    html += '<div class="stat-pill"><div class="pill-val">' + hosp.Laney_Outlier_Count + '</div><div class="pill-lbl">Laney Outliers</div></div>';
    html += '<div class="stat-pill"><div class="pill-val">' + hosp.Trad_Outlier_Count + '</div><div class="pill-lbl">Trad Outliers</div></div>';
    html += '<div class="stat-pill"><div class="pill-val">' + measures.length + '</div><div class="pill-lbl">Measures</div></div>';
    html += '</div>';
    html += '<table><thead><tr><th>Measure</th><th>ERR</th><th>Risk</th><th>Discharges</th><th>Readmissions</th></tr></thead><tbody>';
    for (const m of measures) {
        const r = m.risk.toLowerCase();
        html += '<tr>' +
            '<td style="font-weight:500">' + m.measure + '</td>' +
            '<td style="font-family:monospace;font-weight:600">' + m.err.toFixed(4) + '</td>' +
            '<td><span class="badge badge-' + r + '">' + m.risk + '</span></td>' +
            '<td>' + m.discharges.toLocaleString() + '</td>' +
            '<td>' + m.readmissions.toLocaleString() + '</td>' +
            '</tr>';
    }
    html += '</tbody></table>';
    body.innerHTML = html;
    overlay.classList.add('show');
}

document.getElementById('modal-close').addEventListener('click', () => {
    document.getElementById('modal-overlay').classList.remove('show');
});
document.getElementById('modal-overlay').addEventListener('click', e => {
    if (e.target === document.getElementById('modal-overlay')) {
        document.getElementById('modal-overlay').classList.remove('show');
    }
});
document.addEventListener('keydown', e => {
    if (e.key === 'Escape') document.getElementById('modal-overlay').classList.remove('show');
});
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SPC Hospital Readmission Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
__CSS__
</style>
</head>
<body>

<div class="header">
    <h1>SPC Hospital Readmission Dashboard</h1>
    <p>Laney p' Chart vs Traditional p Chart — Excess Readmission Ratio (ERR) analysis with Wilson Score correction for small samples</p>
</div>

<div class="container">
    <div class="tabs">
        <button class="tab-btn active" data-tab="overview">Overview</button>
        <button class="tab-btn" data-tab="states">State Analysis</button>
        <button class="tab-btn" data-tab="hospitals">Hospital Details</button>
    </div>

    <!-- ═══ OVERVIEW ═══ -->
    <div class="tab-panel active" id="panel-overview">
        <div class="kpi-grid">
            <div class="kpi-card kpi-accent">
                <div class="kpi-icon">&#9878;</div>
                <div class="kpi-value" id="kpi-total">—</div>
                <div class="kpi-label">Total Hospitals</div>
            </div>
            <div class="kpi-card kpi-purple">
                <div class="kpi-icon">&#9670;</div>
                <div class="kpi-value" id="kpi-laney">—</div>
                <div class="kpi-label">Laney p' Outlier Hospitals</div>
            </div>
            <div class="kpi-card kpi-orange">
                <div class="kpi-icon">&#9670;</div>
                <div class="kpi-value" id="kpi-trad">—</div>
                <div class="kpi-label">Traditional Outlier Hospitals</div>
            </div>
            <div class="kpi-card kpi-green">
                <div class="kpi-icon">&#956;</div>
                <div class="kpi-value" id="kpi-mean">—</div>
                <div class="kpi-label">Mean Weighted ERR</div>
            </div>
        </div>

        <div class="chart-grid">
            <div class="card" style="margin-bottom:0;">
                <div class="section-title">Method Performance Comparison</div>
                <table class="compare-table">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th style="color:var(--purple)">Laney p'</th>
                            <th style="color:var(--orange)">Traditional p</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr><td>Outlier Hospitals</td><td class="val-purple" id="cmp-laney-count">—</td><td class="val-orange" id="cmp-trad-count">—</td></tr>
                        <tr><td>Outlier Mean ERR</td><td class="val-purple" id="cmp-laney-mean">—</td><td class="val-orange" id="cmp-trad-mean">—</td></tr>
                        <tr><td>Non-Outlier Mean ERR</td><td style="color:var(--text-muted)" id="cmp-laney-nonmean">—</td><td style="color:var(--text-muted)" id="cmp-trad-nonmean">—</td></tr>
                        <tr><td>Outlier Median ERR</td><td class="val-purple" id="cmp-laney-med">—</td><td class="val-orange" id="cmp-trad-med">—</td></tr>
                        <tr><td>Non-Outlier Median ERR</td><td style="color:var(--text-muted)" id="cmp-laney-nonmed">—</td><td style="color:var(--text-muted)" id="cmp-trad-nonmed">—</td></tr>
                        <tr><td>Separation (MW p-value)</td><td class="val-purple" id="cmp-laney-p">—</td><td class="val-orange" id="cmp-trad-p">—</td></tr>
                    </tbody>
                </table>
            </div>
            <div class="card" style="margin-bottom:0;">
                <div class="section-title">Risk Tier Distribution</div>
                <div style="display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:0.75rem;">
                    <span id="rc-safe" style="font-weight:700;font-size:0.85rem;color:var(--green);">—</span>
                    <span id="rc-mild" style="font-weight:700;font-size:0.85rem;color:var(--amber);">—</span>
                    <span id="rc-moderate" style="font-weight:700;font-size:0.85rem;color:var(--orange);">—</span>
                    <span id="rc-severe" style="font-weight:700;font-size:0.85rem;color:var(--red);">—</span>
                </div>
                <div class="risk-bar">
                    <div class="r-safe" id="bar-safe" style="width:0%"></div>
                    <div class="r-mild" id="bar-mild" style="width:0%"></div>
                    <div class="r-moderate" id="bar-moderate" style="width:0%"></div>
                    <div class="r-severe" id="bar-severe" style="width:0%"></div>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:var(--text-dim);margin-top:0.4rem;">
                    <span>ERR &le; 1.00</span><span>&le; 1.05</span><span>&le; 1.20</span><span>&gt; 1.20</span>
                </div>
            </div>
        </div>

        <div class="chart-grid">
            <div class="chart-box">
                <h3 style="color:var(--purple)">Laney p' Chart — ERR Distribution</h3>
                <canvas id="chart-laney"></canvas>
            </div>
            <div class="chart-box">
                <h3 style="color:var(--orange)">Traditional p Chart — ERR Distribution</h3>
                <canvas id="chart-trad"></canvas>
            </div>
        </div>

        <div class="scatter-grid">
            <div class="chart-box">
                <h3 style="color:var(--purple)">Laney p' — Outlier Count vs ERR</h3>
                <canvas id="scatter-laney"></canvas>
            </div>
            <div class="chart-box">
                <h3 style="color:var(--orange)">Traditional — Outlier Count vs ERR</h3>
                <canvas id="scatter-trad"></canvas>
            </div>
        </div>
    </div>

    <!-- ═══ STATE ANALYSIS ═══ -->
    <div class="tab-panel" id="panel-states">
        <div class="card">
            <div class="section-title">States by Average Weighted ERR</div>
            <p style="color:var(--text-muted);margin-bottom:1rem;">Click any state to drill down into its hospitals.</p>
            <div class="state-grid" id="state-grid"></div>
        </div>
    </div>

    <!-- ═══ HOSPITAL DETAILS ═══ -->
    <div class="tab-panel" id="panel-hospitals">
        <div class="card">
            <div class="section-title">Hospital Performance Matrix</div>
            <div style="display:flex;gap:0.5rem;flex-wrap:wrap;">
                <input type="text" id="hosp-search" class="search-bar" placeholder="Search hospitals or states..." style="flex:1;min-width:200px;">
                <select id="risk-filter" class="filter-select">
                    <option value="all">All Risk Tiers</option>
                    <option value="Safe">Safe</option>
                    <option value="Mild">Mild</option>
                    <option value="Moderate">Moderate</option>
                    <option value="Severe">Severe</option>
                </select>
            </div>
            <div class="table-wrapper">
                <table>
                    <thead>
                        <tr>
                            <th data-sort="Hospital">Hospital <span class="sort-arrow"></span></th>
                            <th data-sort="Weighted_ERR">Weighted ERR <span class="sort-arrow"> ▼</span></th>
                            <th data-sort="State">State <span class="sort-arrow"></span></th>
                            <th data-sort="Risk">Risk <span class="sort-arrow"></span></th>
                            <th data-sort="Laney_Outlier_Count">Laney Outliers <span class="sort-arrow"></span></th>
                            <th data-sort="Trad_Outlier_Count">Trad Outliers <span class="sort-arrow"></span></th>
                            <th data-sort="Total_Measures">Measures <span class="sort-arrow"></span></th>
                        </tr>
                    </thead>
                    <tbody id="hosp-tbody"></tbody>
                </table>
            </div>
        </div>
    </div>
</div>

<!-- MODAL -->
<div class="modal-overlay" id="modal-overlay">
    <div class="modal" id="modal-content">
        <button class="modal-close" id="modal-close">&times;</button>
        <div id="modal-body"></div>
    </div>
</div>

<script>
__JS__
</script>
</body>
</html>"""


def generate_dashboard(csv_path, output_dir=None):
    if output_dir is None:
        base_dir = os.path.dirname(csv_path) or '.'
        output_dir = os.path.join(base_dir, "kpi_page")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data: {csv_path}")
    data_json = build_json_data(csv_path)

    js = JS_TEMPLATE.replace('__DATA_PLACEHOLDER__', json.dumps(data_json, ensure_ascii=False, indent=2))

    html = HTML_TEMPLATE.replace('__CSS__', CSS_TEMPLATE).replace('__JS__', js)

    output_path = os.path.join(output_dir, "dashboard.html")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Dashboard generated: {output_path}")
    return output_path


if __name__ == "__main__":
    input_csv = r"C:\Users\31798\Desktop\毕业论文\data and code\Hospitals_Readmissions_Reduction_Program_ready.csv"
    generate_dashboard(input_csv)
