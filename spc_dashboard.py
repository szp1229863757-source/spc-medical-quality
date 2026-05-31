import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import math

class AdvancedSPCEngine:
    D4 = 3.267
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
        z = (p - p_bar) / sigma_p
        mr = np.abs(np.diff(z))
        mr_bar = np.mean(mr) if len(mr) > 0 else 0
        sigma_z = mr_bar / AdvancedSPCEngine.d2 if mr_bar > 0 else 1.0
        ucl = p_bar + 3 * sigma_z * sigma_p
        lcl = p_bar - 3 * sigma_z * sigma_p
        lcl = np.maximum(lcl, 0)
        small_n_mask = n < 10
        if np.any(small_n_mask):
            lcl_small, ucl_small = AdvancedSPCEngine.calculate_wilson_score_limits(p[small_n_mask], n[small_n_mask], z=3.0)
            ucl[small_n_mask] = ucl_small
            lcl[small_n_mask] = lcl_small
        out_of_control = (p > ucl) | (p < lcl)
        return {
            'p_values': p, 'n_values': n, 'cl': p_bar,
            'ucl': ucl, 'lcl': lcl, 'outliers': out_of_control,
            'sigma_z': sigma_z
        }

    @staticmethod
    def calculate_imr_chart(df):
        x = df['Excess Readmission Ratio'].values
        x_bar = np.mean(x)
        mr = np.abs(np.diff(x))
        mr_bar = np.mean(mr) if len(mr) > 0 else 0
        ucl_x = x_bar + 2.66 * mr_bar
        lcl_x = x_bar - 2.66 * mr_bar
        sigma_process = mr_bar / AdvancedSPCEngine.d2
        USL = 1.0
        Z_upper = (USL - x_bar) / (sigma_process if sigma_process > 0 else 1e-9)
        Ppk = Z_upper / 3.0
        try:
            prob_fail = 0.5 * math.erfc(Z_upper / math.sqrt(2))
        except:
            prob_fail = 0.5
        return {
            'x_values': x, 'mr_values': np.insert(mr, 0, np.nan) if len(mr) > 0 else np.array([np.nan]),
            'cl_x': x_bar, 'ucl_x': ucl_x, 'lcl_x': lcl_x,
            'Ppk': Ppk, 'Z_score': Z_upper,
            'mu': x_bar, 'sigma': sigma_process, 'fail_rate': prob_fail
        }

    @staticmethod
    def calculate_enhanced_kpis(df):
        err = df['Excess Readmission Ratio'].values
        vol = df['Number of Discharges'].values
        weighted_err = np.sum(err * vol) / np.sum(vol) if np.sum(vol) > 0 else 0
        p25, median, p75 = np.percentile(err, [25, 50, 75]) if len(err) > 0 else (0,0,0)
        iqr = p75 - p25
        safe = np.sum(err <= 1.0)
        mild = np.sum((err > 1.0) & (err <= 1.05))
        mod = np.sum((err > 1.05) & (err <= 1.20))
        severe = np.sum(err > 1.20)
        total = len(err)
        return {
            'weighted_err': weighted_err,
            'p25': p25, 'median': median, 'p75': p75, 'iqr': iqr,
            'hrrp': {'safe': safe, 'mild': mild, 'mod': mod, 'severe': severe, 'total': total}
        }

    @staticmethod
    def interpret_performance(stats):
        z = stats['Z_score']
        fail_pct = stats['fail_rate'] * 100
        if z >= 1.0: return ("Elite", f"Highly Capable (Avg {stats['mu']:.3f})", "elite")
        elif 0 <= z < 1.0: return ("Good", f"In Control (Avg {stats['mu']:.3f})", "good")
        elif -1.0 <= z < 0: return ("Warning", f"At Risk (Avg {stats['mu']:.3f})", "warning")
        else: return ("Critical", f"Failing ({fail_pct:.1f}% risk)", "critical")

def create_charts(df, state_name, measure_name):
    spc = AdvancedSPCEngine()
    df_sorted = df.sort_values(by='Number of Discharges').reset_index(drop=True)

    hosp_col = None
    for c in ['Facility Name', 'Hospital Name', 'Provider ID']:
        if c in df_sorted.columns:
            hosp_col = c
            break

    laney_data = spc.calculate_laney_p_chart(df_sorted)
    imr_data = spc.calculate_imr_chart(df_sorted)
    enhanced_kpis = spc.calculate_enhanced_kpis(df_sorted)
    full_stats = {**imr_data, **enhanced_kpis, 'sigma_z': laney_data['sigma_z']}

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"<b>Hybrid Laney P' Funnel</b>: Outlier Detection",
            "<b>Capability Analysis</b>: Ratio Distribution vs Target",
            "<b>I-Chart</b>: Individual Process Stability",
            "<b>MR-Chart</b>: Facility-to-Facility Variation"
        ),
        vertical_spacing=0.15, horizontal_spacing=0.08
    )
    qmul_blue, qmul_light, qmul_red, gray_line = '#00205B', '#00B1C1', '#C8102E', '#64748b'

    n_sorted = laney_data['n_values']
    outliers = laney_data['outliers']

    if hosp_col:
        hosp_names = df_sorted[hosp_col].values
    else:
        hosp_names = np.array([f"Facility {i}" for i in range(len(df_sorted))])

    fig.add_trace(go.Scatter(
        x=n_sorted[~outliers], y=laney_data['p_values'][~outliers],
        mode='markers', marker=dict(color=qmul_blue, opacity=0.6, size=6),
        text=hosp_names[~outliers],
        hovertemplate="<b>%{text}</b><br>Rate: %{y:.2%}<br>Vol: %{x}", name='In Control'
    ), row=1, col=1)
    if any(outliers):
        fig.add_trace(go.Scatter(
            x=n_sorted[outliers], y=laney_data['p_values'][outliers],
            mode='markers', marker=dict(color=qmul_red, size=8, symbol='x', line=dict(width=1, color='white')),
            text=hosp_names[outliers],
            hovertemplate="<b>%{text}</b><br>Rate: %{y:.2%}<br>Vol: %{x} (OUTLIER)", name='Outlier'
        ), row=1, col=1)
    fig.add_trace(go.Scatter(x=n_sorted, y=laney_data['ucl'], mode='lines', line=dict(color=qmul_red, width=1), name='UCL'), row=1, col=1)
    fig.add_trace(go.Scatter(x=n_sorted, y=laney_data['lcl'], mode='lines', line=dict(color=qmul_red, width=1), fill='tonexty', fillcolor='rgba(200, 16, 46, 0.05)', name='LCL'), row=1, col=1)
    fig.add_trace(go.Scatter(x=n_sorted, y=[laney_data['cl']]*len(n_sorted), mode='lines', line=dict(color=gray_line, dash='dash'), name='Mean Rate'), row=1, col=1)

    mu, sigma = imr_data['mu'], imr_data['sigma']
    fig.add_trace(go.Histogram(
        x=df_sorted['Excess Readmission Ratio'], histnorm='probability density',
        marker_color='#e2e8f0', opacity=0.8, name='Observed'
    ), row=1, col=2)
    x_min = df_sorted['Excess Readmission Ratio'].min()
    x_max = df_sorted['Excess Readmission Ratio'].max()
    if pd.isna(x_min) or pd.isna(x_max):
        x_min, x_max = 0.6, 1.4
    x_range = np.linspace(min(x_min, 0.6), max(x_max, 1.4), 200)
    if sigma > 0:
        pdf = (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x_range - mu)/sigma)**2)
        fig.add_trace(go.Scatter(x=x_range, y=pdf, mode='lines', line=dict(color=qmul_blue, width=2), name='Normal Fit'), row=1, col=2)
    fig.add_vline(x=1.0, line_color=qmul_red, line_width=2, line_dash="dash", annotation_text="Target <= 1.0", row=1, col=2)

    x_vals = imr_data['x_values']
    outliers_x = (x_vals > imr_data['ucl_x']) | (x_vals < imr_data['lcl_x'])
    fig.add_trace(go.Scatter(
        y=x_vals, mode='lines+markers', line=dict(color=qmul_light, width=1),
        marker=dict(color=np.where(outliers_x, qmul_red, qmul_blue), size=5),
        text=hosp_names, hovertemplate="<b>%{text}</b><br>Ratio: %{y:.3f}", name='Ratio'
    ), row=2, col=1)
    fig.add_hline(y=imr_data['ucl_x'], line_color=qmul_red, line_dash='dash', row=2, col=1, annotation_text=f"UCL={imr_data['ucl_x']:.3f}")
    fig.add_hline(y=imr_data['lcl_x'], line_color=qmul_red, line_dash='dash', row=2, col=1, annotation_text=f"LCL={imr_data['lcl_x']:.3f}")
    fig.add_hline(y=imr_data['cl_x'], line_color=gray_line, row=2, col=1)

    mr_vals = imr_data['mr_values']
    valid_mr = mr_vals[~np.isnan(mr_vals)]
    mr_mean = np.mean(valid_mr) if len(valid_mr) > 0 else 0
    ucl_mr = AdvancedSPCEngine.D4 * mr_mean if mr_mean > 0 else 0
    fig.add_trace(go.Scatter(
        y=mr_vals, mode='lines+markers', line=dict(color=gray_line, width=1),
        marker=dict(size=4, color=gray_line), name='Moving Range'
    ), row=2, col=2)
    if ucl_mr > 0:
        fig.add_hline(y=ucl_mr, line_color=qmul_red, line_dash='dash', row=2, col=2, annotation_text=f"UCL={ucl_mr:.3f}")
    fig.add_hline(y=mr_mean, line_color=gray_line, row=2, col=2)

    fig.update_xaxes(title_text="Discharge Volume (出院人数)", row=1, col=1)
    fig.update_yaxes(title_text="Readmission Rate (再入院率)", tickformat=".1%", row=1, col=1)
    fig.update_xaxes(title_text="Excess Readmission Ratio (超额再入院比率)", row=1, col=2)
    fig.update_yaxes(title_text="Probability Density (概率密度)", row=1, col=2)
    fig.update_xaxes(title_text="Facility Index (医疗机构序号)", row=2, col=1)
    fig.update_yaxes(title_text="Ratio (比率)", row=2, col=1)
    fig.update_xaxes(title_text="Facility Index (医疗机构序号)", row=2, col=2)
    fig.update_yaxes(title_text="Moving Range (移动极差)", row=2, col=2)

    fig.update_layout(template='plotly_white', height=900, showlegend=False,
                      margin=dict(l=60, r=40, t=60, b=60),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    spc_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    return spc_html, full_stats

GLOBAL_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    :root {
        --qmul-blue: #00205B;
        --qmul-light-blue: #00B1C1;
        --bg-body: #F8FAFC;
    }

    body { font-family: 'Inter', sans-serif; background-color: var(--bg-body); color: #334155; }
    
    .qmul-nav { background: var(--qmul-blue); color: white; padding: 1rem 0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    .qmul-nav a { color: white; text-decoration: none; }
    
    .dashboard-card {
        background: white;
        border-radius: 12px;
        box-shadow: 0 1px 3px 0 rgba(0,0,0,0.1);
        transition: transform 0.2s, box-shadow 0.2s;
        border: 1px solid #f1f5f9;
        height: 100%;
    }
    .dashboard-card:hover { transform: translateY(-3px); box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); }

    .status-card-elite { border-top: 5px solid #059669; background: #ecfdf5; } 
    .status-card-elite .status-badge { background: #059669; color: white; }
    .status-card-good { border-top: 5px solid #2563eb; background: #eff6ff; } 
    .status-card-good .status-badge { background: #2563eb; color: white; }
    .status-card-warning { border-top: 5px solid #d97706; background: #fffbeb; } 
    .status-card-warning .status-badge { background: #d97706; color: white; }
    .status-card-critical { border-top: 5px solid #e11d48; background: #fef2f2; } 
    .status-card-critical .status-badge { background: #e11d48; color: white; }

    .kpi-icon {
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        font-size: 1.2rem;
        margin-bottom: 10px;
    }
    .icon-avg { background-color: #f1f5f9; color: #475569; }
    .icon-weight { background-color: #e0f2fe; color: #0284c7; }
    .icon-disp { background-color: #f3e8ff; color: #9333ea; }
    .icon-hrrp { background-color: #ffe4e6; color: #e11d48; }
    
    .progress-stacked { height: 24px; border-radius: 6px; overflow: hidden; display: flex; }
    .prog-safe { background-color: #10b981; color: white; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: bold;}
    .prog-mild { background-color: #f59e0b; color: white; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: bold;}
    .prog-mod { background-color: #f97316; color: white; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: bold;}
    .prog-sev { background-color: #ef4444; color: white; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: bold;}

    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #f1f5f9; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
</style>
"""

def generate_report_page(state, measure, spc_plot_html, stats, state_full_df):
    status_title, status_desc, status_key = AdvancedSPCEngine.interpret_performance(stats)
    if status_key == "elite": bg_head = "#ecfdf5"; text_head = "#059669"
    elif status_key == "good": bg_head = "#eff6ff"; text_head = "#2563eb"
    elif status_key == "warning": bg_head = "#fffbeb"; text_head = "#d97706"
    else: bg_head = "#fef2f2"; text_head = "#e11d48"
    tot = stats['hrrp']['total']
    pct_safe = (stats['hrrp']['safe'] / tot) * 100 if tot else 0
    pct_mild = (stats['hrrp']['mild'] / tot) * 100 if tot else 0
    pct_mod = (stats['hrrp']['mod'] / tot) * 100 if tot else 0
    pct_sev = (stats['hrrp']['severe'] / tot) * 100 if tot else 0

    hosp_col = None
    for c in ['Facility Name', 'Hospital Name', 'Provider ID']:
        if c in state_full_df.columns:
            hosp_col = c
            break
    if hosp_col is None:
        hosp_col = state_full_df.columns[0]

    pivot_df = state_full_df.pivot_table(
        index=hosp_col,
        columns='Measure Name',
        values='Excess Readmission Ratio',
        aggfunc='first'
    ).reset_index()
    pivot_df = pivot_df.round(3).fillna('-')
    table_headers = "".join([f"<th>{col}</th>" for col in pivot_df.columns])
    table_rows = ""
    for _, row in pivot_df.iterrows():
        row_cells = "".join([f"<td>{val}</td>" for val in row])
        table_rows += f"<tr>{row_cells}</tr>"
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>{state} Report - Advanced SPC</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        {GLOBAL_CSS}
    </head>
    <body>
        <nav class="qmul-nav">
            <div class="container-fluid px-5 d-flex align-items-center">
                <span class="fw-bold fs-4"><i class="fa-solid fa-chart-line me-2"></i> Advanced SPC Analytics</span>
                <div class="ms-auto">
                    <a href="index.html" class="btn btn-outline-light btn-sm rounded-pill px-4"><i class="fa-solid fa-arrow-left me-1"></i> Back to Overview</a>
                </div>
            </div>
        </nav>

        <div class="container py-4">
            <div class="row align-items-center mb-4 pb-3 border-bottom">
                <div class="col-md-8">
                    <div class="small text-uppercase text-muted fw-bold"><i class="fa-solid fa-notes-medical me-1"></i> {measure}</div>
                    <h1 class="display-5 fw-bold text-dark">{state} Performance Report</h1>
                </div>
                <div class="col-md-4 text-md-end">
                    <span class="px-4 py-2 rounded-pill fw-bold text-uppercase" 
                          style="background-color: {bg_head}; color: {text_head}; border: 1px solid {text_head}">
                        <i class="fa-solid fa-circle-check me-1"></i> {status_title}
                    </span>
                    <div class="mt-2 text-muted small">{status_desc}</div>
                </div>
            </div>

            <h5 class="fw-bold mb-3 mt-4 text-dark"><i class="fa-solid fa-gauge-high me-2"></i> Executive Summary & Vol-Weighted Analysis</h5>
            <div class="row g-3 mb-4">
                <div class="col-md-3">
                    <div class="dashboard-card p-3">
                        <div class="kpi-icon icon-avg"><i class="fa-solid fa-calculator"></i></div>
                        <small class="text-muted fw-bold">SIMPLE AVG ERR</small>
                        <div class="fs-3 fw-bold text-dark">{stats['mu']:.3f}</div>
                        <small class="text-muted">Unweighted Arithmetic Mean</small>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="dashboard-card p-3 border-start border-4 border-info">
                        <div class="kpi-icon icon-weight"><i class="fa-solid fa-scale-balanced"></i></div>
                        <small class="text-muted fw-bold">VOLUME-WEIGHTED ERR</small>
                        <div class="fs-3 fw-bold text-info">{stats['weighted_err']:.3f}</div>
                        <small class="text-muted">Actual Patient Burden Risk</small>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="dashboard-card p-3">
                        <div class="kpi-icon icon-disp"><i class="fa-solid fa-arrows-left-right"></i></div>
                        <small class="text-muted fw-bold">DISPERSION (IQR)</small>
                        <div class="fs-3 fw-bold text-secondary">{stats['iqr']:.3f}</div>
                        <div class="d-flex justify-content-between align-items-center mt-1">
                            <small class="text-muted">Median: <b>{stats['median']:.3f}</b></small>
                            <span class="badge bg-light text-dark border">Inconsistency</span>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="dashboard-card p-3">
                        <div class="kpi-icon icon-hrrp"><i class="fa-solid fa-hospital-user"></i></div>
                        <small class="text-muted fw-bold">HRRP PENALTY DISTRIBUTION</small>
                        
                        <div class="progress-stacked mt-2 mb-2 w-100 shadow-sm">
                            <div class="prog-safe" style="width: {pct_safe}%" title="Safe: {pct_safe:.1f}%">{f"{pct_safe:.0f}%" if pct_safe > 10 else ""}</div>
                            <div class="prog-mild" style="width: {pct_mild}%" title="Mild: {pct_mild:.1f}%">{f"{pct_mild:.0f}%" if pct_mild > 10 else ""}</div>
                            <div class="prog-mod" style="width: {pct_mod}%" title="Mod: {pct_mod:.1f}%">{f"{pct_mod:.0f}%" if pct_mod > 10 else ""}</div>
                            <div class="prog-sev" style="width: {pct_sev}%" title="Severe: {pct_sev:.1f}%">{f"{pct_sev:.0f}%" if pct_sev > 10 else ""}</div>
                        </div>
                        
                        <div class="d-flex justify-content-between" style="font-size: 0.7rem;">
                            <span class="text-success fw-bold"><i class="fa-solid fa-square"></i> Safe</span>
                            <span class="text-warning fw-bold"><i class="fa-solid fa-square"></i> Mild</span>
                            <span style="color:#f97316;" class="fw-bold"><i class="fa-solid fa-square"></i> Mod</span>
                            <span class="text-danger fw-bold"><i class="fa-solid fa-square"></i> Sev</span>
                        </div>
                    </div>
                </div>
            </div>

            <h5 class="fw-bold mb-3 mt-5 text-dark"><i class="fa-solid fa-chart-area me-2"></i> Classical SPC Diagnostics</h5>

            <div class="row g-3 mb-4">
                <div class="col-md-3">
                    <div class="dashboard-card p-3 border-start border-4" style="border-color: #00205B !important;">
                        <small class="text-muted fw-bold"><i class="fa-solid fa-chart-line me-1"></i> I-CHART LIMITS</small>
                        <div class="mt-2"><span class="text-muted small">CL:</span> <b>{stats['cl_x']:.4f}</b></div>
                        <div><span class="text-muted small">UCL:</span> <b class="text-danger">{stats['ucl_x']:.4f}</b></div>
                        <div><span class="text-muted small">LCL:</span> <b class="text-danger">{stats['lcl_x']:.4f}</b></div>
                        <div class="mt-2"><span class="text-muted small">Sigma_est:</span> <b>{stats['sigma']:.4f}</b></div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="dashboard-card p-3 border-start border-4" style="border-color: #00B1C1 !important;">
                        <small class="text-muted fw-bold"><i class="fa-solid fa-wave-square me-1"></i> MR-CHART LIMITS</small>
                        <div class="mt-2"><span class="text-muted small">Avg MR:</span> <b>{stats['sigma'] * 1.128:.4f}</b></div>
                        <div><span class="text-muted small">UCL (D4):</span> <b class="text-danger">{stats['sigma'] * 1.128 * 3.267:.4f}</b></div>
                        <div><span class="text-muted small">LCL:</span> <b>0.0000</b></div>
                        <div class="mt-2"><span class="text-muted small">D4 = 3.267</span></div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="dashboard-card p-3 border-start border-4" style="border-color: #059669 !important;">
                        <small class="text-muted fw-bold"><i class="fa-solid fa-gauge-high me-1"></i> CAPABILITY</small>
                        <div class="mt-2"><span class="text-muted small">Ppk:</span> <b>{stats['Ppk']:.4f}</b></div>
                        <div><span class="text-muted small">Z_score:</span> <b>{stats['Z_score']:.4f}</b></div>
                        <div><span class="text-muted small">USL:</span> <b>1.0000</b></div>
                        <div class="mt-2"><span class="text-muted small">Fail_prob:</span> <b>{stats['fail_rate']*100:.2f}%</b></div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="dashboard-card p-3 border-start border-4" style="border-color: #9333ea !important;">
                        <small class="text-muted fw-bold"><i class="fa-solid fa-triangle-exclamation me-1"></i> LANEY P' PARAMS</small>
                        <div class="mt-2"><span class="text-muted small">CL (p_bar):</span> <b>{stats['weighted_err']:.4f}</b></div>
                        <div><span class="text-muted small">sigma_z:</span> <b>{stats.get('sigma_z', 'N/A') if isinstance(stats.get('sigma_z'), (int, float)) else 'N/A'}</b></div>
                        <div class="mt-2"><span class="text-muted small">N (facilities):</span> <b>{stats['hrrp']['total']}</b></div>
                    </div>
                </div>
            </div>

            <div class="dashboard-card p-2 mb-5">
                {spc_plot_html}
            </div>
            
            <h5 class="fw-bold mb-3 mt-5 text-dark"><i class="fa-solid fa-table-list me-2"></i> Hospital Cross-Measure Performance Matrix (ERR)</h5>
            <div class="dashboard-card p-4 mb-5">
                <p class="text-muted small mb-4">Search for any facility in {state} to instantly view their KPI across all evaluated medical conditions. Target is <= 1.0.</p>
                <div class="table-responsive">
                    <table id="hospitalKpiTable" class="table table-striped table-hover border align-middle" style="width:100%">
                        <thead class="table-light">
                            <tr>
                                {table_headers}
                            </tr>
                        </thead>
                        <tbody>
                            {table_rows}
                        </tbody>
                    </table>
                </div>
            </div>
            
        </div>
        
        <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
        <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
        <script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>
        <script>
            $(document).ready(function() {{
                $('#hospitalKpiTable').DataTable({{
                    "pageLength": 10,
                    "language": {{
                        "search": "<i class='fa-solid fa-magnifying-glass'></i> Find Hospital:"
                    }}
                }});
            }});
        </script>
    </body>
    </html>
    """
    return html

def generate_state_index(measure_clean, measure_display, state_stats):
    sorted_stats = sorted(state_stats, key=lambda x: x['z_score'], reverse=True)
    grid_html = ""
    for item in sorted_stats:
        status_title, desc, status_key = AdvancedSPCEngine.interpret_performance(item['full_stats'])
        card_class = f"status-card-{status_key}"
        w_err = item['full_stats']['weighted_err']
        grid_html += f"""
        <div class="col-xl-3 col-lg-4 col-md-6">
            <a href="{item['state']}_Report.html" style="text-decoration:none;">
                <div class="dashboard-card {card_class} p-4 d-flex flex-column h-100">
                    <div class="d-flex justify-content-between align-items-center mb-3">
                        <h4 class="fw-bold mb-0 text-dark">{item['state']}</h4>
                        <span class="badge status-badge rounded-pill px-3">{status_title}</span>
                    </div>
                    
                    <p class="small text-muted mb-4">{desc}</p>
                    
                    <div class="mt-auto d-flex justify-content-between pt-3 border-top border-secondary border-opacity-10">
                        <div>
                            <small class="d-block text-secondary opacity-75" style="font-size:0.7rem">VOL-WEIGHTED ERR</small>
                            <span class="fw-bold text-dark">{w_err:.3f}</span>
                        </div>
                        <div class="text-end">
                            <small class="d-block text-secondary opacity-75" style="font-size:0.7rem">AVG RATIO</small>
                            <span class="fw-bold text-dark">{item['mu']:.3f}</span>
                        </div>
                    </div>
                </div>
            </a>
        </div>
        """
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>{measure_display} Overview</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        {GLOBAL_CSS}
    </head>
    <body>
        <nav class="qmul-nav">
            <div class="container d-flex justify-content-between align-items-center">
                <a class="navbar-brand fw-bold text-white m-0" href="../index.html">SPC Dashboard</a>
                <a href="../index.html" class="btn btn-sm btn-outline-light rounded-pill px-4">Home</a>
            </div>
        </nav>
        
        <div class="bg-white py-5 mb-5 border-bottom">
            <div class="container text-center">
                <h2 class="fw-bold mb-2 text-dark">{measure_display}</h2>
                <p class="text-muted">Advanced Performance Analysis by State</p>
            </div>
        </div>

        <div class="container mb-5">
            <div class="d-flex justify-content-center gap-4 mb-5 flex-wrap">
                <div class="d-flex align-items-center"><span style="width:15px;height:15px;background:#059669;margin-right:8px;border-radius:3px;"></span>Elite</div>
                <div class="d-flex align-items-center"><span style="width:15px;height:15px;background:#2563eb;margin-right:8px;border-radius:3px;"></span>Good</div>
                <div class="d-flex align-items-center"><span style="width:15px;height:15px;background:#d97706;margin-right:8px;border-radius:3px;"></span>Warning</div>
                <div class="d-flex align-items-center"><span style="width:15px;height:15px;background:#e11d48;margin-right:8px;border-radius:3px;"></span>Critical</div>
            </div>

            <div class="row g-4">
                {grid_html}
            </div>
        </div>
    </body>
    </html>
    """
    return html

def generate_home(measures_summary, leaderboard_data):
    measure_cards = ""
    for m in measures_summary:
        measure_cards += f"""
        <div class="col-lg-4 col-md-6 mb-4">
            <a href="./{m['path']}/index.html" style="text-decoration:none;">
                <div class="dashboard-card p-4 h-100 d-flex flex-column">
                    <div class="text-center">
                        <div class="rounded-circle p-3 mb-3 d-inline-block" style="background-color: #f0f9ff; color: #0284c7;">
                            <svg width="32" height="32" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path></svg>
                        </div>
                        <h6 class="fw-bold text-dark">{m['name']}</h6>
                    </div>
                    
                    <hr class="text-secondary opacity-25">
                    
                    <div class="mt-auto mb-3 text-start">
                        <div class="mb-2" style="font-size: 0.85rem; color: #64748b;">
                            <span class="fw-bold text-success">🏆 Top State:</span> 
                            <span class="text-dark fw-bold">{m['best_state']}</span> 
                            (Z-Score: {m['best_state_z']:.2f})
                        </div>
                        <div style="font-size: 0.85rem; color: #64748b;">
                            <span class="fw-bold text-primary">🏥 Top Hospital:</span> 
                            <span class="text-dark fw-bold">{m['best_hosp_name']}</span> 
                            <br><span style="font-size: 0.75rem;">({m['best_hosp_state']} | ERR: {m['best_hosp_err']:.3f})</span>
                        </div>
                    </div>
                    
                    <div class="text-center mt-2">
                        <span class="btn btn-sm btn-outline-primary rounded-pill px-4">View Analysis</span>
                    </div>
                </div>
            </a>
        </div>
        """
    table_rows = ""
    for idx, item in enumerate(leaderboard_data, 1):
        link = f"./{item['path']}/{item['state']}_Report.html"
        z = item['z_score']
        if z >= 1.0: t_color = "text-success"
        elif z >= 0: t_color = "text-primary"
        elif z >= -1.0: t_color = "text-warning"
        else: t_color = "text-danger"
        table_rows += f"""
        <tr>
            <td class="text-center text-muted fw-bold">{idx}</td>
            <td><a href="{link}" class="fw-bold text-decoration-none text-dark">{item['state']}</a></td>
            <td class="text-muted small">{item['measure']}</td>
            <td class="text-center fw-bold {t_color}">{item['z_score']:.2f}</td>
            <td class="text-center text-dark">{item['mu']:.3f}</td>
            <td class="text-center fw-bold text-info">{item['w_err']:.3f}</td>
        </tr>
        """
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Clinical Quality Analytics</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        {GLOBAL_CSS}
        <style> .table-scroll-container {{ height: 600px; overflow-y: auto; }} </style>
    </head>
    <body>
        <div class="bg-dark text-white text-center py-5" style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);">
            <h1 class="display-4 fw-bold mb-2">Clinical Quality Analytics</h1>
            <p class="opacity-75">Advanced SPC Dashboard (Weighted Analysis & HRRP Penalty Integrated)</p>
        </div>

        <div class="container py-5">
            <div class="mb-5">
                <h4 class="fw-bold mb-4 text-dark border-bottom pb-2">Select Measure Category</h4>
                <div class="row g-4">
                    {measure_cards}
                </div>
            </div>

            <div class="row">
                <div class="col-12">
                    <div class="d-flex justify-content-between align-items-end mb-3">
                        <h4 class="fw-bold m-0 text-dark">National Performance Leaderboard</h4>
                        <span class="text-muted small">Scroll to view all {len(leaderboard_data)} entries</span>
                    </div>

                    <div class="table-scroll-container shadow-sm border rounded">
                        <table class="table table-hover table-borderless m-0 table-fixed-head" style="position: relative;">
                            <thead style="position: sticky; top: 0; background-color: var(--qmul-blue); color: white; z-index: 1;">
                                <tr>
                                    <th class="text-center" style="width: 80px;">Rank</th>
                                    <th>State</th>
                                    <th>Measure Category</th>
                                    <th class="text-center">Z-Score</th>
                                    <th class="text-center">Simple Avg</th>
                                    <th class="text-center">Vol-Weighted ERR</th>
                                </tr>
                            </thead>
                            <tbody>
                                {table_rows}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
        
        <footer class="text-center py-4 text-muted small mt-5 border-top">
            &copy; 2026 Advanced SPC Analytics Engine | Enhanced HRRP Module
        </footer>
    </body>
    </html>
    """
    return html

def run_pipeline(csv_path):
    print(f"Loading data from: {csv_path}")
    base_dir = os.path.dirname(csv_path)
    output_dir = os.path.join(base_dir, "SPC page")
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    cols = ['Excess Readmission Ratio', 'Number of Discharges', 'Number of Readmissions']
    for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=cols + ['State', 'Measure Name'])
    df = df[df['Number of Discharges'] > 0]
    measures = df['Measure Name'].unique()
    measures_summary = []
    leaderboard_data = []
    print(f"Found {len(measures)} measures. Generating Advanced Dashboard with Enhanced HRRP Logic...")
    for measure in measures:
        safe_name = measure.replace(' ', '_').replace('/', '_').replace('.', '')[:30]
        measure_dir = os.path.join(output_dir, safe_name)
        if not os.path.exists(measure_dir): os.makedirs(measure_dir)
        m_df = df[df['Measure Name'] == measure]
        states = m_df['State'].unique()
        state_stats_list = []

        hosp_col = None
        for c in ['Facility Name', 'Hospital Name', 'Provider ID']:
            if c in m_df.columns:
                hosp_col = c
                break

        try:
            best_hosp_idx = m_df['Excess Readmission Ratio'].idxmin()
            best_hosp_name = m_df.loc[best_hosp_idx, hosp_col] if hosp_col else 'N/A'
            best_hosp_err = m_df.loc[best_hosp_idx, 'Excess Readmission Ratio']
            best_hosp_state = m_df.loc[best_hosp_idx, 'State']
        except Exception:
            best_hosp_name = 'N/A'
            best_hosp_err = 0
            best_hosp_state = 'N/A'
        for state in states:
            s_df = m_df[m_df['State'] == state]
            if len(s_df) < 3: continue
            state_full_df = df[df['State'] == state]
            spc_plot_html, stats = create_charts(s_df, state, measure)
            if stats:
                report_html = generate_report_page(state, measure, spc_plot_html, stats, state_full_df)
                with open(os.path.join(measure_dir, f"{state}_Report.html"), 'w', encoding='utf-8') as f:
                    f.write(report_html)
                state_stats_list.append({
                    'state': state,
                    'z_score': stats['Z_score'],
                    'mu': stats['mu'],
                    'full_stats': stats
                })
        if state_stats_list:
            index_html = generate_state_index(safe_name, measure, state_stats_list)
            with open(os.path.join(measure_dir, "index.html"), 'w', encoding='utf-8') as f:
                f.write(index_html)
            sorted_state_stats = sorted(state_stats_list, key=lambda x: x['z_score'], reverse=True)
            best_state = sorted_state_stats[0]['state']
            best_state_z = sorted_state_stats[0]['z_score']
            measures_summary.append({
                'name': measure,
                'path': safe_name,
                'best_state': best_state,
                'best_state_z': best_state_z,
                'best_hosp_name': best_hosp_name,
                'best_hosp_err': best_hosp_err,
                'best_hosp_state': best_hosp_state
            })
            for s_item in state_stats_list:
               leaderboard_data.append({
                   'measure': measure,
                   'state': s_item['state'],
                   'z_score': s_item['z_score'],
                   'mu': s_item['mu'],
                   'w_err': s_item['full_stats']['weighted_err'],
                   'path': safe_name
               })
    leaderboard_data.sort(key=lambda x: x['z_score'], reverse=True)
    home_html = generate_home(measures_summary, leaderboard_data)
    with open(os.path.join(output_dir, "index.html"), 'w', encoding='utf-8') as f:
        f.write(home_html)
    print(f"\nSuccess! Dashboard generated at:\n{os.path.join(output_dir, 'index.html')}")

def find_csv_file():
    """Search common locations for the HRRP CSV data file."""
    possible_paths = [
        "Hospitals_Readmissions_Reduction_Program_ready.csv",
        os.path.join("data", "Hospitals_Readmissions_Reduction_Program_ready.csv"),
        os.path.join(os.path.dirname(__file__), "Hospitals_Readmissions_Reduction_Program_ready.csv"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".csv") and "Hospitals_Readmissions_Reduction_Program" in file:
                return os.path.join(root, file)
    return None


if __name__ == "__main__":
    csv_file = find_csv_file()
    if csv_file is None:
        print("Error: Could not find 'Hospitals_Readmissions_Reduction_Program_ready.csv'.")
        print("Please run:  python download_data.py  first, or place the CSV in this directory.")
    else:
        run_pipeline(csv_file)