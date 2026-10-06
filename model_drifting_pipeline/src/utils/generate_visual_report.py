#!/usr/bin/env python3
"""
Generate an executive, professional HTML visualization report for any drift assessment
or training drift experiment without requiring any background servers or Streamlit.
Can be opened in any browser (Chrome, Safari, Firefox, Edge).
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Any, Optional

def generate_inference_html_report(summary_data: Dict[str, Any], output_path: str):
    aid = summary_data.get("assessment_id", "Unknown")
    ts = summary_data.get("timestamp", datetime.now().isoformat())[:19].replace("T", " ")
    model_name = summary_data.get("model_name", "N/A")
    golden_ds = summary_data.get("golden_dataset_name", "N/A")
    new_ds = summary_data.get("new_dataset_name", "N/A")
    sizes = summary_data.get("dataset_sizes", {})
    overall_status = summary_data.get("overall_status", "UNKNOWN")
    risk_summary = summary_data.get("risk_summary", {})
    ratings = risk_summary.get("ratings", {})
    details = risk_summary.get("details", [])
    recommendations = summary_data.get("recommendations", [])
    q_drift = summary_data.get("data_quality_drift", {})
    f_drift = summary_data.get("feature_drift", {})
    p_drift = summary_data.get("prediction_drift", {})
    perf_drift = summary_data.get("performance_drift", {})

    status_color = "#e53935" if "HIGH" in overall_status else ("#fb8c00" if "MEDIUM" in overall_status else "#43a047")
    status_bg = "#ffebee" if "HIGH" in overall_status else ("#fff8e1" if "MEDIUM" in overall_status else "#e8f5e9")

    # Data Quality table rows
    q_rows = ""
    q_labels = []
    q_ref_means = []
    q_curr_means = []
    for metric, mdata in q_drift.items():
        is_drift = mdata.get("drift_detected", False)
        badge = f'<span class="badge badge-danger">DRIFT</span>' if is_drift else '<span class="badge badge-success">STABLE</span>'
        p_val = mdata.get("p_value", 1.0)
        p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001"
        ref_m = round(mdata.get("ref_mean", 0), 2)
        cur_m = round(mdata.get("curr_mean", 0), 2)
        q_labels.append(metric.capitalize())
        q_ref_means.append(ref_m)
        q_curr_means.append(cur_m)

        q_rows += f"""
        <tr>
            <td><strong>{metric.capitalize()}</strong></td>
            <td>{ref_m}</td>
            <td>{cur_m}</td>
            <td>{p_str}</td>
            <td>{round(mdata.get('psi', 0), 3)}</td>
            <td>{badge}</td>
        </tr>
        """

    # Risk Cards
    cards_html = ""
    for r_title, r_val in ratings.items():
        c_badge = "badge-danger" if r_val == "HIGH" else ("badge-warning" if r_val == "MEDIUM" else ("badge-success" if r_val == "LOW" else "badge-info"))
        cards_html += f"""
        <div class="risk-card">
            <div class="risk-card-title">{r_title}</div>
            <div class="risk-badge {c_badge}">{r_val}</div>
        </div>
        """

    # Performance block
    if perf_drift.get("available", False):
        deltas = perf_drift.get("deltas", {})
        perf_rows = ""
        for metric_k, metric_v in deltas.items():
            delta_val = metric_v["delta"] * 100
            d_class = "text-danger" if delta_val < 0 else "text-success"
            perf_rows += f"""
            <tr>
                <td><strong>{metric_k}</strong></td>
                <td>{metric_v['golden']*100:.1f}%</td>
                <td>{metric_v['new']*100:.1f}%</td>
                <td class="{d_class}"><strong>{delta_val:+.1f}%</strong></td>
            </tr>
            """
        perf_content = f"""
        <table class="report-table">
            <thead>
                <tr>
                    <th>Evaluation Metric</th>
                    <th>Golden Baseline</th>
                    <th>New Client Dataset</th>
                    <th>Observed Delta</th>
                </tr>
            </thead>
            <tbody>
                {perf_rows}
            </tbody>
        </table>
        """
    else:
        perf_content = f"""
        <div class="alert alert-warning">
            <strong>⚠️ Ground-Truth Not Available:</strong> {perf_drift.get('message', 'Unlabelled dataset.')}
            <br><small>Actual accuracy or mAP degradation can only be quantitatively measured when labels are supplied.</small>
        </div>
        """

    # Findings and Recommendations
    findings_html = "".join([f"<li>{d}</li>" for d in details])
    recs_html = "".join([f"<li><strong>{r}</strong></li>" for r in recommendations])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Drift Assessment — {aid}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --primary: #1e3a8a;
            --surface: #ffffff;
            --background: #f8fafc;
            --border: #e2e8f0;
            --text-main: #0f172a;
            --text-muted: #64748b;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
        body {{ background-color: var(--background); color: var(--text-main); padding: 32px 24px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid var(--border); padding-bottom: 20px; margin-bottom: 28px; }}
        .brand-title {{ font-size: 24px; font-weight: 700; color: var(--primary); }}
        .brand-sub {{ font-size: 14px; color: var(--text-muted); margin-top: 4px; }}
        .meta-pill {{ font-size: 13px; background: #e2e8f0; padding: 6px 14px; border-radius: 20px; font-weight: 500; }}
        
        .hero-banner {{ display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 20px; background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 24px; margin-bottom: 28px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
        .verdict-box {{ display: flex; flex-direction: column; justify-content: center; }}
        .verdict-badge {{ display: inline-block; padding: 8px 18px; border-radius: 8px; font-size: 20px; font-weight: 800; background: {status_bg}; color: {status_color}; border: 1px solid {status_color}; width: fit-content; margin-top: 8px; }}
        .kpi-box {{ text-align: center; border-left: 1px solid var(--border); display: flex; flex-direction: column; justify-content: center; }}
        .kpi-value {{ font-size: 32px; font-weight: 700; color: var(--text-main); }}
        .kpi-label {{ font-size: 13px; color: var(--text-muted); text-transform: uppercase; margin-top: 4px; letter-spacing: 0.5px; }}

        .risk-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 28px; }}
        .risk-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 18px; text-align: center; }}
        .risk-card-title {{ font-size: 13px; color: var(--text-muted); font-weight: 600; margin-bottom: 10px; }}
        .risk-badge {{ display: inline-block; padding: 6px 14px; border-radius: 16px; font-size: 14px; font-weight: 700; }}
        
        .badge-danger {{ background: #ffebee; color: #c62828; }}
        .badge-warning {{ background: #fff8e1; color: #f57f17; }}
        .badge-success {{ background: #e8f5e9; color: #2e7d32; }}
        .badge-info {{ background: #e0f2fe; color: #0369a1; }}
        .text-danger {{ color: #c62828; }}
        .text-success {{ color: #2e7d32; }}

        .section-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 24px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
        .section-title {{ font-size: 18px; font-weight: 700; color: var(--primary); margin-bottom: 16px; border-bottom: 1px solid var(--border); padding-bottom: 10px; }}

        .insights-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
        ul.insights-list {{ padding-left: 20px; line-height: 1.8; color: #334155; }}

        .report-table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }}
        .report-table th {{ background: #f1f5f9; padding: 12px 14px; text-align: left; font-weight: 600; color: #475569; border-bottom: 2px solid var(--border); }}
        .report-table td {{ padding: 12px 14px; border-bottom: 1px solid var(--border); }}
        .report-table tr:hover {{ background-color: #f8fafc; }}

        .alert {{ padding: 16px 20px; border-radius: 8px; margin-top: 12px; font-size: 14px; line-height: 1.5; }}
        .alert-warning {{ background: #fffbeb; border: 1px solid #fef3c7; color: #92400e; }}

        .chart-container {{ position: relative; height: 320px; width: 100%; margin-top: 16px; }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div>
                <div class="brand-title">🛡️ AI Vision Model Drift Observatory</div>
                <div class="brand-sub">Executive Evaluation Report • Model: <strong>{model_name}</strong></div>
            </div>
            <div class="meta-pill">Run ID: {aid} • {ts}</div>
        </div>

        <!-- Hero Overview -->
        <div class="hero-banner">
            <div class="verdict-box">
                <span style="font-size: 13px; color: var(--text-muted); font-weight: 600; text-transform: uppercase;">Overall Drift Assessment</span>
                <div class="verdict-badge">{overall_status}</div>
                <div style="font-size: 13px; color: var(--text-muted); margin-top: 8px;">
                    Golden: <strong>{golden_ds}</strong> vs New: <strong>{new_ds}</strong>
                </div>
            </div>
            <div class="kpi-box">
                <div class="kpi-value">{sizes.get('golden_samples', 0)}</div>
                <div class="kpi-label">Golden Samples</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-value">{sizes.get('new_samples', 0)}</div>
                <div class="kpi-label">New Samples</div>
            </div>
        </div>

        <!-- 4-Way Risk Dimensions -->
        <div class="risk-grid">
            {cards_html}
        </div>

        <!-- Key Findings & Recommendations -->
        <div class="section-card">
            <div class="section-title">📋 Executive Findings & Recommended Actions</div>
            <div class="insights-grid">
                <div>
                    <h4 style="margin-bottom: 10px; color: #334155;">Key Observations:</h4>
                    <ul class="insights-list">
                        {findings_html}
                    </ul>
                </div>
                <div>
                    <h4 style="margin-bottom: 10px; color: #334155;">Recommended Protocols:</h4>
                    <ul class="insights-list">
                        {recs_html}
                    </ul>
                </div>
            </div>
        </div>

        <!-- Data Quality Section & Chart -->
        <div class="section-card">
            <div class="section-title">🔬 Image Quality & Input Drift Breakdown</div>
            <table class="report-table">
                <thead>
                    <tr>
                        <th>Quality Dimension</th>
                        <th>Golden Baseline Mean</th>
                        <th>New Dataset Mean</th>
                        <th>KS Test p-value</th>
                        <th>PSI (Stability Index)</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {q_rows}
                </tbody>
            </table>

            <div class="chart-container">
                <canvas id="qualityChart"></canvas>
            </div>
        </div>

        <!-- Performance Drift Section -->
        <div class="section-card">
            <div class="section-title">📉 Predictive Performance Drift</div>
            {perf_content}
        </div>
    </div>

    <script>
        const ctx = document.getElementById('qualityChart').getContext('2d');
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(q_labels)},
                datasets: [
                    {{
                        label: 'Golden Reference Dataset',
                        data: {json.dumps(q_ref_means)},
                        backgroundColor: 'rgba(30, 58, 138, 0.75)',
                        borderColor: '#1e3a8a',
                        borderWidth: 1
                    }},
                    {{
                        label: 'New Client Dataset',
                        data: {json.dumps(q_curr_means)},
                        backgroundColor: 'rgba(239, 68, 68, 0.75)',
                        borderColor: '#ef4444',
                        borderWidth: 1
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{ beginAtZero: true, grid: {{ color: '#f1f5f9' }} }},
                    x: {{ grid: {{ display: false }} }}
                }},
                plugins: {{
                    legend: {{ position: 'top' }},
                    title: {{ display: true, text: 'Side-by-Side Image Quality Mean Values' }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ Generated executive visualization report: {output_path}")


def generate_training_drift_html_report(json_path: str, output_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)

    rows = ""
    exp_labels = []
    accs = []
    drops = []

    for item in data:
        noise = item.get("noise_level_pct", 0)
        acc = item.get("accuracy", 0) * 100
        prec = item.get("precision", 0) * 100
        rec = item.get("recall", 0) * 100
        f1 = item.get("f1_score", 0) * 100
        drop = item.get("performance_drop_pp", 0)

        exp_labels.append(f"{noise}% Label Noise")
        accs.append(round(acc, 1))
        drops.append(round(drop, 1))

        rows += f"""
        <tr>
            <td><strong>{noise}% Noise</strong></td>
            <td>{acc:.1f}%</td>
            <td>{prec:.1f}%</td>
            <td>{rec:.1f}%</td>
            <td>{f1:.1f}%</td>
            <td class="{'text-danger' if drop > 0 else 'text-success'}"><strong>-{drop:.1f} pp</strong></td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Training Quality Drift Experiment Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f8fafc; color: #0f172a; padding: 32px 24px; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        .header {{ border-bottom: 2px solid #e2e8f0; padding-bottom: 16px; margin-bottom: 24px; }}
        .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 24px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        th {{ background: #f1f5f9; padding: 12px; text-align: left; }}
        td {{ padding: 12px; border-bottom: 1px solid #e2e8f0; }}
        .text-danger {{ color: #c62828; }}
        .text-success {{ color: #2e7d32; }}
        .chart-box {{ height: 320px; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 style="color: #1e3a8a;">🔬 Training Quality Drift Analysis Report</h1>
            <p style="color: #64748b; margin-top: 4px;">Empirical demonstration of model degradation induced by corrupted training supervision</p>
        </div>

        <div class="card">
            <h3>Experimental Results Summary</h3>
            <table>
                <thead>
                    <tr>
                        <th>Supervision Quality</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1 Score</th>
                        <th>Performance Drop</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>

            <div class="chart-box">
                <canvas id="trainChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('trainChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(exp_labels)},
                datasets: [{{
                    label: 'Clean Golden Test Set Accuracy (%)',
                    data: {json.dumps(accs)},
                    borderColor: '#1e3a8a',
                    backgroundColor: 'rgba(30, 58, 138, 0.1)',
                    fill: true,
                    tension: 0.2,
                    borderWidth: 3,
                    pointRadius: 6
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{ min: 50, max: 100, title: {{ display: true, text: 'Accuracy (%)' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Generated training drift visualization report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate executive visualization report from assessment results")
    parser.add_argument("--assessment_id", type=str, default=None, help="Assessment ID or 'latest'")
    parser.add_argument("--training_drift", action="store_true", help="Generate report for training drift experiments")
    parser.add_argument("--output", type=str, default=None, help="Target HTML file path")
    args = parser.parse_args()

    if args.training_drift:
        json_p = "results/training_drift/training_quality_drift_summary.json"
        out_p = args.output or "results/training_drift/training_drift_report.html"
        generate_training_drift_html_report(json_p, out_p)
        return

    # Inference drift assessment report
    assess_base = "results/assessments"
    if not os.path.exists(assess_base):
        print(f"Error: {assess_base} does not exist.")
        sys.exit(1)

    runs = sorted([d for d in os.listdir(assess_base) if os.path.isdir(os.path.join(assess_base, d))])
    if not runs:
        print("No assessments found in results/assessments.")
        sys.exit(1)

    target_run = runs[-1] if (args.assessment_id is None or args.assessment_id == "latest") else args.assessment_id
    run_dir = os.path.join(assess_base, target_run)
    summary_p = os.path.join(run_dir, "summary.json")

    if not os.path.exists(summary_p):
        print(f"Error: {summary_p} not found.")
        sys.exit(1)

    with open(summary_p, "r") as f:
        sdata = json.load(f)

    out_p = args.output or os.path.join(run_dir, "executive_report.html")
    generate_inference_html_report(sdata, out_p)

if __name__ == "__main__":
    main()

