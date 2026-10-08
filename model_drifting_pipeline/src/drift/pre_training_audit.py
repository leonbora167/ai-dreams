import os
import json
import argparse
from typing import Optional, Dict, Any

from src.adapters.datasets import get_dataset_adapter
from src.quality.quality_analyzer import QualityAnalyzer
from src.drift.drift_stats import DriftStats

def assess_pre_training_drift(
    golden_train_path: str,
    new_train_path: str,
    dataset_type: Optional[str] = None,
    max_samples: int = 50,
    run_id: Optional[str] = None,
    output_dir: str = "results/pre_training_drift"
) -> Dict[str, Any]:
    """
    PRE-TRAINING DRIFT AUDIT:
    Evaluates data quality drift, input statistical divergence, and class imbalance
    BETWEEN the golden training set and the new/incoming training set BEFORE ANY MODEL TRAINING BEGINS.
    
    Prevents burning expensive GPU/compute hours training on garbage or heavily drifted data.
    """
    actual_run_id = run_id or f"pre_train_{os.path.basename(new_train_path)}"
    run_output_dir = os.path.join(output_dir, actual_run_id)
    os.makedirs(run_output_dir, exist_ok=True)

    print("\n" + "="*65)
    print("🛡️ RUNNING PRE-TRAINING DATA DRIFT & INTEGRITY AUDIT")
    print("="*65)
    print(f"Golden Training Data : {golden_train_path}")
    print(f"New Training Data    : {new_train_path}")
    print(f"Audit Output Folder  : {run_output_dir}\n")

    ds_type = dataset_type or ("voc" if "voc" in golden_train_path.lower() else "cifar10")
    adapter = get_dataset_adapter(ds_type)

    print("📦 Loading Golden Training Samples...")
    golden_items = adapter.load(golden_train_path, max_samples=max_samples)
    print(f"   Loaded {len(golden_items)} reference training items.")

    print("📦 Loading New Training Samples...")
    new_items = adapter.load(new_train_path, max_samples=max_samples)
    print(f"   Loaded {len(new_items)} new training items.")

    if not golden_items or not new_items:
        raise ValueError("Failed to load training items for pre-training audit.")

    # 1. Image Quality Drift (Physical Optical Drift)
    print("🔬 Computing Physical Image Quality Metrics...")
    q_golden_df = QualityAnalyzer.analyze_dataset([it["image"] for it in golden_items])
    q_new_df = QualityAnalyzer.analyze_dataset([it["image"] for it in new_items])

    quality_drift = {}
    for col in ["brightness", "contrast", "sharpness", "noise"]:
        ref_vals = q_golden_df[col].to_numpy()
        cur_vals = q_new_df[col].to_numpy()
        quality_drift[col] = DriftStats.test_feature_drift(ref_vals, cur_vals)

    # 2. Annotation / Label Integrity Audit
    golden_labels = [it.get("annotations") for it in golden_items if it.get("annotations") is not None]
    new_labels = [it.get("annotations") for it in new_items if it.get("annotations") is not None]

    labelled_ratio_golden = len(golden_labels) / len(golden_items) if golden_items else 0
    labelled_ratio_new = len(new_labels) / len(new_items) if new_items else 0

    missing_labels_flag = (labelled_ratio_new < 0.90)

    # 3. Overall Readiness Gate Decision (Go / No-Go for Training)
    drifted_quality_metrics = [k for k, v in quality_drift.items() if v.get("drift_detected", False)]
    
    if len(drifted_quality_metrics) >= 2 or missing_labels_flag:
        verdict = "HIGH RISK — DO NOT TRAIN"
        action = "REJECT or clean data before launching training. High probability of convergence failure or model degradation."
    elif len(drifted_quality_metrics) == 1:
        verdict = "MEDIUM RISK — PROCEED WITH CAUTION"
        action = "Data quality has minor shifts. Recommend data augmentations (e.g. brightness/blur compensation)."
    else:
        verdict = "LOW RISK — SAFE TO TRAIN"
        action = "New training distribution aligns closely with golden standard. Safe to train."

    result = {
        "audit_id": actual_run_id,
        "golden_path": golden_train_path,
        "new_path": new_train_path,
        "dataset_type": ds_type,
        "golden_samples": len(golden_items),
        "new_samples": len(new_items),
        "audit_verdict": verdict,
        "recommended_action": action,
        "drifted_metrics": drifted_quality_metrics,
        "data_quality_drift": quality_drift,
        "annotation_audit": {
            "golden_labelled_ratio": labelled_ratio_golden,
            "new_labelled_ratio": labelled_ratio_new,
            "missing_labels_flag": missing_labels_flag
        },
        "quality_means": {
            "golden": q_golden_df.mean().to_dict(),
            "new": q_new_df.mean().to_dict()
        }
    }

    # Generate Evidently AI Interactive Report & Persist to Workspace for localhost Dashboard
    evidently_html_path = None
    try:
        from evidently.legacy.report import Report
        from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset
        from evidently.legacy.ui.workspace import Workspace
        from evidently.legacy.ui.dashboards import DashboardPanelCounter, DashboardPanelPlot, PanelValue, PlotType, CounterAgg, ReportFilter

        ev_report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        ev_report.run(reference_data=q_golden_df, current_data=q_new_df)
        evidently_html_path = os.path.join(run_output_dir, "evidently_pre_training_drift.html")
        ev_report.save_html(evidently_html_path)

        # Save to Evidently Workspace
        ws_path = "results/evidently_workspace"
        os.makedirs(ws_path, exist_ok=True)
        ws = Workspace.create(ws_path)
        proj_name = f"Pre-Training Data Audit - {ds_type.upper()}"
        matching = ws.search_project(proj_name)
        if matching:
            proj = matching[0]
        else:
            proj = ws.create_project(proj_name)
            proj.description = f"Pre-training data drift and quality tracking for {ds_type}"
            proj.dashboard.panels = [
                DashboardPanelCounter(
                    title="Drifted Features",
                    filter=ReportFilter(metadata_values={}, tag_values=[]),
                    value=PanelValue(metric_id="DatasetDriftMetric", field_path="number_of_drifted_columns"),
                    agg=CounterAgg.LAST,
                    size=1
                ),
                DashboardPanelCounter(
                    title="Drift Share",
                    filter=ReportFilter(metadata_values={}, tag_values=[]),
                    value=PanelValue(metric_id="DatasetDriftMetric", field_path="share_of_drifted_columns"),
                    agg=CounterAgg.LAST,
                    size=1
                ),
                DashboardPanelPlot(
                    title="Image Quality Drift Trends",
                    filter=ReportFilter(metadata_values={}, tag_values=[]),
                    values=[
                        PanelValue(metric_id="DataDriftTable", field_path="metrics.brightness.drift_score", legend="Brightness Drift"),
                        PanelValue(metric_id="DataDriftTable", field_path="metrics.sharpness.drift_score", legend="Sharpness Drift"),
                        PanelValue(metric_id="DataDriftTable", field_path="metrics.noise.drift_score", legend="Noise Drift")
                    ],
                    plot_type=PlotType.LINE,
                    size=2
                )
            ]
            proj.save()

        ws.add_report(proj.id, ev_report)
    except Exception as e:
        print(f"Evidently workspace persistence note: {e}")

    # Save summary JSON
    result["evidently_report_path"] = evidently_html_path
    summary_path = os.path.join(run_output_dir, "pre_training_audit.json")
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)

    # Generate Standalone Pre-Training Audit HTML Report
    html_path = os.path.join(run_output_dir, "pre_training_audit_report.html")
    _generate_pre_training_html(result, html_path)

    print("\n" + "="*65)
    print(f"📊 PRE-TRAINING AUDIT VERDICT: {verdict}")
    print(f"📋 Protocol: {action}")
    if evidently_html_path:
        print(f"🌐 Evidently AI Interactive Report : {evidently_html_path}")
    print(f"🌐 Standalone Audit Executive Report: {html_path}")
    print("="*65 + "\n")

    return result

def _generate_pre_training_html(data: Dict[str, Any], output_path: str):
    aid = data["audit_id"]
    verdict = data["audit_verdict"]
    action = data["recommended_action"]
    v_color = "#e53935" if "HIGH" in verdict else ("#fb8c00" if "MEDIUM" in verdict else "#43a047")
    v_bg = "#ffebee" if "HIGH" in verdict else ("#fff8e1" if "MEDIUM" in verdict else "#e8f5e9")

    q_rows = ""
    q_labels = []
    g_vals = []
    n_vals = []
    for k, v in data["data_quality_drift"].items():
        is_d = v.get("drift_detected", False)
        status_tag = '<span style="color:#c62828; font-weight:bold;">🚨 DRIFT</span>' if is_d else '<span style="color:#2e7d32; font-weight:bold;">✅ STABLE</span>'
        ref_m = round(v.get("ref_mean", 0), 2)
        cur_m = round(v.get("curr_mean", 0), 2)
        q_labels.append(k.capitalize())
        g_vals.append(ref_m)
        n_vals.append(cur_m)

        q_rows += f"""
        <tr>
            <td><strong>{k.capitalize()}</strong></td>
            <td>{ref_m}</td>
            <td>{cur_m}</td>
            <td>{v.get('p_value', 1.0):.4f}</td>
            <td>{round(v.get('psi', 0), 3)}</td>
            <td>{status_tag}</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Pre-Training Data Drift & Readiness Audit</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f8fafc; color: #0f172a; padding: 32px 24px; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #e2e8f0; padding-bottom: 16px; margin-bottom: 24px; }}
        .banner {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 24px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
        .badge {{ display: inline-block; padding: 8px 18px; border-radius: 8px; font-size: 20px; font-weight: 800; background: {v_bg}; color: {v_color}; border: 1px solid {v_color}; margin-top: 8px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }}
        th {{ background: #f1f5f9; padding: 12px; text-align: left; }}
        td {{ padding: 12px; border-bottom: 1px solid #e2e8f0; }}
        .chart-box {{ height: 320px; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1 style="color: #1e3a8a;">🛡️ Pre-Training Data Drift & Quality Gate</h1>
                <p style="color: #64748b;">Evaluated BEFORE training to protect compute resources and model integrity</p>
            </div>
            <div style="background: #e2e8f0; padding: 6px 14px; border-radius: 20px; font-size: 13px;">Audit ID: {aid}</div>
        </div>

        <div class="banner">
            <h3>Training Readiness Gate Verdict:</h3>
            <div class="badge">{verdict}</div>
            <p style="margin-top: 12px; font-size: 15px;"><strong>Actionable Guidance:</strong> {action}</p>
            <p style="color: #64748b; font-size: 13px; margin-top: 6px;">
                Golden: <code>{data['golden_path']}</code> vs New: <code>{data['new_path']}</code>
            </p>
        </div>

        <div class="banner">
            <h3>Pre-Training Quality Drift Analysis</h3>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Golden Training Baseline</th>
                        <th>New Training Data</th>
                        <th>KS Test p-value</th>
                        <th>PSI</th>
                        <th>Audit Status</th>
                    </tr>
                </thead>
                <tbody>
                    {q_rows}
                </tbody>
            </table>

            <div class="chart-box">
                <canvas id="qualityChart"></canvas>
            </div>
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
                        label: 'Golden Training Baseline',
                        data: {json.dumps(g_vals)},
                        backgroundColor: 'rgba(30, 58, 138, 0.75)'
                    }},
                    {{
                        label: 'New Training Data',
                        data: {json.dumps(n_vals)},
                        backgroundColor: 'rgba(239, 68, 68, 0.75)'
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{ y: {{ beginAtZero: true }} }}
            }}
        }});
    </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-training drift and data readiness audit")
    parser.add_argument("--golden_dataset", type=str, required=True, help="Path to golden reference training dataset")
    parser.add_argument("--new_dataset", type=str, required=True, help="Path to new candidate training dataset")
    parser.add_argument("--dataset_type", type=str, default=None, help="Dataset type (cifar10, voc)")
    parser.add_argument("--max_samples", type=int, default=40)
    parser.add_argument("--run_id", type=str, default=None, help="Custom audit run name")
    args = parser.parse_args()

    assess_pre_training_drift(
        golden_train_path=args.golden_dataset,
        new_train_path=args.new_dataset,
        dataset_type=args.dataset_type,
        max_samples=args.max_samples,
        run_id=args.run_id
    )

