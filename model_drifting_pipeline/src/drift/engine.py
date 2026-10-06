import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from PIL import Image

from ..adapters.models.base import BaseModelAdapter
from ..adapters.datasets.base import BaseDatasetAdapter
from ..quality.quality_analyzer import QualityAnalyzer
from .drift_stats import DriftStats

try:
    from evidently.legacy.report import Report
    from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.legacy.ui.workspace import Workspace
    EVIDENTLY_AVAILABLE = True
except Exception:
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset, DataSummaryPreset as DataQualityPreset
        from evidently.ui.workspace import Workspace
        EVIDENTLY_AVAILABLE = True
    except Exception:
        EVIDENTLY_AVAILABLE = False


class DriftEngine:
    """
    Core Drift Assessment Engine.
    Executes ad-hoc evaluation comparing Golden vs New datasets across:
    1. Data Quality Drift
    2. Data / Input Drift
    3. Feature / Embedding Drift
    4. Prediction Drift
    5. Performance Drift (if ground-truth available)
    """

    def __init__(self, results_base_dir: str = "results/assessments"):
        self.results_base_dir = results_base_dir
        os.makedirs(self.results_base_dir, exist_ok=True)

    def assess(
        self,
        model_adapter: BaseModelAdapter,
        model_name: str,
        golden_items: List[Dict[str, Any]],
        golden_name: str,
        new_items: List[Dict[str, Any]],
        new_name: str,
        assessment_id: Optional[str] = None
    ) -> Dict[str, Any]:
        
        assessment_id = assessment_id or f"assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:6]}"
        assessment_dir = os.path.join(self.results_base_dir, assessment_id)
        os.makedirs(assessment_dir, exist_ok=True)

        golden_images = [it["image"] for it in golden_items]
        new_images = [it["image"] for it in new_items]

        # 1. DATA QUALITY ASSESSMENT
        quality_golden_df = QualityAnalyzer.analyze_dataset(golden_images)
        quality_new_df = QualityAnalyzer.analyze_dataset(new_images)

        quality_drift_metrics = {}
        for col in ["brightness", "contrast", "sharpness", "noise"]:
            ref_vals = quality_golden_df[col].to_numpy()
            curr_vals = quality_new_df[col].to_numpy()
            quality_drift_metrics[col] = DriftStats.test_feature_drift(ref_vals, curr_vals)

        # 2. FEATURE / EMBEDDING EXTRACTION & DRIFT
        golden_features = model_adapter.extract_features(golden_images)
        new_features = model_adapter.extract_features(new_images)
        feature_drift_metrics = DriftStats.test_multivariate_embedding_drift(golden_features, new_features)

        # 3. PREDICTION INFERENCE & DRIFT
        golden_preds = model_adapter.predict(golden_images)
        new_preds = model_adapter.predict(new_images)

        # Determine prediction drift depending on task
        prediction_drift_metrics = self._calculate_prediction_drift(golden_preds, new_preds)

        # 4. EVIDENTLY TABULAR REPORT & WORKSPACE PERSISTENCE
        evidently_html_path = None
        evidently_summary = {}
        if EVIDENTLY_AVAILABLE and len(quality_golden_df) > 0 and len(quality_new_df) > 0:
            try:
                report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
                report.run(reference_data=quality_golden_df, current_data=quality_new_df)
                evidently_html_path = os.path.join(assessment_dir, "evidently_drift_report.html")
                report.save_html(evidently_html_path)
                evidently_summary = report.as_dict()

                # Also persist to Evidently Workspace for the Evidently Dashboard UI
                ws_path = os.path.join(self.results_base_dir, "..", "evidently_workspace")
                os.makedirs(ws_path, exist_ok=True)
                ws = Workspace.create(ws_path)
                project_name = f"CV Drift - {model_name}"
                matching_projects = ws.search_project(project_name)
                if matching_projects:
                    proj = matching_projects[0]
                else:
                    proj = ws.create_project(project_name)
                    proj.description = f"Model drift tracking and monitoring for {model_name}"
                    proj.save()
                ws.add_report(proj.id, report)
            except Exception as e:
                print(f"Evidently report generation note: {e}")

        # 5. PERFORMANCE DRIFT (If Ground Truth Available)
        golden_labels = [it.get("annotations") for it in golden_items if it.get("annotations") is not None]
        new_labels = [it.get("annotations") for it in new_items if it.get("annotations") is not None]
        
        has_labels = (len(new_labels) == len(new_items) and len(new_labels) > 0)
        performance_metrics = {}

        if has_labels:
            golden_eval = model_adapter.evaluate(golden_preds, golden_labels) if len(golden_labels) == len(golden_items) else {}
            new_eval = model_adapter.evaluate(new_preds, new_labels)
            
            perf_deltas = {}
            for k in new_eval.keys():
                ref_val = golden_eval.get(k, 0.0)
                curr_val = new_eval[k]
                perf_deltas[k] = {
                    "golden": ref_val,
                    "new": curr_val,
                    "delta": curr_val - ref_val
                }

            performance_metrics = {
                "available": True,
                "golden_eval": golden_eval,
                "new_eval": new_eval,
                "deltas": perf_deltas
            }
        else:
            performance_metrics = {
                "available": False,
                "message": "Ground-truth labels not provided in the new dataset. Actual performance degradation cannot be confirmed."
            }

        # 6. OVERALL DECISION LOGIC & RISK SCORING
        overall_status, risk_summary, recommendations = self._compute_overall_risk(
            quality_drift_metrics,
            feature_drift_metrics,
            prediction_drift_metrics,
            performance_metrics
        )

        # Assemble Full Structured Assessment Result
        result = {
            "assessment_id": assessment_id,
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "golden_dataset_name": golden_name,
            "new_dataset_name": new_name,
            "dataset_sizes": {
                "golden_samples": len(golden_items),
                "new_samples": len(new_items)
            },
            "overall_status": overall_status,
            "risk_summary": risk_summary,
            "recommendations": recommendations,
            "data_quality_drift": quality_drift_metrics,
            "feature_drift": feature_drift_metrics,
            "prediction_drift": prediction_drift_metrics,
            "performance_drift": performance_metrics,
            "evidently_report_path": evidently_html_path,
            "summary_tables": {
                "golden_quality_mean": quality_golden_df.mean().to_dict(),
                "new_quality_mean": quality_new_df.mean().to_dict()
            }
        }

        # Persist results
        with open(os.path.join(assessment_dir, "summary.json"), 'w') as f:
            json.dump(result, f, indent=2, default=str)

        return result

    def _calculate_prediction_drift(self, golden_preds: List[Dict[str, Any]], new_preds: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assesses confidence shift, entropy shift, or detection count shift.
        """
        if not golden_preds or not new_preds:
            return {"drift_detected": False}

        # Classification check (confidence & entropy)
        if "confidence" in golden_preds[0]:
            ref_conf = np.array([p["confidence"] for p in golden_preds])
            curr_conf = np.array([p["confidence"] for p in new_preds])
            ref_entropy = np.array([p.get("entropy", 0.0) for p in golden_preds])
            curr_entropy = np.array([p.get("entropy", 0.0) for p in new_preds])

            conf_drift = DriftStats.test_feature_drift(ref_conf, curr_conf)
            ent_drift = DriftStats.test_feature_drift(ref_entropy, curr_entropy)

            # Class distribution frequencies
            ref_classes = pd.Series([p["label"] for p in golden_preds]).value_counts(normalize=True).to_dict()
            curr_classes = pd.Series([p["label"] for p in new_preds]).value_counts(normalize=True).to_dict()

            return {
                "task": "classification",
                "drift_detected": conf_drift["drift_detected"] or ent_drift["drift_detected"],
                "confidence_drift": conf_drift,
                "entropy_drift": ent_drift,
                "golden_class_dist": ref_classes,
                "new_class_dist": curr_classes
            }

        # Detection check (detection count & avg confidence)
        elif "num_detections" in golden_preds[0]:
            ref_counts = np.array([p["num_detections"] for p in golden_preds])
            curr_counts = np.array([p["num_detections"] for p in new_preds])
            ref_conf = np.array([p["avg_confidence"] for p in golden_preds if p["avg_confidence"] > 0])
            curr_conf = np.array([p["avg_confidence"] for p in new_preds if p["avg_confidence"] > 0])

            count_drift = DriftStats.test_feature_drift(ref_counts, curr_counts)
            conf_drift = DriftStats.test_feature_drift(ref_conf, curr_conf) if len(ref_conf) > 0 and len(curr_conf) > 0 else {"drift_detected": False}

            return {
                "task": "detection",
                "drift_detected": count_drift["drift_detected"] or conf_drift.get("drift_detected", False),
                "detection_count_drift": count_drift,
                "confidence_drift": conf_drift
            }

        return {"drift_detected": False}

    def _compute_overall_risk(
        self,
        quality_drift: Dict[str, Any],
        feature_drift: Dict[str, Any],
        prediction_drift: Dict[str, Any],
        performance_drift: Dict[str, Any]
    ) -> tuple:
        
        quality_drifts_count = sum(1 for m in quality_drift.values() if m.get("drift_detected", False))
        feat_drifted = feature_drift.get("drift_detected", False)
        pred_drifted = prediction_drift.get("drift_detected", False)

        risk_scores = {
            "Data Quality Drift": "HIGH" if quality_drifts_count >= 2 else ("MEDIUM" if quality_drifts_count == 1 else "LOW"),
            "Feature Drift": "HIGH" if feat_drifted else "LOW",
            "Prediction Drift": "HIGH" if pred_drifted else "LOW"
        }

        reasons = []
        if quality_drifts_count >= 2:
            reasons.append(f"Significant data quality drift detected across {quality_drifts_count} image quality metrics.")
        elif quality_drifts_count == 1:
            reasons.append("Moderate data quality change detected.")

        if feat_drifted:
            reasons.append("Latent feature distribution significantly deviated from golden baseline.")

        if pred_drifted:
            reasons.append("Model prediction output distribution shows noticeable shift.")

        if performance_drift.get("available", False):
            deltas = performance_drift.get("deltas", {})
            # Look for accuracy or mAP drop
            deg_key = "accuracy" if "accuracy" in deltas else "mAP_50"
            if deg_key in deltas:
                delta_val = deltas[deg_key]["delta"]
                if delta_val < -0.10:
                    risk_scores["Performance Drift"] = "HIGH"
                    reasons.append(f"Severe performance drop detected: {deg_key} dropped by {abs(delta_val)*100:.1f}%.")
                elif delta_val < -0.03:
                    risk_scores["Performance Drift"] = "MEDIUM"
                    reasons.append(f"Noticeable performance degradation: {deg_key} dropped by {abs(delta_val)*100:.1f}%.")
                else:
                    risk_scores["Performance Drift"] = "LOW"
        else:
            risk_scores["Performance Drift"] = "NOT AVAILABLE"
            reasons.append("Performance degradation cannot be confirmed because the new dataset contains no ground-truth labels.")

        # Determine overall status
        high_count = sum(1 for v in risk_scores.values() if v == "HIGH")
        medium_count = sum(1 for v in risk_scores.values() if v == "MEDIUM")

        if high_count >= 2 or risk_scores.get("Performance Drift") == "HIGH":
            overall_status = "HIGH RISK"
        elif high_count == 1 or medium_count >= 1:
            overall_status = "MEDIUM RISK"
        else:
            overall_status = "LOW RISK"

        recommendations = []
        if overall_status == "HIGH RISK":
            recommendations.append("Do NOT auto-promote inferences to production without manual spot checks.")
            recommendations.append("Audit camera / source image acquisition pipelines for blur, lighting, or compression changes.")
            if not performance_drift.get("available", False):
                recommendations.append("Annotate a representative sample (10-20%) of the new dataset to confirm actual mAP / accuracy degradation.")
            else:
                recommendations.append("Retrain or fine-tune model on augmented domain data to recover model accuracy.")
        elif overall_status == "MEDIUM RISK":
            recommendations.append("Monitor incoming stream closely; review low-confidence detections.")
        else:
            recommendations.append("Data is consistent with golden distribution. Production operations safe.")

        return overall_status, {"ratings": risk_scores, "details": reasons}, recommendations
