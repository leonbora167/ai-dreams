import os
import json
import argparse
from typing import Optional

from src.adapters.models import get_model_adapter
from src.adapters.datasets import get_dataset_adapter
from src.drift.engine import DriftEngine

def run_assessment(
    model_name: str,
    golden_path: str,
    new_path: str,
    golden_name: Optional[str] = None,
    new_name: Optional[str] = None,
    dataset_type: Optional[str] = None,
    max_samples: Optional[int] = None
):
    print(f"\n==========================================")
    print(f"🚀 RUNNING AD-HOC MODEL DRIFT ASSESSMENT")
    print(f"==========================================")
    print(f"Model: {model_name}")
    print(f"Golden dataset: {golden_path}")
    print(f"New dataset: {new_path}\n")

    # Determine dataset adapter
    ds_type = dataset_type or ("voc" if "voc" in golden_path.lower() else "cifar10")
    ds_adapter = get_dataset_adapter(ds_type)

    print("📦 Loading reference/golden dataset...")
    golden_items = ds_adapter.load(golden_path, max_samples=max_samples)
    print(f"   Loaded {len(golden_items)} reference items.")

    print("📦 Loading new client dataset...")
    new_items = ds_adapter.load(new_path, max_samples=max_samples)
    print(f"   Loaded {len(new_items)} evaluation items.")

    if not golden_items or not new_items:
        raise ValueError("Failed to load dataset items.")

    print(f"🧠 Initializing model adapter '{model_name}'...")
    model_adapter = get_model_adapter(model_name)

    engine = DriftEngine()
    g_label = golden_name or os.path.basename(golden_path)
    n_label = new_name or os.path.basename(new_path)

    print("🔬 Executing drift assessment engine (Quality, Embeddings, Predictions, Performance, Evidently)...")
    result = engine.assess(
        model_adapter=model_adapter,
        model_name=model_name,
        golden_items=golden_items,
        golden_name=g_label,
        new_items=new_items,
        new_name=n_label
    )

    print("\n" + "=" * 50)
    print(f"📊 DRIFT ASSESSMENT SUMMARY — {result['assessment_id']}")
    print("=" * 50)
    print(f"Overall Risk Status: {result['overall_status']}")
    print("-" * 50)
    for k, v in result['risk_summary']['ratings'].items():
        print(f"  {k:<24}: {v}")
    print("-" * 50)
    print("Findings:")
    for d in result['risk_summary']['details']:
        print(f"  • {d}")
    print("-" * 50)
    print("Recommendations:")
    for r in result['recommendations']:
        print(f"  ➜ {r}")
    
    if result.get('evidently_report_path'):
        print(f"\n🌐 Evidently Interactive Report Saved to:\n  {result['evidently_report_path']}")
    
    # Generate standalone executive visualization report
    try:
        from src.utils.generate_visual_report import generate_inference_html_report
        exec_path = os.path.join("results/assessments", result["assessment_id"], "executive_report.html")
        generate_inference_html_report(result, exec_path)
        print(f"📊 Executive Visual Report Generated:\n  {exec_path}")
    except Exception as e:
        pass
    print("=" * 50 + "\n")

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="inception_v3")
    parser.add_argument("--golden", type=str, default="data/golden/cifar10_golden")
    parser.add_argument("--new", type=str, default="data/new/cifar10_camera_degraded")
    parser.add_argument("--max_samples", type=int, default=50)
    args = parser.parse_args()

    run_assessment(
        model_name=args.model,
        golden_path=args.golden,
        new_path=args.new,
        max_samples=args.max_samples
    )

