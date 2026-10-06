import os
import sys
from src.drift.runner import run_assessment

def run_all_validation_tests():
    print("\n" + "="*60)
    print("🧪 RUNNING END-TO-END ACCEPTANCE VALIDATION SUITE")
    print("="*60)

    # Test 1: Golden vs Identical Copy -> LOW RISK / No Drift
    print("\n▶️ TEST 1: Golden vs Identical Copy (Expected: LOW RISK)")
    res1 = run_assessment(
        model_name="inception_v3",
        golden_path="data/golden/cifar10_golden",
        new_path="data/new/cifar10_identical",
        max_samples=25
    )
    assert res1["overall_status"] == "LOW RISK", f"Test 1 failed: got {res1['overall_status']}"
    print("✅ TEST 1 PASSED!")

    # Test 2: Golden vs Blurred/Camera Degraded -> HIGH RISK (Quality & Input Drift)
    print("\n▶️ TEST 2: Golden vs Camera Degraded (Expected: HIGH RISK / Data Quality Drift)")
    res2 = run_assessment(
        model_name="inception_v3",
        golden_path="data/golden/cifar10_golden",
        new_path="data/new/cifar10_camera_degraded",
        max_samples=25
    )
    assert res2["overall_status"] == "HIGH RISK", f"Test 2 failed: got {res2['overall_status']}"
    assert res2["risk_summary"]["ratings"]["Data Quality Drift"] == "HIGH", "Test 2 failed: Quality drift not detected"
    print("✅ TEST 2 PASSED!")

    # Test 3: Golden vs Distribution Shifted -> Prediction Drift
    print("\n▶️ TEST 3: Golden vs Distribution Shifted (Expected: Prediction Drift)")
    res3 = run_assessment(
        model_name="inception_v3",
        golden_path="data/golden/cifar10_golden",
        new_path="data/new/cifar10_distribution_shifted",
        max_samples=30
    )
    print(f"Test 3 Status: {res3['overall_status']}, Prediction Drift: {res3['risk_summary']['ratings']['Prediction Drift']}")
    print("✅ TEST 3 COMPLETED!")

    # Test 4: Unlabelled New Dataset -> Performance Drift should be NOT AVAILABLE
    print("\n▶️ TEST 4: Golden vs Unlabelled Dataset (Expected: Performance NOT AVAILABLE)")
    res4 = run_assessment(
        model_name="inception_v3",
        golden_path="data/golden/cifar10_golden",
        new_path="data/new/cifar10_unlabelled",
        max_samples=25
    )
    assert res4["performance_drift"]["available"] is False, "Test 4 failed: performance marked available for unlabelled data"
    assert res4["risk_summary"]["ratings"]["Performance Drift"] == "NOT AVAILABLE"
    print("✅ TEST 4 PASSED!")

    # Test 5: Detection RF-DETR with Degraded Dataset -> Performance Degradation Detected
    print("\n▶️ TEST 5: RF-DETR Object Detection with VOC Camera Degraded")
    res5 = run_assessment(
        model_name="rf_detr",
        golden_path="data/golden/voc_golden",
        new_path="data/new/voc_camera_degraded",
        max_samples=15
    )
    assert res5["overall_status"] == "HIGH RISK", f"Test 5 failed: got {res5['overall_status']}"
    print("✅ TEST 5 PASSED!")

    print("\n" + "="*60)
    print("🎉 ALL ACCEPTANCE CRITERIA VALIDATION TESTS PASSED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    run_all_validation_tests()

