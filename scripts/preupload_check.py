#!/usr/bin/env python3
"""
Pre-upload data quality check script.

This script performs comprehensive data quality checks including:
- Dataset audit
- PII scanning
- RTL/rotation analysis
- License checking
- Data cleaning (optional)
- Report generation
"""

import argparse
import sys
from pathlib import Path
import json
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docuagent.data.audit import audit_dataset, save_audit_results
from docuagent.data.pii_scan import scan_pii, generate_pii_report
from docuagent.data.rtl_rot_check import check_rtl_rotation, generate_rtl_report
from docuagent.data.license_check import check_licenses, generate_license_report
from docuagent.data.clean import apply_auto_fixes, validate_cleaned_data
from docuagent.data.report import generate_preupload_report


def main():
    """Main entry point for pre-upload check."""
    parser = argparse.ArgumentParser(
        description="Comprehensive pre-upload data quality check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic check
  python scripts/preupload_check.py --images data/images --labels data/labels --schema yolo

  # Check with auto-fixing
  python scripts/preupload_check.py --images data/images --labels data/labels --schema yolo --fix --out data_clean

  # Check COCO dataset
  python scripts/preupload_check.py --images data/images --labels data/annotations.json --schema coco
        """
    )
    
    # Required arguments
    parser.add_argument("--images", required=True, help="Path to images directory")
    parser.add_argument("--labels", required=True, help="Path to labels (YOLO dir or COCO json)")
    parser.add_argument("--schema", choices=["yolo", "coco"], required=True, help="Dataset schema")
    
    # Optional arguments
    parser.add_argument("--classes", nargs="+", 
                       default=["Text", "Title", "List", "Table", "Figure"],
                       help="Class names (default: Text Title List Table Figure)")
    parser.add_argument("--fix", action="store_true", help="Apply automatic fixes")
    parser.add_argument("--out", default="data_clean", help="Output directory for cleaned data")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--sample-rate", type=float, default=0.03, 
                       help="Fraction of images to sample for RTL/rotation analysis (default: 0.03)")
    parser.add_argument("--no-pii", action="store_true", help="Skip PII scanning")
    parser.add_argument("--no-rtl", action="store_true", help="Skip RTL/rotation analysis")
    parser.add_argument("--no-license", action="store_true", help="Skip license checking")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Validate inputs
    images_path = Path(args.images)
    labels_path = Path(args.labels)
    
    if not images_path.exists():
        print(f"❌ Images directory not found: {args.images}")
        return 1
    
    if not labels_path.exists():
        print(f"❌ Labels path not found: {args.labels}")
        return 1
    
    print("🔍 Starting pre-upload data quality check...")
    print(f"📁 Images: {args.images}")
    print(f"📁 Labels: {args.labels}")
    print(f"📋 Schema: {args.schema}")
    print(f"🏷️  Classes: {', '.join(args.classes)}")
    print(f"🔧 Auto-fix: {'Yes' if args.fix else 'No'}")
    print(f"📤 Output: {args.out}")
    print()
    
    start_time = time.time()
    results = {}
    
    try:
        # 1. Dataset Audit
        print("1️⃣ Running dataset audit...")
        audit_results = audit_dataset(
            images_dir=args.images,
            labels=args.labels,
            schema=args.schema,
            class_names=args.classes,
            fix=args.fix
        )
        results["audit"] = audit_results
        
        # Save audit results
        output_dir = Path(args.out)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_audit_results(audit_results, output_dir)
        
        print(f"✅ Audit complete. Found {len(audit_results.get('duplicates', []))} duplicates, "
              f"{len(audit_results.get('image_sanity', {}).get('unreadable', []))} broken images")
        
        # 2. PII Scanning
        if not args.no_pii:
            print("\n2️⃣ Scanning for PII...")
            pii_results = scan_pii(
                filenames=[str(f) for f in images_path.glob("*.jpg")],
                output_path=str(output_dir / "pii_findings.json")
            )
            results["pii"] = pii_results
            
            print(f"✅ PII scan complete. Found {pii_results['summary']['total_findings']} PII instances")
        else:
            print("\n2️⃣ Skipping PII scan")
            results["pii"] = {"summary": {"total_findings": 0, "files_with_pii": 0}}
        
        # 3. RTL/Rotation Analysis
        if not args.no_rtl:
            print("\n3️⃣ Analyzing RTL and rotation...")
            rtl_results = check_rtl_rotation(
                images_dir=args.images,
                sample_rate=args.sample_rate,
                output_path=str(output_dir / "rtl_rot_report.json"),
                config_path=args.config
            )
            results["rtl"] = rtl_results
            
            print(f"✅ RTL/rotation analysis complete. "
                  f"RTL: {rtl_results['summary']['rtl_percentage']:.1f}%, "
                  f"Skewed: {rtl_results['summary']['skewed_percentage']:.1f}%")
        else:
            print("\n3️⃣ Skipping RTL/rotation analysis")
            results["rtl"] = {"summary": {"rtl_percentage": 0, "skewed_percentage": 0}}
        
        # 4. License Check
        if not args.no_license:
            print("\n4️⃣ Checking licenses...")
            license_results = check_licenses(
                data_root=args.images,
                output_path=str(output_dir / "license_report.json")
            )
            results["license"] = license_results
            
            print(f"✅ License check complete. "
                  f"Licensed folders: {license_results['summary']['licensed_folders']}")
        else:
            print("\n4️⃣ Skipping license check")
            results["license"] = {"compliance_status": {"has_license": False}}
        
        # 5. Data Cleaning (if requested)
        if args.fix:
            print("\n5️⃣ Applying automatic fixes...")
            clean_results = apply_auto_fixes(
                audit_result=audit_results,
                images_dir=args.images,
                labels=args.labels,
                schema=args.schema,
                out_dir=args.out
            )
            results["clean"] = clean_results
            
            print(f"✅ Cleaning complete. "
                  f"Cleaned: {clean_results['cleaned_images']} images, "
                  f"{clean_results['cleaned_labels']} labels")
            
            # Validate cleaned data
            print("\n6️⃣ Validating cleaned data...")
            validation_results = validate_cleaned_data(
                cleaned_dir=args.out,
                schema=args.schema,
                class_names=args.classes
            )
            results["validation"] = validation_results
            
            if validation_results["validation_passed"]:
                print("✅ Validation passed")
            else:
                print(f"⚠️ Validation failed: {len(validation_results['issues_found'])} issues found")
        else:
            print("\n5️⃣ Skipping data cleaning (use --fix to enable)")
        
        # 6. Generate Report
        print("\n7️⃣ Generating comprehensive report...")
        report_path = generate_preupload_report(
            audit_results=audit_results,
            pii_results=results["pii"],
            rtl_results=results["rtl"],
            license_results=results["license"],
            output_dir=args.out,
            sample_images=[str(f) for f in images_path.glob("*.jpg")][:20]
        )
        results["report_path"] = report_path
        
        print(f"✅ Report generated: {report_path}")
        
        # 7. Summary
        elapsed_time = time.time() - start_time
        print(f"\n🎉 Pre-upload check complete in {elapsed_time:.1f}s")
        
        # Determine overall status
        critical_issues = (
            len(audit_results.get("image_sanity", {}).get("unreadable", [])) +
            len(audit_results.get("label_sanity", {}).get("invalid_coords", [])) +
            results["pii"]["summary"]["total_findings"]
        )
        
        needs_attention = (
            results["rtl"]["summary"]["rtl_percentage"] > 20 or
            results["rtl"]["summary"]["skewed_percentage"] > 10 or
            not results["license"]["compliance_status"]["has_license"]
        )
        
        if critical_issues == 0 and not needs_attention:
            print("✅ Status: READY FOR UPLOAD")
            return 0
        else:
            print("⚠️ Status: NEEDS ATTENTION")
            if critical_issues > 0:
                print(f"   - {critical_issues} critical issues found")
            if needs_attention:
                print("   - Dataset needs preprocessing or compliance review")
            return 1
    
    except Exception as e:
        print(f"❌ Error during pre-upload check: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
