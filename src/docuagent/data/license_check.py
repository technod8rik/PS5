"""
License and attribution checking module.

This module checks for:
- License files in the dataset
- Attribution requirements
- Source folder analysis
- Compliance warnings
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict


def check_licenses(
    data_root: str,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Check for license files and attribution requirements.
    
    Args:
        data_root: Root directory of the dataset
        output_path: Optional path to save results
        
    Returns:
        Dict with license analysis results
    """
    data_path = Path(data_root)
    if not data_path.exists():
        raise ValueError(f"Data root not found: {data_root}")
    
    results = {
        "license_files": [],
        "source_folders": [],
        "attribution_requirements": [],
        "compliance_status": {
            "has_license": False,
            "has_attribution": False,
            "warnings": [],
            "recommendations": []
        },
        "summary": {
            "total_folders": 0,
            "licensed_folders": 0,
            "unknown_license_folders": 0
        }
    }
    
    print(f"[LICENSE] Checking licenses in {data_root}")
    
    # Find all subdirectories
    folders = [data_path] + [f for f in data_path.rglob("*") if f.is_dir()]
    results["summary"]["total_folders"] = len(folders)
    
    # Check each folder for license files
    for folder in folders:
        folder_analysis = _analyze_folder_licenses(folder, data_path)
        if folder_analysis:
            results["source_folders"].append(folder_analysis)
            
            if folder_analysis["has_license"]:
                results["license_files"].extend(folder_analysis["license_files"])
                results["summary"]["licensed_folders"] += 1
            else:
                results["summary"]["unknown_license_folders"] += 1
    
    # Analyze license content
    results["attribution_requirements"] = _analyze_license_content(results["license_files"])
    
    # Determine compliance status
    results["compliance_status"] = _assess_compliance(results)
    
    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[LICENSE] Results saved to {output_path}")
    
    return results


def _analyze_folder_licenses(folder: Path, data_root: Path) -> Optional[Dict[str, Any]]:
    """Analyze a single folder for license files."""
    # Common license file names
    license_patterns = [
        "LICENSE*",
        "LICENCE*",
        "COPYING*",
        "COPYRIGHT*",
        "ATTRIBUTION*",
        "README*",
        "NOTICE*",
        "LEGAL*",
        "TERMS*",
        "CONDITIONS*"
    ]
    
    license_files = []
    for pattern in license_patterns:
        license_files.extend(folder.glob(pattern))
    
    # Also check for licenses subdirectory
    licenses_dir = folder / "licenses"
    if licenses_dir.exists():
        license_files.extend(licenses_dir.glob("*"))
    
    if not license_files:
        return {
            "folder": str(folder.relative_to(data_root)),
            "has_license": False,
            "license_files": [],
            "warnings": ["No license files found"]
        }
    
    # Analyze each license file
    analyzed_files = []
    for license_file in license_files:
        try:
            content = license_file.read_text(encoding='utf-8', errors='ignore')
            analyzed_files.append({
                "path": str(license_file.relative_to(data_root)),
                "size": license_file.stat().st_size,
                "content_preview": content[:500],
                "license_type": _detect_license_type(content),
                "has_attribution": _check_attribution_requirements(content)
            })
        except Exception as e:
            analyzed_files.append({
                "path": str(license_file.relative_to(data_root)),
                "error": str(e)
            })
    
    return {
        "folder": str(folder.relative_to(data_root)),
        "has_license": True,
        "license_files": analyzed_files,
        "warnings": []
    }


def _detect_license_type(content: str) -> str:
    """Detect the type of license from content."""
    content_lower = content.lower()
    
    # Common license patterns
    license_patterns = {
        "MIT": ["mit license", "mit copyright", "permission is hereby granted"],
        "Apache": ["apache license", "apache software foundation", "apache 2.0"],
        "GPL": ["gpl", "gnu general public license", "gpl-3.0"],
        "BSD": ["bsd license", "berkeley software distribution", "bsd-3-clause"],
        "CC": ["creative commons", "cc-by", "cc0", "creative commons attribution"],
        "MIT": ["mit license", "mit copyright", "permission is hereby granted"],
        "Apache": ["apache license", "apache software foundation", "apache 2.0"],
        "GPL": ["gpl", "gnu general public license", "gpl-3.0"],
        "BSD": ["bsd license", "berkeley software distribution", "bsd-3-clause"],
        "CC": ["creative commons", "cc-by", "cc0", "creative commons attribution"],
        "Custom": ["custom license", "proprietary", "all rights reserved"],
        "Public Domain": ["public domain", "no copyright", "unrestricted use"]
    }
    
    for license_type, patterns in license_patterns.items():
        if any(pattern in content_lower for pattern in patterns):
            return license_type
    
    return "Unknown"


def _check_attribution_requirements(content: str) -> bool:
    """Check if license requires attribution."""
    content_lower = content.lower()
    
    attribution_keywords = [
        "attribution",
        "credit",
        "acknowledge",
        "cite",
        "reference",
        "author",
        "creator",
        "copyright",
        "license notice",
        "license text"
    ]
    
    return any(keyword in content_lower for keyword in attribution_keywords)


def _analyze_license_content(license_files: List[Dict]) -> List[Dict[str, Any]]:
    """Analyze license content for attribution requirements."""
    requirements = []
    
    for license_file in license_files:
        if "content_preview" in license_file:
            content = license_file["content_preview"]
            license_type = license_file.get("license_type", "Unknown")
            has_attribution = license_file.get("has_attribution", False)
            
            requirements.append({
                "file": license_file["path"],
                "license_type": license_type,
                "requires_attribution": has_attribution,
                "attribution_text": _extract_attribution_text(content),
                "commercial_use": _check_commercial_use(content),
                "modification_allowed": _check_modification_allowed(content),
                "distribution_requirements": _check_distribution_requirements(content)
            })
    
    return requirements


def _extract_attribution_text(content: str) -> str:
    """Extract attribution text from license content."""
    # Look for common attribution patterns
    attribution_patterns = [
        r"attribution[^.]*?\.(.*?)(?:\n\n|\n[A-Z]|$)",
        r"credit[^.]*?\.(.*?)(?:\n\n|\n[A-Z]|$)",
        r"acknowledge[^.]*?\.(.*?)(?:\n\n|\n[A-Z]|$)",
        r"cite[^.]*?\.(.*?)(?:\n\n|\n[A-Z]|$)",
        r"reference[^.]*?\.(.*?)(?:\n\n|\n[A-Z]|$)"
    ]
    
    for pattern in attribution_patterns:
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    
    return ""


def _check_commercial_use(content: str) -> bool:
    """Check if commercial use is allowed."""
    content_lower = content.lower()
    
    # Positive indicators
    positive_keywords = [
        "commercial use",
        "commercial purposes",
        "commercial activity",
        "for profit",
        "commercial distribution"
    ]
    
    # Negative indicators
    negative_keywords = [
        "no commercial use",
        "non-commercial",
        "not for commercial",
        "prohibited commercial",
        "commercial use prohibited"
    ]
    
    has_positive = any(keyword in content_lower for keyword in positive_keywords)
    has_negative = any(keyword in content_lower for keyword in negative_keywords)
    
    if has_negative:
        return False
    elif has_positive:
        return True
    else:
        return True  # Default to allowed if not specified


def _check_modification_allowed(content: str) -> bool:
    """Check if modification is allowed."""
    content_lower = content.lower()
    
    # Positive indicators
    positive_keywords = [
        "modify",
        "adapt",
        "change",
        "alter",
        "derivative works",
        "modification"
    ]
    
    # Negative indicators
    negative_keywords = [
        "no modification",
        "no derivative",
        "no changes",
        "no alteration",
        "modification prohibited"
    ]
    
    has_positive = any(keyword in content_lower for keyword in positive_keywords)
    has_negative = any(keyword in content_lower for keyword in negative_keywords)
    
    if has_negative:
        return False
    elif has_positive:
        return True
    else:
        return True  # Default to allowed if not specified


def _check_distribution_requirements(content: str) -> List[str]:
    """Check distribution requirements."""
    content_lower = content.lower()
    requirements = []
    
    if "share alike" in content_lower or "copyleft" in content_lower:
        requirements.append("ShareAlike")
    
    if "same license" in content_lower:
        requirements.append("SameLicense")
    
    if "source code" in content_lower:
        requirements.append("SourceCode")
    
    if "license notice" in content_lower:
        requirements.append("LicenseNotice")
    
    return requirements


def _assess_compliance(results: Dict[str, Any]) -> Dict[str, Any]:
    """Assess overall compliance status."""
    compliance = {
        "has_license": len(results["license_files"]) > 0,
        "has_attribution": any(req["requires_attribution"] for req in results["attribution_requirements"]),
        "warnings": [],
        "recommendations": []
    }
    
    # Generate warnings
    if not compliance["has_license"]:
        compliance["warnings"].append("No license files found in dataset")
    
    if compliance["has_license"] and not compliance["has_attribution"]:
        compliance["warnings"].append("License files found but attribution requirements unclear")
    
    # Check for mixed licenses
    license_types = [req["license_type"] for req in results["attribution_requirements"]]
    unique_types = set(license_types)
    if len(unique_types) > 1:
        compliance["warnings"].append(f"Multiple license types found: {', '.join(unique_types)}")
    
    # Generate recommendations
    if not compliance["has_license"]:
        compliance["recommendations"].append("Add a LICENSE file to the dataset root")
        compliance["recommendations"].append("Consider using a standard open-source license (MIT, Apache 2.0, etc.)")
    
    if compliance["has_license"] and not compliance["has_attribution"]:
        compliance["recommendations"].append("Clarify attribution requirements in license files")
    
    if len(unique_types) > 1:
        compliance["recommendations"].append("Consider standardizing on a single license type")
    
    return compliance


def generate_license_report(results: Dict[str, Any], output_path: str) -> None:
    """Generate a human-readable license report."""
    report_lines = [
        "# License and Attribution Report",
        "",
        f"**Total folders:** {results['summary']['total_folders']}",
        f"**Licensed folders:** {results['summary']['licensed_folders']}",
        f"**Unknown license folders:** {results['summary']['unknown_license_folders']}",
        "",
        "## License Files Found",
        ""
    ]
    
    if results["license_files"]:
        for license_file in results["license_files"]:
            report_lines.extend([
                f"### {license_file['path']}",
                f"**Type:** {license_file.get('license_type', 'Unknown')}",
                f"**Size:** {license_file.get('size', 0)} bytes",
                f"**Attribution Required:** {license_file.get('has_attribution', False)}",
                "",
                "**Content Preview:**",
                "```",
                license_file.get('content_preview', '')[:300] + "...",
                "```",
                ""
            ])
    else:
        report_lines.append("No license files found.")
        report_lines.append("")
    
    # Attribution requirements
    report_lines.extend([
        "## Attribution Requirements",
        ""
    ])
    
    if results["attribution_requirements"]:
        for req in results["attribution_requirements"]:
            report_lines.extend([
                f"### {req['file']}",
                f"**License Type:** {req['license_type']}",
                f"**Requires Attribution:** {req['requires_attribution']}",
                f"**Commercial Use:** {req['commercial_use']}",
                f"**Modification Allowed:** {req['modification_allowed']}",
                f"**Distribution Requirements:** {', '.join(req['distribution_requirements']) if req['distribution_requirements'] else 'None'}",
                ""
            ])
            
            if req["attribution_text"]:
                report_lines.extend([
                    "**Attribution Text:**",
                    req["attribution_text"],
                    ""
                ])
    else:
        report_lines.append("No attribution requirements found.")
        report_lines.append("")
    
    # Compliance status
    report_lines.extend([
        "## Compliance Status",
        ""
    ])
    
    compliance = results["compliance_status"]
    report_lines.extend([
        f"**Has License:** {compliance['has_license']}",
        f"**Has Attribution:** {compliance['has_attribution']}",
        ""
    ])
    
    if compliance["warnings"]:
        report_lines.extend([
            "### Warnings",
            ""
        ])
        for warning in compliance["warnings"]:
            report_lines.append(f"- ⚠️ {warning}")
        report_lines.append("")
    
    if compliance["recommendations"]:
        report_lines.extend([
            "### Recommendations",
            ""
        ])
        for rec in compliance["recommendations"]:
            report_lines.append(f"- 💡 {rec}")
        report_lines.append("")
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"[LICENSE] Report saved to {output_path}")
