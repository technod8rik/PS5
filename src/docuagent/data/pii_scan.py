"""
PII (Personally Identifiable Information) scanner module.

This module provides fast heuristic scanning for sensitive information in:
- OCR'd text snippets
- Filenames
- Various PII patterns (emails, phones, IDs, etc.)
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib


def scan_pii(
    text_data: Dict[str, str] = None,
    filenames: List[str] = None,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Scan for PII in text data and filenames.
    
    Args:
        text_data: Dict mapping file paths to OCR text content
        filenames: List of filenames to check
        output_path: Optional path to save results
        
    Returns:
        Dict with PII findings
    """
    findings = {
        "emails": [],
        "phone_numbers": [],
        "pan_numbers": [],
        "aadhaar_numbers": [],
        "iban_codes": [],
        "swift_codes": [],
        "dates_of_birth": [],
        "credit_cards": [],
        "ssn": [],
        "summary": {
            "total_files_scanned": 0,
            "total_findings": 0,
            "files_with_pii": 0
        }
    }
    
    # Compile regex patterns
    patterns = _compile_pii_patterns()
    
    # Scan text data
    if text_data:
        for file_path, text in text_data.items():
            file_findings = _scan_text(text, patterns, file_path)
            _merge_findings(findings, file_findings)
            findings["summary"]["total_files_scanned"] += 1
    
    # Scan filenames
    if filenames:
        for filename in filenames:
            file_findings = _scan_text(filename, patterns, filename)
            _merge_findings(findings, file_findings)
            findings["summary"]["total_files_scanned"] += 1
    
    # Calculate summary statistics
    findings["summary"]["total_findings"] = sum(
        len(findings[key]) for key in findings 
        if key != "summary"
    )
    findings["summary"]["files_with_pii"] = len(set(
        finding["file"] for finding in findings["emails"] +
        findings["phone_numbers"] + findings["pan_numbers"] +
        findings["aadhaar_numbers"] + findings["iban_codes"] +
        findings["swift_codes"] + findings["dates_of_birth"] +
        findings["credit_cards"] + findings["ssn"]
    ))
    
    # Save results if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(findings, f, indent=2)
        print(f"[PII] Results saved to {output_path}")
    
    return findings


def _compile_pii_patterns() -> Dict[str, re.Pattern]:
    """Compile regex patterns for PII detection."""
    patterns = {
        # Email addresses
        "email": re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            re.IGNORECASE
        ),
        
        # Phone numbers (various formats)
        "phone": re.compile(
            r'(\+?1[-.\s]?)?(\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}|'
            r'\+?[0-9]{1,4}[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,9}',
            re.IGNORECASE
        ),
        
        # PAN numbers (Indian)
        "pan": re.compile(
            r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b',
            re.IGNORECASE
        ),
        
        # Aadhaar numbers (Indian)
        "aadhaar": re.compile(
            r'\b[0-9]{4}[\s-]?[0-9]{4}[\s-]?[0-9]{4}\b'
        ),
        
        # IBAN codes
        "iban": re.compile(
            r'\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}([A-Z0-9]?){0,16}\b',
            re.IGNORECASE
        ),
        
        # SWIFT codes
        "swift": re.compile(
            r'\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?\b',
            re.IGNORECASE
        ),
        
        # Dates of birth (various formats)
        "dob": re.compile(
            r'\b(0?[1-9]|1[0-2])[-/](0?[1-9]|[12][0-9]|3[01])[-/](19|20)\d{2}\b|'
            r'\b(0?[1-9]|[12][0-9]|3[01])[-/](0?[1-9]|1[0-2])[-/](19|20)\d{2}\b|'
            r'\b(19|20)\d{2}[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12][0-9]|3[01])\b',
            re.IGNORECASE
        ),
        
        # Credit card numbers
        "credit_card": re.compile(
            r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|'
            r'3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'
        ),
        
        # SSN (US)
        "ssn": re.compile(
            r'\b\d{3}-?\d{2}-?\d{4}\b'
        )
    }
    
    return patterns


def _scan_text(text: str, patterns: Dict[str, re.Pattern], file_path: str) -> Dict[str, List[Dict]]:
    """Scan text for PII patterns."""
    findings = {
        "emails": [],
        "phone_numbers": [],
        "pan_numbers": [],
        "aadhaar_numbers": [],
        "iban_codes": [],
        "swift_codes": [],
        "dates_of_birth": [],
        "credit_cards": [],
        "ssn": []
    }
    
    # Map pattern names to finding keys
    pattern_map = {
        "email": "emails",
        "phone": "phone_numbers",
        "pan": "pan_numbers",
        "aadhaar": "aadhaar_numbers",
        "iban": "iban_codes",
        "swift": "swift_codes",
        "dob": "dates_of_birth",
        "credit_card": "credit_cards",
        "ssn": "ssn"
    }
    
    for pattern_name, pattern in patterns.items():
        matches = pattern.findall(text)
        if isinstance(matches[0], tuple) if matches else False:
            matches = [''.join(match) for match in matches]
        
        for match in matches:
            if match.strip():
                redacted = _redact_pii(match, pattern_name)
                findings[pattern_map[pattern_name]].append({
                    "file": file_path,
                    "original": match.strip(),
                    "redacted": redacted,
                    "type": pattern_name,
                    "context": _get_context(text, match, 50)
                })
    
    return findings


def _redact_pii(text: str, pii_type: str) -> str:
    """Redact PII with appropriate masking."""
    if pii_type == "email":
        # Keep first and last character, mask middle
        if len(text) > 2:
            return text[0] + "*" * (len(text) - 2) + text[-1]
        return "*" * len(text)
    
    elif pii_type in ["phone", "credit_card", "ssn"]:
        # Show first 2 and last 2 characters
        if len(text) > 4:
            return text[:2] + "*" * (len(text) - 4) + text[-2:]
        return "*" * len(text)
    
    elif pii_type in ["pan", "aadhaar", "iban", "swift"]:
        # Show first 2 and last 2 characters
        if len(text) > 4:
            return text[:2] + "*" * (len(text) - 4) + text[-2:]
        return "*" * len(text)
    
    elif pii_type == "dob":
        # Mask year
        parts = re.split(r'[-/]', text)
        if len(parts) >= 3:
            year = parts[-1] if parts[-1].isdigit() and len(parts[-1]) == 4 else parts[0]
            if year.isdigit() and len(year) == 4:
                return text.replace(year, "****")
        return "*" * len(text)
    
    else:
        # Generic masking
        return "*" * len(text)


def _get_context(text: str, match: str, context_length: int = 50) -> str:
    """Get context around a PII match."""
    start = text.find(match)
    if start == -1:
        return ""
    
    context_start = max(0, start - context_length)
    context_end = min(len(text), start + len(match) + context_length)
    
    context = text[context_start:context_end]
    if context_start > 0:
        context = "..." + context
    if context_end < len(text):
        context = context + "..."
    
    return context


def _merge_findings(main_findings: Dict, file_findings: Dict) -> None:
    """Merge file findings into main findings."""
    for key in main_findings:
        if key != "summary" and key in file_findings:
            main_findings[key].extend(file_findings[key])


def scan_ocr_text(ocr_results: List[Dict], output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Scan OCR results for PII.
    
    Args:
        ocr_results: List of OCR results with 'text' and 'file' keys
        output_path: Optional path to save results
        
    Returns:
        Dict with PII findings
    """
    text_data = {}
    for result in ocr_results:
        if 'text' in result and 'file' in result:
            text_data[result['file']] = result['text']
    
    return scan_pii(text_data=text_data, output_path=output_path)


def generate_pii_report(findings: Dict[str, Any], output_path: str) -> None:
    """Generate a human-readable PII report."""
    report_lines = [
        "# PII Scan Report",
        "",
        f"**Total files scanned:** {findings['summary']['total_files_scanned']}",
        f"**Total PII findings:** {findings['summary']['total_findings']}",
        f"**Files with PII:** {findings['summary']['files_with_pii']}",
        "",
        "## Findings by Type",
        ""
    ]
    
    # Add findings for each type
    for pii_type in ["emails", "phone_numbers", "pan_numbers", "aadhaar_numbers", 
                     "iban_codes", "swift_codes", "dates_of_birth", "credit_cards", "ssn"]:
        if findings[pii_type]:
            report_lines.extend([
                f"### {pii_type.replace('_', ' ').title()}",
                f"**Count:** {len(findings[pii_type])}",
                ""
            ])
            
            # Add first 5 examples
            for finding in findings[pii_type][:5]:
                report_lines.extend([
                    f"- **File:** {finding['file']}",
                    f"  - **Original:** {finding['original']}",
                    f"  - **Redacted:** {finding['redacted']}",
                    f"  - **Context:** {finding['context']}",
                    ""
                ])
            
            if len(findings[pii_type]) > 5:
                report_lines.append(f"... and {len(findings[pii_type]) - 5} more")
                report_lines.append("")
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"[PII] Report saved to {output_path}")
