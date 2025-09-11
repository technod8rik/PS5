"""Quality assurance for VLM-generated descriptions."""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import statistics


@dataclass
class QAFlag:
    """Represents a quality assurance flag."""
    element_id: str
    page: int
    class_name: str
    issue_type: str
    severity: str  # "low", "medium", "high"
    description: str
    suggestion: str
    metadata: Dict[str, Any]


@dataclass
class QAResult:
    """Results of description quality assurance."""
    total_elements: int
    flagged_elements: int
    flags: List[QAFlag]
    summary: Dict[str, Any]


def load_elements(elements_path: str) -> List[Dict[str, Any]]:
    """Load elements from JSON file."""
    with open(elements_path, 'r') as f:
        data = json.load(f)
    
    if 'elements' in data:
        return data['elements']
    else:
        return data


def check_length_quality(description: str, max_words: int = 80) -> Optional[QAFlag]:
    """Check if description length is appropriate.
    
    Args:
        description: Description text
        max_words: Maximum number of words allowed
        
    Returns:
        QAFlag if issue found, None otherwise
    """
    if not description or not isinstance(description, str):
        return None
    
    words = description.split()
    word_count = len(words)
    
    if word_count > max_words:
        return QAFlag(
            element_id="",
            page=0,
            class_name="",
            issue_type="length_excessive",
            severity="medium",
            description=f"Description is too long ({word_count} words, max {max_words})",
            suggestion="Consider shortening the description to focus on key information",
            metadata={'word_count': word_count, 'max_words': max_words}
        )
    
    if word_count < 3:
        return QAFlag(
            element_id="",
            page=0,
            class_name="",
            issue_type="length_insufficient",
            severity="high",
            description=f"Description is too short ({word_count} words, min 3)",
            suggestion="Provide more detailed description of the element",
            metadata={'word_count': word_count, 'min_words': 3}
        )
    
    return None


def check_language_conformity(
    description: str,
    page_language: Optional[str] = None,
    allowed_languages: List[str] = None
) -> Optional[QAFlag]:
    """Check if description language matches page language.
    
    Args:
        description: Description text
        page_language: Dominant language of the page
        allowed_languages: List of allowed languages
        
    Returns:
        QAFlag if issue found, None otherwise
    """
    if not description or not isinstance(description, str):
        return None
    
    if not page_language or not allowed_languages:
        return None
    
    # Simple language detection based on character patterns
    # This is a basic implementation - in practice, you'd use a proper language detector
    
    # Check for non-Latin characters (indicating non-English)
    has_non_latin = bool(re.search(r'[^\x00-\x7F]', description))
    
    # Check for common English words
    english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
    has_english_words = any(word.lower() in description.lower() for word in english_words)
    
    # Determine if description appears to be in English
    is_english = has_english_words and not has_non_latin
    
    if page_language == 'en' and not is_english:
        return QAFlag(
            element_id="",
            page=0,
            class_name="",
            issue_type="language_mismatch",
            severity="medium",
            description=f"Description appears to be in different language than page ({page_language})",
            suggestion="Ensure description matches the page language",
            metadata={'page_language': page_language, 'detected_english': is_english}
        )
    
    return None


def check_numeric_consistency(
    description: str,
    class_name: str
) -> Optional[QAFlag]:
    """Check if description contains appropriate numeric content for data elements.
    
    Args:
        description: Description text
        class_name: Element class (Table, Figure, etc.)
        
    Returns:
        QAFlag if issue found, None otherwise
    """
    if not description or not isinstance(description, str):
        return None
    
    # Check if this is a data element that should contain numbers
    data_classes = ['Table', 'Figure', 'Chart', 'Graph']
    if class_name not in data_classes:
        return None
    
    # Count numbers in description
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', description)
    number_count = len(numbers)
    
    if number_count == 0:
        return QAFlag(
            element_id="",
            page=0,
            class_name=class_name,
            issue_type="missing_numerics",
            severity="medium",
            description=f"No numbers found in description of {class_name}",
            suggestion="Include key numerical values or statistics from the data",
            metadata={'number_count': number_count, 'class_name': class_name}
        )
    
    return None


def check_sentence_structure(description: str) -> Optional[QAFlag]:
    """Check if description has proper sentence structure.
    
    Args:
        description: Description text
        
    Returns:
        QAFlag if issue found, None otherwise
    """
    if not description or not isinstance(description, str):
        return None
    
    # Check for proper sentence endings
    sentences = re.split(r'[.!?]+', description.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) == 0:
        return QAFlag(
            element_id="",
            page=0,
            class_name="",
            issue_type="no_sentences",
            severity="high",
            description="Description contains no complete sentences",
            suggestion="Write complete sentences with proper punctuation",
            metadata={'sentence_count': 0}
        )
    
    # Check for very long sentences
    long_sentences = [s for s in sentences if len(s.split()) > 30]
    if long_sentences:
        return QAFlag(
            element_id="",
            page=0,
            class_name="",
            issue_type="long_sentences",
            severity="low",
            description=f"Description contains {len(long_sentences)} very long sentences",
            suggestion="Consider breaking long sentences into shorter ones",
            metadata={'sentence_count': len(sentences), 'long_sentence_count': len(long_sentences)}
        )
    
    return None


def check_repetition(description: str) -> Optional[QAFlag]:
    """Check for excessive repetition in description.
    
    Args:
        description: Description text
        
    Returns:
        QAFlag if issue found, None otherwise
    """
    if not description or not isinstance(description, str):
        return None
    
    words = description.lower().split()
    if len(words) < 5:
        return None
    
    # Count word frequencies
    word_counts = Counter(words)
    
    # Check for excessive repetition
    total_words = len(words)
    unique_words = len(word_counts)
    
    repetition_ratio = unique_words / total_words
    
    if repetition_ratio < 0.5:  # Less than 50% unique words
        most_common = word_counts.most_common(3)
        return QAFlag(
            element_id="",
            page=0,
            class_name="",
            issue_type="excessive_repetition",
            severity="medium",
            description=f"Description has high repetition (only {repetition_ratio:.1%} unique words)",
            suggestion="Reduce repetitive language and vary word choice",
            metadata={'repetition_ratio': repetition_ratio, 'most_common_words': most_common}
        )
    
    return None


def check_content_relevance(
    description: str,
    class_name: str,
    bbox: List[int]
) -> Optional[QAFlag]:
    """Check if description content is relevant to the element type.
    
    Args:
        description: Description text
        class_name: Element class
        bbox: Bounding box [x, y, w, h]
        
    Returns:
        QAFlag if issue found, None otherwise
    """
    if not description or not isinstance(description, str):
        return None
    
    # Check for generic or unhelpful descriptions
    generic_phrases = [
        "this is a", "here is a", "the following", "as shown", "see below",
        "figure shows", "table contains", "image displays"
    ]
    
    description_lower = description.lower()
    generic_count = sum(1 for phrase in generic_phrases if phrase in description_lower)
    
    if generic_count > 2:
        return QAFlag(
            element_id="",
            page=0,
            class_name=class_name,
            issue_type="generic_content",
            severity="low",
            description="Description contains generic phrases that don't add value",
            suggestion="Provide specific, informative content about the element",
            metadata={'generic_phrase_count': generic_count, 'class_name': class_name}
        )
    
    # Check for appropriate content based on element type
    if class_name == "Table":
        if not any(word in description_lower for word in ['data', 'table', 'row', 'column', 'value', 'number']):
            return QAFlag(
                element_id="",
                page=0,
                class_name=class_name,
                issue_type="inappropriate_content",
                severity="medium",
                description="Table description doesn't mention data or structure",
                suggestion="Describe the data content and structure of the table",
                metadata={'class_name': class_name}
            )
    
    elif class_name == "Figure":
        if not any(word in description_lower for word in ['image', 'figure', 'chart', 'graph', 'diagram', 'visual']):
            return QAFlag(
                element_id="",
                page=0,
                class_name=class_name,
                issue_type="inappropriate_content",
                severity="medium",
                description="Figure description doesn't mention visual content",
                suggestion="Describe the visual content and purpose of the figure",
                metadata={'class_name': class_name}
            )
    
    return None


def qa_descriptions(
    elements_path: str,
    output_dir: str,
    page_language: Optional[str] = None,
    allowed_languages: List[str] = None
) -> QAResult:
    """Perform quality assurance on element descriptions.
    
    Args:
        elements_path: Path to elements JSON file
        output_dir: Directory to save QA results
        page_language: Dominant language of the page
        allowed_languages: List of allowed languages
        
    Returns:
        QAResult object with QA findings
    """
    print(f"[INFO] Performing QA on descriptions from {elements_path}")
    
    # Load elements
    elements = load_elements(elements_path)
    
    # Initialize results
    flags = []
    total_elements = len(elements)
    flagged_elements = 0
    
    # Process each element
    for i, element in enumerate(elements):
        element_id = f"element_{i}"
        page = element.get('page', 0)
        class_name = element.get('class', 'Unknown')
        description = element.get('content', '')
        bbox = element.get('bbox', [0, 0, 0, 0])
        
        # Skip if no description
        if not description or not isinstance(description, str):
            continue
        
        element_flags = []
        
        # Run all QA checks
        checks = [
            check_length_quality(description),
            check_language_conformity(description, page_language, allowed_languages),
            check_numeric_consistency(description, class_name),
            check_sentence_structure(description),
            check_repetition(description),
            check_content_relevance(description, class_name, bbox)
        ]
        
        # Collect flags for this element
        for flag in checks:
            if flag:
                flag.element_id = element_id
                flag.page = page
                flag.class_name = class_name
                element_flags.append(flag)
        
        # Add flags to overall list
        if element_flags:
            flags.extend(element_flags)
            flagged_elements += 1
    
    # Create summary
    summary = {
        'total_elements': total_elements,
        'flagged_elements': flagged_elements,
        'flag_rate': flagged_elements / total_elements if total_elements > 0 else 0,
        'flags_by_type': Counter(flag.issue_type for flag in flags),
        'flags_by_severity': Counter(flag.severity for flag in flags),
        'flags_by_class': Counter(flag.class_name for flag in flags)
    }
    
    result = QAResult(
        total_elements=total_elements,
        flagged_elements=flagged_elements,
        flags=flags,
        summary=summary
    )
    
    # Save results
    save_qa_results(result, output_dir)
    
    print(f"[INFO] QA complete: {flagged_elements}/{total_elements} elements flagged")
    print(f"[INFO] Results saved to {output_dir}")
    
    return result


def save_qa_results(result: QAResult, output_dir: str) -> None:
    """Save QA results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results as JSON
    results_dict = {
        'total_elements': result.total_elements,
        'flagged_elements': result.flagged_elements,
        'summary': result.summary,
        'flags': [
            {
                'element_id': flag.element_id,
                'page': flag.page,
                'class_name': flag.class_name,
                'issue_type': flag.issue_type,
                'severity': flag.severity,
                'description': flag.description,
                'suggestion': flag.suggestion,
                'metadata': flag.metadata
            }
            for flag in result.flags
        ]
    }
    
    json_path = output_path / "qa_results.json"
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    # Save Markdown report
    markdown_path = output_path / "qa_report.md"
    with open(markdown_path, 'w') as f:
        f.write("# Description Quality Assurance Report\n\n")
        
        f.write(f"**Total Elements:** {result.total_elements}\n")
        f.write(f"**Flagged Elements:** {result.flagged_elements}\n")
        f.write(f"**Flag Rate:** {result.summary['flag_rate']:.1%}\n\n")
        
        f.write("## Summary by Issue Type\n\n")
        for issue_type, count in result.summary['flags_by_type'].most_common():
            f.write(f"- **{issue_type}:** {count}\n")
        
        f.write("\n## Summary by Severity\n\n")
        for severity, count in result.summary['flags_by_severity'].most_common():
            f.write(f"- **{severity}:** {count}\n")
        
        f.write("\n## Summary by Class\n\n")
        for class_name, count in result.summary['flags_by_class'].most_common():
            f.write(f"- **{class_name}:** {count}\n")
        
        f.write("\n## Detailed Flags\n\n")
        for flag in result.flags:
            f.write(f"### {flag.issue_type} (Page {flag.page}, {flag.class_name})\n\n")
            f.write(f"**Severity:** {flag.severity}\n\n")
            f.write(f"**Description:** {flag.description}\n\n")
            f.write(f"**Suggestion:** {flag.suggestion}\n\n")
            if flag.metadata:
                f.write(f"**Metadata:** {flag.metadata}\n\n")
            f.write("---\n\n")
    
    print(f"[INFO] QA results saved to {json_path}")
    print(f"[INFO] QA report saved to {markdown_path}")


def main():
    """Command line interface for description QA."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Perform QA on element descriptions")
    parser.add_argument("--elements", required=True, help="Elements JSON file")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--page-lang", help="Page dominant language")
    parser.add_argument("--allowed-langs", nargs='+', help="Allowed languages")
    
    args = parser.parse_args()
    
    result = qa_descriptions(
        args.elements, args.out, args.page_lang, args.allowed_langs
    )
    
    print(f"[INFO] QA complete: {result.flagged_elements}/{result.total_elements} elements flagged")


if __name__ == "__main__":
    main()
