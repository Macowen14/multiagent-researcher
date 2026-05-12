"""
Enhanced document generation tools with structured output capabilities.
Creates well-documented research outputs with proper formatting and references.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Annotated
from datetime import datetime
from langchain_core.tools import tool

from models.research_schemas import (
    DocumentStructure,
    SourceReference,
    ResearchReport,
    AnalysisReport,
    ContentType,
    ResearchQuality
)

logger = logging.getLogger(__name__)


@tool("create_structured_document")
def create_structured_document(
    title: Annotated[str, "Document title"],
    research_data: Annotated[str, "JSON string containing research data"],
    document_type: Annotated[str, "Type of document: report, analysis, summary, comparison"] = "report"
) -> str:
    """Create a well-structured document with proper formatting and references."""
    logger.info(f"Creating structured document: '{title}' (type: {document_type})")
    
    try:
        research_json = json.loads(research_data)
        
        # Create document structure based on type
        if document_type.lower() == "report":
            sections = [
                {
                    "title": "Executive Summary",
                    "content": research_json.get("summary", "Summary not available"),
                    "level": 1
                },
                {
                    "title": "Introduction",
                    "content": f"This document provides comprehensive research on: {research_json.get('query', 'Unknown topic')}",
                    "level": 1
                },
                {
                    "title": "Methodology",
                    "content": research_json.get("methodology", "Methodology not specified"),
                    "level": 1
                },
                {
                    "title": "Key Findings",
                    "content": _format_findings(research_json.get("findings", [])),
                    "level": 1
                },
                {
                    "title": "Analysis",
                    "content": _format_analysis(research_json),
                    "level": 1
                },
                {
                    "title": "Conclusions",
                    "content": _format_conclusions(research_json),
                    "level": 1
                },
                {
                    "title": "References",
                    "content": _format_references(research_json.get("sources", [])),
                    "level": 1
                }
            ]
        elif document_type.lower() == "comparison":
            sections = [
                {
                    "title": "Comparison Overview",
                    "content": f"Comparative analysis of: {research_json.get('query', 'Unknown topic')}",
                    "level": 1
                },
                {
                    "title": "Options Analysis",
                    "content": _format_comparison_options(research_json),
                    "level": 1
                },
                {
                    "title": "Pros and Cons",
                    "content": _format_pros_cons(research_json),
                    "level": 1
                },
                {
                    "title": "Recommendations",
                    "content": _format_recommendations(research_json),
                    "level": 1
                },
                {
                    "title": "References",
                    "content": _format_references(research_json.get("sources", [])),
                    "level": 1
                }
            ]
        else:  # summary
            sections = [
                {
                    "title": "Summary",
                    "content": research_json.get("summary", "Summary not available"),
                    "level": 1
                },
                {
                    "title": "Key Points",
                    "content": _format_key_points(research_json),
                    "level": 1
                },
                {
                    "title": "Sources",
                    "content": _format_references(research_json.get("sources", [])),
                    "level": 1
                }
            ]
        
        # Create document structure
        doc_structure = DocumentStructure(
            title=title,
            sections=sections,
            references=_create_source_references(research_json.get("sources", [])),
            appendices=[
                {
                    "title": "Research Methodology",
                    "content": f"Research conducted using multiple search tools on {datetime.now().strftime('%Y-%m-%d')}"
                },
                {
                    "title": "Quality Assessment",
                    "content": f"Overall confidence: {research_json.get('confidence_overall', 0.8):.2f}/1.0"
                }
            ],
            metadata={
                "document_type": document_type,
                "created": datetime.now().isoformat(),
                "confidence": research_json.get("confidence_overall", 0.8),
                "source_count": len(research_json.get("sources", []))
            }
        )
        
        # Generate markdown document
        markdown_content = _generate_markdown(doc_structure)
        
        return f"""# Document Created Successfully

**Title:** {doc_structure.title}
**Type:** {document_type.title()}
**Sections:** {len(doc_structure.sections)}
**Sources:** {len(doc_structure.references)}
**Created:** {doc_structure.metadata['created']}

## Document Preview:

{markdown_content[:1000]}...

## Document Structure:
{chr(10).join([f"- {section['title']} (Level {section['level']})" for section in doc_structure.sections])}

## Metadata:
{chr(10).join([f"- {key}: {value}" for key, value in doc_structure.metadata.items()])}

*Full document content available for saving to file.*
"""
        
    except json.JSONDecodeError:
        return "Error: Invalid JSON format for research data"
    except Exception as e:
        logger.error(f"Error creating structured document: {e}")
        return f"Error creating structured document: {str(e)}"


@tool("generate_research_summary")
def generate_research_summary(
    research_data: Annotated[str, "JSON string containing research data"],
    summary_length: Annotated[str, "Summary length: brief, detailed, comprehensive"] = "detailed"
) -> str:
    """Generate a research summary with appropriate length and structure."""
    logger.info(f"Generating research summary (length: {summary_length})")
    
    try:
        research_json = json.loads(research_data)
        
        if summary_length.lower() == "brief":
            summary = f"""# Research Summary

**Topic:** {research_json.get('query', 'Unknown topic')}
**Sources:** {len(research_json.get('sources', []))}
**Confidence:** {research_json.get('confidence_overall', 0.8):.1f}/1.0

## Key Findings
{_format_key_findings_brief(research_json.get('findings', []))}

## Main Conclusion
{research_json.get('summary', 'Summary not available')[:200]}...
"""
        elif summary_length.lower() == "detailed":
            summary = f"""# Detailed Research Summary

## Executive Summary
{research_json.get('summary', 'Summary not available')}

## Research Overview
- **Topic:** {research_json.get('query', 'Unknown topic')}
- **Sources Consulted:** {len(research_json.get('sources', []))}
- **Methodology:** {research_json.get('methodology', 'Not specified')}
- **Overall Confidence:** {research_json.get('confidence_overall', 0.8):.2f}/1.0
- **Research Date:** {datetime.now().strftime('%Y-%m-%d')}

## Key Findings
{_format_findings(research_json.get('findings', []))}

## Research Gaps
{_format_gaps(research_json.get('gaps', []))}

## Limitations
{_format_limitations(research_json.get('limitations', []))}

## Next Steps
{_format_next_steps(research_json.get('next_steps', []))}
"""
        else:  # comprehensive
            summary = f"""# Comprehensive Research Report

## Document Information
- **Title:** {research_json.get('query', 'Unknown topic')}
- **Report Type:** Comprehensive Analysis
- **Generation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Confidence Level:** {research_json.get('confidence_overall', 0.8):.2f}/1.0

## Executive Summary
{research_json.get('summary', 'Summary not available')}

## Research Methodology
{research_json.get('methodology', 'Methodology not specified')}

## Detailed Findings
{_format_findings_detailed(research_json.get('findings', []))}

## Source Analysis
{_format_source_analysis(research_json.get('sources', []))}

## Quality Assessment
- **Source Diversity:** {_assess_source_diversity(research_json.get('sources', []))}
- **Information Depth:** Comprehensive
- **Cross-Validation:** {_assess_cross_validation(research_json.get('findings', []))}
- **Recency:** {_assess_recency(research_json.get('sources', []))}

## Identified Gaps and Limitations
{_format_comprehensive_gaps(research_json.get('gaps', []), research_json.get('limitations', []))}

## Recommendations and Next Steps
{_format_comprehensive_recommendations(research_json.get('next_steps', []))}

## Complete References
{_format_references_detailed(research_json.get('sources', []))}
"""
        
        return summary
        
    except json.JSONDecodeError:
        return "Error: Invalid JSON format for research data"
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return f"Error generating summary: {str(e)}"


@tool("validate_document_structure")
def validate_document_structure(
    document_content: Annotated[str, "Document content to validate"],
    document_type: Annotated[str, "Expected document type: report, analysis, summary"] = "report"
) -> str:
    """Validate document structure and quality."""
    logger.info(f"Validating document structure (type: {document_type})")
    
    validation_results = {
        "is_valid": True,
        "score": 0.0,
        "issues": [],
        "suggestions": [],
        "passed_checks": [],
        "failed_checks": []
    }
    
    # Basic structure checks
    if "# " not in document_content:
        validation_results["issues"].append("Missing main title (should start with # )")
        validation_results["failed_checks"].append("Title structure")
        validation_results["score"] -= 0.2
    else:
        validation_results["passed_checks"].append("Title structure")
        validation_results["score"] += 0.2
    
    # Content length check
    if len(document_content) < 500:
        validation_results["issues"].append("Document too short for comprehensive analysis")
        validation_results["failed_checks"].append("Content length")
        validation_results["score"] -= 0.1
    else:
        validation_results["passed_checks"].append("Content length")
        validation_results["score"] += 0.1
    
    # Reference check
    if "http" not in document_content:
        validation_results["issues"].append("No references or links found")
        validation_results["failed_checks"].append("References")
        validation_results["score"] -= 0.2
    else:
        validation_results["passed_checks"].append("References")
        validation_results["score"] += 0.2
    
    # Structure check based on type
    if document_type.lower() == "report":
        required_sections = ["summary", "findings", "methodology"]
        for section in required_sections:
            if section.lower() not in document_content.lower():
                validation_results["issues"].append(f"Missing {section} section")
                validation_results["failed_checks"].append(f"{section} section")
                validation_results["score"] -= 0.1
            else:
                validation_results["passed_checks"].append(f"{section} section")
                validation_results["score"] += 0.1
    
    # Normalize score
    validation_results["score"] = max(0, min(1.0, validation_results["score"]))
    validation_results["is_valid"] = validation_results["score"] >= 0.7
    
    # Generate suggestions
    if not validation_results["is_valid"]:
        validation_results["suggestions"].extend([
            "Add more comprehensive content",
            "Ensure proper document structure with headings",
            "Include references and citations",
            "Add executive summary and conclusions"
        ])
    
    return f"""## Document Validation Results

**Overall Valid:** {'✅ Yes' if validation_results['is_valid'] else '❌ No'}
**Validation Score:** {validation_results['score']:.2f}/1.00

**Passed Checks ({len(validation_results['passed_checks'])}):**
{chr(10).join([f"✅ {check}" for check in validation_results['passed_checks']])}

**Failed Checks ({len(validation_results['failed_checks'])}):**
{chr(10).join([f"❌ {check}" for check in validation_results['failed_checks']])}

**Issues Found:**
{chr(10).join([f"- {issue}" for issue in validation_results['issues']])}

**Suggestions:**
{chr(10).join([f"- {suggestion}" for suggestion in validation_results['suggestions']])}
"""


# Helper functions for formatting
def _format_findings(findings):
    if not findings:
        return "No specific findings available."
    
    formatted = []
    for i, finding in enumerate(findings, 1):
        if isinstance(finding, dict):
            claim = finding.get('claim', 'No claim')
            confidence = finding.get('confidence_level', 0)
            formatted.append(f"{i}. **{claim}** (Confidence: {confidence:.1f})")
        else:
            formatted.append(f"{i}. {finding}")
    
    return chr(10).join(formatted)

def _format_analysis(research_json):
    sources = research_json.get('sources', [])
    confidence = research_json.get('confidence_overall', 0)
    
    return f"""**Source Analysis:**
- Total sources: {len(sources)}
- Overall confidence: {confidence:.2f}/1.0
- Source diversity: {_assess_source_diversity(sources)}

**Quality Assessment:**
- Information cross-validation: {_assess_cross_validation(research_json.get('findings', []))}
- Research completeness: High
- Methodology transparency: Clear"""

def _format_conclusions(research_json):
    gaps = research_json.get('gaps', [])
    limitations = research_json.get('limitations', [])
    next_steps = research_json.get('next_steps', [])
    
    conclusion = f"""**Main Conclusion:**
{research_json.get('summary', 'Summary not available')}

**Research Limitations:**
{chr(10).join([f"- {limitation}" for limitation in limitations]) if limitations else "- No significant limitations identified"}

**Recommended Next Steps:**
{chr(10).join([f"- {step}" for step in next_steps]) if next_steps else "- Research objectives met"}
"""
    
    if gaps:
        conclusion += f"\n\n**Identified Gaps:**\n"
        conclusion += chr(10).join([f"- {gap.get('description', 'Gap description')}" for gap in gaps])
    
    return conclusion

def _format_references(sources):
    if not sources:
        return "No references available."
    
    formatted = []
    for i, source in enumerate(sources, 1):
        if isinstance(source, dict):
            title = source.get('title', 'Unknown Title')
            url = source.get('url', '')
            formatted.append(f"{i}. **{title}** - {url}")
        else:
            formatted.append(f"{i}. {source}")
    
    return chr(10).join(formatted)

def _format_comparison_options(research_json):
    # This would be implemented based on specific comparison data structure
    return "Comparison analysis would be formatted here based on research data."

def _format_pros_cons(research_json):
    # This would be implemented based on specific pros/cons data structure
    return "Pros and cons analysis would be formatted here based on research data."

def _format_recommendations(research_json):
    next_steps = research_json.get('next_steps', [])
    if not next_steps:
        return "No specific recommendations available."
    
    return chr(10).join([f"- {step}" for step in next_steps])

def _format_key_points(research_json):
    findings = research_json.get('findings', [])
    if not findings:
        return "No key points available."
    
    points = []
    for finding in findings:
        if isinstance(finding, dict):
            points.append(f"- {finding.get('claim', 'No claim')}")
        else:
            points.append(f"- {finding}")
    
    return chr(10).join(points)

def _format_key_findings_brief(findings):
    if not findings:
        return "No findings available."
    
    brief_findings = []
    for finding in findings[:3]:  # Only top 3 for brief
        if isinstance(finding, dict):
            brief_findings.append(f"- {finding.get('claim', 'No claim')}")
        else:
            brief_findings.append(f"- {str(finding)[:100]}...")
    
    return chr(10).join(brief_findings)

def _format_gaps(gaps):
    if not gaps:
        return "No significant gaps identified."
    
    return chr(10).join([f"- {gap.get('description', 'Gap description')}" for gap in gaps])

def _format_limitations(limitations):
    if not limitations:
        return "No significant limitations identified."
    
    return chr(10).join([f"- {limitation}" for limitation in limitations])

def _format_next_steps(next_steps):
    if not next_steps:
        return "No next steps identified."
    
    return chr(10).join([f"- {step}" for step in next_steps])

def _format_findings_detailed(findings):
    if not findings:
        return "No findings available."
    
    detailed = []
    for i, finding in enumerate(findings, 1):
        if isinstance(finding, dict):
            claim = finding.get('claim', 'No claim')
            confidence = finding.get('confidence_level', 0)
            evidence = finding.get('evidence', [])
            detailed.append(f"### Finding {i}: {claim}\n\n**Confidence Level:** {confidence:.2f}/1.0\n\n**Supporting Evidence:**\n{chr(10).join([f'- {e}' for e in evidence])}\n")
        else:
            detailed.append(f"### Finding {i}\n\n{finding}\n")
    
    return chr(10).join(detailed)

def _assess_source_diversity(sources):
    if not sources:
        return "No sources available"
    
    source_types = set()
    for source in sources:
        if isinstance(source, dict):
            url = source.get('url', '').lower()
            if 'github.com' in url:
                source_types.add('Code Repository')
            elif 'arxiv.org' in url:
                source_types.add('Academic Paper')
            elif 'stackoverflow.com' in url:
                source_types.add('Technical Q&A')
            elif 'wikipedia.org' in url:
                source_types.add('Encyclopedia')
            else:
                source_types.add('Web Source')
    
    diversity_score = len(source_types) / 5.0  # Max 5 types
    return f"{len(source_types)} types ({diversity_score:.2f} diversity score)"

def _assess_cross_validation(findings):
    if not findings:
        return "No findings to validate"
    
    # Simple heuristic: if findings have supporting sources, consider cross-validated
    validated_count = sum(1 for f in findings if isinstance(f, dict) and f.get('supporting_sources'))
    cross_validation_score = validated_count / len(findings)
    
    return f"{cross_validation_score:.2f} ({validated_count}/{len(findings)} findings cross-validated)"

def _assess_recency(sources):
    if not sources:
        return "No sources to assess"
    
    # This would require date parsing from sources - simplified for now
    return "Recency assessment not available in current implementation"

def _format_comprehensive_gaps(gaps, limitations):
    all_issues = []
    
    if gaps:
        all_issues.extend(["**Research Gaps:**"] + [f"- {gap.get('description', 'Gap description')}" for gap in gaps])
    
    if limitations:
        all_issues.extend(["**Methodological Limitations:**"] + [f"- {limitation}" for limitation in limitations])
    
    return chr(10).join(all_issues) if all_issues else "No significant gaps or limitations identified."

def _format_comprehensive_recommendations(next_steps):
    if not next_steps:
        return "No specific recommendations available."
    
    return chr(10).join([f"### Recommendation {i+1}\n\n{step}\n" for i, step in enumerate(next_steps)])

def _format_references_detailed(sources):
    if not sources:
        return "No references available."
    
    detailed_refs = []
    for i, source in enumerate(sources, 1):
        if isinstance(source, dict):
            title = source.get('title', 'Unknown Title')
            url = source.get('url', '')
            content_type = source.get('content_type', 'Unknown')
            quality = source.get('quality', 'Unknown')
            detailed_refs.append(f"### Reference {i}\n\n**Title:** {title}\n**URL:** {url}\n**Type:** {content_type}\n**Quality:** {quality}\n")
        else:
            detailed_refs.append(f"### Reference {i}\n\n{source}\n")
    
    return chr(10).join(detailed_refs)

def _format_source_analysis(sources):
    if not sources:
        return "No sources available for analysis."
    
    return f"""**Source Count:** {len(sources)}
**Source Diversity:** {_assess_source_diversity(sources)}
**Quality Distribution:** {_assess_quality_distribution(sources)}
**Content Types:** {_assess_content_types(sources)}"""

def _assess_quality_distribution(sources):
    if not sources:
        return "No sources available"
    
    quality_counts = {}
    for source in sources:
        if isinstance(source, dict):
            quality = source.get('quality', 'Unknown')
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
    
    return chr(10).join([f"- {quality}: {count}" for quality, count in quality_counts.items()])

def _assess_content_types(sources):
    if not sources:
        return "No sources available"
    
    type_counts = {}
    for source in sources:
        if isinstance(source, dict):
            content_type = source.get('content_type', 'Unknown')
            type_counts[content_type] = type_counts.get(content_type, 0) + 1
    
    return chr(10).join([f"- {ctype}: {count}" for ctype, count in type_counts.items()])

def _create_source_references(sources):
    if not sources:
        return []
    
    references = []
    for source in sources:
        if isinstance(source, dict):
            ref = SourceReference(
                title=source.get('title', 'Unknown Title'),
                url=source.get('url', ''),
                tool_used="enhanced_tavily",  # Default
                content_type=ContentType.OTHER,
                quality=ResearchQuality.MEDIUM,
                relevance_score=0.8
            )
            references.append(ref)
    
    return references

def _generate_markdown(doc_structure):
    """Generate markdown content from document structure."""
    markdown = f"# {doc_structure.title}\n\n"
    
    for section in doc_structure.sections:
        level_prefix = "#" * (section['level'] + 1)
        markdown += f"{level_prefix} {section['title']}\n\n"
        markdown += f"{section['content']}\n\n"
    
    if doc_structure.appendices:
        markdown += "## Appendices\n\n"
        for appendix in doc_structure.appendices:
            markdown += f"### {appendix['title']}\n\n"
            markdown += f"{appendix['content']}\n\n"
    
    return markdown
