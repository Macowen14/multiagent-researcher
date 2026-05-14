"""
Enhanced research tools with Pydantic-guided structured outputs.
These tools provide deeper research capabilities with quality assurance.
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional, Annotated
from datetime import datetime
from langchain_core.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel

from models.research_schemas import (
    ResearchStrategy,
    ToolRecommendation,
    ResearchPlan,
    ResearchReport,
    SourceReference,
    ResearchFinding,
    ResearchGap,
    AnalysisReport,
    ValidationResponse,
    SearchToolType,
    ContentType,
    ResearchQuality
)

logger = logging.getLogger(__name__)


def _stringify_source_content(source: Dict[str, Any]) -> str:
    content = source.get("content") or source.get("snippet") or source.get("summary")
    if content:
        return str(content)

    useful_fields = {
        key: value
        for key, value in source.items()
        if key not in {"url", "html_url"} and value not in (None, "", [])
    }
    return json.dumps(useful_fields, ensure_ascii=False)


def _normalize_sources(sources_data: Any) -> List[Dict[str, Any]]:
    """Accept flat or grouped source JSON and return title/url/content records."""
    if isinstance(sources_data, dict):
        sources_data = (
            sources_data.get("sources") or sources_data.get("details") or [sources_data]
        )

    normalized = []
    for source in sources_data or []:
        if not isinstance(source, dict):
            continue

        source_label = source.get("source") or source.get("type") or source.get("tool")
        details = source.get("details")
        detail_items = details if isinstance(details, list) else [source]

        for detail in detail_items:
            if not isinstance(detail, dict):
                continue

            title = (
                detail.get("title")
                or detail.get("repo")
                or detail.get("question")
                or detail.get("name")
                or source_label
                or "Unknown Title"
            )
            url = detail.get("url") or detail.get("html_url") or detail.get("link") or ""
            normalized.append(
                {
                    **detail,
                    "title": str(title),
                    "url": str(url),
                    "content": _stringify_source_content(detail),
                    "source": source_label or detail.get("source") or detail.get("type"),
                }
            )

    return normalized


@tool("create_research_strategy")
def create_research_strategy(
    query: Annotated[str, "The user's research query"],
    query_type: Annotated[str, "Type of query: technical, academic, general, news"] = "general"
) -> str:
    """Create a comprehensive research strategy using Pydantic models."""
    logger.info(f"Creating research strategy for query: '{query}' (type: {query_type})")
    
    try:
        # Analyze query and recommend tools
        tool_recommendations = []
        
        if query_type.lower() in ["technical", "programming"]:
            tool_recommendations.extend([
                ToolRecommendation(
                    tool=SearchToolType.GITHUB,
                    reason="For code repositories and implementations",
                    query_suggestion=query,
                    expected_outcome="Real code examples and implementations",
                    priority=1
                ),
                ToolRecommendation(
                    tool=SearchToolType.STACKOVERFLOW,
                    reason="For programming questions and solutions",
                    query_suggestion=query,
                    expected_outcome="Common issues and solutions",
                    priority=2
                ),
                ToolRecommendation(
                    tool=SearchToolType.ARXIV,
                    reason="For academic papers on technical topics",
                    query_suggestion=query,
                    expected_outcome="Research papers and technical analysis",
                    priority=3
                )
            ])
        
        if query_type.lower() in ["academic", "research"]:
            tool_recommendations.extend([
                ToolRecommendation(
                    tool=SearchToolType.ARXIV,
                    reason="For academic papers and research",
                    query_suggestion=query,
                    expected_outcome="Peer-reviewed research papers",
                    priority=1
                ),
                ToolRecommendation(
                    tool=SearchToolType.WIKIPEDIA,
                    reason="For comprehensive overviews",
                    query_suggestion=query,
                    expected_outcome="Background information and context",
                    priority=2
                )
            ])
        
        # Always include general search tools
        tool_recommendations.extend([
            ToolRecommendation(
                tool=SearchToolType.ENHANCED_TAVILY,
                reason="For comprehensive web search",
                query_suggestion=query,
                expected_outcome="Recent information and diverse sources",
                priority=4
            ),
            ToolRecommendation(
                tool=SearchToolType.DUCKDUCKGO,
                reason="For additional search perspectives",
                query_suggestion=query,
                expected_outcome="Alternative search results",
                priority=5
            )
        ])
        
        # Create research strategy
        strategy = ResearchStrategy(
            query_analysis=f"Query identified as {query_type} type research request",
            tool_recommendations=tool_recommendations,
            search_phases=[
                "Initial broad search using multiple tools",
                "Deep dive into specific technical/academic sources",
                "Cross-reference and validation of findings",
                "Gap identification and additional research"
            ],
            quality_checks=[
                "Source reliability assessment",
                "Information cross-validation",
                "Relevance scoring",
                "Completeness evaluation"
            ],
            success_metrics=[
                "Multiple high-quality sources found",
                "Information cross-validated",
                "Comprehensive coverage of topic",
                "Clear actionable insights"
            ]
        )
        
        return f"""## Research Strategy Created

**Query Analysis:** {strategy.query_analysis}

**Recommended Tools:**
{chr(10).join([f"- {rec.tool}: {rec.reason} (Priority: {rec.priority})" for rec in strategy.tool_recommendations])}

**Research Phases:**
{chr(10).join([f"{i+1}. {phase}" for i, phase in enumerate(strategy.search_phases)])}

**Quality Checks:**
{chr(10).join([f"- {check}" for check in strategy.quality_checks])}

**Success Metrics:**
{chr(10).join([f"- {metric}" for metric in strategy.success_metrics])}

Strategy saved with timestamp: {datetime.now().isoformat()}
"""
        
    except Exception as e:
        logger.error(f"Error creating research strategy: {e}")
        return f"Error creating research strategy: {str(e)}"


@tool("validate_research_sources")
def validate_research_sources(
    sources_json: Annotated[str, "JSON string containing research sources to validate"]
) -> str:
    """Validate research sources for quality and reliability."""
    logger.info("Validating research sources")
    
    try:
        sources_data = _normalize_sources(json.loads(sources_json))
        validation_response = ValidationResponse(
            is_valid=True,
            validation_score=0.0,
            issues_found=[],
            suggestions=[],
            passed_checks=[],
            failed_checks=[]
        )
        
        total_score = 0
        source_count = len(sources_data)
        
        for i, source in enumerate(sources_data):
            source_score = 0
            issues = []
            
            # Check for required fields
            if not source.get('title'):
                issues.append("Missing title")
                validation_response.issues_found.append(f"Source {i+1}: Missing title")
            else:
                source_score += 0.2
                validation_response.passed_checks.append(f"Source {i+1}: Has title")
            
            if not source.get('url'):
                issues.append("Missing URL")
                validation_response.issues_found.append(f"Source {i+1}: Missing URL")
            else:
                source_score += 0.2
                validation_response.passed_checks.append(f"Source {i+1}: Has URL")
            
            if not source.get('content'):
                issues.append("Missing content")
                validation_response.issues_found.append(f"Source {i+1}: Missing content")
            else:
                source_score += 0.3
                validation_response.passed_checks.append(f"Source {i+1}: Has content")
            
            # Content quality checks
            content = source.get('content', '')
            if len(content) < 100:
                issues.append("Content too short")
                validation_response.issues_found.append(f"Source {i+1}: Content too short")
            else:
                source_score += 0.3
                validation_response.passed_checks.append(f"Source {i+1}: Adequate content length")
            
            total_score += source_score
            
            if issues:
                validation_response.failed_checks.extend([f"Source {i+1}: {issue}" for issue in issues])
        
        validation_response.validation_score = total_score / source_count if source_count > 0 else 0
        validation_response.is_valid = validation_response.validation_score >= 0.7
        
        if not validation_response.is_valid:
            validation_response.suggestions.extend([
                "Add more detailed content to sources",
                "Ensure all sources have titles and URLs",
                "Include more diverse source types",
                "Add recent sources for current information"
            ])
        
        return f"""## Source Validation Results

**Overall Valid:** {'✅ Yes' if validation_response.is_valid else '❌ No'}
**Validation Score:** {validation_response.validation_score:.2f}/1.00

**Passed Checks ({len(validation_response.passed_checks)}):**
{chr(10).join([f"- {check}" for check in validation_response.passed_checks])}

**Failed Checks ({len(validation_response.failed_checks)}):**
{chr(10).join([f"- {check}" for check in validation_response.failed_checks])}

**Issues Found:**
{chr(10).join([f"- {issue}" for issue in validation_response.issues_found])}

**Suggestions:**
{chr(10).join([f"- {suggestion}" for suggestion in validation_response.suggestions])}
"""
        
    except json.JSONDecodeError:
        return "Error: Invalid JSON format for sources"
    except Exception as e:
        logger.error(f"Error validating sources: {e}")
        return f"Error validating sources: {str(e)}"


@tool("create_research_report")
def create_research_report(
    query: Annotated[str, "Original research query"],
    sources_json: Annotated[str, "JSON string containing research sources"],
    findings: Annotated[str, "Key findings from research"],
    methodology: Annotated[str, "Research methodology used"]
) -> str:
    """Create a structured research report using Pydantic models."""
    logger.info(f"Creating research report for query: '{query}'")
    
    try:
        sources_data = _normalize_sources(json.loads(sources_json))
        
        # Create source references
        source_references = []
        for source in sources_data:
            # Determine content type and quality based on content and URL
            url = source.get('url', '').lower()
            content = source.get('content', '').lower()
            
            content_type = ContentType.OTHER
            if 'github.com' in url:
                content_type = ContentType.CODE_REPOSITORY
            elif 'stackoverflow.com' in url:
                content_type = ContentType.TECHNICAL_ANSWER
            elif 'arxiv.org' in url:
                content_type = ContentType.ACADEMIC_PAPER
            elif 'wikipedia.org' in url:
                content_type = ContentType.ENCYCLOPEDIC_ARTICLE
            elif 'documentation' in content or 'docs' in url:
                content_type = ContentType.DOCUMENTATION
            
            # Simple quality assessment based on content length and source type
            quality = ResearchQuality.MEDIUM
            if content_type in [ContentType.ACADEMIC_PAPER, ContentType.DOCUMENTATION]:
                quality = ResearchQuality.HIGH
            elif len(source.get('content', '')) < 100:
                quality = ResearchQuality.LOW
            
            source_ref = SourceReference(
                title=source.get('title', 'Unknown Title'),
                url=source.get('url', ''),
                tool_used=SearchToolType.ENHANCED_TAVILY,  # Default, should be determined from context
                content_type=content_type,
                quality=quality,
                relevance_score=0.8,  # Default, should be calculated
                key_points=[],  # Would be extracted from content
                limitations=[]  # Would be analyzed
            )
            source_references.append(source_ref)
        
        # Create research findings (simplified for this example)
        research_findings = [
            ResearchFinding(
                claim=finding.strip(),
                evidence=[f"Based on {len(source_references)} sources"],
                confidence_level=0.8,
                supporting_sources=[src.url for src in source_references],
                contradictory_sources=[],
                context="Research context",
                implications=[]
            )
            for finding in findings.split('\n') if finding.strip()
        ]
        
        # Create research plan (simplified)
        research_plan = ResearchPlan(
            primary_query=query,
            sub_queries=[],
            tools_to_use=[SearchToolType.ENHANCED_TAVILY],
            search_strategy=methodology,
            success_criteria=["Comprehensive coverage", "Multiple sources"],
            expected_outcomes=[]
        )
        
        # Create the report
        report = ResearchReport(
            query=query,
            research_plan=research_plan,
            sources=source_references,
            findings=research_findings,
            gaps=[],
            summary=f"Research conducted on {query} using {len(source_references)} sources",
            methodology=methodology,
            limitations=["Automated quality assessment", "Limited source diversity analysis"],
            next_steps=["Manual review of sources", "Additional research if needed"],
            confidence_overall=0.8
        )
        
        return f"""# Research Report

## Query
{report.query}

## Executive Summary
{report.summary}

## Methodology
{report.methodology}

## Sources ({len(report.sources)})
{chr(10).join([f"- **{src.title}** ({src.content_type.value}) - {src.url}" for src in report.sources])}

## Key Findings
{chr(10).join([f"### {finding.claim}{chr(10)}{finding.context}{chr(10)}**Confidence:** {finding.confidence_level:.1f}" for finding in report.findings])}

## Limitations
{chr(10).join([f"- {limitation}" for limitation in report.limitations])}

## Next Steps
{chr(10).join([f"- {step}" for step in report.next_steps])}

## Overall Confidence
{report.confidence_overall:.1f}/1.0

**Report Generated:** {report.completion_timestamp}
"""
        
    except json.JSONDecodeError:
        return "Error: Invalid JSON format for sources"
    except Exception as e:
        logger.error(f"Error creating research report: {e}")
        return f"Error creating research report: {str(e)}"


@tool("analyze_research_completeness")
def analyze_research_completeness(
    research_report: Annotated[str, "Research report content to analyze"],
    expected_topics: Annotated[str, "Comma-separated list of expected topics"]
) -> str:
    """Analyze research completeness and identify gaps."""
    logger.info("Analyzing research completeness")
    
    try:
        topics = [topic.strip() for topic in expected_topics.split(',')]
        
        # Simple coverage analysis
        coverage_scores = {}
        for topic in topics:
            topic_lower = topic.lower()
            if topic_lower in research_report.lower():
                # Count occurrences as a simple coverage metric
                count = research_report.lower().count(topic_lower)
                coverage_scores[topic] = min(count / 5, 1.0)  # Normalize to 0-1
            else:
                coverage_scores[topic] = 0.0
        
        overall_coverage = sum(coverage_scores.values()) / len(topics) if topics else 0
        
        # Identify gaps
        gaps = []
        for topic, score in coverage_scores.items():
            if score < 0.5:
                gaps.append(ResearchGap(
                    description=f"Insufficient coverage of {topic}",
                    importance=f"Topic {topic} is crucial for comprehensive understanding",
                    suggested_research=[
                        f"Search specifically for {topic} using specialized tools",
                        f"Look for academic papers on {topic}",
                        f"Find practical examples of {topic}"
                    ],
                    tools_to_use=[SearchToolType.ARXIV, SearchToolType.ENHANCED_TAVILY]
                ))
        
        # Create analysis report
        analysis = AnalysisReport(
            research_report=None,  # Would be parsed from input
            quality_assessment=f"Coverage score: {overall_coverage:.2f}",
            completeness_score=overall_coverage,
            reliability_score=0.8,  # Would be calculated based on sources
            recommendations=[
                "Focus on topics with low coverage scores",
                "Add more diverse source types",
                "Include recent sources for current information"
            ],
            approval_status="NEEDS_IMPROVEMENT" if overall_coverage < 0.7 else "APPROVED",
            additional_research_needed=gaps
        )
        
        return f"""## Research Completeness Analysis

**Overall Coverage:** {overall_coverage:.2f}/1.00
**Status:** {analysis.approval_status}

### Topic Coverage
{chr(10).join([f"- **{topic}:** {score:.2f}" for topic, score in coverage_scores.items()])}

### Identified Gaps
{chr(10).join([f"- **{gap.description}**{chr(10)}  Importance: {gap.importance}{chr(10)}  Suggested: {', '.join(gap.suggested_research[:2])}" for gap in gaps])}

### Recommendations
{chr(10).join([f"- {rec}" for rec in analysis.recommendations])}

### Approval Status
{analysis.approval_status}
"""
        
    except Exception as e:
        logger.error(f"Error analyzing completeness: {e}")
        return f"Error analyzing completeness: {str(e)}"
