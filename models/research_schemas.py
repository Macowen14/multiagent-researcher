"""
Pydantic models for structured responses and research validation.
These models guide the AI agents to provide consistent, well-structured outputs.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime


class SearchToolType(str, Enum):
    """Enumeration of available search tools."""
    TAVILY = "tavily"
    ENHANCED_TAVILY = "enhanced_tavily"
    DUCKDUCKGO = "duckduckgo"
    WIKIPEDIA = "wikipedia"
    ARXIV = "arxiv"
    GITHUB = "github"
    STACKOVERFLOW = "stackoverflow"
    WEB_SCRAPING = "web_scraping"


class ResearchQuality(str, Enum):
    """Quality assessment for research sources."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


class ContentType(str, Enum):
    """Types of content in research results."""
    ACADEMIC_PAPER = "academic_paper"
    CODE_REPOSITORY = "code_repository"
    TECHNICAL_ANSWER = "technical_answer"
    ENCYCLOPEDIC_ARTICLE = "encyclopedic_article"
    NEWS_ARTICLE = "news_article"
    DOCUMENTATION = "documentation"
    FORUM_DISCUSSION = "forum_discussion"
    OTHER = "other"


class SourceReference(BaseModel):
    """Structured reference for a research source."""
    title: str = Field(..., description="Title of the source")
    url: str = Field(..., description="URL of the source")
    tool_used: SearchToolType = Field(..., description="Which search tool found this source")
    content_type: ContentType = Field(..., description="Type of content")
    quality: ResearchQuality = Field(..., description="Quality assessment of the source")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score (0-1)")
    date_accessed: str = Field(default_factory=lambda: datetime.now().isoformat())
    key_points: List[str] = Field(default_factory=list, description="Key points from this source")
    limitations: List[str] = Field(default_factory=list, description="Known limitations or biases")
    additional_metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra metadata")

    @validator('relevance_score')
    def validate_relevance_score(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Relevance score must be between 0 and 1')
        return v


class ResearchFinding(BaseModel):
    """Individual research finding with evidence."""
    claim: str = Field(..., description="The main claim or finding")
    evidence: List[str] = Field(..., description="Supporting evidence from sources")
    confidence_level: float = Field(..., ge=0.0, le=1.0, description="Confidence in this finding")
    supporting_sources: List[str] = Field(..., description="Source URLs that support this finding")
    contradictory_sources: List[str] = Field(default_factory=list, description="Sources that contradict this finding")
    context: str = Field(..., description="Context for this finding")
    implications: List[str] = Field(default_factory=list, description="Implications of this finding")


class ResearchGap(BaseModel):
    """Identified gap in current research."""
    description: str = Field(..., description="Description of the research gap")
    importance: str = Field(..., description="Why this gap is important")
    suggested_research: List[str] = Field(..., description="Suggested research to fill this gap")
    tools_to_use: List[SearchToolType] = Field(..., description="Which tools to use for further research")


class ResearchPlan(BaseModel):
    """Structured plan for research execution."""
    primary_query: str = Field(..., description="Main research question")
    sub_queries: List[str] = Field(default_factory=list, description="Sub-questions to explore")
    tools_to_use: List[SearchToolType] = Field(..., description="Tools to use for research")
    search_strategy: str = Field(..., description="Overall search strategy")
    success_criteria: List[str] = Field(..., description="What constitutes successful research")
    expected_outcomes: List[str] = Field(default_factory=list, description="Expected research outcomes")


class ResearchReport(BaseModel):
    """Comprehensive research report structure."""
    query: str = Field(..., description="Original research query")
    research_plan: ResearchPlan = Field(..., description="Plan that guided the research")
    sources: List[SourceReference] = Field(..., description="All sources consulted")
    findings: List[ResearchFinding] = Field(..., description="Key research findings")
    gaps: List[ResearchGap] = Field(default_factory=list, description="Identified research gaps")
    summary: str = Field(..., description="Executive summary of findings")
    methodology: str = Field(..., description="Research methodology used")
    limitations: List[str] = Field(default_factory=list, description="Limitations of the research")
    next_steps: List[str] = Field(default_factory=list, description="Suggested next steps")
    confidence_overall: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in findings")
    completion_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class AnalysisReport(BaseModel):
    """Structured analysis report for the analyst agent."""
    research_report: ResearchReport = Field(..., description="Research report to analyze")
    quality_assessment: str = Field(..., description="Overall quality assessment")
    completeness_score: float = Field(..., ge=0.0, le=1.0, description="How complete the research is")
    reliability_score: float = Field(..., ge=0.0, le=1.0, description="Reliability of sources")
    recommendations: List[str] = Field(..., description="Recommendations for improvement")
    approval_status: str = Field(..., description="Whether to proceed to writing")
    additional_research_needed: List[ResearchGap] = Field(default_factory=list, description="Additional research needed")


class DocumentStructure(BaseModel):
    """Structure for the final document."""
    title: str = Field(..., description="Document title")
    sections: List[Dict[str, Any]] = Field(..., description="Document sections with content")
    references: List[SourceReference] = Field(..., description="All references used")
    appendices: List[Dict[str, Any]] = Field(default_factory=list, description="Additional appendices")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class ToolRecommendation(BaseModel):
    """Recommendation for tool usage."""
    tool: SearchToolType = Field(..., description="Recommended tool")
    reason: str = Field(..., description="Why this tool is recommended")
    query_suggestion: str = Field(..., description="Suggested query for this tool")
    expected_outcome: str = Field(..., description="What we expect to find with this tool")
    priority: int = Field(..., ge=1, le=5, description="Priority (1=highest, 5=lowest)")


class ResearchStrategy(BaseModel):
    """Comprehensive research strategy."""
    query_analysis: str = Field(..., description="Analysis of the user's query")
    tool_recommendations: List[ToolRecommendation] = Field(..., description="Recommended tools and usage")
    search_phases: List[str] = Field(..., description="Phases of research")
    quality_checks: List[str] = Field(..., description="Quality checks to perform")
    success_metrics: List[str] = Field(..., description="Metrics for research success")


class ValidationResponse(BaseModel):
    """Validation response for quality checking."""
    is_valid: bool = Field(..., description="Whether the content passes validation")
    validation_score: float = Field(..., ge=0.0, le=1.0, description="Overall validation score")
    issues_found: List[str] = Field(default_factory=list, description="Issues found during validation")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for improvement")
    passed_checks: List[str] = Field(default_factory=list, description="Checks that passed")
    failed_checks: List[str] = Field(default_factory=list, description="Checks that failed")
