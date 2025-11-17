"""
Recommendation engine for personalized mental health resource recommendations.
"""

from typing import List, Dict, Optional, Any
import logging
from dataclasses import dataclass

from src.recommendations.resource_catalog import (
    ResourceCatalog,
    Resource,
    ResourceType,
    Urgency
)

logger = logging.getLogger(__name__)


@dataclass
class IndividualProfile:
    """Profile information for personalization"""
    anonymized_id: str
    risk_level: str  # low, moderate, high, critical
    contributing_factors: List[str]
    age_group: Optional[str] = None  # youth, adult, senior
    has_therapy_history: bool = False
    has_medication_history: bool = False
    has_support_system: bool = True
    prefers_online: bool = False
    prefers_group: bool = False
    specific_conditions: List[str] = None  # depression, anxiety, etc.
    
    def __post_init__(self):
        if self.specific_conditions is None:
            self.specific_conditions = []


class RecommendationEngine:
    """
    Engine for generating personalized mental health resource recommendations.
    
    Matches resources to individuals based on risk level and profile features.
    """
    
    def __init__(self, catalog: Optional[ResourceCatalog] = None):
        """
        Initialize recommendation engine.
        
        Args:
            catalog: Resource catalog (creates default if not provided)
        """
        self.catalog = catalog or ResourceCatalog()
        logger.info("RecommendationEngine initialized")
    
    def get_recommendations(
        self,
        profile: IndividualProfile,
        max_recommendations: int = 5
    ) -> List[Resource]:
        """
        Generate personalized resource recommendations.
        
        Args:
            profile: Individual profile with risk level and features
            max_recommendations: Maximum number of recommendations to return
        
        Returns:
            List of recommended resources sorted by relevance
        """
        logger.info(
            f"Generating recommendations for risk_level={profile.risk_level}, "
            f"anonymized_id={profile.anonymized_id}"
        )
        
        # Get base recommendations for risk level
        candidates = self.catalog.filter_by_risk_level(profile.risk_level)
        
        # Score and rank candidates based on profile
        scored_resources = []
        for resource in candidates:
            score = self._calculate_relevance_score(resource, profile)
            scored_resources.append((resource, score))
        
        # Sort by score (descending)
        scored_resources.sort(key=lambda x: x[1], reverse=True)
        
        # Return top recommendations
        recommendations = [r for r, _ in scored_resources[:max_recommendations]]
        
        logger.info(
            f"Generated {len(recommendations)} recommendations "
            f"for {profile.anonymized_id}"
        )
        
        return recommendations
    
    def _calculate_relevance_score(
        self,
        resource: Resource,
        profile: IndividualProfile
    ) -> float:
        """
        Calculate relevance score for a resource given individual profile.
        
        Higher scores indicate better match.
        
        Args:
            resource: Resource to score
            profile: Individual profile
        
        Returns:
            Relevance score (0-100)
        """
        score = float(resource.priority)  # Start with base priority
        
        # Boost for exact risk level match
        if profile.risk_level in resource.risk_levels:
            score += 20
        
        # Personalization based on profile features
        
        # Age-appropriate resources
        if profile.age_group:
            if profile.age_group == "youth" and "youth" in resource.tags:
                score += 15
            elif profile.age_group == "senior" and "senior" in resource.tags:
                score += 15
        
        # Therapy history
        if profile.has_therapy_history:
            if resource.resource_type == ResourceType.THERAPY:
                score += 10  # Familiar with therapy
        else:
            # First-time therapy seekers might prefer less intensive options
            if resource.resource_type in [ResourceType.SUPPORT_GROUP, ResourceType.SELF_HELP]:
                score += 5
        
        # Medication history
        if profile.has_medication_history:
            if resource.resource_type == ResourceType.MEDICATION:
                score += 10
        
        # Support system
        if not profile.has_support_system:
            # Prioritize support groups and community resources
            if resource.resource_type in [ResourceType.SUPPORT_GROUP, ResourceType.COMMUNITY]:
                score += 15
        
        # Online preference
        if profile.prefers_online:
            if "online" in resource.tags or "virtual" in resource.tags or "app" in resource.tags:
                score += 20
        
        # Group preference
        if profile.prefers_group:
            if resource.resource_type == ResourceType.SUPPORT_GROUP:
                score += 15
        elif profile.prefers_group is False:
            # Prefers individual resources
            if resource.resource_type == ResourceType.THERAPY:
                score += 10
        
        # Specific conditions matching
        if profile.specific_conditions:
            for condition in profile.specific_conditions:
                if condition.lower() in resource.tags:
                    score += 25  # Strong boost for condition match
                if condition.lower() in resource.description.lower():
                    score += 10
        
        # Contributing factors matching
        if profile.contributing_factors:
            for factor in profile.contributing_factors:
                factor_lower = factor.lower()
                # Check if factor relates to resource tags
                if any(factor_lower in tag for tag in resource.tags):
                    score += 15
                # Check description
                if factor_lower in resource.description.lower():
                    score += 5
        
        return score
    
    def get_recommendations_by_risk_level(
        self,
        risk_level: str,
        contributing_factors: Optional[List[str]] = None,
        max_recommendations: int = 5
    ) -> List[Resource]:
        """
        Get recommendations based on risk level only (simplified interface).
        
        Args:
            risk_level: Risk level (low, moderate, high, critical)
            contributing_factors: Optional list of contributing factors
            max_recommendations: Maximum number of recommendations
        
        Returns:
            List of recommended resources
        """
        profile = IndividualProfile(
            anonymized_id="unknown",
            risk_level=risk_level,
            contributing_factors=contributing_factors or []
        )
        
        return self.get_recommendations(profile, max_recommendations)
    
    def get_recommendations_dict(
        self,
        profile: IndividualProfile,
        max_recommendations: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations as dictionaries (for API responses).
        
        Args:
            profile: Individual profile
            max_recommendations: Maximum number of recommendations
        
        Returns:
            List of resource dictionaries
        """
        resources = self.get_recommendations(profile, max_recommendations)
        
        return [
            {
                "resource_type": resource.resource_type.value,
                "name": resource.name,
                "description": resource.description,
                "contact_info": resource.contact_info,
                "urgency": resource.urgency.value,
                "eligibility_criteria": resource.eligibility_criteria
            }
            for resource in resources
        ]
    
    def filter_by_urgency(
        self,
        resources: List[Resource],
        urgency: Urgency
    ) -> List[Resource]:
        """Filter resources by urgency level"""
        return [r for r in resources if r.urgency == urgency]
    
    def get_immediate_resources(self, risk_level: str) -> List[Resource]:
        """Get immediate/crisis resources for a risk level"""
        resources = self.catalog.filter_by_risk_level(risk_level)
        return self.filter_by_urgency(resources, Urgency.IMMEDIATE)
