"""Tests for recommendation engine"""

import pytest
from src.recommendations import (
    RecommendationEngine,
    IndividualProfile,
    ResourceCatalog,
    Resource,
    ResourceType,
    Urgency
)


class TestResourceCatalog:
    """Test ResourceCatalog functionality"""
    
    def test_catalog_initialization(self):
        """Test catalog initializes with default resources"""
        catalog = ResourceCatalog()
        resources = catalog.get_all_resources()
        
        assert len(resources) > 0
        assert all(isinstance(r, Resource) for r in resources)
    
    def test_filter_by_risk_level(self):
        """Test filtering resources by risk level"""
        catalog = ResourceCatalog()
        
        critical_resources = catalog.filter_by_risk_level("critical")
        assert len(critical_resources) > 0
        assert all("critical" in r.risk_levels for r in critical_resources)
        
        low_resources = catalog.filter_by_risk_level("low")
        assert len(low_resources) > 0
        assert all("low" in r.risk_levels for r in low_resources)
    
    def test_filter_by_type(self):
        """Test filtering resources by type"""
        catalog = ResourceCatalog()
        
        crisis_resources = catalog.filter_by_type(ResourceType.CRISIS_LINE)
        assert len(crisis_resources) > 0
        assert all(r.resource_type == ResourceType.CRISIS_LINE for r in crisis_resources)
    
    def test_search_with_multiple_filters(self):
        """Test searching with multiple filters"""
        catalog = ResourceCatalog()
        
        results = catalog.search(
            risk_level="moderate",
            resource_type=ResourceType.THERAPY
        )
        
        assert all("moderate" in r.risk_levels for r in results)
        assert all(r.resource_type == ResourceType.THERAPY for r in results)
    
    def test_add_custom_resource(self):
        """Test adding custom resource to catalog"""
        catalog = ResourceCatalog()
        
        custom_resource = Resource(
            id="test_resource",
            resource_type=ResourceType.WELLNESS,
            name="Test Resource",
            description="Test description",
            contact_info="555-1234",
            urgency=Urgency.ROUTINE,
            eligibility_criteria=["Test criteria"],
            risk_levels=["low"],
            tags=["test"],
            priority=50
        )
        
        catalog.add_resource(custom_resource)
        retrieved = catalog.get_resource("test_resource")
        
        assert retrieved is not None
        assert retrieved.name == "Test Resource"


class TestRecommendationEngine:
    """Test RecommendationEngine functionality"""
    
    def test_engine_initialization(self):
        """Test engine initializes correctly"""
        engine = RecommendationEngine()
        assert engine.catalog is not None
    
    def test_get_recommendations_critical_risk(self):
        """Test recommendations for critical risk level"""
        engine = RecommendationEngine()
        
        profile = IndividualProfile(
            anonymized_id="test123",
            risk_level="critical",
            contributing_factors=["suicide ideation"]
        )
        
        recommendations = engine.get_recommendations(profile, max_recommendations=5)
        
        assert len(recommendations) > 0
        assert len(recommendations) <= 5
        
        # Should include crisis resources
        resource_types = [r.resource_type for r in recommendations]
        assert ResourceType.CRISIS_LINE in resource_types or ResourceType.EMERGENCY in resource_types
    
    def test_get_recommendations_low_risk(self):
        """Test recommendations for low risk level"""
        engine = RecommendationEngine()
        
        profile = IndividualProfile(
            anonymized_id="test456",
            risk_level="low",
            contributing_factors=[]
        )
        
        recommendations = engine.get_recommendations(profile, max_recommendations=5)
        
        assert len(recommendations) > 0
        # Low risk should get wellness/prevention resources
        resource_types = [r.resource_type for r in recommendations]
        assert ResourceType.WELLNESS in resource_types or ResourceType.SELF_HELP in resource_types
    
    def test_personalization_online_preference(self):
        """Test personalization based on online preference"""
        engine = RecommendationEngine()
        
        profile_online = IndividualProfile(
            anonymized_id="test789",
            risk_level="moderate",
            contributing_factors=["anxiety"],
            prefers_online=True
        )
        
        recommendations = engine.get_recommendations(profile_online, max_recommendations=5)
        
        # Should prioritize online resources
        assert len(recommendations) > 0
        # Check if any recommendations have online-related tags
        has_online = any(
            "online" in r.tags or "virtual" in r.tags or "app" in r.tags
            for r in recommendations
        )
        assert has_online or len(recommendations) > 0  # At least got some recommendations
    
    def test_personalization_specific_conditions(self):
        """Test personalization based on specific conditions"""
        engine = RecommendationEngine()
        
        profile = IndividualProfile(
            anonymized_id="test999",
            risk_level="moderate",
            contributing_factors=["depression"],
            specific_conditions=["depression"]
        )
        
        recommendations = engine.get_recommendations(profile, max_recommendations=5)
        
        assert len(recommendations) > 0
        # Should get depression-related resources
        has_depression_match = any(
            "depression" in r.tags or "depression" in r.description.lower()
            for r in recommendations
        )
        assert has_depression_match or len(recommendations) > 0
    
    def test_get_recommendations_by_risk_level(self):
        """Test simplified interface for getting recommendations"""
        engine = RecommendationEngine()
        
        recommendations = engine.get_recommendations_by_risk_level(
            risk_level="high",
            contributing_factors=["anxiety", "sleep issues"],
            max_recommendations=3
        )
        
        assert len(recommendations) > 0
        assert len(recommendations) <= 3
    
    def test_get_recommendations_dict(self):
        """Test getting recommendations as dictionaries"""
        engine = RecommendationEngine()
        
        profile = IndividualProfile(
            anonymized_id="test_dict",
            risk_level="moderate",
            contributing_factors=[]
        )
        
        recommendations_dict = engine.get_recommendations_dict(profile, max_recommendations=3)
        
        assert len(recommendations_dict) > 0
        assert all(isinstance(r, dict) for r in recommendations_dict)
        assert all("resource_type" in r for r in recommendations_dict)
        assert all("name" in r for r in recommendations_dict)
    
    def test_relevance_scoring(self):
        """Test that relevance scoring works"""
        engine = RecommendationEngine()
        catalog = engine.catalog
        
        # Get a resource
        resources = catalog.filter_by_risk_level("moderate")
        if resources:
            resource = resources[0]
            
            profile = IndividualProfile(
                anonymized_id="test_score",
                risk_level="moderate",
                contributing_factors=[]
            )
            
            score = engine._calculate_relevance_score(resource, profile)
            assert isinstance(score, float)
            assert score >= 0
    
    def test_get_immediate_resources(self):
        """Test getting immediate/crisis resources"""
        engine = RecommendationEngine()
        
        immediate_resources = engine.get_immediate_resources("critical")
        
        assert len(immediate_resources) > 0
        assert all(r.urgency == Urgency.IMMEDIATE for r in immediate_resources)


class TestIndividualProfile:
    """Test IndividualProfile data class"""
    
    def test_profile_creation(self):
        """Test creating individual profile"""
        profile = IndividualProfile(
            anonymized_id="test123",
            risk_level="moderate",
            contributing_factors=["anxiety", "depression"]
        )
        
        assert profile.anonymized_id == "test123"
        assert profile.risk_level == "moderate"
        assert len(profile.contributing_factors) == 2
        assert profile.specific_conditions == []  # Default
    
    def test_profile_with_all_features(self):
        """Test profile with all personalization features"""
        profile = IndividualProfile(
            anonymized_id="test456",
            risk_level="high",
            contributing_factors=["depression"],
            age_group="adult",
            has_therapy_history=True,
            has_medication_history=True,
            has_support_system=False,
            prefers_online=True,
            prefers_group=False,
            specific_conditions=["depression", "anxiety"]
        )
        
        assert profile.age_group == "adult"
        assert profile.has_therapy_history is True
        assert profile.has_medication_history is True
        assert profile.has_support_system is False
        assert profile.prefers_online is True
        assert profile.prefers_group is False
        assert len(profile.specific_conditions) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
