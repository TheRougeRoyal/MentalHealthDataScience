"""
Resource catalog for managing mental health resources.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum


class ResourceType(str, Enum):
    """Types of mental health resources"""
    CRISIS_LINE = "crisis_line"
    EMERGENCY = "emergency"
    THERAPY = "therapy"
    SUPPORT_GROUP = "support_group"
    WELLNESS = "wellness"
    MEDICATION = "medication"
    SELF_HELP = "self_help"
    COMMUNITY = "community"


class Urgency(str, Enum):
    """Urgency levels for resources"""
    IMMEDIATE = "immediate"
    SOON = "soon"
    ROUTINE = "routine"


@dataclass
class Resource:
    """Mental health resource"""
    id: str
    resource_type: ResourceType
    name: str
    description: str
    contact_info: str
    urgency: Urgency
    eligibility_criteria: List[str]
    risk_levels: List[str]  # low, moderate, high, critical
    tags: List[str] = field(default_factory=list)
    priority: int = 0  # Higher priority shown first


class ResourceCatalog:
    """
    Catalog of mental health resources with filtering and retrieval capabilities.
    """
    
    def __init__(self):
        """Initialize resource catalog with default resources"""
        self.resources: Dict[str, Resource] = {}
        self._initialize_default_resources()
    
    def _initialize_default_resources(self):
        """Initialize catalog with default mental health resources"""
        
        # Crisis resources
        self.add_resource(Resource(
            id="crisis_line_988",
            resource_type=ResourceType.CRISIS_LINE,
            name="National Suicide Prevention Lifeline",
            description="24/7 crisis support and suicide prevention",
            contact_info="988",
            urgency=Urgency.IMMEDIATE,
            eligibility_criteria=["Available to all"],
            risk_levels=["critical"],
            tags=["suicide", "crisis", "24/7"],
            priority=100
        ))
        
        self.add_resource(Resource(
            id="emergency_911",
            resource_type=ResourceType.EMERGENCY,
            name="Emergency Services",
            description="Immediate emergency response",
            contact_info="911",
            urgency=Urgency.IMMEDIATE,
            eligibility_criteria=["Emergency situations"],
            risk_levels=["critical"],
            tags=["emergency", "immediate"],
            priority=100
        ))
        
        self.add_resource(Resource(
            id="crisis_text_line",
            resource_type=ResourceType.CRISIS_LINE,
            name="Crisis Text Line",
            description="Text-based crisis support available 24/7",
            contact_info="Text HOME to 741741",
            urgency=Urgency.IMMEDIATE,
            eligibility_criteria=["Available to all"],
            risk_levels=["critical", "high"],
            tags=["crisis", "text", "24/7"],
            priority=95
        ))
        
        # High-risk therapy resources
        self.add_resource(Resource(
            id="crisis_intervention_therapy",
            resource_type=ResourceType.THERAPY,
            name="Crisis Intervention Therapy",
            description="Immediate therapeutic intervention for acute crisis",
            contact_info="Contact your healthcare provider or local crisis center",
            urgency=Urgency.IMMEDIATE,
            eligibility_criteria=["High risk individuals", "Acute crisis"],
            risk_levels=["critical", "high"],
            tags=["therapy", "crisis", "intervention"],
            priority=90
        ))
        
        self.add_resource(Resource(
            id="intensive_outpatient",
            resource_type=ResourceType.THERAPY,
            name="Intensive Outpatient Program (IOP)",
            description="Structured daily therapy program for high-risk individuals",
            contact_info="Contact your healthcare provider",
            urgency=Urgency.SOON,
            eligibility_criteria=["High risk", "Recent hospitalization"],
            risk_levels=["high"],
            tags=["therapy", "intensive", "structured"],
            priority=85
        ))
        
        # Moderate-risk therapy resources
        self.add_resource(Resource(
            id="cbt_therapy",
            resource_type=ResourceType.THERAPY,
            name="Cognitive Behavioral Therapy (CBT)",
            description="Evidence-based therapy for depression and anxiety",
            contact_info="Contact your healthcare provider",
            urgency=Urgency.SOON,
            eligibility_criteria=["Diagnosed mental health condition"],
            risk_levels=["moderate", "high"],
            tags=["therapy", "cbt", "evidence-based", "depression", "anxiety"],
            priority=80
        ))
        
        self.add_resource(Resource(
            id="dbt_therapy",
            resource_type=ResourceType.THERAPY,
            name="Dialectical Behavior Therapy (DBT)",
            description="Therapy focused on emotional regulation and distress tolerance",
            contact_info="Contact your healthcare provider",
            urgency=Urgency.SOON,
            eligibility_criteria=["Emotional dysregulation", "Self-harm history"],
            risk_levels=["moderate", "high"],
            tags=["therapy", "dbt", "emotional-regulation"],
            priority=80
        ))
        
        # Support groups
        self.add_resource(Resource(
            id="peer_support_group",
            resource_type=ResourceType.SUPPORT_GROUP,
            name="Peer Support Groups",
            description="Group support from others with similar experiences",
            contact_info="Local mental health center",
            urgency=Urgency.ROUTINE,
            eligibility_criteria=["Open to all"],
            risk_levels=["moderate", "high", "low"],
            tags=["support", "peer", "group"],
            priority=70
        ))
        
        self.add_resource(Resource(
            id="depression_support_group",
            resource_type=ResourceType.SUPPORT_GROUP,
            name="Depression Support Group",
            description="Support group specifically for individuals with depression",
            contact_info="Local mental health center",
            urgency=Urgency.ROUTINE,
            eligibility_criteria=["Depression diagnosis or symptoms"],
            risk_levels=["moderate", "high"],
            tags=["support", "depression", "group"],
            priority=70
        ))
        
        # Medication management
        self.add_resource(Resource(
            id="psychiatric_medication",
            resource_type=ResourceType.MEDICATION,
            name="Psychiatric Medication Management",
            description="Medication evaluation and management by psychiatrist",
            contact_info="Contact your healthcare provider or psychiatrist",
            urgency=Urgency.SOON,
            eligibility_criteria=["Moderate to severe symptoms"],
            risk_levels=["moderate", "high"],
            tags=["medication", "psychiatrist"],
            priority=75
        ))
        
        # Wellness and prevention
        self.add_resource(Resource(
            id="wellness_program",
            resource_type=ResourceType.WELLNESS,
            name="Mental Health Wellness Programs",
            description="Preventive mental health and wellness activities",
            contact_info="Community health center",
            urgency=Urgency.ROUTINE,
            eligibility_criteria=["Open to all"],
            risk_levels=["low"],
            tags=["wellness", "prevention", "self-care"],
            priority=60
        ))
        
        self.add_resource(Resource(
            id="mindfulness_program",
            resource_type=ResourceType.WELLNESS,
            name="Mindfulness and Meditation Programs",
            description="Stress reduction through mindfulness practices",
            contact_info="Community health center or online platforms",
            urgency=Urgency.ROUTINE,
            eligibility_criteria=["Open to all"],
            risk_levels=["low", "moderate"],
            tags=["wellness", "mindfulness", "meditation", "stress"],
            priority=60
        ))
        
        # Self-help resources
        self.add_resource(Resource(
            id="self_help_apps",
            resource_type=ResourceType.SELF_HELP,
            name="Mental Health Apps",
            description="Mobile apps for mood tracking, meditation, and coping skills",
            contact_info="App stores (e.g., Headspace, Calm, Moodpath)",
            urgency=Urgency.ROUTINE,
            eligibility_criteria=["Open to all"],
            risk_levels=["low", "moderate"],
            tags=["self-help", "app", "technology"],
            priority=50
        ))
        
        self.add_resource(Resource(
            id="online_therapy",
            resource_type=ResourceType.THERAPY,
            name="Online Therapy Platforms",
            description="Virtual therapy sessions with licensed therapists",
            contact_info="BetterHelp, Talkspace, or similar platforms",
            urgency=Urgency.ROUTINE,
            eligibility_criteria=["Internet access", "Mild to moderate symptoms"],
            risk_levels=["low", "moderate"],
            tags=["therapy", "online", "virtual"],
            priority=65
        ))
        
        # Community resources
        self.add_resource(Resource(
            id="community_mental_health",
            resource_type=ResourceType.COMMUNITY,
            name="Community Mental Health Center",
            description="Local mental health services and resources",
            contact_info="Find your local community mental health center",
            urgency=Urgency.ROUTINE,
            eligibility_criteria=["Open to all"],
            risk_levels=["low", "moderate", "high"],
            tags=["community", "local"],
            priority=65
        ))
    
    def add_resource(self, resource: Resource):
        """Add a resource to the catalog"""
        self.resources[resource.id] = resource
    
    def get_resource(self, resource_id: str) -> Optional[Resource]:
        """Get a specific resource by ID"""
        return self.resources.get(resource_id)
    
    def filter_by_risk_level(self, risk_level: str) -> List[Resource]:
        """Filter resources by risk level"""
        return [
            resource for resource in self.resources.values()
            if risk_level in resource.risk_levels
        ]
    
    def filter_by_type(self, resource_type: ResourceType) -> List[Resource]:
        """Filter resources by type"""
        return [
            resource for resource in self.resources.values()
            if resource.resource_type == resource_type
        ]
    
    def filter_by_tags(self, tags: List[str]) -> List[Resource]:
        """Filter resources by tags (any match)"""
        return [
            resource for resource in self.resources.values()
            if any(tag in resource.tags for tag in tags)
        ]
    
    def search(
        self,
        risk_level: Optional[str] = None,
        resource_type: Optional[ResourceType] = None,
        tags: Optional[List[str]] = None,
        eligibility: Optional[List[str]] = None
    ) -> List[Resource]:
        """
        Search resources with multiple filters.
        
        Args:
            risk_level: Filter by risk level
            resource_type: Filter by resource type
            tags: Filter by tags (any match)
            eligibility: Filter by eligibility criteria (any match)
        
        Returns:
            List of matching resources sorted by priority
        """
        results = list(self.resources.values())
        
        if risk_level:
            results = [r for r in results if risk_level in r.risk_levels]
        
        if resource_type:
            results = [r for r in results if r.resource_type == resource_type]
        
        if tags:
            results = [r for r in results if any(tag in r.tags for tag in tags)]
        
        if eligibility:
            results = [
                r for r in results
                if any(crit in r.eligibility_criteria for crit in eligibility)
            ]
        
        # Sort by priority (descending)
        results.sort(key=lambda r: r.priority, reverse=True)
        
        return results
    
    def get_all_resources(self) -> List[Resource]:
        """Get all resources sorted by priority"""
        resources = list(self.resources.values())
        resources.sort(key=lambda r: r.priority, reverse=True)
        return resources
