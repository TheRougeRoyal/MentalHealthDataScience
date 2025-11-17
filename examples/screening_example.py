"""
Example usage of the ScreeningService for mental health risk assessment.

This example demonstrates:
1. Creating a screening request
2. Performing a screening
3. Accessing results and recommendations
4. Batch processing multiple screenings
"""

from datetime import datetime
from src.screening_service import ScreeningServiceSync, ScreeningRequest


def example_single_screening():
    """Example of screening a single individual."""
    print("=" * 60)
    print("Example 1: Single Individual Screening")
    print("=" * 60)
    
    # Initialize the screening service
    service = ScreeningServiceSync(
        alert_threshold=75.0,
        human_review_threshold=75.0,
        use_cache=True
    )
    
    # Create a screening request with survey and wearable data
    request = ScreeningRequest(
        anonymized_id="example_user_001",
        survey_data={
            "responses": {
                "mood": "low",
                "sleep_quality": "poor",
                "anxiety_level": "high",
                "suicidal_ideation": False,
                "self_harm": False,
                "social_withdrawal": True
            },
            "timestamp": datetime.utcnow().isoformat()
        },
        wearable_data={
            "metrics": {
                "sleep_duration_hours": 4.5,
                "sleep_efficiency": 0.65,
                "heart_rate_avg": 85,
                "heart_rate_resting": 72,
                "hrv_rmssd": 25,
                "activity_minutes": 15
            },
            "data_quality": {
                "completeness_percent": 92,
                "wear_time_minutes": 1200
            }
        },
        user_id="clinician_001"
    )
    
    try:
        # Perform the screening
        print("\nPerforming screening...")
        response = service.screen_individual(request)
        
        # Display results
        print(f"\n{'Results':^60}")
        print("-" * 60)
        print(f"Anonymized ID: {response.anonymized_id}")
        print(f"Risk Score: {response.risk_score:.2f}/100")
        print(f"Risk Level: {response.risk_level.upper()}")
        print(f"Confidence: {response.confidence:.2%}")
        print(f"Processing Time: {response.processing_time_seconds:.3f}s")
        print(f"Requires Human Review: {response.requires_human_review}")
        print(f"Alert Triggered: {response.alert_triggered}")
        
        # Display contributing factors
        print(f"\n{'Contributing Factors':^60}")
        print("-" * 60)
        for i, factor in enumerate(response.contributing_factors[:5], 1):
            print(f"{i}. {factor}")
        
        # Display recommendations
        print(f"\n{'Recommendations':^60}")
        print("-" * 60)
        for i, rec in enumerate(response.recommendations, 1):
            print(f"\n{i}. {rec['name']}")
            print(f"   Type: {rec['resource_type']}")
            print(f"   Urgency: {rec['urgency']}")
            print(f"   Description: {rec['description'][:100]}...")
        
        # Display explanations
        print(f"\n{'Explanations':^60}")
        print("-" * 60)
        print(f"Counterfactual: {response.explanations.get('counterfactual', 'N/A')}")
        print(f"Rule: {response.explanations.get('rule_approximation', 'N/A')}")
        
    except Exception as e:
        print(f"\nError during screening: {e}")
        print(f"Error type: {type(e).__name__}")


def example_batch_screening():
    """Example of batch screening multiple individuals."""
    print("\n\n" + "=" * 60)
    print("Example 2: Batch Screening")
    print("=" * 60)
    
    # Initialize service
    service = ScreeningServiceSync()
    
    # Preload models for better performance
    print("\nPreloading models...")
    service.preload_models()
    
    # Create multiple screening requests
    requests = []
    for i in range(5):
        request = ScreeningRequest(
            anonymized_id=f"batch_user_{i:03d}",
            survey_data={
                "responses": {
                    "mood": ["low", "moderate", "high"][i % 3],
                    "sleep_quality": ["poor", "fair", "good"][i % 3],
                    "anxiety_level": ["low", "moderate", "high"][i % 3],
                    "suicidal_ideation": False,
                    "self_harm": False
                },
                "timestamp": datetime.utcnow().isoformat()
            },
            user_id="clinician_batch"
        )
        requests.append(request)
    
    try:
        # Process batch
        print(f"\nProcessing {len(requests)} screenings in batch...")
        responses = service.screen_batch(requests, batch_size=3)
        
        # Display summary
        print(f"\n{'Batch Results Summary':^60}")
        print("-" * 60)
        print(f"Total Screenings: {len(responses)}")
        
        # Count by risk level
        risk_counts = {}
        for response in responses:
            risk_counts[response.risk_level] = risk_counts.get(response.risk_level, 0) + 1
        
        print("\nRisk Level Distribution:")
        for level in ["low", "moderate", "high", "critical"]:
            count = risk_counts.get(level, 0)
            print(f"  {level.capitalize()}: {count}")
        
        # High risk cases
        high_risk = [r for r in responses if r.risk_level in ["high", "critical"]]
        print(f"\nHigh Risk Cases: {len(high_risk)}")
        for response in high_risk:
            print(f"  - {response.anonymized_id}: {response.risk_score:.1f} ({response.risk_level})")
        
        # Average processing time
        avg_time = sum(r.processing_time_seconds for r in responses) / len(responses)
        print(f"\nAverage Processing Time: {avg_time:.3f}s per screening")
        
    except Exception as e:
        print(f"\nError during batch screening: {e}")


def example_service_statistics():
    """Example of accessing service statistics."""
    print("\n\n" + "=" * 60)
    print("Example 3: Service Statistics")
    print("=" * 60)
    
    service = ScreeningServiceSync()
    
    # Get statistics
    stats = service.get_screening_statistics()
    
    print(f"\n{'Model Registry':^60}")
    print("-" * 60)
    print(f"Active Models: {stats['model_registry']['active_models']}")
    print(f"Total Models: {stats['model_registry']['total_models']}")
    
    print(f"\n{'Human Review Queue':^60}")
    print("-" * 60)
    print(f"Total Cases: {stats['review_queue']['total_cases']}")
    print(f"Pending Cases: {stats['review_queue']['pending_count']}")
    print(f"Overdue Cases: {stats['review_queue']['overdue_count']}")
    
    if stats['review_queue']['average_review_time_hours']:
        print(f"Average Review Time: {stats['review_queue']['average_review_time_hours']:.2f}h")
    
    if stats['review_queue']['oldest_pending_hours']:
        print(f"Oldest Pending Case: {stats['review_queue']['oldest_pending_hours']:.2f}h")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Mental Health Risk Assessment System")
    print("Screening Service Examples")
    print("=" * 60)
    
    # Run examples
    example_single_screening()
    example_batch_screening()
    example_service_statistics()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
