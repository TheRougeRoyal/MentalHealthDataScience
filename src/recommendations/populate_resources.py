"""
Utility script to populate the resources database with default resources.
"""

import json
import logging
from typing import Dict, Any

from src.database.connection import DatabaseConnection
from src.database.repositories import ResourceRepository
from src.recommendations.resource_catalog import ResourceCatalog

logger = logging.getLogger(__name__)


def resource_to_dict(resource) -> Dict[str, Any]:
    """
    Convert Resource object to dictionary for database insertion.
    
    Args:
        resource: Resource object
    
    Returns:
        Dictionary with resource data
    """
    return {
        'id': resource.id,
        'resource_type': resource.resource_type.value,
        'name': resource.name,
        'description': resource.description,
        'contact_info': resource.contact_info,
        'urgency': resource.urgency.value,
        'eligibility_criteria': json.dumps(resource.eligibility_criteria),
        'risk_levels': json.dumps(resource.risk_levels),
        'tags': json.dumps(resource.tags),
        'priority': resource.priority
    }


def populate_resources(db_connection: DatabaseConnection) -> int:
    """
    Populate the resources table with default resources from catalog.
    
    Args:
        db_connection: Database connection
    
    Returns:
        Number of resources inserted
    """
    catalog = ResourceCatalog()
    repository = ResourceRepository(db_connection)
    
    count = 0
    for resource in catalog.get_all_resources():
        try:
            resource_dict = resource_to_dict(resource)
            repository.create(resource_dict)
            count += 1
            logger.info(f"Inserted resource: {resource.name}")
        except Exception as e:
            logger.warning(f"Resource {resource.id} may already exist or error occurred: {e}")
    
    logger.info(f"Populated {count} resources")
    return count


def main():
    """Main function to run population script"""
    logging.basicConfig(level=logging.INFO)
    
    try:
        db = DatabaseConnection()
        count = populate_resources(db)
        print(f"Successfully populated {count} resources")
    except Exception as e:
        logger.error(f"Failed to populate resources: {e}")
        raise


if __name__ == "__main__":
    main()
