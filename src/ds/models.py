"""Pydantic models for data science components."""

from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID
from pydantic import BaseModel, Field


class Experiment(BaseModel):
    """Experiment model."""
    
    experiment_id: Optional[UUID] = None
    experiment_name: str
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    tags: Optional[Dict[str, str]] = None


class Run(BaseModel):
    """Experiment run model."""
    
    run_id: Optional[UUID] = None
    experiment_id: UUID
    run_name: Optional[str] = None
    status: str = Field(pattern="^(RUNNING|FINISHED|FAILED)$")
    start_time: datetime
    end_time: Optional[datetime] = None
    params: Optional[Dict[str, Any]] = None
    tags: Optional[Dict[str, str]] = None
    git_commit: Optional[str] = None
    code_version: Optional[str] = None
    created_at: Optional[datetime] = None


class Metric(BaseModel):
    """Metric log model."""
    
    metric_id: Optional[UUID] = None
    run_id: UUID
    metric_name: str
    metric_value: float
    step: Optional[int] = None
    timestamp: Optional[datetime] = None


class Artifact(BaseModel):
    """Artifact model."""
    
    artifact_id: Optional[UUID] = None
    run_id: UUID
    artifact_type: str
    artifact_path: str
    size_bytes: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
