-- Migration 001: Initial schema for MHRAS
-- Creates tables for predictions, audit_log, consent, and human_review_queue

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    anonymized_id VARCHAR(64) NOT NULL,
    risk_score FLOAT NOT NULL CHECK (risk_score >= 0 AND risk_score <= 100),
    risk_level VARCHAR(20) NOT NULL CHECK (risk_level IN ('LOW', 'MODERATE', 'HIGH', 'CRITICAL')),
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    model_version VARCHAR(50) NOT NULL,
    features_hash VARCHAR(64) NOT NULL,
    contributing_factors JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for predictions table
CREATE INDEX idx_predictions_anonymized_id ON predictions(anonymized_id);
CREATE INDEX idx_predictions_created_at ON predictions(created_at DESC);
CREATE INDEX idx_predictions_risk_level ON predictions(risk_level);
CREATE INDEX idx_predictions_model_version ON predictions(model_version);

-- Audit log table
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(50) NOT NULL,
    anonymized_id VARCHAR(64),
    user_id VARCHAR(64),
    details JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for audit_log table
CREATE INDEX idx_audit_log_event_type ON audit_log(event_type);
CREATE INDEX idx_audit_log_created_at ON audit_log(created_at DESC);
CREATE INDEX idx_audit_log_anonymized_id ON audit_log(anonymized_id);
CREATE INDEX idx_audit_log_user_id ON audit_log(user_id);

-- Consent table
CREATE TABLE IF NOT EXISTS consent (
    anonymized_id VARCHAR(64) PRIMARY KEY,
    data_types TEXT[] NOT NULL,
    granted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    revoked_at TIMESTAMP,
    metadata JSONB
);

-- Indexes for consent table
CREATE INDEX idx_consent_expires_at ON consent(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX idx_consent_revoked_at ON consent(revoked_at) WHERE revoked_at IS NOT NULL;
CREATE INDEX idx_consent_data_types ON consent USING GIN(data_types);

-- Human review queue table
CREATE TABLE IF NOT EXISTS human_review_queue (
    case_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    anonymized_id VARCHAR(64) NOT NULL,
    risk_score FLOAT NOT NULL CHECK (risk_score >= 0 AND risk_score <= 100),
    prediction_id UUID REFERENCES predictions(id),
    assigned_to VARCHAR(64),
    status VARCHAR(20) NOT NULL CHECK (status IN ('PENDING', 'IN_REVIEW', 'COMPLETED', 'ESCALATED')),
    decision VARCHAR(20) CHECK (decision IN ('CONFIRMED', 'MODIFIED', 'OVERRIDDEN')),
    decision_notes TEXT,
    priority VARCHAR(20) NOT NULL DEFAULT 'NORMAL' CHECK (priority IN ('LOW', 'NORMAL', 'HIGH', 'CRITICAL')),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    reviewed_at TIMESTAMP,
    escalated_at TIMESTAMP
);

-- Indexes for human_review_queue table
CREATE INDEX idx_hrq_status ON human_review_queue(status);
CREATE INDEX idx_hrq_assigned_to ON human_review_queue(assigned_to) WHERE assigned_to IS NOT NULL;
CREATE INDEX idx_hrq_created_at ON human_review_queue(created_at DESC);
CREATE INDEX idx_hrq_priority ON human_review_queue(priority);
CREATE INDEX idx_hrq_anonymized_id ON human_review_queue(anonymized_id);

-- Create a composite index for pending cases query
CREATE INDEX idx_hrq_status_priority ON human_review_queue(status, priority DESC, created_at ASC);
