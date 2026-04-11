-- Migration 002: Resources schema
-- Creates table for mental health resources

-- Resources table
CREATE TABLE IF NOT EXISTS resources (
    id VARCHAR(100) PRIMARY KEY,
    resource_type VARCHAR(50) NOT NULL CHECK (resource_type IN (
        'crisis_line', 'emergency', 'therapy', 'support_group', 
        'wellness', 'medication', 'self_help', 'community'
    )),
    name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    contact_info TEXT NOT NULL,
    urgency VARCHAR(20) NOT NULL CHECK (urgency IN ('immediate', 'soon', 'routine')),
    eligibility_criteria JSONB NOT NULL,
    risk_levels JSONB NOT NULL,
    tags JSONB,
    priority INTEGER NOT NULL DEFAULT 0,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for resources table
CREATE INDEX idx_resources_resource_type ON resources(resource_type);
CREATE INDEX idx_resources_urgency ON resources(urgency);
CREATE INDEX idx_resources_priority ON resources(priority DESC);
CREATE INDEX idx_resources_active ON resources(active) WHERE active = TRUE;
CREATE INDEX idx_resources_risk_levels ON resources USING GIN(risk_levels);
CREATE INDEX idx_resources_tags ON resources USING GIN(tags);

-- Resource recommendations tracking table
CREATE TABLE IF NOT EXISTS resource_recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    anonymized_id VARCHAR(64) NOT NULL,
    prediction_id UUID REFERENCES predictions(id),
    resource_id VARCHAR(100) REFERENCES resources(id),
    relevance_score FLOAT,
    recommended_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    accessed BOOLEAN DEFAULT FALSE,
    accessed_at TIMESTAMP
);

-- Indexes for resource_recommendations table
CREATE INDEX idx_resource_recs_anonymized_id ON resource_recommendations(anonymized_id);
CREATE INDEX idx_resource_recs_prediction_id ON resource_recommendations(prediction_id);
CREATE INDEX idx_resource_recs_resource_id ON resource_recommendations(resource_id);
CREATE INDEX idx_resource_recs_recommended_at ON resource_recommendations(recommended_at DESC);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to automatically update updated_at
CREATE TRIGGER update_resources_updated_at 
    BEFORE UPDATE ON resources
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
