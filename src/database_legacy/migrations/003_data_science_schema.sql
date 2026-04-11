-- Migration 003: Data Science Enhancements Schema
-- Creates tables for experiment tracking, data versioning, and feature store

-- Enable UUID extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- EXPERIMENT TRACKING TABLES
-- ============================================================================

-- Experiments table
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_name VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    tags JSONB
);

CREATE INDEX idx_experiments_name ON experiments(experiment_name);
CREATE INDEX idx_experiments_created_at ON experiments(created_at DESC);

-- Runs table
CREATE TABLE IF NOT EXISTS runs (
    run_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    run_name VARCHAR(255),
    status VARCHAR(20) NOT NULL CHECK (status IN ('RUNNING', 'FINISHED', 'FAILED', 'KILLED')),
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    params JSONB,
    tags JSONB,
    git_commit VARCHAR(40),
    code_version VARCHAR(50),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_runs_experiment ON runs(experiment_id);
CREATE INDEX idx_runs_status ON runs(status);
CREATE INDEX idx_runs_start_time ON runs(start_time DESC);
CREATE INDEX idx_runs_tags ON runs USING GIN(tags);

-- Metrics table
CREATE TABLE IF NOT EXISTS metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID REFERENCES runs(run_id) ON DELETE CASCADE,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    step INTEGER,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_metrics_run ON metrics(run_id, metric_name);
CREATE INDEX idx_metrics_run_step ON metrics(run_id, step);
CREATE INDEX idx_metrics_timestamp ON metrics(timestamp DESC);

-- Artifacts table
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID REFERENCES runs(run_id) ON DELETE CASCADE,
    artifact_type VARCHAR(50) NOT NULL,
    artifact_path TEXT NOT NULL,
    size_bytes BIGINT,
    metadata JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_artifacts_run ON artifacts(run_id);
CREATE INDEX idx_artifacts_type ON artifacts(artifact_type);
CREATE INDEX idx_artifacts_created_at ON artifacts(created_at DESC);

-- ============================================================================
-- DATA VERSIONING TABLES
-- ============================================================================

-- Dataset versions table
CREATE TABLE IF NOT EXISTS dataset_versions (
    version_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    source VARCHAR(255),
    num_rows INTEGER,
    num_columns INTEGER,
    schema JSONB,
    statistics JSONB,
    storage_uri TEXT NOT NULL,
    metadata JSONB,
    parent_version_id UUID REFERENCES dataset_versions(version_id) ON DELETE SET NULL,
    content_hash VARCHAR(64),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(dataset_name, version)
);

CREATE INDEX idx_dataset_versions_name ON dataset_versions(dataset_name);
CREATE INDEX idx_dataset_versions_created_at ON dataset_versions(created_at DESC);
CREATE INDEX idx_dataset_versions_parent ON dataset_versions(parent_version_id);
CREATE INDEX idx_dataset_versions_hash ON dataset_versions(dataset_name, content_hash);

-- Data lineage table
CREATE TABLE IF NOT EXISTS data_lineage (
    lineage_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    input_version_id UUID REFERENCES dataset_versions(version_id) ON DELETE CASCADE,
    output_version_id UUID REFERENCES dataset_versions(version_id) ON DELETE CASCADE,
    transformation_type VARCHAR(100),
    transformation_code TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_data_lineage_input ON data_lineage(input_version_id);
CREATE INDEX idx_data_lineage_output ON data_lineage(output_version_id);
CREATE INDEX idx_data_lineage_created_at ON data_lineage(created_at DESC);

-- Drift reports table
CREATE TABLE IF NOT EXISTS drift_reports (
    report_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    version_id1 UUID REFERENCES dataset_versions(version_id) ON DELETE CASCADE,
    version_id2 UUID REFERENCES dataset_versions(version_id) ON DELETE CASCADE,
    drift_detected BOOLEAN NOT NULL,
    drift_score FLOAT,
    feature_drifts JSONB,
    statistical_tests JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_drift_reports_version1 ON drift_reports(version_id1);
CREATE INDEX idx_drift_reports_version2 ON drift_reports(version_id2);
CREATE INDEX idx_drift_reports_created_at ON drift_reports(created_at DESC);

-- ============================================================================
-- FEATURE STORE TABLES
-- ============================================================================

-- Features table
CREATE TABLE IF NOT EXISTS features (
    feature_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    feature_name VARCHAR(255) UNIQUE NOT NULL,
    feature_type VARCHAR(50) NOT NULL CHECK (feature_type IN ('numeric', 'categorical', 'embedding', 'text', 'boolean')),
    description TEXT,
    transformation_code TEXT,
    input_schema JSONB,
    output_schema JSONB,
    version VARCHAR(50) NOT NULL,
    dependencies JSONB,
    owner VARCHAR(100),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_features_name ON features(feature_name);
CREATE INDEX idx_features_type ON features(feature_type);
CREATE INDEX idx_features_owner ON features(owner);
CREATE INDEX idx_features_created_at ON features(created_at DESC);

-- Feature values table
CREATE TABLE IF NOT EXISTS feature_values (
    value_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    feature_id UUID REFERENCES features(feature_id) ON DELETE CASCADE,
    entity_id VARCHAR(255) NOT NULL,
    feature_value JSONB NOT NULL,
    computed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    dataset_version_id UUID REFERENCES dataset_versions(version_id) ON DELETE SET NULL
);

CREATE INDEX idx_feature_values_entity ON feature_values(entity_id, feature_id);
CREATE INDEX idx_feature_values_feature ON feature_values(feature_id);
CREATE INDEX idx_feature_values_computed_at ON feature_values(computed_at DESC);
CREATE INDEX idx_feature_values_dataset ON feature_values(dataset_version_id);

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Function to update updated_at timestamp for features
CREATE OR REPLACE FUNCTION update_features_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to automatically update updated_at for features
CREATE TRIGGER update_features_updated_at_trigger
    BEFORE UPDATE ON features
    FOR EACH ROW
    EXECUTE FUNCTION update_features_updated_at();

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View for latest runs per experiment
CREATE OR REPLACE VIEW latest_runs AS
SELECT DISTINCT ON (experiment_id)
    r.*,
    e.experiment_name
FROM runs r
JOIN experiments e ON r.experiment_id = e.experiment_id
ORDER BY experiment_id, start_time DESC;

-- View for latest dataset versions
CREATE OR REPLACE VIEW latest_dataset_versions AS
SELECT DISTINCT ON (dataset_name)
    *
FROM dataset_versions
ORDER BY dataset_name, created_at DESC;
