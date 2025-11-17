"""Performance tests for MHRAS components"""

import pytest
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import tempfile
from pathlib import Path

from src.ingestion.validation import DataValidator
from src.processing.etl_pipeline import ETLPipeline, ETLPipelineConfig
from src.ml.feature_pipeline import FeatureEngineeringPipeline
from src.governance.anonymization import Anonymizer


class TestAPIEndpointPerformance:
    """Test API endpoint performance requirements"""
    
    def test_screening_request_latency(self):
        """Test that screening request completes within 5 seconds"""
        # Simulate screening workflow components
        validator = DataValidator()
        anonymizer = Anonymizer()
        
        # Create test data
        survey_data = {
            "survey_id": "perf_test",
            "timestamp": "2025-11-17T10:30:00Z",
            "responses": {
                "mood_score": 5,
                "anxiety_level": 7,
                "depression_level": 8,
                "sleep_quality": 3
            }
        }
        
        start_time = time.time()
        
        # 1. Validation (should be < 100ms)
        validation_start = time.time()
        result = validator.validate_survey(survey_data)
        validation_time = time.time() - validation_start
        
        assert result.is_valid
        assert validation_time < 0.1  # 100ms
        
        # 2. Anonymization
        anon_start = time.time()
        anon_data = anonymizer.anonymize_record(
            {"patient_id": "patient_123", "data": survey_data},
            pii_fields=["patient_id"]
        )
        anon_time = time.time() - anon_start
        
        assert anon_time < 0.05  # 50ms
        
        total_time = time.time() - start_time
        
        # Total workflow should be well under 5 seconds
        assert total_time < 5.0
        
        print(f"\nScreening request performance:")
        print(f"  Validation: {validation_time*1000:.2f}ms")
        print(f"  Anonymization: {anon_time*1000:.2f}ms")
        print(f"  Total: {total_time*1000:.2f}ms")
    
    def test_concurrent_screening_requests(self):
        """Test handling concurrent screening requests"""
        validator = DataValidator()
        
        def process_request(request_id):
            """Simulate processing a single request"""
            start = time.time()
            
            survey_data = {
                "survey_id": f"concurrent_{request_id}",
                "timestamp": "2025-11-17T10:30:00Z",
                "responses": {"mood_score": 5}
            }
            
            result = validator.validate_survey(survey_data)
            elapsed = time.time() - start
            
            return {
                "request_id": request_id,
                "success": result.is_valid,
                "latency": elapsed
            }
        
        # Test with 10 concurrent requests
        num_requests = 10
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_request, i) for i in range(num_requests)]
            results = [future.result() for future in as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # All requests should succeed
        assert all(r["success"] for r in results)
        
        # Calculate statistics
        latencies = [r["latency"] for r in results]
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        print(f"\nConcurrent request performance ({num_requests} requests):")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Avg latency: {avg_latency*1000:.2f}ms")
        print(f"  Max latency: {max_latency*1000:.2f}ms")
        print(f"  P95 latency: {p95_latency*1000:.2f}ms")
        
        # Performance targets
        assert avg_latency < 0.2  # 200ms average
        assert p95_latency < 0.5  # 500ms p95


class TestETLPipelinePerformance:
    """Test ETL pipeline performance requirements"""
    
    def test_batch_processing_1000_records(self):
        """Test that ETL pipeline processes 1000 records within 60 seconds"""
        # Create 1000 records
        np.random.seed(42)
        n_records = 1000
        
        df = pd.DataFrame({
            "anonymized_id": [f"user_{i % 100}" for i in range(n_records)],
            "timestamp": pd.date_range("2025-01-01", periods=n_records, freq="H"),
            "heart_rate": np.random.normal(70, 10, n_records),
            "sleep_hours": np.random.normal(7, 1.5, n_records),
            "mood_score": np.random.randint(1, 11, n_records),
            "activity_level": np.random.choice(["low", "medium", "high"], n_records)
        })
        
        # Configure pipeline
        config = ETLPipelineConfig(
            outlier_method="iqr",
            imputation_strategies={
                "heart_rate": "median",
                "sleep_hours": "median",
                "mood_score": "forward_fill"
            },
            standardize_columns=["heart_rate", "sleep_hours", "mood_score"],
            group_by_column="anonymized_id"
        )
        
        pipeline = ETLPipeline(config)
        
        # Measure processing time
        start_time = time.time()
        result = pipeline.fit_transform(df)
        elapsed_time = time.time() - start_time
        
        # Verify results
        assert result is not None
        assert len(result) > 0
        assert pipeline.is_fitted
        
        # Performance requirement: < 60 seconds for 1000 records
        assert elapsed_time < 60.0
        
        # Calculate throughput
        throughput = n_records / elapsed_time
        
        print(f"\nETL pipeline performance (1000 records):")
        print(f"  Processing time: {elapsed_time:.3f}s")
        print(f"  Throughput: {throughput:.1f} records/second")
        print(f"  Per-record time: {(elapsed_time/n_records)*1000:.2f}ms")
        
        # Get pipeline statistics
        stats = pipeline.get_pipeline_stats()
        print(f"  Input records: {stats['input_records']}")
        print(f"  Output records: {stats['output_records']}")
    
    def test_batch_processing_scalability(self):
        """Test ETL pipeline scalability with different batch sizes"""
        config = ETLPipelineConfig(
            standardize_columns=["value"],
            group_by_column="anonymized_id"
        )
        
        batch_sizes = [100, 500, 1000]
        results = []
        
        for batch_size in batch_sizes:
            # Create data
            df = pd.DataFrame({
                "anonymized_id": [f"user_{i % 10}" for i in range(batch_size)],
                "timestamp": pd.date_range("2025-01-01", periods=batch_size, freq="H"),
                "value": np.random.normal(0, 1, batch_size)
            })
            
            # Process
            pipeline = ETLPipeline(config)
            start_time = time.time()
            pipeline.fit_transform(df)
            elapsed_time = time.time() - start_time
            
            throughput = batch_size / elapsed_time
            results.append({
                "batch_size": batch_size,
                "time": elapsed_time,
                "throughput": throughput
            })
        
        print(f"\nETL scalability test:")
        for r in results:
            print(f"  {r['batch_size']} records: {r['time']:.3f}s ({r['throughput']:.1f} rec/s)")
        
        # Verify reasonable scalability (not exponential growth)
        # Time should scale roughly linearly
        time_ratio = results[-1]["time"] / results[0]["time"]
        size_ratio = results[-1]["batch_size"] / results[0]["batch_size"]
        
        # Time ratio should not be much worse than size ratio
        assert time_ratio < size_ratio * 2


class TestFeatureEngineeringPerformance:
    """Test feature engineering performance requirements"""
    
    def test_feature_extraction_per_individual(self):
        """Test that feature extraction completes within 2 seconds per individual"""
        feature_pipeline = FeatureEngineeringPipeline(use_parallel=False)
        
        # Create data for one individual
        behavioral_data = pd.DataFrame({
            "anonymized_id": ["user_1"] * 30,
            "activity_type": np.random.choice(["exercise", "social", "work"], 30),
            "timestamp": pd.date_range("2025-01-01", periods=30, freq="D")
        })
        
        # Measure extraction time
        start_time = time.time()
        features = feature_pipeline.extract_features(behavioral_df=behavioral_data)
        elapsed_time = time.time() - start_time
        
        assert features is not None
        assert len(features) > 0
        
        # Should complete within 2 seconds
        assert elapsed_time < 2.0
        
        print(f"\nFeature extraction performance (1 individual):")
        print(f"  Extraction time: {elapsed_time*1000:.2f}ms")
        print(f"  Features generated: {len(features.columns)}")
    
    def test_feature_extraction_batch(self):
        """Test feature extraction for multiple individuals"""
        feature_pipeline = FeatureEngineeringPipeline(use_parallel=False)
        
        # Create data for 10 individuals
        n_individuals = 10
        records_per_individual = 20
        
        behavioral_data = pd.DataFrame({
            "anonymized_id": [f"user_{i}" for i in range(n_individuals) for _ in range(records_per_individual)],
            "activity_type": np.random.choice(["exercise", "social", "work"], n_individuals * records_per_individual),
            "timestamp": pd.date_range("2025-01-01", periods=n_individuals * records_per_individual, freq="H")
        })
        
        # Measure extraction time
        start_time = time.time()
        features = feature_pipeline.extract_features(behavioral_df=behavioral_data)
        elapsed_time = time.time() - start_time
        
        assert features is not None
        assert len(features) == n_individuals
        
        # Calculate per-individual time
        per_individual_time = elapsed_time / n_individuals
        
        print(f"\nFeature extraction batch performance ({n_individuals} individuals):")
        print(f"  Total time: {elapsed_time:.3f}s")
        print(f"  Per-individual time: {per_individual_time*1000:.2f}ms")
        
        # Each individual should be processed within 2 seconds
        assert per_individual_time < 2.0
    
    def test_parallel_feature_extraction(self):
        """Test parallel feature extraction performance"""
        # Sequential processing
        pipeline_seq = FeatureEngineeringPipeline(use_parallel=False)
        
        behavioral_data = pd.DataFrame({
            "anonymized_id": [f"user_{i}" for i in range(5) for _ in range(10)],
            "activity_type": np.random.choice(["exercise", "social", "work"], 50),
            "timestamp": pd.date_range("2025-01-01", periods=50, freq="H")
        })
        
        start_seq = time.time()
        features_seq = pipeline_seq.extract_features(behavioral_df=behavioral_data)
        time_seq = time.time() - start_seq
        
        # Parallel processing
        pipeline_par = FeatureEngineeringPipeline(use_parallel=True, max_workers=2)
        
        start_par = time.time()
        features_par = pipeline_par.extract_features(behavioral_df=behavioral_data)
        time_par = time.time() - start_par
        
        print(f"\nParallel vs Sequential feature extraction:")
        print(f"  Sequential: {time_seq*1000:.2f}ms")
        print(f"  Parallel: {time_par*1000:.2f}ms")
        print(f"  Speedup: {time_seq/time_par:.2f}x")
        
        # Both should produce results
        assert features_seq is not None
        assert features_par is not None


class TestValidationPerformance:
    """Test data validation performance requirements"""
    
    def test_validation_latency_survey(self):
        """Test survey validation completes within 100ms"""
        validator = DataValidator()
        
        survey_data = {
            "survey_id": "perf_survey",
            "timestamp": "2025-11-17T10:30:00Z",
            "responses": {
                "mood_score": 5,
                "anxiety_level": 7,
                "depression_level": 8,
                "stress_level": 6,
                "sleep_quality": 3,
                "social_support": 4,
                "suicidal_ideation": False,
                "self_harm": False,
                "substance_use": "none",
                "medication_adherence": "good",
                "therapy_engagement": 75,
                "life_events": ["job_change"],
                "free_text_response": "Feeling stressed but managing"
            }
        }
        
        # Measure validation time
        start_time = time.time()
        result = validator.validate_survey(survey_data)
        elapsed_time = time.time() - start_time
        
        assert result.is_valid
        assert result.validation_time_ms < 100
        assert elapsed_time < 0.1  # 100ms
        
        print(f"\nSurvey validation performance:")
        print(f"  Validation time: {elapsed_time*1000:.2f}ms")
        print(f"  Reported time: {result.validation_time_ms:.2f}ms")
    
    def test_validation_latency_wearable(self):
        """Test wearable validation completes within 100ms"""
        validator = DataValidator()
        
        wearable_data = {
            "device_id": "device_perf",
            "device_type": "smartwatch",
            "start_timestamp": "2025-11-17T00:00:00Z",
            "end_timestamp": "2025-11-17T23:59:59Z",
            "metrics": {
                "heart_rate": {
                    "average_bpm": 72,
                    "resting_bpm": 58,
                    "max_bpm": 145,
                    "min_bpm": 52
                },
                "hrv": {
                    "rmssd": 45.5,
                    "sdnn": 62.3,
                    "lf_hf_ratio": 1.8
                },
                "sleep": {
                    "total_minutes": 420,
                    "deep_sleep_minutes": 90,
                    "rem_sleep_minutes": 105,
                    "light_sleep_minutes": 210,
                    "awake_minutes": 15,
                    "interruptions": 2,
                    "efficiency_percent": 96.4
                },
                "activity": {
                    "steps": 8500,
                    "distance_meters": 6800,
                    "calories_burned": 2200,
                    "active_minutes": 45
                }
            }
        }
        
        # Measure validation time
        start_time = time.time()
        result = validator.validate_wearable(wearable_data)
        elapsed_time = time.time() - start_time
        
        assert result.is_valid
        assert result.validation_time_ms < 100
        assert elapsed_time < 0.1  # 100ms
        
        print(f"\nWearable validation performance:")
        print(f"  Validation time: {elapsed_time*1000:.2f}ms")
        print(f"  Reported time: {result.validation_time_ms:.2f}ms")
    
    def test_validation_batch_performance(self):
        """Test validation performance for batch of records"""
        validator = DataValidator()
        
        # Create 100 survey records
        n_records = 100
        
        start_time = time.time()
        
        for i in range(n_records):
            survey_data = {
                "survey_id": f"batch_{i}",
                "timestamp": "2025-11-17T10:30:00Z",
                "responses": {"mood_score": 5}
            }
            result = validator.validate_survey(survey_data)
            assert result.is_valid
        
        elapsed_time = time.time() - start_time
        per_record_time = elapsed_time / n_records
        
        print(f"\nBatch validation performance ({n_records} records):")
        print(f"  Total time: {elapsed_time:.3f}s")
        print(f"  Per-record time: {per_record_time*1000:.2f}ms")
        print(f"  Throughput: {n_records/elapsed_time:.1f} records/second")
        
        # Each record should validate within 100ms
        assert per_record_time < 0.1


class TestAnonymizationPerformance:
    """Test anonymization performance"""
    
    def test_hash_identifier_performance(self):
        """Test identifier hashing performance"""
        anonymizer = Anonymizer(salt="perf_test")
        
        n_identifiers = 1000
        identifiers = [f"patient_{i}" for i in range(n_identifiers)]
        
        start_time = time.time()
        
        for identifier in identifiers:
            hash_value = anonymizer.hash_identifier(identifier)
            assert len(hash_value) == 64
        
        elapsed_time = time.time() - start_time
        per_hash_time = elapsed_time / n_identifiers
        
        print(f"\nIdentifier hashing performance ({n_identifiers} identifiers):")
        print(f"  Total time: {elapsed_time:.3f}s")
        print(f"  Per-hash time: {per_hash_time*1000:.3f}ms")
        print(f"  Throughput: {n_identifiers/elapsed_time:.1f} hashes/second")
        
        # Should be very fast
        assert per_hash_time < 0.001  # 1ms per hash
    
    def test_text_anonymization_performance(self):
        """Test text anonymization performance"""
        anonymizer = Anonymizer()
        
        text = "Contact me at john.doe@example.com or call 555-123-4567. " * 10
        
        n_iterations = 100
        
        start_time = time.time()
        
        for _ in range(n_iterations):
            anon_text = anonymizer.anonymize_text(text)
            assert "[REDACTED]" in anon_text
        
        elapsed_time = time.time() - start_time
        per_text_time = elapsed_time / n_iterations
        
        print(f"\nText anonymization performance ({n_iterations} texts):")
        print(f"  Total time: {elapsed_time:.3f}s")
        print(f"  Per-text time: {per_text_time*1000:.2f}ms")
        
        # Should be reasonably fast
        assert per_text_time < 0.01  # 10ms per text


class TestMemoryUsage:
    """Test memory usage for large datasets"""
    
    def test_etl_memory_efficiency(self):
        """Test ETL pipeline memory usage with large dataset"""
        # Create large dataset (10,000 records)
        n_records = 10000
        
        df = pd.DataFrame({
            "anonymized_id": [f"user_{i % 100}" for i in range(n_records)],
            "timestamp": pd.date_range("2025-01-01", periods=n_records, freq="min"),
            "value": np.random.normal(0, 1, n_records)
        })
        
        config = ETLPipelineConfig(
            standardize_columns=["value"],
            group_by_column="anonymized_id"
        )
        
        pipeline = ETLPipeline(config)
        
        # Process data
        start_time = time.time()
        result = pipeline.fit_transform(df)
        elapsed_time = time.time() - start_time
        
        assert result is not None
        assert len(result) > 0
        
        print(f"\nLarge dataset processing ({n_records} records):")
        print(f"  Processing time: {elapsed_time:.3f}s")
        print(f"  Throughput: {n_records/elapsed_time:.1f} records/second")


class TestLatencyPercentiles:
    """Test latency percentiles for various operations"""
    
    def test_validation_latency_distribution(self):
        """Test validation latency distribution"""
        validator = DataValidator()
        
        latencies = []
        n_iterations = 100
        
        for i in range(n_iterations):
            survey_data = {
                "survey_id": f"latency_{i}",
                "timestamp": "2025-11-17T10:30:00Z",
                "responses": {"mood_score": 5}
            }
            
            start = time.time()
            validator.validate_survey(survey_data)
            latency = time.time() - start
            latencies.append(latency * 1000)  # Convert to ms
        
        # Calculate percentiles
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        avg = np.mean(latencies)
        max_latency = np.max(latencies)
        
        print(f"\nValidation latency distribution ({n_iterations} iterations):")
        print(f"  Average: {avg:.2f}ms")
        print(f"  P50: {p50:.2f}ms")
        print(f"  P95: {p95:.2f}ms")
        print(f"  P99: {p99:.2f}ms")
        print(f"  Max: {max_latency:.2f}ms")
        
        # Performance targets
        assert p50 < 50  # 50ms median
        assert p95 < 100  # 100ms p95
        assert p99 < 150  # 150ms p99


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
