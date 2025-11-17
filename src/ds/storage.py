"""Storage backend abstraction for artifacts and datasets"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, BinaryIO
import shutil
import hashlib
from datetime import datetime


class StorageBackend(ABC):
    """Abstract storage backend for artifacts and datasets"""
    
    @abstractmethod
    def save_artifact(self, artifact: bytes, path: str, metadata: Optional[dict] = None) -> str:
        """
        Save artifact and return URI
        
        Args:
            artifact: Binary content to save
            path: Relative path for the artifact
            metadata: Optional metadata dictionary
            
        Returns:
            URI string for the saved artifact
        """
        pass
    
    @abstractmethod
    def load_artifact(self, uri: str) -> bytes:
        """
        Load artifact from URI
        
        Args:
            uri: URI of the artifact to load
            
        Returns:
            Binary content of the artifact
        """
        pass
    
    @abstractmethod
    def delete_artifact(self, uri: str) -> None:
        """
        Delete artifact at URI
        
        Args:
            uri: URI of the artifact to delete
        """
        pass
    
    @abstractmethod
    def artifact_exists(self, uri: str) -> bool:
        """
        Check if artifact exists at URI
        
        Args:
            uri: URI of the artifact to check
            
        Returns:
            True if artifact exists, False otherwise
        """
        pass
    
    @abstractmethod
    def get_artifact_size(self, uri: str) -> int:
        """
        Get size of artifact in bytes
        
        Args:
            uri: URI of the artifact
            
        Returns:
            Size in bytes
        """
        pass


class FileSystemStorage(StorageBackend):
    """Local filesystem storage implementation"""
    
    def __init__(self, base_path: str = "experiments/artifacts"):
        """
        Initialize filesystem storage
        
        Args:
            base_path: Base directory for artifact storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _get_full_path(self, path: str) -> Path:
        """Get full filesystem path from relative path"""
        return self.base_path / path
    
    def _generate_uri(self, path: str) -> str:
        """Generate URI for a given path"""
        return f"file://{self.base_path.absolute()}/{path}"
    
    def _parse_uri(self, uri: str) -> Path:
        """Parse URI to get filesystem path"""
        if uri.startswith("file://"):
            return Path(uri[7:])
        return Path(uri)
    
    def save_artifact(self, artifact: bytes, path: str, metadata: Optional[dict] = None) -> str:
        """
        Save artifact to filesystem
        
        Args:
            artifact: Binary content to save
            path: Relative path for the artifact
            metadata: Optional metadata (not used in filesystem storage)
            
        Returns:
            URI string for the saved artifact
        """
        full_path = self._get_full_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, 'wb') as f:
            f.write(artifact)
        
        return self._generate_uri(path)
    
    def save_artifact_from_file(self, source_path: str, dest_path: str, metadata: Optional[dict] = None) -> str:
        """
        Save artifact by copying from source file
        
        Args:
            source_path: Path to source file
            dest_path: Relative destination path
            metadata: Optional metadata (not used in filesystem storage)
            
        Returns:
            URI string for the saved artifact
        """
        full_dest_path = self._get_full_path(dest_path)
        full_dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(source_path, full_dest_path)
        
        return self._generate_uri(dest_path)
    
    def load_artifact(self, uri: str) -> bytes:
        """
        Load artifact from filesystem
        
        Args:
            uri: URI of the artifact to load
            
        Returns:
            Binary content of the artifact
            
        Raises:
            FileNotFoundError: If artifact doesn't exist
        """
        path = self._parse_uri(uri)
        
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found at {uri}")
        
        with open(path, 'rb') as f:
            return f.read()
    
    def load_artifact_to_file(self, uri: str, dest_path: str) -> None:
        """
        Load artifact and save to destination file
        
        Args:
            uri: URI of the artifact to load
            dest_path: Destination file path
            
        Raises:
            FileNotFoundError: If artifact doesn't exist
        """
        source_path = self._parse_uri(uri)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Artifact not found at {uri}")
        
        shutil.copy2(source_path, dest_path)
    
    def delete_artifact(self, uri: str) -> None:
        """
        Delete artifact from filesystem
        
        Args:
            uri: URI of the artifact to delete
        """
        path = self._parse_uri(uri)
        
        if path.exists():
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
    
    def artifact_exists(self, uri: str) -> bool:
        """
        Check if artifact exists
        
        Args:
            uri: URI of the artifact to check
            
        Returns:
            True if artifact exists, False otherwise
        """
        path = self._parse_uri(uri)
        return path.exists()
    
    def get_artifact_size(self, uri: str) -> int:
        """
        Get size of artifact in bytes
        
        Args:
            uri: URI of the artifact
            
        Returns:
            Size in bytes
            
        Raises:
            FileNotFoundError: If artifact doesn't exist
        """
        path = self._parse_uri(uri)
        
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found at {uri}")
        
        if path.is_file():
            return path.stat().st_size
        elif path.is_dir():
            # Calculate total size of directory
            total_size = 0
            for item in path.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
            return total_size
        
        return 0
    
    def compute_hash(self, uri: str, algorithm: str = "sha256") -> str:
        """
        Compute hash of artifact content
        
        Args:
            uri: URI of the artifact
            algorithm: Hash algorithm (md5, sha1, sha256)
            
        Returns:
            Hex digest of the hash
            
        Raises:
            FileNotFoundError: If artifact doesn't exist
        """
        path = self._parse_uri(uri)
        
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found at {uri}")
        
        if not path.is_file():
            raise ValueError(f"Cannot compute hash for directory: {uri}")
        
        hash_obj = hashlib.new(algorithm)
        
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    def list_artifacts(self, prefix: str = "") -> list[str]:
        """
        List all artifacts with given prefix
        
        Args:
            prefix: Path prefix to filter artifacts
            
        Returns:
            List of artifact URIs
        """
        search_path = self._get_full_path(prefix) if prefix else self.base_path
        
        if not search_path.exists():
            return []
        
        artifacts = []
        for item in search_path.rglob('*'):
            if item.is_file():
                rel_path = item.relative_to(self.base_path)
                artifacts.append(self._generate_uri(str(rel_path)))
        
        return artifacts
