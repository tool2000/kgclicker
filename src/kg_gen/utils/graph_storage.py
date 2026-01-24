"""
Graph storage module for managing knowledge graphs as JSON files.
Supports saving, loading, merging, and version history.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

from kg_gen.models import Graph

logger = logging.getLogger(__name__)

# Default storage directory
DEFAULT_STORAGE_DIR = Path.home() / ".kg_gen" / "graphs"


class GraphStorage:
    """
    Manages knowledge graph persistence with version history support.
    """

    def __init__(self, storage_dir: Optional[str | Path] = None):
        """
        Initialize graph storage.

        Args:
            storage_dir: Directory to store graphs. Defaults to ~/.kg_gen/graphs
        """
        self.storage_dir = Path(storage_dir) if storage_dir else DEFAULT_STORAGE_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.graphs_dir = self.storage_dir / "current"
        self.history_dir = self.storage_dir / "history"
        self.graphs_dir.mkdir(exist_ok=True)
        self.history_dir.mkdir(exist_ok=True)

        # Metadata file for tracking graphs
        self.metadata_file = self.storage_dir / "metadata.json"
        self._load_metadata()

    def _load_metadata(self):
        """Load or initialize metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "graphs": {},
                "created_at": datetime.now().isoformat(),
            }
            self._save_metadata()

    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def _get_graph_path(self, graph_id: str) -> Path:
        """Get the file path for a graph."""
        return self.graphs_dir / f"{graph_id}.json"

    def _get_history_dir(self, graph_id: str) -> Path:
        """Get the history directory for a graph."""
        history_dir = self.history_dir / graph_id
        history_dir.mkdir(exist_ok=True)
        return history_dir

    def _graph_to_dict(self, graph: Graph) -> dict:
        """Convert Graph to serializable dict."""
        return {
            "entities": list(graph.entities),
            "relations": [list(r) for r in graph.relations],
            "edges": list(graph.edges),
            "entity_clusters": {k: list(v) for k, v in graph.entity_clusters.items()}
            if graph.entity_clusters
            else None,
            "edge_clusters": {k: list(v) for k, v in graph.edge_clusters.items()}
            if graph.edge_clusters
            else None,
            "entity_metadata": graph.entity_metadata,
        }

    def _dict_to_graph(self, data: dict) -> Graph:
        """Convert dict to Graph."""
        return Graph(
            entities=set(data.get("entities", [])),
            relations={tuple(r) for r in data.get("relations", [])},
            edges=set(data.get("edges", [])),
            entity_clusters={k: set(v) for k, v in data.get("entity_clusters", {}).items()}
            if data.get("entity_clusters")
            else None,
            edge_clusters={k: set(v) for k, v in data.get("edge_clusters", {}).items()}
            if data.get("edge_clusters")
            else None,
            entity_metadata=data.get("entity_metadata"),
        )

    def save(
        self,
        graph: Graph,
        graph_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        source_documents: Optional[list[str]] = None,
        create_version: bool = True,
    ) -> dict:
        """
        Save a graph to storage.

        Args:
            graph: The Graph object to save
            graph_id: Unique identifier for the graph
            name: Human-readable name for the graph
            description: Description of the graph
            source_documents: List of source document names
            create_version: Whether to create a version history entry

        Returns:
            Updated metadata for the graph
        """
        graph_path = self._get_graph_path(graph_id)
        now = datetime.now().isoformat()

        # Create version history if graph exists and create_version is True
        if create_version and graph_path.exists():
            self._create_version(graph_id)

        # Save graph data
        graph_data = self._graph_to_dict(graph)
        with open(graph_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)

        # Update metadata
        if graph_id not in self.metadata["graphs"]:
            self.metadata["graphs"][graph_id] = {
                "id": graph_id,
                "name": name or graph_id,
                "description": description or "",
                "created_at": now,
                "source_documents": source_documents or [],
                "versions": [],
            }

        graph_meta = self.metadata["graphs"][graph_id]
        graph_meta["updated_at"] = now
        graph_meta["stats"] = {
            "entities": len(graph.entities),
            "relations": len(graph.relations),
            "edges": len(graph.edges),
        }

        if name:
            graph_meta["name"] = name
        if description:
            graph_meta["description"] = description
        if source_documents:
            existing_docs = set(graph_meta.get("source_documents", []))
            existing_docs.update(source_documents)
            graph_meta["source_documents"] = list(existing_docs)

        self._save_metadata()
        logger.info(f"Saved graph '{graph_id}' with {len(graph.entities)} entities")

        return graph_meta

    def _create_version(self, graph_id: str):
        """Create a version snapshot of the current graph."""
        current_path = self._get_graph_path(graph_id)
        if not current_path.exists():
            return

        history_dir = self._get_history_dir(graph_id)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_path = history_dir / f"{timestamp}.json"

        shutil.copy(current_path, version_path)

        # Update version list in metadata
        if graph_id in self.metadata["graphs"]:
            versions = self.metadata["graphs"][graph_id].get("versions", [])
            versions.append({
                "timestamp": timestamp,
                "created_at": datetime.now().isoformat(),
            })
            # Keep only last 50 versions
            self.metadata["graphs"][graph_id]["versions"] = versions[-50:]
            self._save_metadata()

        logger.info(f"Created version snapshot for graph '{graph_id}'")

    def load(self, graph_id: str, version: Optional[str] = None) -> Optional[Graph]:
        """
        Load a graph from storage.

        Args:
            graph_id: The graph identifier
            version: Optional version timestamp to load (defaults to current)

        Returns:
            The loaded Graph or None if not found
        """
        if version:
            graph_path = self._get_history_dir(graph_id) / f"{version}.json"
        else:
            graph_path = self._get_graph_path(graph_id)

        if not graph_path.exists():
            logger.warning(f"Graph '{graph_id}' not found")
            return None

        with open(graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return self._dict_to_graph(data)

    def delete(self, graph_id: str, delete_history: bool = False) -> bool:
        """
        Delete a graph from storage.

        Args:
            graph_id: The graph identifier
            delete_history: Whether to also delete version history

        Returns:
            True if deleted, False if not found
        """
        graph_path = self._get_graph_path(graph_id)

        if not graph_path.exists():
            return False

        graph_path.unlink()

        if delete_history:
            history_dir = self._get_history_dir(graph_id)
            if history_dir.exists():
                shutil.rmtree(history_dir)

        if graph_id in self.metadata["graphs"]:
            del self.metadata["graphs"][graph_id]
            self._save_metadata()

        logger.info(f"Deleted graph '{graph_id}'")
        return True

    def list_graphs(self) -> list[dict]:
        """
        List all saved graphs.

        Returns:
            List of graph metadata dictionaries
        """
        graphs = []
        for graph_id, meta in self.metadata["graphs"].items():
            graph_path = self._get_graph_path(graph_id)
            if graph_path.exists():
                graphs.append(meta)

        # Sort by updated_at descending
        graphs.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return graphs

    def get_versions(self, graph_id: str) -> list[dict]:
        """
        Get version history for a graph.

        Args:
            graph_id: The graph identifier

        Returns:
            List of version metadata
        """
        if graph_id not in self.metadata["graphs"]:
            return []

        return self.metadata["graphs"][graph_id].get("versions", [])

    def exists(self, graph_id: str) -> bool:
        """Check if a graph exists."""
        return self._get_graph_path(graph_id).exists()

    def get_metadata(self, graph_id: str) -> Optional[dict]:
        """Get metadata for a graph."""
        return self.metadata["graphs"].get(graph_id)

    def update_metadata(
        self,
        graph_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Update graph metadata without changing the graph data.

        Args:
            graph_id: The graph identifier
            name: New name for the graph
            description: New description for the graph

        Returns:
            Updated metadata or None if graph not found
        """
        if graph_id not in self.metadata["graphs"]:
            return None

        if name:
            self.metadata["graphs"][graph_id]["name"] = name
        if description:
            self.metadata["graphs"][graph_id]["description"] = description

        self.metadata["graphs"][graph_id]["updated_at"] = datetime.now().isoformat()
        self._save_metadata()

        return self.metadata["graphs"][graph_id]

    def update_vector_index(self, graph_id: str, vector_index: dict) -> Optional[dict]:
        if graph_id not in self.metadata["graphs"]:
            return None

        self.metadata["graphs"][graph_id]["vector_index"] = vector_index
        self.metadata["graphs"][graph_id]["updated_at"] = datetime.now().isoformat()
        self._save_metadata()

        return self.metadata["graphs"][graph_id]


# Global storage instance
_storage: Optional[GraphStorage] = None


def get_storage(storage_dir: Optional[str | Path] = None) -> GraphStorage:
    """Get or create the global storage instance."""
    global _storage
    if _storage is None or (storage_dir and Path(storage_dir) != _storage.storage_dir):
        _storage = GraphStorage(storage_dir)
    return _storage
