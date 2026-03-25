"""
GraphStorage — abstrakte Schnittstelle für Graph-Speicher-Backends.

Alle Zep-Cloud-Aufrufe werden durch diese Abstraktion ersetzt.
Aktuelle Implementierung: Neo4jStorage (neo4j_storage.py).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable


class GraphStorage(ABC):
    """Abstrakte Schnittstelle für Graph-Speicher-Backends."""

    # --- Graph-Lebenszyklus ---

    @abstractmethod
    def create_graph(self, name: str, description: str = "") -> str:
        """Neuen Graph erstellen. Gibt graph_id zurück."""

    @abstractmethod
    def delete_graph(self, graph_id: str) -> None:
        """Graph und alle seine Knoten/Kanten löschen."""

    @abstractmethod
    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]) -> None:
        """Ontologie (Entitätstypen + Beziehungstypen) für einen Graph speichern."""

    @abstractmethod
    def get_ontology(self, graph_id: str) -> Dict[str, Any]:
        """Gespeicherte Ontologie für einen Graph abrufen."""

    # --- Daten hinzufügen ---

    @abstractmethod
    def add_text(self, graph_id: str, text: str) -> str:
        """
        Text verarbeiten: NER/RE → Knoten/Kanten erstellen → episode_id zurückgeben.
        Dies ist synchron (im Gegensatz zu Zep Clouds asynchronen Episoden).
        """

    @abstractmethod
    def add_text_batch(
        self,
        graph_id: str,
        chunks: List[str],
        batch_size: int = 3,
        progress_callback: Optional[Callable] = None,
    ) -> List[str]:
        """Textabschnitte batchweise hinzufügen. Gibt Liste von episode_ids zurück."""

    @abstractmethod
    def wait_for_processing(
        self,
        episode_ids: List[str],
        progress_callback: Optional[Callable] = None,
        timeout: int = 600,
    ) -> None:
        """
        Auf Verarbeitung von Episoden warten.
        Für Neo4j: No-Op (synchrone Verarbeitung).
        Aus API-Kompatibilität mit Zep-Ära-Aufrufern beibehalten.
        """

    # --- Knoten lesen ---

    @abstractmethod
    def get_all_nodes(self, graph_id: str, limit: int = 2000) -> List[Dict[str, Any]]:
        """Alle Knoten in einem Graph abrufen (mit optionalem Limit)."""

    @abstractmethod
    def get_node(self, uuid: str) -> Optional[Dict[str, Any]]:
        """Einzelnen Knoten über UUID abrufen."""

    @abstractmethod
    def get_node_edges(self, node_uuid: str) -> List[Dict[str, Any]]:
        """Alle mit einem Knoten verbundenen Kanten abrufen (O(1) über Cypher, kein Full-Scan)."""

    @abstractmethod
    def get_nodes_by_label(self, graph_id: str, label: str) -> List[Dict[str, Any]]:
        """Knoten nach Entitätstyp-Label gefiltert abrufen."""

    # --- Kanten lesen ---

    @abstractmethod
    def get_all_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        """Alle Kanten in einem Graph abrufen."""

    # --- Suche ---

    @abstractmethod
    def search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
    ):
        """
        Hybridsuche (Vektor + Schlüsselwort) über Graph-Daten.

        Args:
            graph_id: Graph zum Durchsuchen
            query: Suchanfragetext
            limit: Maximale Ergebnisse
            scope: "edges", "nodes" oder "both"

        Returns:
            Dict mit 'edges'- und/oder 'nodes'-Listen (wird von GraphToolsService in SearchResult verpackt)
        """

    # --- Graph-Informationen ---

    @abstractmethod
    def get_graph_info(self, graph_id: str) -> Dict[str, Any]:
        """Graph-Metadaten abrufen (Knotenanzahl, Kantenanzahl, Entitätstypen)."""

    @abstractmethod
    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        """
        Vollständige Graph-Daten abrufen (angereichertes Format für Frontend).

        Gibt Dict zurück mit:
            graph_id, nodes, edges, node_count, edge_count
        Kanten-Dicts enthalten abgeleitete Felder: fact_type, source_node_name, target_node_name
        """
