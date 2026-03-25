"""
Entitätslese- und Filterdienst.
Liest Knoten aus dem Neo4j-Graph und filtert aussagekräftige Entitätstyp-Knoten heraus.

Ersetzt zep_entity_reader.py — alle Zep-Cloud-Aufrufe durch GraphStorage ersetzt.
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field

from ..utils.logger import get_logger
from ..storage import GraphStorage

logger = get_logger('mirofish.entity_reader')


@dataclass
class EntityNode:
    """Entitätsknoten-Datenstruktur"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    # Zugehörige Kanten
    related_edges: List[Dict[str, Any]] = field(default_factory=list)
    # Zugehörige andere Knoten
    related_nodes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
            "related_edges": self.related_edges,
            "related_nodes": self.related_nodes,
        }

    def get_entity_type(self) -> Optional[str]:
        """Entitätstyp abrufen (Standard-Entity-Label ausschließen)"""
        for label in self.labels:
            if label not in ["Entity", "Node"]:
                return label
        return None


@dataclass
class FilteredEntities:
    """Gefilterte Entitätsmenge"""
    entities: List[EntityNode]
    entity_types: Set[str]
    total_count: int
    filtered_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "entity_types": list(self.entity_types),
            "total_count": self.total_count,
            "filtered_count": self.filtered_count,
        }


class EntityReader:
    """
    Entitätslese- und Filterdienst (über GraphStorage / Neo4j)

    Hauptfunktionen:
    1. Alle Knoten aus dem Graph lesen
    2. Aussagekräftige Entitätstyp-Knoten herausfiltern (Knoten, deren Labels nicht nur "Entity" sind)
    3. Zugehörige Kanten und verknüpfte Knoteninformationen für jede Entität abrufen
    """

    def __init__(self, storage: GraphStorage):
        self.storage = storage

    def get_all_nodes(self, graph_id: str) -> List[Dict[str, Any]]:
        """
        Alle Knoten aus dem Graph abrufen.

        Args:
            graph_id: Graph-ID

        Returns:
            Liste von Knoten.
        """
        logger.info(f"Rufe alle Knoten im Graph {graph_id} ab...")
        nodes = self.storage.get_all_nodes(graph_id)
        logger.info(f"{len(nodes)} Knoten insgesamt abgerufen")
        return nodes

    def get_all_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        """
        Alle Kanten aus dem Graph abrufen.

        Args:
            graph_id: Graph-ID

        Returns:
            Liste von Kanten.
        """
        logger.info(f"Rufe alle Kanten im Graph {graph_id} ab...")
        edges = self.storage.get_all_edges(graph_id)
        logger.info(f"{len(edges)} Kanten insgesamt abgerufen")
        return edges

    def get_node_edges(self, node_uuid: str) -> List[Dict[str, Any]]:
        """
        Alle zugehörigen Kanten für einen bestimmten Knoten abrufen.

        Args:
            node_uuid: Knoten-UUID

        Returns:
            Liste von Kanten.
        """
        try:
            return self.storage.get_node_edges(node_uuid)
        except Exception as e:
            logger.warning(f"Fehler beim Abrufen der Kanten für Knoten {node_uuid}: {str(e)}")
            return []

    def filter_defined_entities(
        self,
        graph_id: str,
        defined_entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True
    ) -> FilteredEntities:
        """
        Knoten mit aussagekräftigen Entitätstypen filtern und extrahieren.

        Filterlogik:
        - Wenn die Labels eines Knotens nur "Entity" enthalten, hat er keinen aussagekräftigen Typ und wird übersprungen.
        - Wenn die Labels eines Knotens andere Labels als "Entity" und "Node" enthalten, hat er einen aussagekräftigen Typ und wird beibehalten.

        Args:
            graph_id: Graph-ID
            defined_entity_types: Optionale Liste von Entitätstypen zum Filtern. Wenn angegeben, werden nur Entitäten beibehalten, die einem dieser Typen entsprechen.
            enrich_with_edges: Ob zugehörige Kanteninformationen für jede Entität abgerufen werden sollen.

        Returns:
            FilteredEntities: Gefilterte Entitätssammlung.
        """
        logger.info(f"Starte Filterung von Entitäten im Graph {graph_id}...")

        # Alle Knoten abrufen
        all_nodes = self.get_all_nodes(graph_id)
        total_count = len(all_nodes)

        # Alle Kanten abrufen (für nachfolgende Zuordnungssuche)
        all_edges = self.get_all_edges(graph_id) if enrich_with_edges else []

        # Zuordnung von Knoten-UUID zu Knotendaten erstellen
        node_map = {n["uuid"]: n for n in all_nodes}

        # Entitäten filtern, die den Kriterien entsprechen
        filtered_entities = []
        entity_types_found: Set[str] = set()

        for node in all_nodes:
            labels = node.get("labels", [])

            # Filterlogik: Labels müssen Labels außer "Entity" und "Node" enthalten
            custom_labels = [la for la in labels if la not in ["Entity", "Node"]]

            if not custom_labels:
                # Nur Standard-Labels, überspringen
                continue

            # Wenn vordefinierte Typen angegeben, prüfen ob übereinstimmend
            if defined_entity_types:
                matching_labels = [la for la in custom_labels if la in defined_entity_types]
                if not matching_labels:
                    continue
                entity_type = matching_labels[0]
            else:
                entity_type = custom_labels[0]

            entity_types_found.add(entity_type)

            # Entitätsknoten-Objekt erstellen
            entity = EntityNode(
                uuid=node["uuid"],
                name=node["name"],
                labels=labels,
                summary=node.get("summary", ""),
                attributes=node.get("attributes", {}),
            )

            # Zugehörige Kanten und Knoten abrufen
            if enrich_with_edges:
                related_edges = []
                related_node_uuids: Set[str] = set()

                for edge in all_edges:
                    if edge["source_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "outgoing",
                            "edge_name": edge["name"],
                            "fact": edge.get("fact", ""),
                            "target_node_uuid": edge["target_node_uuid"],
                        })
                        related_node_uuids.add(edge["target_node_uuid"])
                    elif edge["target_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "incoming",
                            "edge_name": edge["name"],
                            "fact": edge.get("fact", ""),
                            "source_node_uuid": edge["source_node_uuid"],
                        })
                        related_node_uuids.add(edge["source_node_uuid"])

                entity.related_edges = related_edges

                # Zugehörige verknüpfte Knoten mit ihren Informationen abrufen
                related_nodes = []
                for related_uuid in related_node_uuids:
                    if related_uuid in node_map:
                        related_node = node_map[related_uuid]
                        related_nodes.append({
                            "uuid": related_node["uuid"],
                            "name": related_node["name"],
                            "labels": related_node.get("labels", []),
                            "summary": related_node.get("summary", ""),
                        })

                entity.related_nodes = related_nodes

            filtered_entities.append(entity)

        logger.info(f"Filterung abgeschlossen: Knoten gesamt {total_count}, gefiltert {len(filtered_entities)}, "
                     f"Entitätstypen: {entity_types_found}")

        return FilteredEntities(
            entities=filtered_entities,
            entity_types=entity_types_found,
            total_count=total_count,
            filtered_count=len(filtered_entities),
        )

    def get_entity_with_context(
        self,
        graph_id: str,
        entity_uuid: str
    ) -> Optional[EntityNode]:
        """
        Eine einzelne Entität mit ihrem vollständigen Kontext (Kanten und zugehörige Knoten) abrufen.

        Optimiert: Verwendet get_node() + get_node_edges() anstatt ALLE Knoten zu laden.
        Ruft zugehörige Knoten nur einzeln nach Bedarf ab.

        Args:
            graph_id: Graph-ID
            entity_uuid: Entitäts-UUID

        Returns:
            EntityNode oder None.
        """
        try:
            # Knoten direkt über UUID abrufen (O(1)-Suche)
            node = self.storage.get_node(entity_uuid)
            if not node:
                return None

            # Kanten für diesen Knoten abrufen (O(Grad) über Cypher)
            edges = self.storage.get_node_edges(entity_uuid)

            # Zugehörige Kanten verarbeiten und zugehörige Knoten-UUIDs sammeln
            related_edges = []
            related_node_uuids: Set[str] = set()

            for edge in edges:
                if edge["source_node_uuid"] == entity_uuid:
                    related_edges.append({
                        "direction": "outgoing",
                        "edge_name": edge["name"],
                        "fact": edge.get("fact", ""),
                        "target_node_uuid": edge["target_node_uuid"],
                    })
                    related_node_uuids.add(edge["target_node_uuid"])
                else:
                    related_edges.append({
                        "direction": "incoming",
                        "edge_name": edge["name"],
                        "fact": edge.get("fact", ""),
                        "source_node_uuid": edge["source_node_uuid"],
                    })
                    related_node_uuids.add(edge["source_node_uuid"])

            # Zugehörige Knoten einzeln abrufen (vermeidet das Laden ALLER Knoten)
            related_nodes = []
            for related_uuid in related_node_uuids:
                related_node = self.storage.get_node(related_uuid)
                if related_node:
                    related_nodes.append({
                        "uuid": related_node["uuid"],
                        "name": related_node["name"],
                        "labels": related_node.get("labels", []),
                        "summary": related_node.get("summary", ""),
                    })

            return EntityNode(
                uuid=node["uuid"],
                name=node["name"],
                labels=node.get("labels", []),
                summary=node.get("summary", ""),
                attributes=node.get("attributes", {}),
                related_edges=related_edges,
                related_nodes=related_nodes,
            )

        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Entität {entity_uuid}: {str(e)}")
            return None

    def get_entities_by_type(
        self,
        graph_id: str,
        entity_type: str,
        enrich_with_edges: bool = True
    ) -> List[EntityNode]:
        """
        Alle Entitäten eines bestimmten Typs abrufen.

        Args:
            graph_id: Graph-ID
            entity_type: Entitätstyp (z.B. "Student", "PublicFigure" usw.)
            enrich_with_edges: Ob zugehörige Kanteninformationen für jede Entität abgerufen werden sollen.

        Returns:
            Liste von Entitäten des angegebenen Typs.
        """
        result = self.filter_defined_entities(
            graph_id=graph_id,
            defined_entity_types=[entity_type],
            enrich_with_edges=enrich_with_edges
        )
        return result.entities
