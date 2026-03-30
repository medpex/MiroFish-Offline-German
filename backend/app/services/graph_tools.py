"""
Graph-Abruf-Werkzeugdienst
Kapselt Graph-Suche, Knotenabruf, Kantenabfragen und weitere Werkzeuge für den Report Agent.

Ersetzt zep_tools.py — alle Zep-Cloud-Aufrufe durch GraphStorage ersetzt.

Kern-Abrufwerkzeuge (optimiert):
1. InsightForge (Tiefenanalyse-Abruf) - Leistungsstärkste Hybridsuche, generiert automatisch Unterfragen und mehrdimensionalen Abruf
2. PanoramaSearch (Breitensuche) - Umfassende Übersicht erhalten, einschließlich abgelaufener Inhalte
3. QuickSearch (Einfache Suche) - Schnellabruf
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from ..utils.logger import get_logger
from ..utils.llm_client import LLMClient
from ..storage import GraphStorage

logger = get_logger('mirofish.graph_tools')


@dataclass
class SearchResult:
    """Suchergebnis"""
    facts: List[str]
    edges: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    query: str
    total_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": self.facts,
            "edges": self.edges,
            "nodes": self.nodes,
            "query": self.query,
            "total_count": self.total_count
        }

    def to_text(self) -> str:
        """In Textformat für LLM-Verständnis umwandeln"""
        text_parts = [f"Suchanfrage: {self.query}", f"{self.total_count} verwandte Ergebnisse gefunden"]

        if self.facts:
            text_parts.append("\n### Verwandte Fakten:")
            for i, fact in enumerate(self.facts, 1):
                text_parts.append(f"{i}. {fact}")

        return "\n".join(text_parts)


@dataclass
class NodeInfo:
    """Knoteninformation"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes
        }

    def to_text(self) -> str:
        """In Textformat umwandeln"""
        entity_type = next((la for la in self.labels if la not in ["Entität", "Node"]), "Unbekannter Typ")
        return f"Entität: {self.name} (Typ: {entity_type})\nZusammenfassung: {self.summary}"


@dataclass
class EdgeInfo:
    """Kanteninformation"""
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    source_node_name: Optional[str] = None
    target_node_name: Optional[str] = None
    # Zeitliche Informationen (können in Neo4j fehlen — für Schnittstellenkompatibilität beibehalten)
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "fact": self.fact,
            "source_node_uuid": self.source_node_uuid,
            "target_node_uuid": self.target_node_uuid,
            "source_node_name": self.source_node_name,
            "target_node_name": self.target_node_name,
            "created_at": self.created_at,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "expired_at": self.expired_at
        }

    def to_text(self, include_temporal: bool = False) -> str:
        """In Textformat umwandeln"""
        source = self.source_node_name or self.source_node_uuid[:8]
        target = self.target_node_name or self.target_node_uuid[:8]
        base_text = f"Beziehung: {source} --[{self.name}]--> {target}\nFakt: {self.fact}"

        if include_temporal:
            valid_at = self.valid_at or "Unbekannt"
            invalid_at = self.invalid_at or "Gegenwart"
            base_text += f"\nZeitraum: {valid_at} - {invalid_at}"
            if self.expired_at:
                base_text += f" (Abgelaufen: {self.expired_at})"

        return base_text

    @property
    def is_expired(self) -> bool:
        """Ob bereits abgelaufen"""
        return self.expired_at is not None

    @property
    def is_invalid(self) -> bool:
        """Ob bereits ungültig"""
        return self.invalid_at is not None


@dataclass
class InsightForgeResult:
    """
    Tiefenanalyse-Abrufergebnis (InsightForge)
    Enthält Abrufergebnisse aus mehreren Unterfragen und integrierte Analyse
    """
    query: str
    simulation_requirement: str
    sub_queries: List[str]

    # Abrufergebnisse nach Dimension
    semantic_facts: List[str] = field(default_factory=list)
    entity_insights: List[Dict[str, Any]] = field(default_factory=list)
    relationship_chains: List[str] = field(default_factory=list)

    # Statistische Informationen
    total_facts: int = 0
    total_entities: int = 0
    total_relationships: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "simulation_requirement": self.simulation_requirement,
            "sub_queries": self.sub_queries,
            "semantic_facts": self.semantic_facts,
            "entity_insights": self.entity_insights,
            "relationship_chains": self.relationship_chains,
            "total_facts": self.total_facts,
            "total_entities": self.total_entities,
            "total_relationships": self.total_relationships
        }

    def to_text(self) -> str:
        """In detailliertes Textformat für LLM-Verständnis umwandeln"""
        text_parts = [
            f"## Zukunftsprognose – Tiefenanalyse",
            f"Analyseabfrage: {self.query}",
            f"Prognoseszenario: {self.simulation_requirement}",
            f"\n### Prognosedaten-Statistik",
            f"- Verwandte Prognosefakten: {self.total_facts}",
            f"- Beteiligte Entitäten: {self.total_entities}",
            f"- Beziehungsketten: {self.total_relationships}"
        ]

        if self.sub_queries:
            text_parts.append(f"\n### Analyse-Unterfragen")
            for i, sq in enumerate(self.sub_queries, 1):
                text_parts.append(f"{i}. {sq}")

        if self.semantic_facts:
            text_parts.append(f"\n### Schlüsselfakten (Bitte im Bericht wörtlich zitieren)")
            for i, fact in enumerate(self.semantic_facts, 1):
                text_parts.append(f'{i}. "{fact}"')

        if self.entity_insights:
            text_parts.append(f"\n### Kernentitäten")
            for entity in self.entity_insights:
                text_parts.append(f"- **{entity.get('name', 'Unbekannt')}** ({entity.get('type', 'Entität')})")
                if entity.get('summary'):
                    text_parts.append(f"  Zusammenfassung: \"{entity.get('summary')}\"")
                if entity.get('related_facts'):
                    text_parts.append(f"  Verwandte Fakten: {len(entity.get('related_facts', []))} Fakten")

        if self.relationship_chains:
            text_parts.append(f"\n### Beziehungsketten")
            for chain in self.relationship_chains:
                text_parts.append(f"- {chain}")

        return "\n".join(text_parts)


@dataclass
class PanoramaResult:
    """
    Breitensuchergebnis (Panorama)
    Enthält alle verwandten Informationen, einschließlich abgelaufener Inhalte
    """
    query: str

    all_nodes: List[NodeInfo] = field(default_factory=list)
    all_edges: List[EdgeInfo] = field(default_factory=list)
    active_facts: List[str] = field(default_factory=list)
    historical_facts: List[str] = field(default_factory=list)

    total_nodes: int = 0
    total_edges: int = 0
    active_count: int = 0
    historical_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "all_nodes": [n.to_dict() for n in self.all_nodes],
            "all_edges": [e.to_dict() for e in self.all_edges],
            "active_facts": self.active_facts,
            "historical_facts": self.historical_facts,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "active_count": self.active_count,
            "historical_count": self.historical_count
        }

    def to_text(self) -> str:
        """In Textformat umwandeln (vollständige Version, ohne Kürzung)"""
        text_parts = [
            f"## Breitensuchergebnisse (Zukunfts-Panoramaansicht)",
            f"Abfrage: {self.query}",
            f"\n### Statistik",
            f"- Gesamtanzahl Knoten: {self.total_nodes}",
            f"- Gesamtanzahl Kanten: {self.total_edges}",
            f"- Aktuell gültige Fakten: {self.active_count}",
            f"- Historische/abgelaufene Fakten: {self.historical_count}"
        ]

        if self.active_facts:
            text_parts.append(f"\n### Aktuell gültige Fakten (Simulationsergebnisse im Wortlaut)")
            for i, fact in enumerate(self.active_facts, 1):
                text_parts.append(f'{i}. "{fact}"')

        if self.historical_facts:
            text_parts.append(f"\n### Historische/abgelaufene Fakten (Entwicklungsverlauf)")
            for i, fact in enumerate(self.historical_facts, 1):
                text_parts.append(f'{i}. "{fact}"')

        if self.all_nodes:
            text_parts.append(f"\n### Beteiligte Entitäten")
            for node in self.all_nodes:
                entity_type = next((la for la in node.labels if la not in ["Entität", "Node"]), "Entität")
                text_parts.append(f"- **{node.name}** ({entity_type})")

        return "\n".join(text_parts)


@dataclass
class AgentInterview:
    """Einzelnes Agent-Interviewergebnis"""
    agent_name: str
    agent_role: str
    agent_bio: str
    question: str
    response: str
    key_quotes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "agent_bio": self.agent_bio,
            "question": self.question,
            "response": self.response,
            "key_quotes": self.key_quotes
        }

    def to_text(self) -> str:
        text = f"**{self.agent_name}** ({self.agent_role})\n"
        text += f"_Biografie: {self.agent_bio}_\n\n"
        text += f"**F:** {self.question}\n\n"
        text += f"**A:** {self.response}\n"
        if self.key_quotes:
            text += "\n**Schlüsselzitate:**\n"
            for quote in self.key_quotes:
                clean_quote = quote.replace('\u201c', '').replace('\u201d', '').replace('"', '')
                clean_quote = clean_quote.replace('\u300c', '').replace('\u300d', '')
                clean_quote = clean_quote.strip()
                while clean_quote and clean_quote[0] in '，,；;：:、。！？\n\r\t ':
                    clean_quote = clean_quote[1:]
                skip = False
                for d in '123456789':
                    if f'\u95ee\u9898{d}' in clean_quote:
                        skip = True
                        break
                if skip:
                    continue
                if len(clean_quote) > 150:
                    dot_pos = clean_quote.find('\u3002', 80)
                    if dot_pos > 0:
                        clean_quote = clean_quote[:dot_pos + 1]
                    else:
                        clean_quote = clean_quote[:147] + "..."
                if clean_quote and len(clean_quote) >= 10:
                    text += f'> "{clean_quote}"\n'
        return text


@dataclass
class InterviewResult:
    """
    Interviewergebnis
    Enthält Interviewantworten von mehreren simulierten Agents
    """
    interview_topic: str
    interview_questions: List[str]

    selected_agents: List[Dict[str, Any]] = field(default_factory=list)
    interviews: List[AgentInterview] = field(default_factory=list)

    selection_reasoning: str = ""
    summary: str = ""

    total_agents: int = 0
    interviewed_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interview_topic": self.interview_topic,
            "interview_questions": self.interview_questions,
            "selected_agents": self.selected_agents,
            "interviews": [i.to_dict() for i in self.interviews],
            "selection_reasoning": self.selection_reasoning,
            "summary": self.summary,
            "total_agents": self.total_agents,
            "interviewed_count": self.interviewed_count
        }

    def to_text(self) -> str:
        """In detailliertes Textformat für LLM-Verständnis und Berichtsreferenz umwandeln"""
        text_parts = [
            "## Tiefeninterview-Bericht",
            f"**Interviewthema:** {self.interview_topic}",
            f"**Befragte:** {self.interviewed_count} / {self.total_agents} simulierte Agents",
            "\n### Auswahlbegründung",
            self.selection_reasoning or "(Automatische Auswahl)",
            "\n---",
            "\n### Interviewprotokolle",
        ]

        if self.interviews:
            for i, interview in enumerate(self.interviews, 1):
                text_parts.append(f"\n#### Interview Nr. {i}: {interview.agent_name}")
                text_parts.append(interview.to_text())
                text_parts.append("\n---")
        else:
            text_parts.append("(Keine Interviewaufzeichnungen)\n\n---")

        text_parts.append("\n### Interviewzusammenfassung & Schlüsselerkenntnisse")
        text_parts.append(self.summary or "(Keine Zusammenfassung)")

        return "\n".join(text_parts)


class GraphToolsService:
    """
    Graph-Abruf-Werkzeugdienst (via GraphStorage / Neo4j)

    [Kern-Abrufwerkzeuge - Optimiert]
    1. insight_forge - Tiefenanalyse-Abruf (Leistungsstärkstes Werkzeug, generiert automatisch Unterfragen, mehrdimensionaler Abruf)
    2. panorama_search - Breitensuche (Umfassende Übersicht erhalten, einschließlich abgelaufener Inhalte)
    3. quick_search - Einfache Suche (Schnellabruf)
    4. interview_agents - Tiefeninterview (Simulierte Agents interviewen, Mehrperspektiven-Erkenntnisse gewinnen)

    [Basiswerkzeuge]
    - search_graph - Graph-Semantiksuche
    - get_all_nodes - Alle Knoten im Graph abrufen
    - get_all_edges - Alle Kanten im Graph abrufen (mit zeitlichen Informationen)
    - get_node_detail - Detaillierte Knoteninformationen abrufen
    - get_node_edges - Knotenbeziehungskanten abrufen
    - get_entities_by_type - Entitäten nach Typ abrufen
    - get_entity_summary - Entitäts-Beziehungszusammenfassung abrufen
    """

    def __init__(self, storage: GraphStorage, llm_client: Optional[LLMClient] = None):
        self.storage = storage
        self._llm_client = llm_client
        logger.info("GraphToolsService-Initialisierung abgeschlossen")

    @property
    def llm(self) -> LLMClient:
        """Verzögerte Initialisierung des LLM-Clients"""
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    # ========== Basiswerkzeuge ==========

    def search_graph(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        """
        Graph-Semantiksuche (Hybrid: Vektor + BM25 via Neo4j)

        Args:
            graph_id: Graph-ID
            query: Suchanfrage
            limit: Anzahl der zurückzugebenden Ergebnisse
            scope: Suchbereich, "edges" oder "nodes" oder "both"

        Returns:
            SearchResult
        """
        logger.info(f"Graph-Suche: graph_id={graph_id}, query={query[:50]}...")

        try:
            search_results = self.storage.search(
                graph_id=graph_id,
                query=query,
                limit=limit,
                scope=scope,
            )

            facts = []
            edges = []
            nodes = []

            # Kantenergebnisse parsen
            if hasattr(search_results, 'edges'):
                edge_list = search_results.edges
            elif isinstance(search_results, dict) and 'edges' in search_results:
                edge_list = search_results['edges']
            else:
                edge_list = []

            for edge in edge_list:
                if isinstance(edge, dict):
                    fact = edge.get('fact', '')
                    if fact:
                        facts.append(fact)
                    edges.append({
                        "uuid": edge.get('uuid', ''),
                        "name": edge.get('name', ''),
                        "fact": fact,
                        "source_node_uuid": edge.get('source_node_uuid', ''),
                        "target_node_uuid": edge.get('target_node_uuid', ''),
                    })

            # Knotenergebnisse parsen
            if hasattr(search_results, 'nodes'):
                node_list = search_results.nodes
            elif isinstance(search_results, dict) and 'nodes' in search_results:
                node_list = search_results['nodes']
            else:
                node_list = []

            for node in node_list:
                if isinstance(node, dict):
                    nodes.append({
                        "uuid": node.get('uuid', ''),
                        "name": node.get('name', ''),
                        "labels": node.get('labels', []),
                        "summary": node.get('summary', ''),
                    })
                    summary = node.get('summary', '')
                    if summary:
                        facts.append(f"[{node.get('name', '')}]: {summary}")

            logger.info(f"Suche abgeschlossen: {len(facts)} verwandte Fakten gefunden")

            return SearchResult(
                facts=facts,
                edges=edges,
                nodes=nodes,
                query=query,
                total_count=len(facts)
            )

        except Exception as e:
            logger.warning(f"Graph-Suche fehlgeschlagen, Fallback auf lokale Suche: {str(e)}")
            return self._local_search(graph_id, query, limit, scope)

    def _local_search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        """
        Lokale Schlüsselwort-Suche (Fallback-Verfahren)
        """
        logger.info(f"Verwende lokale Suche: query={query[:30]}...")

        facts = []
        edges_result = []
        nodes_result = []

        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('，', ' ').split() if len(w.strip()) > 1]

        def match_score(text: str) -> int:
            if not text:
                return 0
            text_lower = text.lower()
            if query_lower in text_lower:
                return 100
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 10
            return score

        try:
            if scope in ["edges", "both"]:
                all_edges = self.storage.get_all_edges(graph_id)
                scored_edges = []
                for edge in all_edges:
                    score = match_score(edge.get("fact", "")) + match_score(edge.get("name", ""))
                    if score > 0:
                        scored_edges.append((score, edge))

                scored_edges.sort(key=lambda x: x[0], reverse=True)

                for score, edge in scored_edges[:limit]:
                    fact = edge.get("fact", "")
                    if fact:
                        facts.append(fact)
                    edges_result.append({
                        "uuid": edge.get("uuid", ""),
                        "name": edge.get("name", ""),
                        "fact": fact,
                        "source_node_uuid": edge.get("source_node_uuid", ""),
                        "target_node_uuid": edge.get("target_node_uuid", ""),
                    })

            if scope in ["nodes", "both"]:
                all_nodes = self.storage.get_all_nodes(graph_id)
                scored_nodes = []
                for node in all_nodes:
                    score = match_score(node.get("name", "")) + match_score(node.get("summary", ""))
                    if score > 0:
                        scored_nodes.append((score, node))

                scored_nodes.sort(key=lambda x: x[0], reverse=True)

                for score, node in scored_nodes[:limit]:
                    nodes_result.append({
                        "uuid": node.get("uuid", ""),
                        "name": node.get("name", ""),
                        "labels": node.get("labels", []),
                        "summary": node.get("summary", ""),
                    })
                    summary = node.get("summary", "")
                    if summary:
                        facts.append(f"[{node.get('name', '')}]: {summary}")

            logger.info(f"Lokale Suche abgeschlossen: {len(facts)} verwandte Fakten gefunden")

        except Exception as e:
            logger.error(f"Lokale Suche fehlgeschlagen: {str(e)}")

        return SearchResult(
            facts=facts,
            edges=edges_result,
            nodes=nodes_result,
            query=query,
            total_count=len(facts)
        )

    def get_all_nodes(self, graph_id: str) -> List[NodeInfo]:
        """Alle Knoten im Graph abrufen"""
        logger.info(f"Abruf aller Knoten im Graph {graph_id}...")

        raw_nodes = self.storage.get_all_nodes(graph_id)

        result = []
        for node in raw_nodes:
            result.append(NodeInfo(
                uuid=node.get("uuid", ""),
                name=node.get("name", ""),
                labels=node.get("labels", []),
                summary=node.get("summary", ""),
                attributes=node.get("attributes", {})
            ))

        logger.info(f"{len(result)} Knoten abgerufen")
        return result

    def get_all_edges(self, graph_id: str, include_temporal: bool = True) -> List[EdgeInfo]:
        """Alle Kanten im Graph abrufen (mit zeitlichen Informationen)"""
        logger.info(f"Abruf aller Kanten im Graph {graph_id}...")

        raw_edges = self.storage.get_all_edges(graph_id)

        result = []
        for edge in raw_edges:
            edge_info = EdgeInfo(
                uuid=edge.get("uuid", ""),
                name=edge.get("name", ""),
                fact=edge.get("fact", ""),
                source_node_uuid=edge.get("source_node_uuid", ""),
                target_node_uuid=edge.get("target_node_uuid", "")
            )

            if include_temporal:
                edge_info.created_at = edge.get("created_at")
                edge_info.valid_at = edge.get("valid_at")
                edge_info.invalid_at = edge.get("invalid_at")
                edge_info.expired_at = edge.get("expired_at")

            result.append(edge_info)

        logger.info(f"{len(result)} Kanten abgerufen")
        return result

    def get_node_detail(self, node_uuid: str) -> Optional[NodeInfo]:
        """Detaillierte Informationen zu einem einzelnen Knoten abrufen"""
        logger.info(f"Abruf der Knotendetails: {node_uuid[:8]}...")

        try:
            node = self.storage.get_node(node_uuid)
            if not node:
                return None

            return NodeInfo(
                uuid=node.get("uuid", ""),
                name=node.get("name", ""),
                labels=node.get("labels", []),
                summary=node.get("summary", ""),
                attributes=node.get("attributes", {})
            )
        except Exception as e:
            logger.error(f"Abruf der Knotendetails fehlgeschlagen: {str(e)}")
            return None

    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[EdgeInfo]:
        """
        Alle mit einem Knoten verbundenen Kanten abrufen

        Optimiert: Verwendet storage.get_node_edges() (O(degree) Cypher)
        anstatt ALLE Kanten zu laden und zu filtern.
        """
        logger.info(f"Abruf der Kanten für Knoten {node_uuid[:8]}...")

        try:
            raw_edges = self.storage.get_node_edges(node_uuid)

            result = []
            for edge in raw_edges:
                result.append(EdgeInfo(
                    uuid=edge.get("uuid", ""),
                    name=edge.get("name", ""),
                    fact=edge.get("fact", ""),
                    source_node_uuid=edge.get("source_node_uuid", ""),
                    target_node_uuid=edge.get("target_node_uuid", ""),
                    created_at=edge.get("created_at"),
                    valid_at=edge.get("valid_at"),
                    invalid_at=edge.get("invalid_at"),
                    expired_at=edge.get("expired_at"),
                ))

            logger.info(f"{len(result)} mit dem Knoten verbundene Kanten gefunden")
            return result

        except Exception as e:
            logger.warning(f"Abruf der Knotenkanten fehlgeschlagen: {str(e)}")
            return []

    def get_entities_by_type(
        self,
        graph_id: str,
        entity_type: str
    ) -> List[NodeInfo]:
        """Entitäten nach Typ abrufen"""
        logger.info(f"Abruf der Entitäten vom Typ {entity_type}...")

        # Optimierte Label-basierte Abfrage aus dem Speicher verwenden
        raw_nodes = self.storage.get_nodes_by_label(graph_id, entity_type)

        result = []
        for node in raw_nodes:
            result.append(NodeInfo(
                uuid=node.get("uuid", ""),
                name=node.get("name", ""),
                labels=node.get("labels", []),
                summary=node.get("summary", ""),
                attributes=node.get("attributes", {})
            ))

        logger.info(f"{len(result)} Entitäten vom Typ {entity_type} gefunden")
        return result

    def get_entity_summary(
        self,
        graph_id: str,
        entity_name: str
    ) -> Dict[str, Any]:
        """Beziehungszusammenfassung für eine bestimmte Entität abrufen"""
        logger.info(f"Abruf der Beziehungszusammenfassung für Entität {entity_name}...")

        search_result = self.search_graph(
            graph_id=graph_id,
            query=entity_name,
            limit=20
        )

        all_nodes = self.get_all_nodes(graph_id)
        entity_node = None
        for node in all_nodes:
            if node.name.lower() == entity_name.lower():
                entity_node = node
                break

        related_edges = []
        if entity_node:
            related_edges = self.get_node_edges(graph_id, entity_node.uuid)

        return {
            "entity_name": entity_name,
            "entity_info": entity_node.to_dict() if entity_node else None,
            "related_facts": search_result.facts,
            "related_edges": [e.to_dict() for e in related_edges],
            "total_relations": len(related_edges)
        }

    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """Statistiken für den Graph abrufen"""
        logger.info(f"Abruf der Statistiken für Graph {graph_id}...")

        nodes = self.get_all_nodes(graph_id)
        edges = self.get_all_edges(graph_id)

        entity_types = {}
        for node in nodes:
            for label in node.labels:
                if label not in ["Entität", "Node"]:
                    entity_types[label] = entity_types.get(label, 0) + 1

        relation_types = {}
        for edge in edges:
            relation_types[edge.name] = relation_types.get(edge.name, 0) + 1

        return {
            "graph_id": graph_id,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entity_types": entity_types,
            "relation_types": relation_types
        }

    def get_simulation_context(
        self,
        graph_id: str,
        simulation_requirement: str,
        limit: int = 30
    ) -> Dict[str, Any]:
        """Simulationsbezogene Kontextinformationen abrufen"""
        logger.info(f"Abruf des Simulationskontexts: {simulation_requirement[:50]}...")

        search_result = self.search_graph(
            graph_id=graph_id,
            query=simulation_requirement,
            limit=limit
        )

        stats = self.get_graph_statistics(graph_id)

        all_nodes = self.get_all_nodes(graph_id)

        entities = []
        for node in all_nodes:
            custom_labels = [la for la in node.labels if la not in ["Entität", "Node"]]
            if custom_labels:
                entities.append({
                    "name": node.name,
                    "type": custom_labels[0],
                    "summary": node.summary
                })

        return {
            "simulation_requirement": simulation_requirement,
            "related_facts": search_result.facts,
            "graph_statistics": stats,
            "entities": entities[:limit],
            "total_entities": len(entities)
        }

    # ========== Kern-Abrufwerkzeuge (optimiert) ==========

    def insight_forge(
        self,
        graph_id: str,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_sub_queries: int = 5
    ) -> InsightForgeResult:
        """
        [InsightForge - Tiefenanalyse-Abruf]

        Die leistungsstärkste Hybrid-Abruffunktion, zerlegt Probleme automatisch und führt mehrdimensionalen Abruf durch:
        1. LLM verwenden, um das Problem in mehrere Unterfragen zu zerlegen
        2. Semantische Suche für jede Unterfrage durchführen
        3. Verwandte Entitäten extrahieren und deren detaillierte Informationen abrufen
        4. Beziehungsketten verfolgen
        5. Alle Ergebnisse integrieren und tiefgehende Erkenntnisse generieren
        """
        logger.info(f"InsightForge Tiefenanalyse-Abruf: {query[:50]}...")

        result = InsightForgeResult(
            query=query,
            simulation_requirement=simulation_requirement,
            sub_queries=[]
        )

        # Schritt 1: LLM zur Generierung von Unterfragen verwenden
        sub_queries = self._generate_sub_queries(
            query=query,
            simulation_requirement=simulation_requirement,
            report_context=report_context,
            max_queries=max_sub_queries
        )
        result.sub_queries = sub_queries
        logger.info(f"{len(sub_queries)} Unterfragen generiert")

        # Schritt 2: Semantische Suche für jede Unterfrage durchführen
        all_facts = []
        all_edges = []
        seen_facts = set()

        for sub_query in sub_queries:
            search_result = self.search_graph(
                graph_id=graph_id,
                query=sub_query,
                limit=15,
                scope="edges"
            )

            for fact in search_result.facts:
                if fact not in seen_facts:
                    all_facts.append(fact)
                    seen_facts.add(fact)

            all_edges.extend(search_result.edges)

        # Auch die ursprüngliche Frage durchsuchen
        main_search = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=20,
            scope="edges"
        )
        for fact in main_search.facts:
            if fact not in seen_facts:
                all_facts.append(fact)
                seen_facts.add(fact)

        result.semantic_facts = all_facts
        result.total_facts = len(all_facts)

        # Schritt 3: Verwandte Entitäts-UUIDs aus Kanten extrahieren
        entity_uuids = set()
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                if source_uuid:
                    entity_uuids.add(source_uuid)
                if target_uuid:
                    entity_uuids.add(target_uuid)

        # Verwandte Entitätsdetails abrufen
        entity_insights = []
        node_map = {}

        for uuid in list(entity_uuids):
            if not uuid:
                continue
            try:
                node = self.get_node_detail(uuid)
                if node:
                    node_map[uuid] = node
                    entity_type = next((la for la in node.labels if la not in ["Entität", "Node"]), "Entität")

                    related_facts = [
                        f for f in all_facts
                        if node.name.lower() in f.lower()
                    ]

                    entity_insights.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "type": entity_type,
                        "summary": node.summary,
                        "related_facts": related_facts
                    })
            except Exception as e:
                logger.debug(f"Knoten {uuid} konnte nicht abgerufen werden: {e}")
                continue

        result.entity_insights = entity_insights
        result.total_entities = len(entity_insights)

        # Schritt 4: Beziehungsketten aufbauen
        relationship_chains = []
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                relation_name = edge_data.get('name', '')

                source_name = node_map.get(source_uuid, NodeInfo('', '', [], '', {})).name or source_uuid[:8]
                target_name = node_map.get(target_uuid, NodeInfo('', '', [], '', {})).name or target_uuid[:8]

                chain = f"{source_name} --[{relation_name}]--> {target_name}"
                if chain not in relationship_chains:
                    relationship_chains.append(chain)

        result.relationship_chains = relationship_chains
        result.total_relationships = len(relationship_chains)

        logger.info(f"InsightForge abgeschlossen: {result.total_facts} Fakten, {result.total_entities} Entitäten, {result.total_relationships} Beziehungen")
        return result

    def _generate_sub_queries(
        self,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_queries: int = 5
    ) -> List[str]:
        """LLM zur Generierung von Unterfragen verwenden"""
        system_prompt = """Du bist ein professioneller Fragenanalyse-Experte. Deine Aufgabe ist es, eine komplexe Frage in mehrere Unterfragen zu zerlegen, die unabhängig in einer simulierten Welt beobachtet werden können.

Anforderungen:
1. Jede Unterfrage sollte spezifisch genug sein, um verwandtes Agent-Verhalten oder Ereignisse in der simulierten Welt zu finden
2. Unterfragen sollten verschiedene Dimensionen der Originalfrage abdecken (z.B. wer, was, warum, wie, wann, wo)
3. Unterfragen sollten relevant für das Simulationsszenario sein
4. Rückgabe im JSON-Format: {"sub_queries": ["Unterfrage 1", "Unterfrage 2", ...]}"""

        user_prompt = f"""Hintergrund der Simulationsanforderung:
{simulation_requirement}

{f"Berichtskontext: {report_context[:500]}" if report_context else ""}

Bitte zerlege die folgende Frage in {max_queries} Unterfragen:
{query}

Gib die Unterfragen als JSON-Liste zurück."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )

            sub_queries = response.get("sub_queries", [])
            return [str(sq) for sq in sub_queries[:max_queries]]

        except Exception as e:
            logger.warning(f"Generierung der Unterfragen fehlgeschlagen: {str(e)}, verwende Standard-Unterfragen")
            return [
                query,
                f"Hauptbeteiligte bei {query}",
                f"Ursachen und Auswirkungen von {query}",
                f"Entwicklungsverlauf von {query}"
            ][:max_queries]

    def panorama_search(
        self,
        graph_id: str,
        query: str,
        include_expired: bool = True,
        limit: int = 50
    ) -> PanoramaResult:
        """
        [PanoramaSearch - Breitensuche]

        Umfassende Panoramaansicht erhalten, einschließlich aller verwandten Inhalte und historischer/abgelaufener Informationen.
        """
        logger.info(f"PanoramaSearch Breitensuche: {query[:50]}...")

        result = PanoramaResult(query=query)

        # Alle Knoten abrufen
        all_nodes = self.get_all_nodes(graph_id)
        node_map = {n.uuid: n for n in all_nodes}
        result.all_nodes = all_nodes
        result.total_nodes = len(all_nodes)

        # Alle Kanten abrufen (einschließlich zeitlicher Informationen)
        all_edges = self.get_all_edges(graph_id, include_temporal=True)
        result.all_edges = all_edges
        result.total_edges = len(all_edges)

        # Fakten kategorisieren
        active_facts = []
        historical_facts = []

        for edge in all_edges:
            if not edge.fact:
                continue

            source_name = node_map.get(edge.source_node_uuid, NodeInfo('', '', [], '', {})).name or edge.source_node_uuid[:8]
            target_name = node_map.get(edge.target_node_uuid, NodeInfo('', '', [], '', {})).name or edge.target_node_uuid[:8]

            is_historical = edge.is_expired or edge.is_invalid

            if is_historical:
                valid_at = edge.valid_at or "Unbekannt"
                invalid_at = edge.invalid_at or edge.expired_at or "Unbekannt"
                fact_with_time = f"[{valid_at} - {invalid_at}] {edge.fact}"
                historical_facts.append(fact_with_time)
            else:
                active_facts.append(edge.fact)

        # Nach Relevanz basierend auf der Abfrage sortieren
        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('，', ' ').split() if len(w.strip()) > 1]

        def relevance_score(fact: str) -> int:
            fact_lower = fact.lower()
            score = 0
            if query_lower in fact_lower:
                score += 100
            for kw in keywords:
                if kw in fact_lower:
                    score += 10
            return score

        active_facts.sort(key=relevance_score, reverse=True)
        historical_facts.sort(key=relevance_score, reverse=True)

        result.active_facts = active_facts[:limit]
        result.historical_facts = historical_facts[:limit] if include_expired else []
        result.active_count = len(active_facts)
        result.historical_count = len(historical_facts)

        logger.info(f"PanoramaSearch abgeschlossen: {result.active_count} gültig, {result.historical_count} historisch")
        return result

    def quick_search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10
    ) -> SearchResult:
        """
        [QuickSearch - Einfache Suche]
        Schnelles und leichtgewichtiges Abrufwerkzeug.
        """
        logger.info(f"QuickSearch Einfache Suche: {query[:50]}...")

        result = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=limit,
            scope="edges"
        )

        logger.info(f"QuickSearch abgeschlossen: {result.total_count} Ergebnisse")
        return result

    def interview_agents(
        self,
        simulation_id: str,
        interview_requirement: str,
        simulation_requirement: str = "",
        max_agents: int = 5,
        custom_questions: List[str] = None
    ) -> InterviewResult:
        """
        [InterviewAgents - Tiefeninterview]

        Ruft die echte OASIS-Interview-API auf, um laufende Simulations-Agents zu interviewen.
        Diese Methode verwendet NICHT GraphStorage — sie ruft SimulationRunner auf
        und liest Agent-Profile von der Festplatte.
        """
        from .simulation_runner import SimulationRunner

        logger.info(f"InterviewAgents Tiefeninterview (echte API): {interview_requirement[:50]}...")

        result = InterviewResult(
            interview_topic=interview_requirement,
            interview_questions=custom_questions or []
        )

        # Schritt 1: Agent-Profildateien einlesen
        profiles = self._load_agent_profiles(simulation_id)

        if not profiles:
            logger.warning(f"Keine Profildateien für Simulation {simulation_id} gefunden")
            result.summary = "Keine Agent-Profildateien für das Interview gefunden"
            return result

        result.total_agents = len(profiles)
        logger.info(f"{len(profiles)} Agent-Profile geladen")

        # Schritt 2: LLM zur Auswahl der zu interviewenden Agents verwenden
        selected_agents, selected_indices, selection_reasoning = self._select_agents_for_interview(
            profiles=profiles,
            interview_requirement=interview_requirement,
            simulation_requirement=simulation_requirement,
            max_agents=max_agents
        )

        result.selected_agents = selected_agents
        result.selection_reasoning = selection_reasoning
        logger.info(f"{len(selected_agents)} Agents für Interview ausgewählt: {selected_indices}")

        # Schritt 3: Interviewfragen generieren
        if not result.interview_questions:
            result.interview_questions = self._generate_interview_questions(
                interview_requirement=interview_requirement,
                simulation_requirement=simulation_requirement,
                selected_agents=selected_agents
            )
            logger.info(f"{len(result.interview_questions)} Interviewfragen generiert")

        combined_prompt = "\n".join([f"{i+1}. {q}" for i, q in enumerate(result.interview_questions)])

        INTERVIEW_PROMPT_PREFIX = (
            "Du wirst interviewt. Bitte kombiniere dein Charakterprofil, alle bisherigen Erinnerungen und Handlungen, "
            "und beantworte die folgenden Fragen direkt als Fließtext.\n"
            "Antwortanforderungen:\n"
            "1. Antworte direkt in natürlicher Sprache, rufe keine Werkzeuge auf\n"
            "2. Gib kein JSON-Format oder Werkzeugaufruf-Format zurück\n"
            "3. Verwende keine Markdown-Überschriften (z.B. #, ##, ###)\n"
            "4. Beantworte die Fragen der Reihe nach, jede Antwort beginnt mit 'Frage X:' (X ist die Fragennummer)\n"
            "5. Trenne jede Antwort durch eine Leerzeile\n"
            "6. Gib inhaltliche Antworten, mindestens 2-3 Sätze pro Frage\n\n"
        )
        optimized_prompt = f"{INTERVIEW_PROMPT_PREFIX}{combined_prompt}"

        # Schritt 4: Echte Interview-API aufrufen
        try:
            interviews_request = []
            for agent_idx in selected_indices:
                interviews_request.append({
                    "agent_id": agent_idx,
                    "prompt": optimized_prompt
                })

            logger.info(f"Batch-Interview-API aufrufen (Duale Plattform): {len(interviews_request)} Agents")

            api_result = SimulationRunner.interview_agents_batch(
                simulation_id=simulation_id,
                interviews=interviews_request,
                platform=None,
                timeout=180.0
            )

            logger.info(f"Interview-API zurückgegeben: {api_result.get('interviews_count', 0)} Ergebnisse, success={api_result.get('success')}")

            if not api_result.get("success", False):
                error_msg = api_result.get("error", "Unbekannter Fehler")
                logger.warning(f"Interview-API-Aufruf fehlgeschlagen: {error_msg}")
                result.summary = f"Interview-API-Aufruf fehlgeschlagen: {error_msg}. Bitte den Status der OASIS-Simulationsumgebung prüfen."
                return result

            # Schritt 5: API-Antwort parsen
            api_data = api_result.get("result", {})
            results_dict = api_data.get("results", {}) if isinstance(api_data, dict) else {}

            for i, agent_idx in enumerate(selected_indices):
                agent = selected_agents[i]
                agent_name = agent.get("realname", agent.get("username", f"Agent_{agent_idx}"))
                agent_role = agent.get("profession", "Unbekannt")
                agent_bio = agent.get("bio", "")

                twitter_result = results_dict.get(f"twitter_{agent_idx}", {})
                reddit_result = results_dict.get(f"reddit_{agent_idx}", {})

                twitter_response = twitter_result.get("response", "")
                reddit_response = reddit_result.get("response", "")

                twitter_response = self._clean_tool_call_response(twitter_response)
                reddit_response = self._clean_tool_call_response(reddit_response)

                twitter_text = twitter_response if twitter_response else "(Keine Antwort von dieser Plattform)"
                reddit_text = reddit_response if reddit_response else "(Keine Antwort von dieser Plattform)"
                response_text = f"[Twitter-Plattform-Antwort]\n{twitter_text}\n\n[Reddit-Plattform-Antwort]\n{reddit_text}"

                import re
                combined_responses = f"{twitter_response} {reddit_response}"

                clean_text = re.sub(r'#{1,6}\s+', '', combined_responses)
                clean_text = re.sub(r'\{[^}]*tool_name[^}]*\}', '', clean_text)
                clean_text = re.sub(r'[*_`|>~\-]{2,}', '', clean_text)
                clean_text = re.sub(r'Question\d+[：:]\s*', '', clean_text)
                clean_text = re.sub(r'【[^】]+】', '', clean_text)

                sentences = re.split(r'[。！？]', clean_text)
                meaningful = [
                    s.strip() for s in sentences
                    if 20 <= len(s.strip()) <= 150
                    and not re.match(r'^[\s\W，,；;：:、]+', s.strip())
                    and not s.strip().startswith(('{', 'Question'))
                ]
                meaningful.sort(key=len, reverse=True)
                key_quotes = [s + "。" for s in meaningful[:3]]

                if not key_quotes:
                    paired = re.findall(r'\u201c([^\u201c\u201d]{15,100})\u201d', clean_text)
                    paired += re.findall(r'\u300c([^\u300c\u300d]{15,100})\u300d', clean_text)
                    key_quotes = [q for q in paired if not re.match(r'^[，,；;：:、]', q)][:3]

                interview = AgentInterview(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    agent_bio=agent_bio[:1000],
                    question=combined_prompt,
                    response=response_text,
                    key_quotes=key_quotes[:5]
                )
                result.interviews.append(interview)

            result.interviewed_count = len(result.interviews)

        except ValueError as e:
            logger.warning(f"Interview-API-Aufruf fehlgeschlagen (Umgebung nicht aktiv?): {e}")
            result.summary = f"Interview fehlgeschlagen: {str(e)}. Die Simulationsumgebung ist möglicherweise geschlossen. Bitte sicherstellen, dass die OASIS-Umgebung läuft."
            return result
        except Exception as e:
            logger.error(f"Interview-API-Aufruf Ausnahme: {e}")
            import traceback
            logger.error(traceback.format_exc())
            result.summary = f"Während des Interviewprozesses ist ein Fehler aufgetreten: {str(e)}"
            return result

        # Schritt 6: Interviewzusammenfassung generieren
        if result.interviews:
            result.summary = self._generate_interview_summary(
                interviews=result.interviews,
                interview_requirement=interview_requirement
            )

        logger.info(f"InterviewAgents abgeschlossen: {result.interviewed_count} Agents interviewt (Duale Plattform)")
        return result

    @staticmethod
    def _clean_tool_call_response(response: str) -> str:
        """JSON-Werkzeugaufruf-Wrapper in Agent-Antworten bereinigen und tatsächlichen Inhalt extrahieren"""
        if not response or not response.strip().startswith('{'):
            return response
        text = response.strip()
        if 'tool_name' not in text[:80]:
            return response
        import re as _re
        try:
            data = json.loads(text)
            if isinstance(data, dict) and 'arguments' in data:
                for key in ('content', 'text', 'body', 'message', 'reply'):
                    if key in data['arguments']:
                        return str(data['arguments'][key])
        except (json.JSONDecodeError, KeyError, TypeError):
            match = _re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            if match:
                return match.group(1).replace('\\n', '\n').replace('\\"', '"')
        return response

    def _load_agent_profiles(self, simulation_id: str) -> List[Dict[str, Any]]:
        """Agent-Profildateien für die Simulation laden"""
        import os
        import csv

        sim_dir = os.path.join(
            os.path.dirname(__file__),
            f'../../uploads/simulations/{simulation_id}'
        )

        profiles = []

        # Bevorzugt Reddit-JSON-Format lesen
        reddit_profile_path = os.path.join(sim_dir, "reddit_profiles.json")
        if os.path.exists(reddit_profile_path):
            try:
                with open(reddit_profile_path, 'r', encoding='utf-8') as f:
                    profiles = json.load(f)
                logger.info(f"{len(profiles)} Profile aus reddit_profiles.json geladen")
                return profiles
            except Exception as e:
                logger.warning(f"Lesen von reddit_profiles.json fehlgeschlagen: {e}")

        # Twitter-CSV-Format versuchen
        twitter_profile_path = os.path.join(sim_dir, "twitter_profiles.csv")
        if os.path.exists(twitter_profile_path):
            try:
                with open(twitter_profile_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        profiles.append({
                            "realname": row.get("name", ""),
                            "username": row.get("username", ""),
                            "bio": row.get("description", ""),
                            "persona": row.get("user_char", ""),
                            "profession": "Unbekannt"
                        })
                logger.info(f"{len(profiles)} Profile aus twitter_profiles.csv geladen")
                return profiles
            except Exception as e:
                logger.warning(f"Lesen von twitter_profiles.csv fehlgeschlagen: {e}")

        return profiles

    def _select_agents_for_interview(
        self,
        profiles: List[Dict[str, Any]],
        interview_requirement: str,
        simulation_requirement: str,
        max_agents: int
    ) -> tuple:
        """LLM zur Auswahl der zu interviewenden Agents verwenden"""

        agent_summaries = []
        for i, profile in enumerate(profiles):
            summary = {
                "index": i,
                "name": profile.get("realname", profile.get("username", f"Agent_{i}")),
                "profession": profile.get("profession", "Unbekannt"),
                "bio": profile.get("bio", "")[:200],
                "interested_topics": profile.get("interested_topics", [])
            }
            agent_summaries.append(summary)

        system_prompt = """Du bist ein professioneller Interviewplanungsexperte. Deine Aufgabe ist es, basierend auf den Interviewanforderungen die am besten geeigneten Agents aus der simulierten Agent-Liste für das Interview auszuwählen.

Auswahlkriterien:
1. Identität/Beruf des Agents ist relevant für das Interviewthema
2. Agent könnte einzigartige oder wertvolle Perspektiven haben
3. Diverse Perspektiven auswählen (z.B. Befürworter, Gegner, Neutrale, Experten usw.)
4. Rollen mit direktem Bezug zum Ereignis bevorzugen

Rückgabe im JSON-Format:
{
    "selected_indices": [Liste der Indizes der ausgewählten Agents],
    "reasoning": "Erklärung der Auswahlbegründung"
}"""

        user_prompt = f"""Interviewanforderung:
{interview_requirement}

Simulationshintergrund:
{simulation_requirement if simulation_requirement else "Nicht angegeben"}

Verfügbare Agent-Liste ({len(agent_summaries)} insgesamt):
{json.dumps(agent_summaries, ensure_ascii=False, indent=2)}

Bitte wähle bis zu {max_agents} am besten geeignete Agents für das Interview aus und erkläre deine Auswahlbegründung."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )

            selected_indices = response.get("selected_indices", [])[:max_agents]
            reasoning = response.get("reasoning", "Automatisch basierend auf Relevanz ausgewählt")

            selected_agents = []
            valid_indices = []
            for idx in selected_indices:
                if 0 <= idx < len(profiles):
                    selected_agents.append(profiles[idx])
                    valid_indices.append(idx)

            return selected_agents, valid_indices, reasoning

        except Exception as e:
            logger.warning(f"LLM-Agent-Auswahl fehlgeschlagen, verwende Standardauswahl: {e}")
            selected = profiles[:max_agents]
            indices = list(range(min(max_agents, len(profiles))))
            return selected, indices, "Verwende Standard-Auswahlstrategie"

    def _generate_interview_questions(
        self,
        interview_requirement: str,
        simulation_requirement: str,
        selected_agents: List[Dict[str, Any]]
    ) -> List[str]:
        """LLM zur Generierung von Interviewfragen verwenden"""

        agent_roles = [a.get("profession", "Unbekannt") for a in selected_agents]

        system_prompt = """Du bist ein professioneller Journalist/Interviewer. Generiere basierend auf den Interviewanforderungen 3-5 tiefgehende Interviewfragen.

Fragenanforderungen:
1. Offene Fragen, die zu detaillierten Antworten ermutigen
2. Fragen, die für verschiedene Rollen unterschiedliche Antworten haben könnten
3. Mehrere Dimensionen abdecken: Fakten, Standpunkte, Gefühle usw.
4. Natürliche Sprache, wie bei echten Interviews
5. Jede Frage unter 50 Zeichen halten, prägnant und klar
6. Direkt fragen, keine Hintergrundinformationen oder Präfixe einfügen

Rückgabe im JSON-Format: {"questions": ["Frage1", "Frage2", ...]}"""

        user_prompt = f"""Interviewanforderung: {interview_requirement}

Simulationshintergrund: {simulation_requirement if simulation_requirement else "Nicht angegeben"}

Rollen der Interviewpartner: {', '.join(agent_roles)}

Bitte generiere 3-5 Interviewfragen."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5
            )

            return response.get("questions", [f"Was ist Ihre Perspektive zu {interview_requirement}?"])

        except Exception as e:
            logger.warning(f"Generierung der Interviewfragen fehlgeschlagen: {e}")
            return [
                f"Was ist Ihre Perspektive zu {interview_requirement}?",
                "Welche Auswirkungen hat dies auf Sie oder die Gruppe, die Sie vertreten?",
                "Wie sollte dieses Problem Ihrer Meinung nach gelöst oder verbessert werden?"
            ]

    def _generate_interview_summary(
        self,
        interviews: List[AgentInterview],
        interview_requirement: str
    ) -> str:
        """Interviewzusammenfassung generieren"""

        if not interviews:
            return "Keine Interviews abgeschlossen"

        interview_texts = []
        for interview in interviews:
            interview_texts.append(f"[{interview.agent_name} ({interview.agent_role})]\n{interview.response[:500]}")

        system_prompt = """Du bist ein professioneller Nachrichtenredakteur. Bitte generiere eine Interviewzusammenfassung basierend auf den Antworten mehrerer Befragter.

Zusammenfassungsanforderungen:
1. Hauptstandpunkte aller Parteien extrahieren
2. Konsens und Meinungsverschiedenheiten zwischen den Standpunkten aufzeigen
3. Wertvolle Zitate hervorheben
4. Objektiv und neutral bleiben, keine Seite bevorzugen
5. Unter 1000 Wörter halten

Formatvorgaben (müssen eingehalten werden):
- Fließtextabsätze verwenden, durch Leerzeilen getrennt
- Keine Markdown-Überschriften verwenden (z.B. #, ##, ###)
- Keine Trennlinien verwenden (z.B. ---, ***)
- Angemessene Zitate verwenden, wenn Befragte zitiert werden
- **Fettdruck** zur Markierung von Schlüsselwörtern verwenden, aber keine andere Markdown-Syntax"""

        user_prompt = f"""Interviewthema: {interview_requirement}

Interviewinhalt:
{"".join(interview_texts)}

Bitte generiere eine Interviewzusammenfassung."""

        try:
            summary = self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            return summary

        except Exception as e:
            logger.warning(f"Generierung der Interviewzusammenfassung fehlgeschlagen: {e}")
            return f"{len(interviews)} Befragte interviewt, darunter: " + ", ".join([i.agent_name for i in interviews])
