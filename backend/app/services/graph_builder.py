"""
Graph-Erstellungsdienst.
Verwendet GraphStorage (Neo4j) anstelle der Zep Cloud API.
"""

import time
import logging
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from ..config import Config
from ..models.task import TaskManager, TaskStatus
from ..storage import GraphStorage
from .text_processor import TextProcessor

logger = logging.getLogger('mirofish.graph_builder')


@dataclass
class GraphInfo:
    """Graph-Informationen"""
    graph_id: str
    node_count: int
    edge_count: int
    entity_types: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": self.entity_types,
        }


class GraphBuilderService:
    """
    Graph-Erstellungsdienst
    Erstellt Wissensgraph über die GraphStorage-Schnittstelle
    """

    def __init__(self, storage: GraphStorage):
        self.storage = storage
        self.task_manager = TaskManager()

    def build_graph_async(
        self,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str = "MiroFish Graph",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 3
    ) -> str:
        """
        Graph asynchron erstellen

        Args:
            text: Eingabetext zur Verarbeitung
            ontology: Ontologie-Definition (aus Ontologie-Generator-Ausgabe)
            graph_name: Name für den Graph
            chunk_size: Textabschnittgröße
            chunk_overlap: Überlappungsgröße der Abschnitte
            batch_size: Anzahl der Abschnitte pro Batch

        Returns:
            Aufgaben-ID
        """
        # Aufgabe erstellen
        task_id = self.task_manager.create_task(
            task_type="graph_build",
            metadata={
                "graph_name": graph_name,
                "chunk_size": chunk_size,
                "text_length": len(text),
            }
        )

        # Erstellung im Hintergrund-Thread ausführen
        thread = threading.Thread(
            target=self._build_graph_worker,
            args=(task_id, text, ontology, graph_name, chunk_size, chunk_overlap, batch_size)
        )
        thread.daemon = True
        thread.start()

        return task_id

    def _build_graph_worker(
        self,
        task_id: str,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int
    ):
        """Graph-Erstellungs-Worker-Thread"""
        try:
            self.task_manager.update_task(
                task_id,
                status=TaskStatus.PROCESSING,
                progress=5,
                message="Starte Graph-Erstellung..."
            )

            # 1. Graph erstellen
            graph_id = self.create_graph(graph_name)
            self.task_manager.update_task(
                task_id,
                progress=10,
                message=f"Graph erstellt: {graph_id}"
            )

            # 2. Ontologie setzen
            self.set_ontology(graph_id, ontology)
            self.task_manager.update_task(
                task_id,
                progress=15,
                message="Ontologie gesetzt"
            )

            # 3. Text aufteilen
            chunks = TextProcessor.split_text(text, chunk_size, chunk_overlap)
            total_chunks = len(chunks)
            self.task_manager.update_task(
                task_id,
                progress=20,
                message=f"Text in {total_chunks} Abschnitte aufgeteilt"
            )

            # 4. Daten batchweise senden (NER + Embedding + Neo4j-Einfügung — synchron)
            episode_uuids = self.add_text_batches(
                graph_id, chunks, batch_size,
                lambda msg, prog: self.task_manager.update_task(
                    task_id,
                    progress=20 + int(prog * 0.6),  # 20-80%
                    message=msg
                )
            )

            # 5. Auf Verarbeitung warten (bei Neo4j nicht nötig — bereits synchron)
            self.storage.wait_for_processing(episode_uuids)

            self.task_manager.update_task(
                task_id,
                progress=85,
                message="Datenverarbeitung abgeschlossen, rufe Graph-Informationen ab..."
            )

            # 6. Graph-Informationen abrufen
            graph_info = self._get_graph_info(graph_id)

            # Abgeschlossen
            self.task_manager.complete_task(task_id, {
                "graph_id": graph_id,
                "graph_info": graph_info.to_dict(),
                "chunks_processed": total_chunks,
            })

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.task_manager.fail_task(task_id, error_msg)

    def create_graph(self, name: str) -> str:
        """Graph erstellen"""
        return self.storage.create_graph(
            name=name,
            description="MiroFish Social Simulation Graph"
        )

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        """
        Graph-Ontologie setzen

        Speichert die Ontologie einfach als JSON im Graph-Knoten.
        Keine dynamische Pydantic-Klassenerstellung mehr (war Zep-spezifisch).
        Der NER-Extraktor liest diese Ontologie, um die Extraktion zu steuern.
        """
        self.storage.set_ontology(graph_id, ontology)

    def add_text_batches(
        self,
        graph_id: str,
        chunks: List[str],
        batch_size: int = 3,
        progress_callback: Optional[Callable] = None
    ) -> List[str]:
        """Text batchweise zum Graph hinzufügen, UUID-Liste aller Episoden zurückgeben"""
        episode_uuids = []
        total_chunks = len(chunks)
        total_batches = (total_chunks + batch_size - 1) // batch_size

        logger.info(f"[graph_build] Starte: {total_chunks} Abschnitte, {total_batches} Batches (batch_size={batch_size})")

        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1

            if progress_callback:
                progress = (i + len(batch_chunks)) / total_chunks
                progress_callback(
                    f"Verarbeite Batch {batch_num}/{total_batches} ({len(batch_chunks)} Abschnitte)...",
                    progress
                )

            for j, chunk in enumerate(batch_chunks):
                chunk_idx = i + j + 1
                chunk_preview = chunk[:80].replace('\n', ' ')
                logger.info(
                    f"[graph_build] Abschnitt {chunk_idx}/{total_chunks} "
                    f"({len(chunk)} Zeichen): \"{chunk_preview}...\""
                )
                t0 = time.time()
                try:
                    episode_id = self.storage.add_text(graph_id, chunk)
                    episode_uuids.append(episode_id)
                    elapsed = time.time() - t0
                    logger.info(
                        f"[graph_build] Abschnitt {chunk_idx}/{total_chunks} fertig in {elapsed:.1f}s"
                    )
                except Exception as e:
                    elapsed = time.time() - t0
                    logger.error(
                        f"[graph_build] Abschnitt {chunk_idx}/{total_chunks} FEHLGESCHLAGEN "
                        f"nach {elapsed:.1f}s: {e}"
                    )
                    if progress_callback:
                        progress_callback(f"Batch {batch_num} Verarbeitung fehlgeschlagen: {str(e)}", 0)
                    raise

        logger.info(f"[graph_build] Alle {total_chunks} Abschnitte erfolgreich verarbeitet")
        return episode_uuids

    def _get_graph_info(self, graph_id: str) -> GraphInfo:
        """Graph-Informationen abrufen"""
        info = self.storage.get_graph_info(graph_id)
        return GraphInfo(
            graph_id=info["graph_id"],
            node_count=info["node_count"],
            edge_count=info["edge_count"],
            entity_types=info.get("entity_types", []),
        )

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        """Vollständige Graph-Daten abrufen (einschließlich Details)"""
        return self.storage.get_graph_data(graph_id)

    def delete_graph(self, graph_id: str):
        """Graph löschen"""
        self.storage.delete_graph(graph_id)
