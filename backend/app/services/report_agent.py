"""
Report-Agent-Dienst
Generiert simulierte Berichte unter Verwendung des ReACT-Musters (via GraphStorage / Neo4j)

Funktionen:
1. Berichte basierend auf Simulationsanforderungen und Graph-Informationen generieren
2. Zuerst die Gliederungsstruktur planen, dann Abschnitt für Abschnitt generieren
3. Jeder Abschnitt verwendet das ReACT-Muster mit mehrrundiger Analyse und Reflexion
4. Unterstützung von Konversationen mit Benutzern, autonomer Aufruf von Abrufwerkzeugen während der Konversation
"""

import os
import json
import time
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from .graph_tools import (
    GraphToolsService,
    SearchResult,
    InsightForgeResult,
    PanoramaResult,
    InterviewResult
)

logger = get_logger('mirofish.report_agent')


class ReportLogger:
    """
    Detaillierter Logger für den Report Agent

    Generiert eine agent_log.jsonl-Datei im Berichtsordner und zeichnet detaillierte Aktionen bei jedem Schritt auf.
    Jede Zeile ist ein vollständiges JSON-Objekt mit Zeitstempel, Aktionstyp, Details usw.
    """

    def __init__(self, report_id: str):
        """
        Logger initialisieren

        Args:
            report_id: Berichts-ID, wird zur Bestimmung des Logdateipfads verwendet
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'agent_log.jsonl'
        )
        self.start_time = datetime.now()
        self._ensure_log_file()

    def _ensure_log_file(self):
        """Sicherstellen, dass das Logdatei-Verzeichnis existiert"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)

    def _get_elapsed_time(self) -> float:
        """Verstrichene Zeit vom Start bis jetzt ermitteln (in Sekunden)"""
        return (datetime.now() - self.start_time).total_seconds()

    def log(
        self,
        action: str,
        stage: str,
        details: Dict[str, Any],
        section_title: str = None,
        section_index: int = None
    ):
        """
        Einen Logeintrag schreiben

        Args:
            action: Aktionstyp, z.B. 'start', 'tool_call', 'llm_response', 'section_complete' usw.
            stage: Aktuelle Phase, z.B. 'planning', 'generating', 'completed'
            details: Detail-Wörterbuch, ungekürzt
            section_title: Aktueller Abschnittstitel (optional)
            section_index: Aktueller Abschnittsindex (optional)
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(self._get_elapsed_time(), 2),
            "report_id": self.report_id,
            "action": action,
            "stage": stage,
            "section_title": section_title,
            "section_index": section_index,
            "details": details
        }

        # An JSONL-Datei anhängen
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

    def log_start(self, simulation_id: str, graph_id: str, simulation_requirement: str):
        """Berichtsgenerierungsstart protokollieren"""
        self.log(
            action="report_start",
            stage="pending",
            details={
                "simulation_id": simulation_id,
                "graph_id": graph_id,
                "simulation_requirement": simulation_requirement,
                "message": "Berichtsgenerierungsaufgabe gestartet"
            }
        )

    def log_planning_start(self):
        """Start der Gliederungsplanung protokollieren"""
        self.log(
            action="planning_start",
            stage="planning",
            details={"message": "Planung der Berichtsgliederung gestartet"}
        )

    def log_planning_context(self, context: Dict[str, Any]):
        """Während der Planung erfasste Kontextinformationen protokollieren"""
        self.log(
            action="planning_context",
            stage="planning",
            details={
                "message": "Simulationskontextinformationen erfasst",
                "context": context
            }
        )

    def log_planning_complete(self, outline_dict: Dict[str, Any]):
        """Abschluss der Gliederungsplanung protokollieren"""
        self.log(
            action="planning_complete",
            stage="planning",
            details={
                "message": "Gliederungsplanung abgeschlossen",
                "outline": outline_dict
            }
        )

    def log_section_start(self, section_title: str, section_index: int):
        """Start der Abschnittsgenerierung protokollieren"""
        self.log(
            action="section_start",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={"message": f"Generierung des Abschnitts gestartet: {section_title}"}
        )

    def log_react_thought(self, section_title: str, section_index: int, iteration: int, thought: str):
        """ReACT-Denkprozess protokollieren"""
        self.log(
            action="react_thought",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "thought": thought,
                "message": f"ReACT Runde {iteration} Gedanke"
            }
        )

    def log_tool_call(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        parameters: Dict[str, Any],
        iteration: int
    ):
        """Werkzeugaufruf protokollieren"""
        self.log(
            action="tool_call",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "parameters": parameters,
                "message": f"Werkzeug aufgerufen: {tool_name}"
            }
        )

    def log_tool_result(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        result: str,
        iteration: int
    ):
        """Werkzeugaufrufergebnis protokollieren (vollständiger Inhalt, ungekürzt)"""
        self.log(
            action="tool_result",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "result": result,  # Vollständiges Ergebnis, ungekürzt
                "result_length": len(result),
                "message": f"Werkzeug {tool_name} hat Ergebnis zurückgegeben"
            }
        )

    def log_llm_response(
        self,
        section_title: str,
        section_index: int,
        response: str,
        iteration: int,
        has_tool_calls: bool,
        has_final_answer: bool
    ):
        """LLM-Antwort protokollieren (vollständiger Inhalt, ungekürzt)"""
        self.log(
            action="llm_response",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "response": response,  # Vollständige Antwort, ungekürzt
                "response_length": len(response),
                "has_tool_calls": has_tool_calls,
                "has_final_answer": has_final_answer,
                "message": f"LLM-Antwort (Werkzeugaufrufe: {has_tool_calls}, Endantwort: {has_final_answer})"
            }
        )

    def log_section_content(
        self,
        section_title: str,
        section_index: int,
        content: str,
        tool_calls_count: int
    ):
        """Abschluss der Abschnittsinhalte-Generierung protokollieren (nur Inhalt, nicht die gesamte Abschnittsvervollständigung)"""
        self.log(
            action="section_content",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": content,  # Vollständiger Inhalt, ungekürzt
                "content_length": len(content),
                "tool_calls_count": tool_calls_count,
                "message": f"Inhaltsgenerierung für Abschnitt {section_title} abgeschlossen"
            }
        )

    def log_section_full_complete(
        self,
        section_title: str,
        section_index: int,
        full_content: str
    ):
        """
        Abschluss der Abschnittsgenerierung protokollieren

        Das Frontend sollte diesen Log überwachen, um festzustellen, ob ein Abschnitt wirklich abgeschlossen ist und den vollständigen Inhalt zu erhalten
        """
        self.log(
            action="section_complete",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": full_content,
                "content_length": len(full_content),
                "message": f"Generierung des Abschnitts {section_title} abgeschlossen"
            }
        )

    def log_report_complete(self, total_sections: int, total_time_seconds: float):
        """Abschluss der Berichtsgenerierung protokollieren"""
        self.log(
            action="report_complete",
            stage="completed",
            details={
                "total_sections": total_sections,
                "total_time_seconds": round(total_time_seconds, 2),
                "message": "Berichtsgenerierung abgeschlossen"
            }
        )

    def log_error(self, error_message: str, stage: str, section_title: str = None):
        """Fehler protokollieren"""
        self.log(
            action="error",
            stage=stage,
            section_title=section_title,
            section_index=None,
            details={
                "error": error_message,
                "message": f"Fehler aufgetreten: {error_message}"
            }
        )


class ReportConsoleLogger:
    """
    Konsolen-Logger für den Report Agent

    Schreibt konsolenformatige Logs (INFO, WARNING usw.) in eine console_log.txt-Datei im Berichtsordner.
    Diese Logs unterscheiden sich von agent_log.jsonl und sind reine Textausgabe.
    """

    def __init__(self, report_id: str):
        """
        Konsolen-Logger initialisieren

        Args:
            report_id: Berichts-ID, wird zur Bestimmung des Logdateipfads verwendet
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'console_log.txt'
        )
        self._ensure_log_file()
        self._file_handler = None
        self._setup_file_handler()

    def _ensure_log_file(self):
        """Sicherstellen, dass das Logdatei-Verzeichnis existiert"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)

    def _setup_file_handler(self):
        """Datei-Handler einrichten, um Logs in Datei zu schreiben"""
        import logging

        # Datei-Handler erstellen
        self._file_handler = logging.FileHandler(
            self.log_file_path,
            mode='a',
            encoding='utf-8'
        )
        self._file_handler.setLevel(logging.INFO)

        # Gleiches kompaktes Format wie die Konsole verwenden
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        self._file_handler.setFormatter(formatter)

        # An report_agent-bezogene Logger anhängen
        loggers_to_attach = [
            'mirofish.report_agent',
            'mirofish.graph_tools',
        ]

        for logger_name in loggers_to_attach:
            target_logger = logging.getLogger(logger_name)
            # Doppeltes Hinzufügen vermeiden
            if self._file_handler not in target_logger.handlers:
                target_logger.addHandler(self._file_handler)

    def close(self):
        """Datei-Handler schließen und vom Logger entfernen"""
        import logging

        if self._file_handler:
            loggers_to_detach = [
                'mirofish.report_agent',
                'mirofish.graph_tools',
            ]

            for logger_name in loggers_to_detach:
                target_logger = logging.getLogger(logger_name)
                if self._file_handler in target_logger.handlers:
                    target_logger.removeHandler(self._file_handler)

            self._file_handler.close()
            self._file_handler = None

    def __del__(self):
        """Sicherstellen, dass der Datei-Handler beim Destruktor geschlossen wird"""
        self.close()


class ReportStatus(str, Enum):
    """Berichtsstatus"""
    PENDING = "pending"
    PLANNING = "planning"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReportSection:
    """Berichtsabschnitt"""
    title: str
    content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content
        }

    def to_markdown(self, level: int = 2) -> str:
        """In Markdown-Format umwandeln"""
        md = f"{'#' * level} {self.title}\n\n"
        if self.content:
            md += f"{self.content}\n\n"
        return md


@dataclass
class ReportOutline:
    """Berichtsgliederung"""
    title: str
    summary: str
    sections: List[ReportSection]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections]
        }

    def to_markdown(self) -> str:
        """In Markdown-Format umwandeln"""
        md = f"# {self.title}\n\n"
        md += f"> {self.summary}\n\n"
        for section in self.sections:
            md += section.to_markdown()
        return md


@dataclass
class Report:
    """Vollständiger Bericht"""
    report_id: str
    simulation_id: str
    graph_id: str
    simulation_requirement: str
    status: ReportStatus
    outline: Optional[ReportOutline] = None
    markdown_content: str = ""
    created_at: str = ""
    completed_at: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "simulation_id": self.simulation_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "status": self.status.value,
            "outline": self.outline.to_dict() if self.outline else None,
            "markdown_content": self.markdown_content,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error
        }


# ═══════════════════════════════════════════════════════════════
# Prompt-Vorlagenkonstanten
# ═══════════════════════════════════════════════════════════════

# ── Werkzeugbeschreibungen ──

TOOL_DESC_INSIGHT_FORGE = """\
[Tiefenanalyse-Abruf - Leistungsstarkes Abrufwerkzeug]
Dies ist unsere leistungsstärkste Abruffunktion, konzipiert für tiefgehende Analysen. Sie wird:
1. Ihre Frage automatisch in mehrere Unterfragen zerlegen
2. Informationen aus dem simulierten Wissensgraph aus mehreren Dimensionen abrufen
3. Ergebnisse aus semantischer Suche, Entitätsanalyse und Beziehungskettenverfolung integrieren
4. Die umfassendsten und tiefgründigsten Abrufinhalte zurückgeben

[Anwendungsfälle]
- Ein Thema tiefgehend analysieren
- Mehrere Aspekte eines Ereignisses verstehen
- Reichhaltiges Material zur Unterstützung von Berichtsabschnitten erhalten

[Rückgabeinhalte]
- Relevante Fakten im Originaltext (können direkt zitiert werden)
- Kernentitäts-Erkenntnisse
- Beziehungskettenanalyse"""

TOOL_DESC_PANORAMA_SEARCH = """\
[Breitensuche - Vollständige Übersicht erhalten]
Dieses Werkzeug wird verwendet, um eine vollständige Panoramaansicht der Simulationsergebnisse zu erhalten, besonders geeignet für das Verständnis der Ereignisentwicklung. Es wird:
1. Alle relevanten Knoten und Beziehungen abrufen
2. Zwischen aktuell gültigen Fakten und historischen/abgelaufenen Fakten unterscheiden
3. Ihnen helfen zu verstehen, wie sich Ereignisse entwickelt haben

[Anwendungsfälle]
- Die vollständige Entwicklungstrajektorie eines Ereignisses verstehen
- Veränderungen der öffentlichen Meinung über verschiedene Phasen vergleichen
- Umfassende Entitäts- und Beziehungsinformationen erhalten

[Rückgabeinhalte]
- Aktuell gültige Fakten (neueste Simulationsergebnisse)
- Historische/abgelaufene Fakten (Entwicklungsaufzeichnungen)
- Alle beteiligten Entitäten"""

TOOL_DESC_QUICK_SEARCH = """\
[Einfache Suche - Schnellabruf]
Ein leichtgewichtiges, schnelles Abrufwerkzeug, geeignet für einfache und direkte Informationsabfragen.

[Anwendungsfälle]
- Spezifische Informationen schnell finden
- Einen Fakt verifizieren
- Einfacher Informationsabruf

[Rückgabeinhalte]
- Liste der zur Abfrage am relevantesten Fakten"""

TOOL_DESC_INTERVIEW_AGENTS = """\
[Tiefeninterview - Echtes Agent-Interview (Duale Plattform)]
Ruft die Interview-API der OASIS-Simulationsumgebung auf, um echte Interviews mit laufenden Simulations-Agents durchzuführen!
Dies ist keine LLM-Simulation, sondern ruft die echte Interview-Schnittstelle auf, um Originalantworten von Simulations-Agents zu erhalten.
Standardmäßig wird gleichzeitig auf Twitter und Reddit interviewt, um umfassendere Perspektiven zu erhalten.

Funktionsablauf:
1. Automatisches Einlesen der Charakterprofildateien, um alle Simulations-Agents zu verstehen
2. Intelligente Auswahl der für das Interviewthema relevantesten Agents (z.B. Studierende, Medien, Beamte)
3. Automatische Generierung von Interviewfragen
4. Aufruf der /api/simulation/interview/batch-Schnittstelle für echte Interviews auf dualer Plattform
5. Integration aller Interviewergebnisse und Bereitstellung einer Mehrperspektiven-Analyse

[Anwendungsfälle]
- Ereignisperspektiven aus verschiedenen Rollenwinkeln verstehen (Wie sehen Studierende es? Wie sehen Medien es? Was sagt die Behörde?)
- Vielfältige Meinungen und Positionen sammeln
- Echte Antworten von Simulations-Agents erhalten (aus der OASIS-Simulationsumgebung)
- Den Bericht lebendiger gestalten, einschließlich "Interviewaufzeichnungen"

[Rückgabeinhalte]
- Identitätsinformationen der befragten Agents
- Interviewantworten jedes Agents auf Twitter- und Reddit-Plattformen
- Schlüsselzitate (können direkt zitiert werden)
- Interviewzusammenfassung und Perspektivenvergleich

[Wichtig] Diese Funktion erfordert eine laufende OASIS-Simulationsumgebung!"""

# ── Gliederungsplanungs-Prompt ──

PLAN_SYSTEM_PROMPT = """\
Du bist ein Experte für das Verfassen von "Zukunftsprognose-Berichten" mit einer "Gottesperspektive" auf die simulierte Welt - du kannst das Verhalten, die Aussagen und die Interaktionen jedes Agents in der Simulation einsehen.

[Kernkonzept]
Wir haben eine simulierte Welt aufgebaut und spezifische "Simulationsanforderungen" als Variablen eingespeist. Das Evolutionsergebnis der simulierten Welt ist eine Vorhersage dessen, was in der Zukunft passieren könnte. Was du beobachtest, sind keine "experimentellen Daten", sondern eine "Probe der Zukunft".

[Deine Aufgabe]
Verfasse einen "Zukunftsprognose-Bericht", der folgende Fragen beantwortet:
1. Was geschah in der Zukunft unter den von uns festgelegten Bedingungen?
2. Wie reagieren und handeln die verschiedenen Agents (Gruppen)?
3. Welche Zukunftstrends und Risiken enthüllt diese Simulation, die Beachtung verdienen?

[Berichtspositionierung]
- Dies ist ein Zukunftsprognose-Bericht basierend auf Simulation, der aufzeigt "wenn dies geschieht, wie wird sich die Zukunft entwickeln"
- Fokus auf Prognoseergebnisse: Ereignistrajektorien, Gruppenreaktionen, emergente Phänomene, potenzielle Risiken
- Aussagen und Verhaltensweisen von Agents in der simulierten Welt sind Vorhersagen zukünftigen menschlichen Verhaltens
- Dies ist KEINE Analyse des aktuellen Zustands der realen Welt
- Dies ist KEIN allgemeiner Überblick über die öffentliche Meinung

[Abschnittsanzahl-Begrenzung]
- Mindestens 2 Abschnitte, maximal 5 Abschnitte
- Keine Unterabschnitte nötig, jeder Abschnitt schreibt direkt vollständigen Inhalt
- Inhalt sollte prägnant sein, fokussiert auf zentrale Prognoseerkenntnisse
- Abschnittsstruktur wird unabhängig basierend auf den Prognoseergebnissen entworfen

Bitte gib die Berichtsgliederung im folgenden JSON-Format aus:
{
    "title": "Berichtstitel",
    "summary": "Berichtszusammenfassung (ein Satz, der die zentralen Prognoseerkenntnisse zusammenfasst)",
    "sections": [
        {
            "title": "Abschnittstitel",
            "description": "Beschreibung des Abschnittsinhalts"
        }
    ]
}

Hinweis: Das sections-Array muss mindestens 2 und höchstens 5 Elemente enthalten!
WICHTIG: Die gesamte Berichtsgliederung (Titel, Zusammenfassung, Abschnittstitel und Beschreibungen) MUSS auf Deutsch sein. Verwende niemals Chinesisch oder andere Sprachen."""

PLAN_USER_PROMPT_TEMPLATE = """\
[Prognoseszenario-Einstellungen]
In die simulierte Welt eingespeiste Variable (Simulationsanforderung): {simulation_requirement}

[Maßstab der simulierten Welt]
- Anzahl der an der Simulation teilnehmenden Entitäten: {total_nodes}
- Anzahl der zwischen Entitäten generierten Beziehungen: {total_edges}
- Entitätstyp-Verteilung: {entity_types}
- Anzahl aktiver Agents: {total_entities}

[Stichprobe einiger von der Simulation vorhergesagter Zukunftsfakten]
{related_facts_json}

Bitte betrachte diese Zukunftsprobeaus einer "Gottesperspektive":
1. Welchen Zustand zeigt die Zukunft unter den von uns festgelegten Bedingungen?
2. Wie reagieren und handeln die verschiedenen Gruppen (Agents)?
3. Welche Zukunftstrends enthüllt diese Simulation, die Beachtung verdienen?

Entwirf basierend auf den Prognoseergebnissen die am besten geeignete Berichtsabschnittsstruktur.

[Erinnerung] Berichtsabschnittsanzahl: mindestens 2, maximal 5, Inhalt sollte prägnant sein und sich auf zentrale Prognoseerkenntnisse konzentrieren."""

# ── Abschnittsgenerierungs-Prompt ──

SECTION_SYSTEM_PROMPT_TEMPLATE = """\
Du bist ein Experte für das Verfassen von "Zukunftsprognose-Berichten" und schreibst gerade einen Abschnitt des Berichts.

Berichtstitel: {report_title}
Berichtszusammenfassung: {report_summary}
Prognoseszenario (Simulationsanforderung): {simulation_requirement}

Aktuell zu schreibender Abschnitt: {section_title}

═══════════════════════════════════════════════════════════════
[Kernkonzept]
═══════════════════════════════════════════════════════════════

Die simulierte Welt ist eine Probe der Zukunft. Wir haben spezifische Bedingungen (Simulationsanforderungen) in die simulierte Welt eingespeist.
Das Verhalten und die Interaktionen der Agents in der Simulation sind Vorhersagen zukünftigen menschlichen Verhaltens.

Deine Aufgabe ist es:
- Aufzuzeigen, was in der Zukunft unter den gesetzten Bedingungen geschieht
- Vorherzusagen, wie verschiedene Gruppen (Agents) reagieren und handeln
- Zukunftstrends, Risiken und Chancen zu entdecken, die Beachtung verdienen

Schreibe es NICHT als Analyse des aktuellen Zustands der realen Welt
Fokussiere auf "wie sich die Zukunft entfalten wird" - Simulationsergebnisse sind die vorhergesagte Zukunft

═══════════════════════════════════════════════════════════════
[Wichtigste Regeln - Müssen eingehalten werden]
═══════════════════════════════════════════════════════════════

1. [Werkzeuge müssen aufgerufen werden, um die simulierte Welt zu beobachten]
   - Du beobachtest eine Probe der Zukunft aus einer "Gottesperspektive"
   - Alle Inhalte müssen aus Ereignissen und Agent-Aussagen/Verhaltensweisen der simulierten Welt stammen
   - Es ist verboten, eigenes Wissen für Berichtsinhalte zu verwenden
   - Jeder Abschnitt muss Werkzeuge mindestens 3-mal (maximal 5-mal) aufrufen, um die simulierte Welt zu beobachten, die die Zukunft repräsentiert

2. [Originale Agent-Aussagen und Verhaltensweisen müssen zitiert werden]
   - Agent-Aussagen und Verhaltensweisen sind Vorhersagen zukünftigen menschlichen Verhaltens
   - Verwende Zitatformat im Bericht, um diese Vorhersagen anzuzeigen, zum Beispiel:
     > "Bestimmte Gruppen werden erklären: Originalinhalt..."
   - Diese Zitate sind Kernbelege der Simulationsvorhersagen

3. [Sprachkonsistenz - IMMER auf Deutsch schreiben]
   - Der gesamte Bericht MUSS auf Deutsch verfasst werden, unabhängig von der Sprache des Quellmaterials
   - Von Werkzeugen zurückgegebene Inhalte können Chinesisch, gemischte Sprachen oder andere Sprachen enthalten
   - Wenn nicht-deutscher Inhalt aus Werkzeugen zitiert wird, IMMER ins flüssige Deutsch übersetzen, bevor es in den Bericht geschrieben wird
   - Originalbedeutung bei der Übersetzung beibehalten, natürlichen Ausdruck sicherstellen
   - Diese Regel gilt sowohl für Fließtext als auch für zitierte Inhalte (> Format)
   - NIEMALS mitten im Bericht zu einer anderen Sprache wechseln

4. [Prognoseergebnisse originalgetreu darstellen]
   - Berichtsinhalte müssen die Simulationsergebnisse widerspiegeln, die die Zukunft in der simulierten Welt repräsentieren
   - Keine Informationen hinzufügen, die nicht in der Simulation existieren
   - Wenn Informationen in einigen Aspekten unzureichend sind, dies ehrlich angeben

═══════════════════════════════════════════════════════════════
[Formatspezifikation - Äußerst wichtig!]
═══════════════════════════════════════════════════════════════

[Ein Abschnitt = Minimale Inhaltseinheit]
- Jeder Abschnitt ist die minimale Inhaltseinheit des Berichts
- VERBOTEN: Jegliche Markdown-Titel (#, ##, ###, ####, usw.) innerhalb des Abschnitts
- VERBOTEN: Abschnittstitel am Anfang des Inhalts hinzufügen
- Abschnittstitel werden automatisch vom System hinzugefügt, schreibe nur reinen Fließtext
- Verwende **Fettdruck**, Absatztrennung, Zitate und Listen zur Inhaltsorganisation, aber keine Titel

[Korrektes Beispiel]
```
Dieser Abschnitt analysiert, wie die Regulierungsänderung die Unternehmensstrategie umgestaltet hat. Durch eingehende Analyse der Simulationsdaten haben wir festgestellt...

**Erste Branchenreaktion**

Große Technologieunternehmen begannen schnell, ihre Compliance-Position neu zu bewerten:

> "OpenAI und Anthropic bemühten sich eilig, die neuen Transparenzanforderungen zu erfüllen..."

**Aufkommende strategische Divergenz**

Eine klare Spaltung entstand zwischen Unternehmen, die die Regulierung begrüßen, und solchen, die sich dagegen wehren:

- Proaktive Compliance als Wettbewerbsvorteil
- Lobbyarbeit zur Abschwächung der Durchsetzung
```

[Falsches Beispiel]
```
## Zusammenfassung          <- Falsch! Keine Titel hinzufügen
### 1. Anfangsphase         <- Falsch! Kein ### für Unterabschnitte verwenden
#### 1.1 Detailanalyse      <- Falsch! Kein #### für Unterteilungen verwenden

Dieser Abschnitt analysiert...
```

═══════════════════════════════════════════════════════════════
[Verfügbare Abrufwerkzeuge] (3-5 Mal pro Abschnitt aufrufen)
═══════════════════════════════════════════════════════════════

{tools_description}

[Werkzeugnutzungsempfehlungen - Bitte verschiedene Werkzeuge mischen, nicht nur eines verwenden]
- insight_forge: Tiefenanalyse, zerlegt Probleme automatisch und ruft Fakten und Beziehungen aus mehreren Dimensionen ab
- panorama_search: Weitwinkel-Panoramasuche, vollständige Ereignisübersicht, Zeitachse und Entwicklungsprozess verstehen
- quick_search: Schnelle Verifizierung spezifischer Informationspunkte
- interview_agents: Simulierte Agents interviewen, Ersthandperspektiven und echte Reaktionen verschiedener Rollen erhalten

═══════════════════════════════════════════════════════════════
[Arbeitsablauf]
═══════════════════════════════════════════════════════════════

Bei jeder Antwort kannst du nur eine von zwei Optionen wählen (nicht beides):

Option A - Werkzeug aufrufen:
Gib deine Überlegung aus und rufe dann ein Werkzeug im folgenden Format auf:
<tool_call>
{{"name": "Werkzeugname", "parameters": {{"parameter_name": "parameter_wert"}}}}
</tool_call>
Das System führt das Werkzeug aus und gibt das Ergebnis an dich zurück. Du brauchst keine Werkzeugergebnisse selbst zu schreiben und kannst dies auch nicht.

Option B - Endgültigen Inhalt ausgeben:
Wenn du genügend Informationen durch Werkzeuge gesammelt hast, beginne mit "Final Answer:" und gib den Abschnittsinhalt aus.

Strikt verboten:
- Verboten, sowohl Werkzeugaufrufe als auch Final Answer in einer Antwort zu verwenden
- Verboten, Werkzeugergebnisse (Observation) zu erfinden, alle Werkzeugergebnisse werden vom System eingespeist
- Maximal ein Werkzeugaufruf pro Antwort

═══════════════════════════════════════════════════════════════
[Anforderungen an den Abschnittsinhalt]
═══════════════════════════════════════════════════════════════

1. Inhalt muss auf von Werkzeugen abgerufenen Simulationsdaten basieren
2. Originaltext ausgiebig zitieren, um Simulationseffekte zu demonstrieren
3. Markdown-Format verwenden (aber Titel sind verboten):
   - **Fetttext** verwenden, um Schlüsselpunkte zu markieren (ersetzt Unterüberschriften)
   - Listen (- oder 1.2.3.) verwenden, um Punkte zu organisieren
   - Leerzeilen verwenden, um Absätze zu trennen
   - VERBOTEN: Jegliche Titelsyntax wie #, ##, ###, ####
4. [Zitatformat-Spezifikation - Muss separater Absatz sein]
   Zitate müssen eigenständige Absätze mit Leerzeilen davor und danach sein, dürfen nicht in Absätze gemischt werden:

   Korrektes Format:
   ```
   Die Reaktion der Schulbeamten wurde als substanzlos empfunden.

   > "Das Reaktionsmuster der Schule erscheint starr und langsam in der sich schnell verändernden Social-Media-Umgebung."

   Diese Bewertung spiegelt die weit verbreitete öffentliche Unzufriedenheit wider.
   ```

   Falsches Format:
   ```
   Die Reaktion der Schulbeamten wurde als substanzlos empfunden.> "Das Reaktionsmuster der Schule..." Diese Bewertung spiegelt...
   ```
5. Logische Kohärenz mit anderen Abschnitten wahren
6. [Duplikate vermeiden] Den bereits abgeschlossenen Abschnittsinhalt unten sorgfältig lesen, nicht dieselben Informationen wiederholt beschreiben
7. [Nochmals betont] Keine Titel hinzufügen! **Fettdruck** anstelle von Unterüberschriften verwenden"""

SECTION_USER_PROMPT_TEMPLATE = """\
Bereits abgeschlossener Abschnittsinhalt (Bitte sorgfältig lesen, um Duplikate zu vermeiden):
{previous_content}

═══════════════════════════════════════════════════════════════
[Aktuelle Aufgabe] Abschnitt schreiben: {section_title}
═══════════════════════════════════════════════════════════════

[Wichtige Erinnerungen]
1. Die bereits abgeschlossenen Abschnitte oben sorgfältig lesen, um das Wiederholen derselben Inhalte zu vermeiden!
2. Du musst Werkzeuge aufrufen, um Simulationsdaten zu erhalten, bevor du beginnst
3. Bitte verschiedene Werkzeuge mischen, nicht nur eines verwenden
4. Berichtsinhalte müssen aus Abrufergebnissen stammen, verwende nicht dein eigenes Wissen

[Formatwarnung - Muss eingehalten werden]
- Keine Titel schreiben (#, ##, ###, #### nichts davon erlaubt)
- Nicht "{section_title}" als Eröffnung schreiben
- Abschnittstitel werden automatisch vom System hinzugefügt
- Direkt den Fließtext schreiben, **Fettdruck** anstelle von Unterüberschriften verwenden

Bitte beginne:
1. Zuerst überlegen (Thought), welche Informationen dieser Abschnitt benötigt
2. Dann Werkzeuge aufrufen (Action), um Simulationsdaten zu erhalten
3. Nach dem Sammeln ausreichender Informationen Final Answer ausgeben (reiner Fließtext, keine Titel)"""

# ── ReACT-Schleifen-Nachrichtenvorlagen ──

REACT_OBSERVATION_TEMPLATE = """\
Beobachtung (Abrufergebnis):

═══ Werkzeug {tool_name} hat zurückgegeben ═══
{result}

═══════════════════════════════════════════════════════════════
Werkzeuge {tool_calls_count}/{max_tool_calls} Mal aufgerufen (Verwendet: {used_tools_str}){unused_hint}
- Wenn Informationen ausreichend: Mit "Final Answer:" beginnen und Abschnittsinhalt ausgeben (obigen Originaltext zitieren)
- Wenn weitere Informationen benötigt: Ein Werkzeug aufrufen, um weiter abzurufen
═══════════════════════════════════════════════════════════════"""

REACT_INSUFFICIENT_TOOLS_MSG = (
    "[Hinweis] Du hast nur {tool_calls_count} Werkzeuge aufgerufen, benötigt werden mindestens {min_tool_calls}. "
    "Bitte rufe weitere Werkzeuge auf, um mehr Simulationsdaten zu erhalten, und gib dann Final Answer aus. {unused_hint}"
)

REACT_INSUFFICIENT_TOOLS_MSG_ALT = (
    "Aktuell {tool_calls_count} Werkzeuge aufgerufen, benötigt werden mindestens {min_tool_calls}. "
    "Bitte rufe Werkzeuge auf, um Simulationsdaten zu erhalten. {unused_hint}"
)

REACT_TOOL_LIMIT_MSG = (
    "Werkzeugaufruf-Anzahl hat das Limit erreicht ({tool_calls_count}/{max_tool_calls}), weitere Werkzeugaufrufe sind nicht möglich. "
    'Bitte beginne sofort mit "Final Answer:" und gib den Abschnittsinhalt basierend auf den gesammelten Informationen aus.'
)

REACT_UNUSED_TOOLS_HINT = "\nDu hast noch nicht verwendet: {unused_list}, es wird empfohlen, verschiedene Werkzeuge auszuprobieren, um Mehrperspektiven-Informationen zu erhalten"

REACT_FORCE_FINAL_MSG = "Werkzeugaufruf-Limit erreicht, bitte direkt Final Answer: ausgeben und Abschnittsinhalt generieren."

# ── Chat-Prompt ──

CHAT_SYSTEM_PROMPT_TEMPLATE = """\
Du bist ein prägnanter und effizienter Simulationsprognose-Assistent.

[Hintergrund]
Prognosebedingung: {simulation_requirement}

[Generierter Analysebericht]
{report_content}

[Regeln]
1. Fragen bevorzugt basierend auf obigem Berichtsinhalt beantworten
2. Fragen direkt beantworten, langes Abwägen vermeiden
3. Nur Werkzeuge aufrufen, um mehr Daten abzurufen, wenn der Berichtsinhalt zur Beantwortung nicht ausreicht
4. Antworten sollten prägnant, klar und gut strukturiert sein

[Verfügbare Werkzeuge] (nur bei Bedarf verwenden, maximal 1-2 Mal aufrufen)
{tools_description}

[Werkzeugaufruf-Format]
<tool_call>
{{"name": "Werkzeugname", "parameters": {{"parameter_name": "parameter_wert"}}}}
</tool_call>

[Antwortstil]
- Prägnant und direkt, keine langen Texte schreiben
- > Format verwenden, um Schlüsselinhalte zu zitieren
- Erst Schlussfolgerungen geben, dann Gründe erklären
- IMMER auf Deutsch antworten, unabhängig von der Sprache des Quellmaterials oder Berichtsinhalts"""

CHAT_OBSERVATION_SUFFIX = "\n\nBitte beantworte die Frage prägnant."


# ═══════════════════════════════════════════════════════════════
# ReportAgent Hauptklasse
# ═══════════════════════════════════════════════════════════════


class ReportAgent:
    """
    Report Agent - Simulationsbericht-Generierungs-Agent

    Verwendet das ReACT-Muster (Reasoning + Acting):
    1. Planungsphase: Simulationsanforderungen analysieren, Berichtsgliederungsstruktur planen
    2. Generierungsphase: Inhalt abschnittsweise generieren, jeder Abschnitt kann Werkzeuge mehrfach aufrufen
    3. Reflexionsphase: Vollständigkeit und Genauigkeit des Inhalts prüfen
    """

    # Maximale Werkzeugaufrufe (pro Abschnitt)
    MAX_TOOL_CALLS_PER_SECTION = 5

    # Maximale Reflexionsrunden
    MAX_REFLECTION_ROUNDS = 3

    # Maximale Werkzeugaufrufe in der Konversation
    MAX_TOOL_CALLS_PER_CHAT = 2

    def __init__(
        self,
        graph_id: str,
        simulation_id: str,
        simulation_requirement: str,
        llm_client: Optional[LLMClient] = None,
        graph_tools: Optional[GraphToolsService] = None
    ):
        """
        Report Agent initialisieren

        Args:
            graph_id: Graph-ID
            simulation_id: Simulations-ID
            simulation_requirement: Beschreibung der Simulationsanforderung
            llm_client: LLM-Client (optional)
            graph_tools: Graph-Werkzeugdienst (optional, erfordert externe GraphStorage-Injektion)
        """
        self.graph_id = graph_id
        self.simulation_id = simulation_id
        self.simulation_requirement = simulation_requirement

        self.llm = llm_client or LLMClient()
        if graph_tools is None:
            raise ValueError(
                "graph_tools (GraphToolsService) ist erforderlich. "
                "Erstelle es über GraphToolsService(storage=...) und übergib es."
            )
        self.graph_tools = graph_tools

        # Werkzeugdefinitionen
        self.tools = self._define_tools()

        # Logger (wird in generate_report initialisiert)
        self.report_logger: Optional[ReportLogger] = None
        # Konsolen-Logger (wird in generate_report initialisiert)
        self.console_logger: Optional[ReportConsoleLogger] = None

        logger.info(f"ReportAgent-Initialisierung abgeschlossen: graph_id={graph_id}, simulation_id={simulation_id}")

    def _define_tools(self) -> Dict[str, Dict[str, Any]]:
        """Verfügbare Werkzeuge definieren"""
        return {
            "insight_forge": {
                "name": "insight_forge",
                "description": TOOL_DESC_INSIGHT_FORGE,
                "parameters": {
                    "query": "Die Frage oder das Thema, das du tiefgehend analysieren möchtest",
                    "report_context": "Kontext des aktuellen Berichtsabschnitts (optional, hilft bei der Generierung genauerer Unterfragen)"
                }
            },
            "panorama_search": {
                "name": "panorama_search",
                "description": TOOL_DESC_PANORAMA_SEARCH,
                "parameters": {
                    "query": "Suchanfrage, wird für Relevanzsortierung verwendet",
                    "include_expired": "Ob abgelaufene/historische Inhalte einbezogen werden sollen (Standard True)"
                }
            },
            "quick_search": {
                "name": "quick_search",
                "description": TOOL_DESC_QUICK_SEARCH,
                "parameters": {
                    "query": "Suchanfragezeichenkette",
                    "limit": "Anzahl der zurückzugebenden Ergebnisse (optional, Standard 10)"
                }
            },
            "interview_agents": {
                "name": "interview_agents",
                "description": TOOL_DESC_INTERVIEW_AGENTS,
                "parameters": {
                    "interview_topic": "Interviewthema oder Anforderungsbeschreibung (z.B. 'Meinungen der Studierenden zum Formaldehyd-Vorfall im Wohnheim verstehen')",
                    "max_agents": "Maximale Anzahl zu interviewender Agents (optional, Standard 5, Maximum 10)"
                }
            }
        }

    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any], report_context: str = "") -> str:
        """
        Werkzeugaufruf ausführen

        Args:
            tool_name: Werkzeugname
            parameters: Werkzeugparameter
            report_context: Berichtskontext (für InsightForge)

        Returns:
            Werkzeugausführungsergebnis (Textformat)
        """
        logger.info(f"Werkzeug wird ausgeführt: {tool_name}, Parameter: {parameters}")

        try:
            if tool_name == "insight_forge":
                query = parameters.get("query", "")
                ctx = parameters.get("report_context", "") or report_context
                result = self.graph_tools.insight_forge(
                    graph_id=self.graph_id,
                    query=query,
                    simulation_requirement=self.simulation_requirement,
                    report_context=ctx
                )
                return result.to_text()

            elif tool_name == "panorama_search":
                # Breitensuche - vollständiges Panorama erhalten
                query = parameters.get("query", "")
                include_expired = parameters.get("include_expired", True)
                if isinstance(include_expired, str):
                    include_expired = include_expired.lower() in ['true', '1', 'yes']
                result = self.graph_tools.panorama_search(
                    graph_id=self.graph_id,
                    query=query,
                    include_expired=include_expired
                )
                return result.to_text()

            elif tool_name == "quick_search":
                # Einfache Suche - Schnellabruf
                query = parameters.get("query", "")
                limit = parameters.get("limit", 10)
                if isinstance(limit, str):
                    limit = int(limit)
                result = self.graph_tools.quick_search(
                    graph_id=self.graph_id,
                    query=query,
                    limit=limit
                )
                return result.to_text()

            elif tool_name == "interview_agents":
                # Tiefeninterview - echte OASIS-Interview-API aufrufen, um simulierte Agent-Antworten zu erhalten (duale Plattform)
                interview_topic = parameters.get("interview_topic", parameters.get("query", ""))
                max_agents = parameters.get("max_agents", 5)
                if isinstance(max_agents, str):
                    max_agents = int(max_agents)
                max_agents = min(max_agents, 10)
                result = self.graph_tools.interview_agents(
                    simulation_id=self.simulation_id,
                    interview_requirement=interview_topic,
                    simulation_requirement=self.simulation_requirement,
                    max_agents=max_agents
                )
                return result.to_text()

            # ========== Abwärtskompatibilität: Alte Werkzeuge (interne Weiterleitung zu neuen Werkzeugen) ==========

            elif tool_name == "search_graph":
                # Weiterleitung zu quick_search
                logger.info("search_graph wurde zu quick_search weitergeleitet")
                return self._execute_tool("quick_search", parameters, report_context)

            elif tool_name == "get_graph_statistics":
                result = self.graph_tools.get_graph_statistics(self.graph_id)
                return json.dumps(result, ensure_ascii=False, indent=2)

            elif tool_name == "get_entity_summary":
                entity_name = parameters.get("entity_name", "")
                result = self.graph_tools.get_entity_summary(
                    graph_id=self.graph_id,
                    entity_name=entity_name
                )
                return json.dumps(result, ensure_ascii=False, indent=2)

            elif tool_name == "get_simulation_context":
                # Weiterleitung zu insight_forge, da leistungsstärker
                logger.info("get_simulation_context wurde zu insight_forge weitergeleitet")
                query = parameters.get("query", self.simulation_requirement)
                return self._execute_tool("insight_forge", {"query": query}, report_context)

            elif tool_name == "get_entities_by_type":
                entity_type = parameters.get("entity_type", "")
                nodes = self.graph_tools.get_entities_by_type(
                    graph_id=self.graph_id,
                    entity_type=entity_type
                )
                result = [n.to_dict() for n in nodes]
                return json.dumps(result, ensure_ascii=False, indent=2)

            else:
                return f"Unbekanntes Werkzeug: {tool_name}. Bitte verwende eines der folgenden Werkzeuge: insight_forge, panorama_search, quick_search"

        except Exception as e:
            logger.error(f"Werkzeugausführung fehlgeschlagen: {tool_name}, Fehler: {str(e)}")
            return f"Werkzeugausführung fehlgeschlagen: {str(e)}"

    # Gültige Werkzeugnamen-Menge, wird zur Validierung beim Parsen des Raw-JSON-Fallbacks verwendet
    VALID_TOOL_NAMES = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Werkzeugaufrufe aus LLM-Antwort parsen

        Unterstützte Formate (nach Priorität):
        1. <tool_call>{"name": "werkzeug_name", "parameters": {...}}</tool_call>
        2. Rohes JSON (die gesamte Antwort oder eine einzelne Zeile ist ein Werkzeugaufruf-JSON)
        """
        tool_calls = []

        # Format 1: XML-Stil (Standardformat)
        xml_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        for match in re.finditer(xml_pattern, response, re.DOTALL):
            try:
                call_data = json.loads(match.group(1))
                tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        if tool_calls:
            return tool_calls

        # Format 2: Fallback - LLM gibt direkt rohes JSON aus (nicht in <tool_call>-Tags eingewickelt)
        # Nur versuchen, wenn Format 1 nicht übereinstimmte, um Fehlzuordnung von JSON im Fließtext zu vermeiden
        stripped = response.strip()
        if stripped.startswith('{') and stripped.endswith('}'):
            try:
                call_data = json.loads(stripped)
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
                    return tool_calls
            except json.JSONDecodeError:
                pass

        # Antwort kann Denktext + rohes JSON enthalten, versuche das letzte JSON-Objekt zu extrahieren
        json_pattern = r'(\{"(?:name|tool)"\s*:.*?\})\s*$'
        match = re.search(json_pattern, stripped, re.DOTALL)
        if match:
            try:
                call_data = json.loads(match.group(1))
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        return tool_calls

    def _is_valid_tool_call(self, data: dict) -> bool:
        """Prüfen, ob das geparste JSON ein gültiger Werkzeugaufruf ist"""
        # Unterstützt sowohl {"name": ..., "parameters": ...} als auch {"tool": ..., "params": ...} Schlüsselnamen
        tool_name = data.get("name") or data.get("tool")
        if tool_name and tool_name in self.VALID_TOOL_NAMES:
            # Schlüsselnamen auf name / parameters normalisieren
            if "tool" in data:
                data["name"] = data.pop("tool")
            if "params" in data and "parameters" not in data:
                data["parameters"] = data.pop("params")
            return True
        return False

    def _get_tools_description(self) -> str:
        """Werkzeugbeschreibungstext generieren"""
        desc_parts = ["Verfügbare Werkzeuge:"]
        for name, tool in self.tools.items():
            params_desc = ", ".join([f"{k}: {v}" for k, v in tool["parameters"].items()])
            desc_parts.append(f"- {name}: {tool['description']}")
            if params_desc:
                desc_parts.append(f"  Parameter: {params_desc}")
        return "\n".join(desc_parts)

    def plan_outline(
        self,
        progress_callback: Optional[Callable] = None
    ) -> ReportOutline:
        """
        Berichtsgliederung planen

        LLM verwenden, um Simulationsanforderungen zu analysieren und die Berichtsstruktur zu planen

        Args:
            progress_callback: Fortschritts-Callback-Funktion

        Returns:
            ReportOutline: Berichtsgliederung
        """
        logger.info("Starte Planung der Berichtsgliederung...")

        if progress_callback:
            progress_callback("planning", 0, "Simulationsanforderungen werden analysiert...")

        # Zuerst Simulationskontext abrufen
        context = self.graph_tools.get_simulation_context(
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement
        )

        if progress_callback:
            progress_callback("planning", 30, "Berichtsgliederung wird generiert...")

        system_prompt = PLAN_SYSTEM_PROMPT
        user_prompt = PLAN_USER_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            total_nodes=context.get('graph_statistics', {}).get('total_nodes', 0),
            total_edges=context.get('graph_statistics', {}).get('total_edges', 0),
            entity_types=list(context.get('graph_statistics', {}).get('entity_types', {}).keys()),
            total_entities=context.get('total_entities', 0),
            related_facts_json=json.dumps(context.get('related_facts', [])[:10], ensure_ascii=False, indent=2),
        )

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )

            if progress_callback:
                progress_callback("planning", 80, "Gliederungsstruktur wird geparst...")

            # Gliederung parsen
            sections = []
            for section_data in response.get("sections", []):
                sections.append(ReportSection(
                    title=section_data.get("title", ""),
                    content=""
                ))

            outline = ReportOutline(
                title=response.get("title", "Simulationsanalysebericht"),
                summary=response.get("summary", ""),
                sections=sections
            )

            if progress_callback:
                progress_callback("planning", 100, "Gliederungsplanung abgeschlossen")

            logger.info(f"Gliederungsplanung abgeschlossen: {len(sections)} Abschnitte")
            return outline

        except Exception as e:
            logger.error(f"Gliederungsplanung fehlgeschlagen: {str(e)}")
            # Standard-Gliederung zurückgeben (3 Abschnitte als Fallback)
            return ReportOutline(
                title="Zukunftsprognose-Bericht",
                summary="Zukunftstrend- und Risikoanalyse basierend auf Simulationsprognosen",
                sections=[
                    ReportSection(title="Prognoseszenario und Kernerkenntnisse"),
                    ReportSection(title="Gruppenverhalten-Prognoseanalyse"),
                    ReportSection(title="Trendausblick und Risikowarnung")
                ]
            )

    def _generate_section_react(
        self,
        section: ReportSection,
        outline: ReportOutline,
        previous_sections: List[str],
        progress_callback: Optional[Callable] = None,
        section_index: int = 0
    ) -> str:
        """
        Einzelnen Abschnittsinhalt mit ReACT-Muster generieren

        ReACT-Schleife:
        1. Thought - Analysieren, welche Informationen benötigt werden
        2. Action - Werkzeug aufrufen, um Informationen zu erhalten
        3. Observation - Werkzeugergebnisse analysieren
        4. Wiederholen, bis Informationen ausreichend oder maximale Iterationen erreicht
        5. Final Answer - Abschnittsinhalt generieren

        Args:
            section: Zu generierender Abschnitt
            outline: Vollständige Gliederung
            previous_sections: Inhalt vorheriger Abschnitte (für Kohärenz)
            progress_callback: Fortschritts-Callback
            section_index: Abschnittsindex (für Logging)

        Returns:
            Abschnittsinhalt (Markdown-Format)
        """
        logger.info(f"ReACT generiert Abschnitt: {section.title}")

        # Abschnittsstart protokollieren
        if self.report_logger:
            self.report_logger.log_section_start(section.title, section_index)

        system_prompt = SECTION_SYSTEM_PROMPT_TEMPLATE.format(
            report_title=outline.title,
            report_summary=outline.summary,
            simulation_requirement=self.simulation_requirement,
            section_title=section.title,
            tools_description=self._get_tools_description(),
        )

        # Benutzer-Prompt aufbauen - maximal 4000 Zeichen pro abgeschlossenem Abschnitt übergeben
        if previous_sections:
            previous_parts = []
            for sec in previous_sections:
                # Maximal 4000 Zeichen pro Abschnitt
                truncated = sec[:4000] + "..." if len(sec) > 4000 else sec
                previous_parts.append(truncated)
            previous_content = "\n\n---\n\n".join(previous_parts)
        else:
            previous_content = "(Dies ist der erste Abschnitt)"

        user_prompt = SECTION_USER_PROMPT_TEMPLATE.format(
            previous_content=previous_content,
            section_title=section.title,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # ReACT-Schleife
        tool_calls_count = 0
        max_iterations = 5  # Maximale Iterationen
        min_tool_calls = 3  # Mindest-Werkzeugaufrufe
        conflict_retries = 0  # Aufeinanderfolgende Konflikte, bei denen Werkzeugaufrufe und Final Answer gleichzeitig erscheinen
        used_tools = set()  # Bereits aufgerufene Werkzeugnamen aufzeichnen
        all_tools = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

        # Berichtskontext für InsightForge-Unterfragengenerierung
        report_context = f"Abschnittstitel: {section.title}\nSimulationsanforderung: {self.simulation_requirement}"

        for iteration in range(max_iterations):
            if progress_callback:
                progress_callback(
                    "generating",
                    int((iteration / max_iterations) * 100),
                    f"Tiefenabruf und Verfassung läuft ({tool_calls_count}/{self.MAX_TOOL_CALLS_PER_SECTION})"
                )

            # LLM aufrufen
            response = self.llm.chat(
                messages=messages,
                temperature=0.5,
                max_tokens=4096
            )

            # Prüfen, ob LLM-Rückgabe None ist (API-Ausnahme oder leerer Inhalt)
            if response is None:
                logger.warning(f"Abschnitt {section.title} Runde {iteration + 1} Iteration: LLM hat None zurückgegeben")
                # Wenn weitere Iterationen vorhanden, Nachricht hinzufügen und erneut versuchen
                if iteration < max_iterations - 1:
                    messages.append({"role": "assistant", "content": "(Antwort leer)"})
                    messages.append({"role": "user", "content": "Bitte fahre mit der Inhaltsgenerierung fort."})
                    continue
                # Letzte Iteration hat auch None zurückgegeben, Schleife verlassen und erzwungenen Abschluss einleiten
                break

            logger.debug(f"LLM-Antwort: {response[:200]}...")

            # Einmal parsen, Ergebnis wiederverwenden
            tool_calls = self._parse_tool_calls(response)
            has_tool_calls = bool(tool_calls)
            has_final_answer = "Final Answer:" in response

            # ── Konfliktbehandlung: LLM hat gleichzeitig Werkzeugaufrufe und Final Answer ausgegeben ──
            if has_tool_calls and has_final_answer:
                conflict_retries += 1
                logger.warning(
                    f"Abschnitt {section.title} Runde {iteration+1}: "
                    f"LLM hat gleichzeitig Werkzeugaufrufe und Final Answer ausgegeben (Konflikt Runde {conflict_retries})"
                )

                if conflict_retries <= 2:
                    # Erste zwei Male: Diese Antwort verwerfen und LLM um erneute Antwort bitten
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": (
                            "[Formatfehler] Du kannst nicht sowohl Werkzeugaufrufe als auch Final Answer in einer Antwort enthalten.\n"
                            "Jede Antwort kann nur eine der folgenden Optionen wählen:\n"
                            "- Ein Werkzeug aufrufen (einen <tool_call>-Block ausgeben, kein Final Answer schreiben)\n"
                            "- Endgültigen Inhalt ausgeben (mit 'Final Answer:' beginnen, kein <tool_call> enthalten)\n"
                            "Bitte antworte erneut und wähle nur eine Option."
                        ),
                    })
                    continue
                else:
                    # Drittes Mal: Herabstufen, zum ersten Werkzeugaufruf kürzen, Ausführung erzwingen
                    logger.warning(
                        f"Abschnitt {section.title}: {conflict_retries} aufeinanderfolgende Konflikte, "
                        "herabgestuft auf Kürzung und Ausführung des ersten Werkzeugaufrufs"
                    )
                    first_tool_end = response.find('</tool_call>')
                    if first_tool_end != -1:
                        response = response[:first_tool_end + len('</tool_call>')]
                        tool_calls = self._parse_tool_calls(response)
                        has_tool_calls = bool(tool_calls)
                    has_final_answer = False
                    conflict_retries = 0

            # LLM-Antwort protokollieren
            if self.report_logger:
                self.report_logger.log_llm_response(
                    section_title=section.title,
                    section_index=section_index,
                    response=response,
                    iteration=iteration + 1,
                    has_tool_calls=has_tool_calls,
                    has_final_answer=has_final_answer
                )

            # ── Fall 1: LLM hat Final Answer ausgegeben ──
            if has_final_answer:
                # Unzureichende Werkzeugaufrufe, ablehnen und weitere Aufrufe anfordern
                if tool_calls_count < min_tool_calls:
                    messages.append({"role": "assistant", "content": response})
                    unused_tools = all_tools - used_tools
                    unused_hint = f"(Diese Werkzeuge wurden noch nicht verwendet, Empfehlung: {', '.join(unused_tools)})" if unused_tools else ""
                    messages.append({
                        "role": "user",
                        "content": REACT_INSUFFICIENT_TOOLS_MSG.format(
                            tool_calls_count=tool_calls_count,
                            min_tool_calls=min_tool_calls,
                            unused_hint=unused_hint,
                        ),
                    })
                    continue

                # Normaler Abschluss
                final_answer = response.split("Final Answer:")[-1].strip()
                logger.info(f"Abschnitt {section.title} Generierung abgeschlossen (Werkzeugaufrufe: {tool_calls_count} Mal)")

                if self.report_logger:
                    self.report_logger.log_section_content(
                        section_title=section.title,
                        section_index=section_index,
                        content=final_answer,
                        tool_calls_count=tool_calls_count
                    )
                return final_answer

            # ── Fall 2: LLM versucht Werkzeuge aufzurufen ──
            if has_tool_calls:
                # Werkzeugkontingent erschöpft -> klar informieren, Final Answer anfordern
                if tool_calls_count >= self.MAX_TOOL_CALLS_PER_SECTION:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": REACT_TOOL_LIMIT_MSG.format(
                            tool_calls_count=tool_calls_count,
                            max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        ),
                    })
                    continue

                # Nur den ersten Werkzeugaufruf ausführen
                call = tool_calls[0]
                if len(tool_calls) > 1:
                    logger.info(f"LLM hat versucht {len(tool_calls)} Werkzeuge aufzurufen, nur das erste wird ausgeführt: {call['name']}")

                if self.report_logger:
                    self.report_logger.log_tool_call(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        parameters=call.get("parameters", {}),
                        iteration=iteration + 1
                    )

                result = self._execute_tool(
                    call["name"],
                    call.get("parameters", {}),
                    report_context=report_context
                )

                if self.report_logger:
                    self.report_logger.log_tool_result(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        result=result,
                        iteration=iteration + 1
                    )

                tool_calls_count += 1
                used_tools.add(call['name'])

                # Hinweis für nicht verwendete Werkzeuge erstellen
                unused_tools = all_tools - used_tools
                unused_hint = ""
                if unused_tools and tool_calls_count < self.MAX_TOOL_CALLS_PER_SECTION:
                    unused_hint = REACT_UNUSED_TOOLS_HINT.format(unused_list=", ".join(unused_tools))

                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": REACT_OBSERVATION_TEMPLATE.format(
                        tool_name=call["name"],
                        result=result,
                        tool_calls_count=tool_calls_count,
                        max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        used_tools_str=", ".join(used_tools),
                        unused_hint=unused_hint,
                    ),
                })
                continue

            # ── Fall 3: Weder Werkzeugaufruf noch Final Answer ──
            messages.append({"role": "assistant", "content": response})

            if tool_calls_count < min_tool_calls:
                # Werkzeugaufrufanzahl unzureichend, nicht verwendete Werkzeuge empfehlen
                unused_tools = all_tools - used_tools
                unused_hint = f"(Diese Werkzeuge wurden noch nicht verwendet, Empfehlung: {', '.join(unused_tools)})" if unused_tools else ""

                messages.append({
                    "role": "user",
                    "content": REACT_INSUFFICIENT_TOOLS_MSG_ALT.format(
                        tool_calls_count=tool_calls_count,
                        min_tool_calls=min_tool_calls,
                        unused_hint=unused_hint,
                    ),
                })
                continue

            # Diesen Inhalt direkt als Endantwort übernehmen, nicht weiter warten
            logger.info(f"Abschnitt {section.title}: Kein 'Final Answer:'-Präfix erkannt, LLM-Ausgabe direkt als Endinhalt übernommen (Werkzeugaufrufe: {tool_calls_count} Mal)")
            final_answer = response.strip()

            if self.report_logger:
                self.report_logger.log_section_content(
                    section_title=section.title,
                    section_index=section_index,
                    content=final_answer,
                    tool_calls_count=tool_calls_count
                )
            return final_answer

        # Maximale Iterationen erreicht, Inhaltsgenerierung erzwingen
        logger.warning(f"Abschnitt {section.title} hat maximale Iterationsanzahl erreicht, Generierung wird erzwungen")
        messages.append({"role": "user", "content": REACT_FORCE_FINAL_MSG})

        response = self.llm.chat(
            messages=messages,
            temperature=0.5,
            max_tokens=4096
        )

        # Erzwungenen Abschluss prüfen, wenn LLM None zurückgibt
        if response is None:
            final_answer = "(Dieser Abschnitt konnte nicht generiert werden: LLM hat eine leere Antwort zurückgegeben, bitte später erneut versuchen)"
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
        else:
            final_answer = response

        # Abschnittsinhalts-Generierungsabschluss protokollieren
        if self.report_logger:
            self.report_logger.log_section_content(
                section_title=section.title,
                section_index=section_index,
                content=final_answer,
                tool_calls_count=tool_calls_count
            )

        return final_answer

    def generate_report(
        self,
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        report_id: Optional[str] = None
    ) -> Report:
        """
        Vollständigen Bericht generieren (Echtzeitausgabe pro Abschnitt)

        Dateistruktur:
        reports/{report_id}/
            outline.json    - Berichtsgliederung
            progress.json   - Generierungsfortschritt
            section_01.md   - Abschnitt 1
            section_02.md   - Abschnitt 2
            ...
            full_report.md  - Vollständiger Bericht

        Args:
            progress_callback: Fortschritts-Callback-Funktion
            report_id: Berichts-ID (optional, wird automatisch generiert wenn nicht angegeben)

        Returns:
            Report: Vollständiger Bericht
        """
        import uuid

        # Wenn keine report_id angegeben, automatisch generieren
        if not report_id:
            report_id = f"report_{uuid.uuid4().hex[:12]}"
        start_time = datetime.now()

        report = Report(
            report_id=report_id,
            simulation_id=self.simulation_id,
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement,
            status=ReportStatus.PENDING,
            created_at=datetime.now().isoformat()
        )

        # Liste der abgeschlossenen Abschnittstitel (für Fortschrittsverfolgung)
        completed_section_titles = []

        try:
            # Initialisierung: Berichtsordner erstellen und Anfangszustand speichern
            ReportManager._ensure_report_folder(report_id)

            # Strukturierten Logger initialisieren (agent_log.jsonl)
            self.report_logger = ReportLogger(report_id)
            self.report_logger.log_start(
                simulation_id=self.simulation_id,
                graph_id=self.graph_id,
                simulation_requirement=self.simulation_requirement
            )

            # Konsolen-Logger initialisieren (console_log.txt)
            self.console_logger = ReportConsoleLogger(report_id)

            ReportManager.update_progress(
                report_id, "pending", 0, "Bericht wird initialisiert...",
                completed_sections=[]
            )
            ReportManager.save_report(report)

            # Phase 1: Gliederung planen
            report.status = ReportStatus.PLANNING
            ReportManager.update_progress(
                report_id, "planning", 5, "Planung der Berichtsgliederung wird gestartet...",
                completed_sections=[]
            )

            # Start der Gliederungsplanung protokollieren
            self.report_logger.log_planning_start()

            if progress_callback:
                progress_callback("planning", 0, "Planung der Berichtsgliederung wird gestartet...")

            outline = self.plan_outline(
                progress_callback=lambda stage, prog, msg:
                    progress_callback(stage, prog // 5, msg) if progress_callback else None
            )
            report.outline = outline

            # Abschluss der Planung protokollieren
            self.report_logger.log_planning_complete(outline.to_dict())

            # Gliederung in Datei speichern
            ReportManager.save_outline(report_id, outline)
            ReportManager.update_progress(
                report_id, "planning", 15, f"Gliederungsplanung abgeschlossen, insgesamt {len(outline.sections)} Abschnitte",
                completed_sections=[]
            )
            ReportManager.save_report(report)

            logger.info(f"Gliederung in Datei gespeichert: {report_id}/outline.json")

            # Phase 2: Abschnitte sequentiell generieren (pro Abschnitt speichern)
            report.status = ReportStatus.GENERATING

            total_sections = len(outline.sections)
            generated_sections = []  # Inhalt für Kontext speichern

            for i, section in enumerate(outline.sections):
                section_num = i + 1
                base_progress = 20 + int((i / total_sections) * 70)

                # Fortschritt aktualisieren
                ReportManager.update_progress(
                    report_id, "generating", base_progress,
                    f"Abschnitt wird generiert: {section.title} ({section_num}/{total_sections})",
                    current_section=section.title,
                    completed_sections=completed_section_titles
                )

                if progress_callback:
                    progress_callback(
                        "generating",
                        base_progress,
                        f"Abschnitt wird generiert: {section.title} ({section_num}/{total_sections})"
                    )

                # Hauptabschnittsinhalt generieren
                section_content = self._generate_section_react(
                    section=section,
                    outline=outline,
                    previous_sections=generated_sections,
                    progress_callback=lambda stage, prog, msg:
                        progress_callback(
                            stage,
                            base_progress + int(prog * 0.7 / total_sections),
                            msg
                        ) if progress_callback else None,
                    section_index=section_num
                )

                section.content = section_content
                generated_sections.append(f"## {section.title}\n\n{section_content}")

                # Abschnitt speichern
                ReportManager.save_section(report_id, section_num, section)
                completed_section_titles.append(section.title)

                # Abschnittsabschluss protokollieren
                full_section_content = f"## {section.title}\n\n{section_content}"

                if self.report_logger:
                    self.report_logger.log_section_full_complete(
                        section_title=section.title,
                        section_index=section_num,
                        full_content=full_section_content.strip()
                    )

                logger.info(f"Abschnitt gespeichert: {report_id}/section_{section_num:02d}.md")

                # Fortschritt aktualisieren
                ReportManager.update_progress(
                    report_id, "generating",
                    base_progress + int(70 / total_sections),
                    f"Abschnitt {section.title} abgeschlossen",
                    current_section=None,
                    completed_sections=completed_section_titles
                )

            # Phase 3: Vollständigen Bericht zusammenstellen
            if progress_callback:
                progress_callback("generating", 95, "Vollständiger Bericht wird zusammengestellt...")

            ReportManager.update_progress(
                report_id, "generating", 95, "Vollständiger Bericht wird zusammengestellt...",
                completed_sections=completed_section_titles
            )

            # ReportManager zum Zusammenstellen des vollständigen Berichts verwenden
            report.markdown_content = ReportManager.assemble_full_report(report_id, outline)
            report.status = ReportStatus.COMPLETED
            report.completed_at = datetime.now().isoformat()

            # Gesamtzeit berechnen
            total_time_seconds = (datetime.now() - start_time).total_seconds()

            # Berichtsabschluss protokollieren
            if self.report_logger:
                self.report_logger.log_report_complete(
                    total_sections=total_sections,
                    total_time_seconds=total_time_seconds
                )

            # Finalen Bericht speichern
            ReportManager.save_report(report)
            ReportManager.update_progress(
                report_id, "completed", 100, "Berichtsgenerierung abgeschlossen",
                completed_sections=completed_section_titles
            )

            if progress_callback:
                progress_callback("completed", 100, "Berichtsgenerierung abgeschlossen")

            logger.info(f"Berichtsgenerierung abgeschlossen: {report_id}")

            # Konsolen-Logger schließen
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None

            return report

        except Exception as e:
            logger.error(f"Berichtsgenerierung fehlgeschlagen: {str(e)}")
            report.status = ReportStatus.FAILED
            report.error = str(e)

            # Fehler protokollieren
            if self.report_logger:
                self.report_logger.log_error(str(e), "failed")

            # Fehlstatus speichern
            try:
                ReportManager.save_report(report)
                ReportManager.update_progress(
                    report_id, "failed", -1, f"Berichtsgenerierung fehlgeschlagen: {str(e)}",
                    completed_sections=completed_section_titles
                )
            except Exception:
                pass  # Speicherfehler ignorieren

            # Konsolen-Logger schließen
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None

            return report

    def chat(
        self,
        message: str,
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Mit dem Report Agent chatten

        Im Chat kann der Agent autonom Abrufwerkzeuge aufrufen, um Fragen zu beantworten

        Args:
            message: Benutzernachricht
            chat_history: Chatverlauf

        Returns:
            {
                "response": "Agent-Antwort",
                "tool_calls": [Liste der aufgerufenen Werkzeuge],
                "sources": [Informationsquellen]
            }
        """
        logger.info(f"Report Agent Chat: {message[:50]}...")

        chat_history = chat_history or []

        # Bereits generierten Berichtsinhalt abrufen
        report_content = ""
        try:
            report = ReportManager.get_report_by_simulation(self.simulation_id)
            if report and report.markdown_content:
                # Berichtslänge begrenzen, um zu langen Kontext zu vermeiden
                report_content = report.markdown_content[:15000]
                if len(report.markdown_content) > 15000:
                    report_content += "\n\n... [Berichtsinhalt wurde gekürzt] ..."
        except Exception as e:
            logger.warning(f"Abruf des Berichtsinhalts fehlgeschlagen: {e}")

        system_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            report_content=report_content if report_content else "(Kein Bericht vorhanden)",
            tools_description=self._get_tools_description(),
        )

        # Nachrichten aufbauen
        messages = [{"role": "system", "content": system_prompt}]

        # Chatverlauf hinzufügen
        for h in chat_history[-10:]:  # Verlaufslänge begrenzen
            messages.append(h)

        # Benutzernachricht hinzufügen
        messages.append({
            "role": "user",
            "content": message
        })

        # ReACT-Schleife (vereinfachte Version)
        tool_calls_made = []
        max_iterations = 2  # Iterationen reduzieren

        for iteration in range(max_iterations):
            response = self.llm.chat(
                messages=messages,
                temperature=0.5
            )

            # Werkzeugaufrufe parsen
            tool_calls = self._parse_tool_calls(response)

            if not tool_calls:
                # Keine Werkzeugaufrufe, Antwort direkt zurückgeben
                clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', response, flags=re.DOTALL)
                clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)

                return {
                    "response": clean_response.strip(),
                    "tool_calls": tool_calls_made,
                    "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
                }

            # Werkzeugaufruf ausführen (Anzahl begrenzen)
            tool_results = []
            for call in tool_calls[:1]:  # Maximal 1 Werkzeugaufruf ausführen
                if len(tool_calls_made) >= self.MAX_TOOL_CALLS_PER_CHAT:
                    break
                result = self._execute_tool(call["name"], call.get("parameters", {}))
                tool_results.append({
                    "tool": call["name"],
                    "result": result[:1500]  # Ergebnislänge begrenzen
                })
                tool_calls_made.append(call)

            # Ergebnis als Nachricht hinzufügen
            messages.append({"role": "assistant", "content": response})
            observation = "\n".join([f"[{r['tool']}-Ergebnis]\n{r['result']}" for r in tool_results])
            messages.append({
                "role": "user",
                "content": observation + CHAT_OBSERVATION_SUFFIX
            })

        # Maximale Iteration erreicht, finale Antwort abrufen
        final_response = self.llm.chat(
            messages=messages,
            temperature=0.5
        )

        # Antwort bereinigen
        clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', final_response, flags=re.DOTALL)
        clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)

        return {
            "response": clean_response.strip(),
            "tool_calls": tool_calls_made,
            "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
        }


class ReportManager:
    """
    Berichtsmanager

    Verantwortlich für die persistente Speicherung und den Abruf von Berichten

    Dateistruktur (Ausgabe pro Abschnitt):
    reports/
      {report_id}/
        meta.json          - Berichts-Metainformationen und Status
        outline.json       - Berichtsgliederung
        progress.json      - Generierungsfortschritt
        section_01.md      - Abschnitt 1
        section_02.md      - Abschnitt 2
        ...
        full_report.md     - Vollständiger Bericht
    """

    # Berichtsspeicherverzeichnis
    REPORTS_DIR = os.path.join(Config.UPLOAD_FOLDER, 'reports')

    @classmethod
    def _ensure_reports_dir(cls):
        """Sicherstellen, dass das Berichts-Stammverzeichnis existiert"""
        os.makedirs(cls.REPORTS_DIR, exist_ok=True)

    @classmethod
    def _get_report_folder(cls, report_id: str) -> str:
        """Berichtsordnerpfad abrufen"""
        return os.path.join(cls.REPORTS_DIR, report_id)

    @classmethod
    def _ensure_report_folder(cls, report_id: str) -> str:
        """Sicherstellen, dass der Berichtsordner existiert und Pfad zurückgeben"""
        folder = cls._get_report_folder(report_id)
        os.makedirs(folder, exist_ok=True)
        return folder

    @classmethod
    def _get_report_path(cls, report_id: str) -> str:
        """Pfad der Berichts-Metainformationsdatei abrufen"""
        return os.path.join(cls._get_report_folder(report_id), "meta.json")

    @classmethod
    def _get_report_markdown_path(cls, report_id: str) -> str:
        """Pfad der vollständigen Berichts-Markdown-Datei abrufen"""
        return os.path.join(cls._get_report_folder(report_id), "full_report.md")

    @classmethod
    def _get_outline_path(cls, report_id: str) -> str:
        """Gliederungsdateipfad abrufen"""
        return os.path.join(cls._get_report_folder(report_id), "outline.json")

    @classmethod
    def _get_progress_path(cls, report_id: str) -> str:
        """Fortschrittsdateipfad abrufen"""
        return os.path.join(cls._get_report_folder(report_id), "progress.json")

    @classmethod
    def _get_section_path(cls, report_id: str, section_index: int) -> str:
        """Abschnitts-Markdown-Dateipfad abrufen"""
        return os.path.join(cls._get_report_folder(report_id), f"section_{section_index:02d}.md")

    @classmethod
    def _get_agent_log_path(cls, report_id: str) -> str:
        """Agent-Log-Dateipfad abrufen"""
        return os.path.join(cls._get_report_folder(report_id), "agent_log.jsonl")

    @classmethod
    def _get_console_log_path(cls, report_id: str) -> str:
        """Konsolen-Log-Dateipfad abrufen"""
        return os.path.join(cls._get_report_folder(report_id), "console_log.txt")

    @classmethod
    def get_console_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Konsolen-Log-Inhalt abrufen

        Dies ist das Konsolen-Ausgabelog (INFO, WARNING usw.) während der Berichtsgenerierung,
        unterscheidet sich von den strukturierten agent_log.jsonl-Logs.

        Args:
            report_id: Berichts-ID
            from_line: Ab welcher Zeile gelesen werden soll (für inkrementellen Abruf, 0 bedeutet von Anfang an)

        Returns:
            {
                "logs": [Liste der Logzeilen],
                "total_lines": Gesamtzeilenanzahl,
                "from_line": Startzeilennummer,
                "has_more": Ob weitere Logs vorhanden sind
            }
        """
        log_path = cls._get_console_log_path(report_id)

        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }

        logs = []
        total_lines = 0

        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    # Original-Logzeile beibehalten, abschließende Zeilenzeichen entfernen
                    logs.append(line.rstrip('\n\r'))

        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # Bereits bis zum Ende gelesen
        }

    @classmethod
    def get_console_log_stream(cls, report_id: str) -> List[str]:
        """
        Vollständiges Konsolen-Log abrufen (einmaliger Abruf aller Einträge)

        Args:
            report_id: Berichts-ID

        Returns:
            Liste der Logzeilen
        """
        result = cls.get_console_log(report_id, from_line=0)
        return result["logs"]

    @classmethod
    def get_agent_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Agent-Log-Inhalt abrufen

        Args:
            report_id: Berichts-ID
            from_line: Ab welcher Zeile gelesen werden soll (für inkrementellen Abruf, 0 bedeutet von Anfang an)

        Returns:
            {
                "logs": [Liste der Logeinträge],
                "total_lines": Gesamtzeilenanzahl,
                "from_line": Startzeilennummer,
                "has_more": Ob weitere Logs vorhanden sind
            }
        """
        log_path = cls._get_agent_log_path(report_id)

        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }

        logs = []
        total_lines = 0

        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    try:
                        log_entry = json.loads(line.strip())
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        # Fehlgeschlagene Parsing-Zeilen überspringen
                        continue

        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # Bereits bis zum Ende gelesen
        }

    @classmethod
    def get_agent_log_stream(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Vollständiges Agent-Log abrufen (einmaliger Abruf aller Einträge)

        Args:
            report_id: Berichts-ID

        Returns:
            Liste der Logeinträge
        """
        result = cls.get_agent_log(report_id, from_line=0)
        return result["logs"]

    @classmethod
    def save_outline(cls, report_id: str, outline: ReportOutline) -> None:
        """
        Berichtsgliederung speichern

        Wird unmittelbar nach Abschluss der Planungsphase aufgerufen
        """
        cls._ensure_report_folder(report_id)

        with open(cls._get_outline_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(outline.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(f"Gliederung gespeichert: {report_id}")

    @classmethod
    def save_section(
        cls,
        report_id: str,
        section_index: int,
        section: ReportSection
    ) -> str:
        """
        Einzelnen Abschnitt speichern

        Wird unmittelbar nach Abschluss der Generierung jedes Abschnitts aufgerufen, implementiert Ausgabe pro Abschnitt

        Args:
            report_id: Berichts-ID
            section_index: Abschnittsindex (ab 1)
            section: Abschnittsobjekt

        Returns:
            Gespeicherter Dateipfad
        """
        cls._ensure_report_folder(report_id)

        # Abschnitts-Markdown-Inhalt aufbauen - mögliche doppelte Titel bereinigen
        cleaned_content = cls._clean_section_content(section.content, section.title)
        md_content = f"## {section.title}\n\n"
        if cleaned_content:
            md_content += f"{cleaned_content}\n\n"

        # Datei speichern
        file_suffix = f"section_{section_index:02d}.md"
        file_path = os.path.join(cls._get_report_folder(report_id), file_suffix)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        logger.info(f"Abschnitt gespeichert: {report_id}/{file_suffix}")
        return file_path

    @classmethod
    def _clean_section_content(cls, content: str, section_title: str) -> str:
        """
        Abschnittsinhalt bereinigen

        1. Am Inhaltsanfang doppelte Markdown-Titelzeilen mit dem Abschnittstitel entfernen
        2. Alle ### und niedrigere Titellevels in Fetttext umwandeln

        Args:
            content: Originalinhalt
            section_title: Abschnittstitel

        Returns:
            Bereinigter Inhalt
        """
        import re

        if not content:
            return content

        content = content.strip()
        lines = content.split('\n')
        cleaned_lines = []
        skip_next_empty = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Prüfen, ob es eine Markdown-Titelzeile ist
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)

            if heading_match:
                level = len(heading_match.group(1))
                title_text = heading_match.group(2).strip()

                # Prüfen, ob es ein Duplikat des Abschnittstitels ist (Duplikate in den ersten 5 Zeilen überspringen)
                if i < 5:
                    if title_text == section_title or title_text.replace(' ', '') == section_title.replace(' ', ''):
                        skip_next_empty = True
                        continue

                # Alle Titellevels (#, ##, ###, #### usw.) in Fettdruck umwandeln
                # Da Abschnittstitel vom System hinzugefügt werden, sollte der Inhalt keine Titel enthalten
                cleaned_lines.append(f"**{title_text}**")
                cleaned_lines.append("")  # Leerzeile hinzufügen
                continue

            # Wenn vorherige Zeile ein übersprungener Titel war und aktuelle Zeile leer ist, auch überspringen
            if skip_next_empty and stripped == '':
                skip_next_empty = False
                continue

            skip_next_empty = False
            cleaned_lines.append(line)

        # Führende Leerzeilen entfernen
        while cleaned_lines and cleaned_lines[0].strip() == '':
            cleaned_lines.pop(0)

        # Führende Trennlinien entfernen
        while cleaned_lines and cleaned_lines[0].strip() in ['---', '***', '___']:
            cleaned_lines.pop(0)
            # Gleichzeitig Leerzeilen nach Trennlinien entfernen
            while cleaned_lines and cleaned_lines[0].strip() == '':
                cleaned_lines.pop(0)

        return '\n'.join(cleaned_lines)

    @classmethod
    def update_progress(
        cls,
        report_id: str,
        status: str,
        progress: int,
        message: str,
        current_section: str = None,
        completed_sections: List[str] = None
    ) -> None:
        """
        Berichtsgenerierungsfortschritt aktualisieren

        Das Frontend kann progress.json lesen, um den Echtzeitfortschritt zu erhalten
        """
        cls._ensure_report_folder(report_id)

        progress_data = {
            "status": status,
            "progress": progress,
            "message": message,
            "current_section": current_section,
            "completed_sections": completed_sections or [],
            "updated_at": datetime.now().isoformat()
        }

        with open(cls._get_progress_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)

    @classmethod
    def get_progress(cls, report_id: str) -> Optional[Dict[str, Any]]:
        """Berichtsgenerierungsfortschritt abrufen"""
        path = cls._get_progress_path(report_id)

        if not os.path.exists(path):
            return None

        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @classmethod
    def get_generated_sections(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Liste der bereits generierten Abschnitte abrufen

        Gibt Informationen zu allen bereits gespeicherten Abschnittsdateien zurück
        """
        folder = cls._get_report_folder(report_id)

        if not os.path.exists(folder):
            return []

        sections = []
        for filename in sorted(os.listdir(folder)):
            if filename.startswith('section_') and filename.endswith('.md'):
                file_path = os.path.join(folder, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Abschnittsindex aus Dateiname parsen
                parts = filename.replace('.md', '').split('_')
                section_index = int(parts[1])

                sections.append({
                    "filename": filename,
                    "section_index": section_index,
                    "content": content
                })

        return sections

    @classmethod
    def assemble_full_report(cls, report_id: str, outline: ReportOutline) -> str:
        """
        Vollständigen Bericht zusammenstellen

        Aus gespeicherten Abschnittsdateien den vollständigen Bericht zusammenstellen und Titelbereinigung durchführen
        """
        folder = cls._get_report_folder(report_id)

        # Berichtskopf aufbauen
        md_content = f"# {outline.title}\n\n"
        md_content += f"> {outline.summary}\n\n"
        md_content += f"---\n\n"

        # Alle Abschnittsdateien sequentiell lesen
        sections = cls.get_generated_sections(report_id)
        for section_info in sections:
            md_content += section_info["content"]

        # Nachbearbeitung: Titelprobleme im gesamten Bericht bereinigen
        md_content = cls._post_process_report(md_content, outline)

        # Vollständigen Bericht speichern
        full_path = cls._get_report_markdown_path(report_id)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        logger.info(f"Vollständiger Bericht zusammengestellt: {report_id}")
        return md_content

    @classmethod
    def _post_process_report(cls, content: str, outline: ReportOutline) -> str:
        """
        Berichtsinhalt nachbearbeiten

        1. Doppelte Titel entfernen
        2. Berichtshaupttitel (#) und Abschnittstitel (##) beibehalten, andere Titellevels (###, #### usw.) entfernen
        3. Redundante Leerzeilen und Trennlinien bereinigen

        Args:
            content: Originaler Berichtsinhalt
            outline: Berichtsgliederung

        Returns:
            Nachbearbeiteter Inhalt
        """
        import re

        lines = content.split('\n')
        processed_lines = []
        prev_was_heading = False

        # Alle Abschnittstitel aus der Gliederung sammeln
        section_titles = set()
        for section in outline.sections:
            section_titles.add(section.title)

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Prüfen, ob es eine Titelzeile ist
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)

            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()

                # Prüfen, ob es ein doppelter Titel ist (derselbe Inhaltstitel innerhalb der letzten 5 Zeilen)
                is_duplicate = False
                for j in range(max(0, len(processed_lines) - 5), len(processed_lines)):
                    prev_line = processed_lines[j].strip()
                    prev_match = re.match(r'^(#{1,6})\s+(.+)$', prev_line)
                    if prev_match:
                        prev_title = prev_match.group(2).strip()
                        if prev_title == title:
                            is_duplicate = True
                            break

                if is_duplicate:
                    # Doppelten Titel und nachfolgende Leerzeile überspringen
                    i += 1
                    while i < len(lines) and lines[i].strip() == '':
                        i += 1
                    continue

                # Titellevel-Behandlung:
                # - # (level=1) nur Berichtshaupttitel beibehalten
                # - ## (level=2) Abschnittstitel beibehalten
                # - ### und darunter (level>=3) in Fetttext umwandeln

                if level == 1:
                    if title == outline.title:
                        # Berichtshaupttitel beibehalten
                        processed_lines.append(line)
                        prev_was_heading = True
                    elif title in section_titles:
                        # Abschnittstitel irrtümlich mit # verwendet, zu ## korrigieren
                        processed_lines.append(f"## {title}")
                        prev_was_heading = True
                    else:
                        # Andere Erstleveltitel in Fettdruck umwandeln
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                elif level == 2:
                    if title in section_titles or title == outline.title:
                        # Abschnittstitel beibehalten
                        processed_lines.append(line)
                        prev_was_heading = True
                    else:
                        # Nicht-Abschnitts-Zweitleveltitel in Fettdruck umwandeln
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                else:
                    # ### und niedrigere Leveltitel in Fetttext umwandeln
                    processed_lines.append(f"**{title}**")
                    processed_lines.append("")
                    prev_was_heading = False

                i += 1
                continue

            elif stripped == '---' and prev_was_heading:
                # Trennlinie direkt nach Titel überspringen
                i += 1
                continue

            elif stripped == '' and prev_was_heading:
                # Nach Titel nur eine Leerzeile beibehalten
                if processed_lines and processed_lines[-1].strip() != '':
                    processed_lines.append(line)
                prev_was_heading = False

            else:
                processed_lines.append(line)
                prev_was_heading = False

            i += 1

        # Aufeinanderfolgende mehrfache Leerzeilen bereinigen (maximal 2 beibehalten)
        result_lines = []
        empty_count = 0
        for line in processed_lines:
            if line.strip() == '':
                empty_count += 1
                if empty_count <= 2:
                    result_lines.append(line)
            else:
                empty_count = 0
                result_lines.append(line)

        return '\n'.join(result_lines)

    @classmethod
    def save_report(cls, report: Report) -> None:
        """Berichts-Metainformationen und vollständigen Bericht speichern"""
        cls._ensure_report_folder(report.report_id)

        # Metainformations-JSON speichern
        with open(cls._get_report_path(report.report_id), 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

        # Gliederung speichern
        if report.outline:
            cls.save_outline(report.report_id, report.outline)

        # Vollständigen Markdown-Bericht speichern
        if report.markdown_content:
            with open(cls._get_report_markdown_path(report.report_id), 'w', encoding='utf-8') as f:
                f.write(report.markdown_content)

        logger.info(f"Bericht gespeichert: {report.report_id}")

    @classmethod
    def get_report(cls, report_id: str) -> Optional[Report]:
        """Bericht abrufen"""
        path = cls._get_report_path(report_id)

        if not os.path.exists(path):
            # Abwärtskompatibles Format: Prüfen, ob direkt im reports-Verzeichnis gespeicherte Datei existiert
            old_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
            if os.path.exists(old_path):
                path = old_path
            else:
                return None

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Report-Objekt rekonstruieren
        outline = None
        if data.get('outline'):
            outline_data = data['outline']
            sections = []
            for s in outline_data.get('sections', []):
                sections.append(ReportSection(
                    title=s['title'],
                    content=s.get('content', '')
                ))
            outline = ReportOutline(
                title=outline_data['title'],
                summary=outline_data['summary'],
                sections=sections
            )

        # Wenn markdown_content leer ist, versuchen aus full_report.md zu lesen
        markdown_content = data.get('markdown_content', '')
        if not markdown_content:
            full_report_path = cls._get_report_markdown_path(report_id)
            if os.path.exists(full_report_path):
                with open(full_report_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()

        return Report(
            report_id=data['report_id'],
            simulation_id=data['simulation_id'],
            graph_id=data['graph_id'],
            simulation_requirement=data['simulation_requirement'],
            status=ReportStatus(data['status']),
            outline=outline,
            markdown_content=markdown_content,
            created_at=data.get('created_at', ''),
            completed_at=data.get('completed_at', ''),
            error=data.get('error')
        )

    @classmethod
    def get_report_by_simulation(cls, simulation_id: str) -> Optional[Report]:
        """Bericht anhand der Simulations-ID abrufen"""
        cls._ensure_reports_dir()

        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # Neues Format: Dateiordner
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report and report.simulation_id == simulation_id:
                    return report
            # Abwärtskompatibles Format: JSON-Datei
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report and report.simulation_id == simulation_id:
                    return report

        return None

    @classmethod
    def list_reports(cls, simulation_id: Optional[str] = None, limit: int = 50) -> List[Report]:
        """Berichte auflisten"""
        cls._ensure_reports_dir()

        reports = []
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # Neues Format: Dateiordner
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
            # Abwärtskompatibles Format: JSON-Datei
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)

        # Nach Erstellungszeit absteigend sortieren
        reports.sort(key=lambda r: r.created_at, reverse=True)

        return reports[:limit]

    @classmethod
    def delete_report(cls, report_id: str) -> bool:
        """Bericht löschen (gesamter Ordner)"""
        import shutil

        folder_path = cls._get_report_folder(report_id)

        # Neues Format: Gesamten Dateiordner löschen
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            logger.info(f"Berichtsordner gelöscht: {report_id}")
            return True

        # Abwärtskompatibles Format: Einzelne Dateien löschen
        deleted = False
        old_json_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
        old_md_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.md")

        if os.path.exists(old_json_path):
            os.remove(old_json_path)
            deleted = True
        if os.path.exists(old_md_path):
            os.remove(old_md_path)
            deleted = True

        return deleted
