"""
Simulationsbezogene API-Routen
Schritt 2: Entitätslesen und -filterung, OASIS-Simulationsvorbereitung und -ausführung (vollautomatisch)
"""

import os
import traceback
from flask import request, jsonify, send_file, current_app

from . import simulation_bp
from ..config import Config
from ..services.entity_reader import EntityReader
from ..services.oasis_profile_generator import OasisProfileGenerator
from ..services.simulation_manager import SimulationManager, SimulationStatus
from ..services.simulation_runner import SimulationRunner, RunnerStatus
from ..utils.logger import get_logger
from ..models.project import ProjectManager

logger = get_logger('mirofish.api.simulation')


# Interview-Prompt-Optimierungspräfix
# Das Hinzufügen dieses Präfixes verhindert, dass Agents Tools aufrufen und stattdessen direkt mit Text antworten
INTERVIEW_PROMPT_PREFIX = "Basierend auf deiner Persona, all deinen bisherigen Erinnerungen und Handlungen, antworte mir direkt mit Text, ohne Tools aufzurufen:"


def optimize_interview_prompt(prompt: str) -> str:
    """
    Interview-Fragen optimieren, Präfix hinzufügen um Agent-Tool-Aufrufe zu vermeiden

    Args:
        prompt: Originalfrage

    Returns:
        Optimierte Frage
    """
    if not prompt:
        return prompt
    # Vermeidung doppelter Präfixe
    if prompt.startswith(INTERVIEW_PROMPT_PREFIX):
        return prompt
    return f"{INTERVIEW_PROMPT_PREFIX}{prompt}"


# ============== Entitätslese-Schnittstelle ==============

@simulation_bp.route('/entities/<graph_id>', methods=['GET'])
def get_graph_entities(graph_id: str):
    """
    Alle Entitäten aus dem Wissensgraph abrufen (gefiltert)

    Gibt nur Knoten zurück, die vordefinierten Entitätstypen entsprechen (Knoten, deren Labels nicht nur Entität sind)

    Abfrageparameter:
        entity_types: Kommagetrennte Liste von Entitätstypen (optional, zur weiteren Filterung)
        enrich: Ob zugehörige Kanteninformationen abgerufen werden sollen (Standard true)
    """
    try:
        entity_types_str = request.args.get('entity_types', '')
        entity_types = [t.strip() for t in entity_types_str.split(',') if t.strip()] if entity_types_str else None
        enrich = request.args.get('enrich', 'true').lower() == 'true'

        logger.info(f"Rufe Wissensgraph-Entitäten ab: graph_id={graph_id}, entity_types={entity_types}, enrich={enrich}")

        storage = current_app.extensions.get('neo4j_storage')
        if not storage:
            raise ValueError("GraphStorage nicht initialisiert")
        reader = EntityReader(storage)
        result = reader.filter_defined_entities(
            graph_id=graph_id,
            defined_entity_types=entity_types,
            enrich_with_edges=enrich
        )

        return jsonify({
            "success": True,
            "data": result.to_dict()
        })

    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Wissensgraph-Entitäten: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/entities/<graph_id>/<entity_uuid>', methods=['GET'])
def get_entity_detail(graph_id: str, entity_uuid: str):
    """Detailinformationen einer einzelnen Entität abrufen"""
    try:
        storage = current_app.extensions.get('neo4j_storage')
        if not storage:
            raise ValueError("GraphStorage nicht initialisiert")
        reader = EntityReader(storage)
        entity = reader.get_entity_with_context(graph_id, entity_uuid)

        if not entity:
            return jsonify({
                "success": False,
                "error": f"Entität existiert nicht: {entity_uuid}"
            }), 404

        return jsonify({
            "success": True,
            "data": entity.to_dict()
        })

    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Entitätsdetails: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/entities/<graph_id>/by-type/<entity_type>', methods=['GET'])
def get_entities_by_type(graph_id: str, entity_type: str):
    """Alle Entitäten eines bestimmten Typs abrufen"""
    try:
        enrich = request.args.get('enrich', 'true').lower() == 'true'

        storage = current_app.extensions.get('neo4j_storage')
        if not storage:
            raise ValueError("GraphStorage nicht initialisiert")
        reader = EntityReader(storage)
        entities = reader.get_entities_by_type(
            graph_id=graph_id,
            entity_type=entity_type,
            enrich_with_edges=enrich
        )

        return jsonify({
            "success": True,
            "data": {
                "entity_type": entity_type,
                "count": len(entities),
                "entities": [e.to_dict() for e in entities]
            }
        })

    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Entitäten: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Simulationsverwaltungs-Schnittstelle ==============

@simulation_bp.route('/create', methods=['POST'])
def create_simulation():
    """
    Neue Simulation erstellen

    Hinweis: Parameter wie max_rounds werden intelligent vom LLM generiert, keine manuelle Einstellung nötig

    Anfrage (JSON):
        {
            "project_id": "proj_xxxx",      // Erforderlich
            "graph_id": "mirofish_xxxx",    // Optional, wenn nicht angegeben, vom Projekt abrufen
            "enable_twitter": true,          // Optional, Standard true
            "enable_reddit": true            // Optional, Standard true
        }

    Rückgabe:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "project_id": "proj_xxxx",
                "graph_id": "mirofish_xxxx",
                "status": "created",
                "enable_twitter": true,
                "enable_reddit": true,
                "created_at": "2025-12-01T10:00:00"
            }
        }
    """
    try:
        data = request.get_json() or {}

        project_id = data.get('project_id')
        if not project_id:
            return jsonify({
                "success": False,
                "error": "Bitte geben Sie project_id an"
            }), 400

        project = ProjectManager.get_project(project_id)
        if not project:
            return jsonify({
                "success": False,
                "error": f"Projekt existiert nicht: {project_id}"
            }), 404

        graph_id = data.get('graph_id') or project.graph_id
        if not graph_id:
            return jsonify({
                "success": False,
                "error": "Projekt hat noch keinen Wissensgraph erstellt, bitte rufen Sie zuerst /api/graph/build auf"
            }), 400

        manager = SimulationManager()
        state = manager.create_simulation(
            project_id=project_id,
            graph_id=graph_id,
            enable_twitter=data.get('enable_twitter', True),
            enable_reddit=data.get('enable_reddit', True),
        )

        return jsonify({
            "success": True,
            "data": state.to_dict()
        })

    except Exception as e:
        logger.error(f"Fehler beim Erstellen der Simulation: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


def _check_simulation_prepared(simulation_id: str) -> tuple:
    """
    Prüfen ob Simulation bereit ist

    Prüfbedingungen:
    1. state.json existiert und Status ist "ready"
    2. Erforderliche Dateien existieren: reddit_profiles.json, twitter_profiles.csv, simulation_config.json

    Hinweis: Ausführungsskripte (run_*.py) verbleiben im backend/scripts/-Verzeichnis, werden nicht mehr ins Simulationsverzeichnis kopiert

    Args:
        simulation_id: Simulations-ID

    Returns:
        (is_prepared: bool, info: dict)
    """
    import os
    from ..config import Config

    simulation_dir = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)

    # Prüfen ob Verzeichnis existiert
    if not os.path.exists(simulation_dir):
        return False, {"reason": "Simulationsverzeichnis existiert nicht"}

    # Liste erforderlicher Dateien (Skripte nicht enthalten, Skripte sind in backend/scripts/)
    required_files = [
        "state.json",
        "simulation_config.json",
        "reddit_profiles.json",
        "twitter_profiles.csv"
    ]

    # Prüfen ob Dateien existieren
    existing_files = []
    missing_files = []
    for f in required_files:
        file_path = os.path.join(simulation_dir, f)
        if os.path.exists(file_path):
            existing_files.append(f)
        else:
            missing_files.append(f)

    if missing_files:
        return False, {
            "reason": "Erforderliche Dateien fehlen",
            "missing_files": missing_files,
            "existing_files": existing_files
        }

    # Status in state.json prüfen
    state_file = os.path.join(simulation_dir, "state.json")
    try:
        import json
        with open(state_file, 'r', encoding='utf-8') as f:
            state_data = json.load(f)

        status = state_data.get("status", "")
        config_generated = state_data.get("config_generated", False)

        # Detaillierte Logs
        logger.debug(f"Prüfe Simulationsvorbereitungsstatus: {simulation_id}, status={status}, config_generated={config_generated}")

        # Wenn config_generated=True und Dateien existieren, Vorbereitung als abgeschlossen betrachten
        # Folgende Status zeigen an, dass die Vorbereitung abgeschlossen ist:
        # - ready: Vorbereitung abgeschlossen, kann ausgeführt werden
        # - preparing: Wenn config_generated=True, Beschreibung zeigt abgeschlossen
        # - running: Läuft, Vorbereitung bereits abgeschlossen
        # - completed: Ausführung abgeschlossen, Vorbereitung bereits abgeschlossen
        # - stopped: Gestoppt, Vorbereitung bereits abgeschlossen
        # - failed: Ausführung fehlgeschlagen (aber Vorbereitung nicht abgeschlossen)
        prepared_statuses = ["ready", "preparing", "running", "completed", "stopped", "failed"]
        if status in prepared_statuses and config_generated:
            # Dateistatistiken abrufen
            profiles_file = os.path.join(simulation_dir, "reddit_profiles.json")
            config_file = os.path.join(simulation_dir, "simulation_config.json")

            profiles_count = 0
            if os.path.exists(profiles_file):
                with open(profiles_file, 'r', encoding='utf-8') as f:
                    profiles_data = json.load(f)
                    profiles_count = len(profiles_data) if isinstance(profiles_data, list) else 0

            # Wenn Status "preparing" aber Dateien vollständig, Status auf "ready" aktualisieren
            if status == "preparing":
                try:
                    state_data["status"] = "ready"
                    from datetime import datetime
                    state_data["updated_at"] = datetime.now().isoformat()
                    with open(state_file, 'w', encoding='utf-8') as f:
                        json.dump(state_data, f, ensure_ascii=False, indent=2)
                    logger.info(f"Simulationsstatus automatisch aktualisiert: {simulation_id} preparing -> ready")
                    status = "ready"
                except Exception as e:
                    logger.warning(f"Fehler beim automatischen Aktualisieren des Status: {e}")

            logger.info(f"Simulation {simulation_id} Prüfergebnis: Vorbereitung abgeschlossen (status={status}, config_generated={config_generated})")
            return True, {
                "status": status,
                "entities_count": state_data.get("entities_count", 0),
                "profiles_count": profiles_count,
                "entity_types": state_data.get("entity_types", []),
                "config_generated": config_generated,
                "created_at": state_data.get("created_at"),
                "updated_at": state_data.get("updated_at"),
                "existing_files": existing_files
            }
        else:
            logger.warning(f"Simulation {simulation_id} Prüfergebnis: Vorbereitung nicht abgeschlossen (status={status}, config_generated={config_generated})")
            return False, {
                "reason": f"Status nicht in vorbereiteter Liste oder config_generated ist false: status={status}, config_generated={config_generated}",
                "status": status,
                "config_generated": config_generated
            }

    except Exception as e:
        return False, {"reason": f"Fehler beim Lesen der Statusdatei: {str(e)}"}


@simulation_bp.route('/prepare', methods=['POST'])
def prepare_simulation():
    """
    Simulationsumgebung vorbereiten (asynchrone Aufgabe mit LLM-intelligenter Konfigurationsgenerierung).

    Dies ist ein zeitaufwendiger Vorgang. Die Schnittstelle gibt sofort eine task_id zurück.
    Verwenden Sie GET /api/simulation/prepare/status, um den Fortschritt abzufragen.

    Funktionen:
    - Automatische Erkennung abgeschlossener Vorbereitungen zur Vermeidung doppelter Generierung
    - Wenn bereits vorbereitet, bestehende Ergebnisse direkt zurückgeben
    - Unterstützung erzwungener Neugenerierung (force_regenerate=true)

    Schritte:
    1. Prüfen ob Vorbereitung bereits abgeschlossen
    2. Entitäten aus Wissensgraph lesen und filtern
    3. OASIS-Agent-Profil für jede Entität generieren (mit Wiederholungsmechanismus)
    4. LLM generiert intelligent Simulationskonfiguration (mit Wiederholungsmechanismus)
    5. Konfigurationsdateien und vorgegebene Skripte speichern

    Anfrage (JSON):
        {
            "simulation_id": "sim_xxxx",                   // Erforderlich, Simulations-ID
            "entity_types": ["Student", "Prominenter"],  // Optional, angegebene Entitätstypen
            "use_llm_for_profiles": true,                 // Optional, ob LLM zur Profilgenerierung verwendet werden soll
            "parallel_profile_count": 5,                  // Optional, Anzahl paralleler Profilgenerierungen, Standard 5
            "force_regenerate": false                     // Optional, Neugenerierung erzwingen, Standard false
        }

    Rückgabe:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "task_id": "task_xxxx",           // Rückgabe für neue Aufgaben
                "status": "preparing|ready",
                "message": "Vorbereitungsaufgabe gestartet|Vorbereitung bereits abgeschlossen",
                "already_prepared": true|false    // Ist Vorbereitung abgeschlossen
            }
        }
    """
    import threading
    import os
    from ..models.task import TaskManager, TaskStatus
    from ..config import Config

    try:
        data = request.get_json() or {}

        simulation_id = data.get('simulation_id')
        if not simulation_id:
            return jsonify({
                "success": False,
                "error": "Bitte geben Sie simulation_id an"
            }), 400

        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)

        if not state:
            return jsonify({
                "success": False,
                "error": f"Simulation existiert nicht: {simulation_id}"
            }), 404

        # Prüfen ob erzwungene Neugenerierung
        force_regenerate = data.get('force_regenerate', False)
        logger.info(f"Starte Verarbeitung der /prepare-Anfrage: simulation_id={simulation_id}, force_regenerate={force_regenerate}")

        # Prüfen ob bereits vorbereitet (doppelte Generierung vermeiden)
        if not force_regenerate:
            logger.debug(f"Prüfe ob Simulation {simulation_id} vorbereitet ist...")
            is_prepared, prepare_info = _check_simulation_prepared(simulation_id)
            logger.debug(f"Prüfergebnis: is_prepared={is_prepared}, prepare_info={prepare_info}")
            if is_prepared:
                logger.info(f"Simulation {simulation_id} Vorbereitung abgeschlossen, keine Neugenerierung nötig")
                return jsonify({
                    "success": True,
                    "data": {
                        "simulation_id": simulation_id,
                        "status": "ready",
                        "message": "Vorbereitung bereits abgeschlossen, keine erneute Generierung nötig",
                        "already_prepared": True,
                        "prepare_info": prepare_info
                    }
                })
            else:
                logger.info(f"Simulation {simulation_id} Vorbereitung nicht abgeschlossen, wird jetzt vorbereitet")

        # Erforderliche Informationen aus Projekt abrufen
        project = ProjectManager.get_project(state.project_id)
        if not project:
            return jsonify({
                "success": False,
                "error": f"Projekt existiert nicht: {state.project_id}"
            }), 404

        # Simulationsanforderungen abrufen
        simulation_requirement = project.simulation_requirement or ""
        if not simulation_requirement:
            return jsonify({
                "success": False,
                "error": "Projekt hat keine Simulationsanforderungsbeschreibung (simulation_requirement)"
            }), 400

        # Dokumenttext abrufen
        document_text = ProjectManager.get_extracted_text(state.project_id) or ""

        entity_types_list = data.get('entity_types')
        use_llm_for_profiles = data.get('use_llm_for_profiles', True)
        parallel_profile_count = data.get('parallel_profile_count', 5)

        # ========== GraphStorage abrufen (Referenz vor Start der Hintergrundaufgabe erfassen) ==========
        storage = current_app.extensions.get('neo4j_storage')
        if not storage:
            raise ValueError("GraphStorage nicht initialisiert — Neo4j-Verbindung prüfen")

        # ========== Synchron Entitätenanzahl abrufen (vor Start der Hintergrundaufgabe) ==========
        # Damit das Frontend beim Aufruf von prepare sofort die erwartete Gesamtzahl der Agents erhalten kann
        try:
            logger.info(f"Rufe synchron Entitätenanzahl ab: graph_id={state.graph_id}")
            reader = EntityReader(storage)
            # Schnelles Lesen der Entitäten (keine Kanteninformationen, nur Statistiken benötigt)
            filtered_preview = reader.filter_defined_entities(
                graph_id=state.graph_id,
                defined_entity_types=entity_types_list,
                enrich_with_edges=False  # Keine Kanteninformationen, beschleunigt
            )
            # Entitätenanzahl im Status speichern (damit Frontend sofort abrufen kann)
            state.entities_count = filtered_preview.filtered_count
            state.entity_types = list(filtered_preview.entity_types)
            logger.info(f"Erwartete Entitätenanzahl: {filtered_preview.filtered_count}, Typen: {filtered_preview.entity_types}")
        except Exception as e:
            logger.warning(f"Synchrones Abrufen der Entitätenanzahl fehlgeschlagen (wird in Hintergrundaufgabe wiederholt): {e}")
            # Fehler beeinflusst nachfolgenden Prozess nicht, Hintergrundaufgabe wird wiederholen

        # Asynchrone Aufgabe erstellen
        task_manager = TaskManager()
        task_id = task_manager.create_task(
            task_type="simulation_prepare",
            metadata={
                "simulation_id": simulation_id,
                "project_id": state.project_id
            }
        )

        # Simulationsstatus aktualisieren (einschließlich vorab abgerufener Entitätenanzahl)
        state.status = SimulationStatus.PREPARING
        manager._save_simulation_state(state)

        # Hintergrundaufgabe definieren
        def run_prepare():
            try:
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.PROCESSING,
                    progress=0,
                    message="Starte Vorbereitung der Simulationsumgebung..."
                )

                # Simulation vorbereiten (mit Fortschrittsrückruf)
                # Phasenfortschrittsdetails speichern
                stage_details = {}

                def progress_callback(stage, progress, message, **kwargs):
                    # Gesamtfortschritt berechnen
                    stage_weights = {
                        "reading": (0, 20),           # 0-20%
                        "generating_profiles": (20, 70),  # 20-70%
                        "generating_config": (70, 90),    # 70-90%
                        "copying_scripts": (90, 100)       # 90-100%
                    }

                    start, end = stage_weights.get(stage, (0, 100))
                    current_progress = int(start + (end - start) * progress / 100)

                    # Detaillierte Fortschrittsinformationen erstellen
                    stage_names = {
                        "reading": "Wissensgraph-Entitäten lesen",
                        "generating_profiles": "Agent-Profile generieren",
                        "generating_config": "Simulationskonfiguration generieren",
                        "copying_scripts": "Simulationsskripte vorbereiten"
                    }

                    stage_index = list(stage_weights.keys()).index(stage) + 1 if stage in stage_weights else 1
                    total_stages = len(stage_weights)

                    # Phasendetails aktualisieren
                    stage_details[stage] = {
                        "stage_name": stage_names.get(stage, stage),
                        "stage_progress": progress,
                        "current": kwargs.get("current", 0),
                        "total": kwargs.get("total", 0),
                        "item_name": kwargs.get("item_name", "")
                    }

                    # Detaillierte Fortschrittsinformationen erstellen
                    detail = stage_details[stage]
                    progress_detail_data = {
                        "current_stage": stage,
                        "current_stage_name": stage_names.get(stage, stage),
                        "stage_index": stage_index,
                        "total_stages": total_stages,
                        "stage_progress": progress,
                        "current_item": detail["current"],
                        "total_items": detail["total"],
                        "item_description": message
                    }

                    # Kompakte Nachricht erstellen
                    if detail["total"] > 0:
                        detailed_message = (
                            f"[{stage_index}/{total_stages}] {stage_names.get(stage, stage)}: "
                            f"{detail['current']}/{detail['total']} - {message}"
                        )
                    else:
                        detailed_message = f"[{stage_index}/{total_stages}] {stage_names.get(stage, stage)}: {message}"

                    task_manager.update_task(
                        task_id,
                        progress=current_progress,
                        message=detailed_message,
                        progress_detail=progress_detail_data
                    )

                result_state = manager.prepare_simulation(
                    simulation_id=simulation_id,
                    simulation_requirement=simulation_requirement,
                    document_text=document_text,
                    defined_entity_types=entity_types_list,
                    use_llm_for_profiles=use_llm_for_profiles,
                    progress_callback=progress_callback,
                    parallel_profile_count=parallel_profile_count,
                    storage=storage,
                )

                # Aufgabe abgeschlossen
                task_manager.complete_task(
                    task_id,
                    result=result_state.to_simple_dict()
                )

            except Exception as e:
                logger.error(f"Fehler bei der Simulationsvorbereitung: {str(e)}")
                task_manager.fail_task(task_id, str(e))

                # Simulationsstatus auf fehlgeschlagen aktualisieren
                state = manager.get_simulation(simulation_id)
                if state:
                    state.status = SimulationStatus.FAILED
                    state.error = str(e)
                    manager._save_simulation_state(state)

        # Hintergrund-Thread starten (kein Daemon, damit er nicht bei Server-Neustart gekillt wird)
        thread = threading.Thread(target=run_prepare, daemon=False, name=f"prepare-{simulation_id}")
        thread.start()

        return jsonify({
            "success": True,
            "data": {
                "simulation_id": simulation_id,
                "task_id": task_id,
                "status": "preparing",
                "message": "Vorbereitungsaufgabe gestartet, bitte Fortschritt abfragen über /api/simulation/prepare/status",
                "already_prepared": False,
                "expected_entities_count": state.entities_count,  # Erwartete Anzahl zu verarbeitender Entitäten
                "entity_types": state.entity_types  # Entitätstypliste
            }
        })

    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 404

    except Exception as e:
        logger.error(f"Fehler beim Starten der Vorbereitungsaufgabe: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/prepare/status', methods=['POST'])
def get_prepare_status():
    """
    Vorbereitungsaufgaben-Fortschritt abfragen

    Unterstützt zwei Abfragemethoden:
    1. Über task_id den laufenden Aufgabenfortschritt abfragen
    2. Über simulation_id prüfen, ob die Vorbereitung bereits abgeschlossen ist

    Anfrage (JSON):
        {
            "task_id": "task_xxxx",          // Optional, vom vorherigen /prepare-Aufruf
            "simulation_id": "sim_xxxx"      // Optional, Simulations-ID (zur Prüfung abgeschlossener Vorbereitung)
        }

    Rückgabe:
        {
            "success": true,
            "data": {
                "task_id": "task_xxxx",
                "status": "processing|completed|ready",
                "progress": 45,
                "message": "...",
                "already_prepared": true|false,  // Gibt es abgeschlossene Vorbereitung
                "prepare_info": {...}            // Detailinfos bei abgeschlossener Vorbereitung
            }
        }
    """
    from ..models.task import TaskManager

    try:
        data = request.get_json() or {}

        task_id = data.get('task_id')
        simulation_id = data.get('simulation_id')

        # Wenn simulation_id angegeben, prüfen ob Vorbereitung abgeschlossen
        if simulation_id:
            is_prepared, prepare_info = _check_simulation_prepared(simulation_id)
            if is_prepared:
                return jsonify({
                    "success": True,
                    "data": {
                        "simulation_id": simulation_id,
                        "status": "ready",
                        "progress": 100,
                        "message": "Vorbereitung bereits abgeschlossen",
                        "already_prepared": True,
                        "prepare_info": prepare_info
                    }
                })

        # Wenn keine task_id, Fehler zurückgeben
        if not task_id:
            if simulation_id:
                # simulation_id vorhanden aber Vorbereitung nicht abgeschlossen
                return jsonify({
                    "success": True,
                    "data": {
                        "simulation_id": simulation_id,
                        "status": "not_started",
                        "progress": 0,
                        "message": "Vorbereitung noch nicht gestartet, bitte /api/simulation/prepare aufrufen",
                        "already_prepared": False
                    }
                })
            return jsonify({
                "success": False,
                "error": "Bitte geben Sie task_id oder simulation_id an"
            }), 400

        task_manager = TaskManager()
        task = task_manager.get_task(task_id)

        if not task:
            # Aufgabe existiert nicht, aber wenn simulation_id angegeben, prüfen ob Vorbereitung abgeschlossen
            if simulation_id:
                is_prepared, prepare_info = _check_simulation_prepared(simulation_id)
                if is_prepared:
                    return jsonify({
                        "success": True,
                        "data": {
                            "simulation_id": simulation_id,
                            "task_id": task_id,
                            "status": "ready",
                            "progress": 100,
                            "message": "Aufgabe abgeschlossen (Vorbereitungsarbeit existiert bereits)",
                            "already_prepared": True,
                            "prepare_info": prepare_info
                        }
                    })

            return jsonify({
                "success": False,
                "error": f"Aufgabe existiert nicht: {task_id}"
            }), 404

        task_dict = task.to_dict()
        task_dict["already_prepared"] = False

        return jsonify({
            "success": True,
            "data": task_dict
        })

    except Exception as e:
        logger.error(f"Fehler beim Abfragen des Aufgabenstatus: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@simulation_bp.route('/<simulation_id>', methods=['GET'])
def get_simulation(simulation_id: str):
    """Simulationsstatus abrufen"""
    try:
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)

        if not state:
            return jsonify({
                "success": False,
                "error": f"Simulation existiert nicht: {simulation_id}"
            }), 404

        result = state.to_dict()

        # Wenn Simulation bereit ist, zusätzliche Ausführungsanweisungen
        if state.status == SimulationStatus.READY:
            result["run_instructions"] = manager.get_run_instructions(simulation_id)

        return jsonify({
            "success": True,
            "data": result
        })

    except Exception as e:
        logger.error(f"Fehler beim Abrufen des Simulationsstatus: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/list', methods=['GET'])
def list_simulations():
    """
    Alle Simulationen auflisten

    Abfrageparameter:
        project_id: Nach Projekt-ID filtern (optional)
    """
    try:
        project_id = request.args.get('project_id')

        manager = SimulationManager()
        simulations = manager.list_simulations(project_id=project_id)

        return jsonify({
            "success": True,
            "data": [s.to_dict() for s in simulations],
            "count": len(simulations)
        })

    except Exception as e:
        logger.error(f"Fehler beim Auflisten der Simulationen: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


def _get_report_id_for_simulation(simulation_id: str) -> str:
    """
    Neueste report_id für die Simulation abrufen

    Durchsucht das Berichtsverzeichnis und findet den Bericht, der der simulation_id entspricht.
    Wenn mehrere existieren, wird der neueste zurückgegeben (nach created_at Zeitstempel).

    Args:
        simulation_id: Simulations-ID

    Returns:
        report_id oder None
    """
    import json
    from datetime import datetime

    # Berichtsverzeichnispfad: backend/uploads/reports
    reports_dir = os.path.join(os.path.dirname(__file__), '../../uploads/reports')
    if not os.path.exists(reports_dir):
        return None

    matching_reports = []

    try:
        for report_folder in os.listdir(reports_dir):
            report_path = os.path.join(reports_dir, report_folder)
            if not os.path.isdir(report_path):
                continue

            meta_file = os.path.join(report_path, "meta.json")
            if not os.path.exists(meta_file):
                continue

            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)

                if meta.get("simulation_id") == simulation_id:
                    matching_reports.append({
                        "report_id": meta.get("report_id"),
                        "created_at": meta.get("created_at", ""),
                        "status": meta.get("status", "")
                    })
            except Exception:
                continue

        if not matching_reports:
            return None

        # Nach Erstellungszeit absteigend sortieren, neuesten zurückgeben
        matching_reports.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return matching_reports[0].get("report_id")

    except Exception as e:
        logger.warning(f"Fehler beim Suchen des Berichts für Simulation {simulation_id}: {e}")
        return None


@simulation_bp.route('/history', methods=['GET'])
def get_simulation_history():
    """
    Historische Simulationsliste abrufen (mit Projektdetails)

    Für Anzeige historischer Projekte auf der Startseite. Gibt Projektname und weitere Informationen zur Simulation zurück.

    Abfrageparameter:
        limit: Rückgabemengenlimit (Standard 20)

    Rückgabe:
        {
            "success": true,
            "data": [
                {
                    "simulation_id": "sim_xxxx",
                    "project_id": "proj_xxxx",
                    "project_name": "WDU Meinungsanalyse",
                    "simulation_requirement": "Wenn die Universität Wuhan veröffentlicht...",
                    "status": "completed",
                    "entities_count": 68,
                    "profiles_count": 68,
                    "entity_types": ["Student", "Professor", ...],  // Deutsch
                    "created_at": "2024-12-10",
                    "updated_at": "2024-12-10",
                    "total_rounds": 120,
                    "current_round": 120,
                    "report_id": "report_xxxx",
                    "version": "v1.0.2"
                },
                ...
            ],
            "count": 7
        }
    """
    try:
        limit = request.args.get('limit', 20, type=int)

        manager = SimulationManager()
        simulations = manager.list_simulations()[:limit]

        # Simulationsdaten erweitern, nur aus Simulationsdateien lesen
        enriched_simulations = []
        for sim in simulations:
            sim_dict = sim.to_dict()

            # Simulationskonfigurationsinformationen abrufen (simulation_requirement aus simulation_config.json lesen)
            config = manager.get_simulation_config(sim.simulation_id)
            if config:
                sim_dict["simulation_requirement"] = config.get("simulation_requirement", "")
                time_config = config.get("time_config", {})
                sim_dict["total_simulation_hours"] = time_config.get("total_simulation_hours", 0)
                # Empfohlene Runden (Fallback-Wert)
                recommended_rounds = int(
                    time_config.get("total_simulation_hours", 0) * 60 /
                    max(time_config.get("minutes_per_round", 60), 1)
                )
            else:
                sim_dict["simulation_requirement"] = ""
                sim_dict["total_simulation_hours"] = 0
                recommended_rounds = 0

            # Laufstatus abrufen (aus run_state.json)
            run_state = SimulationRunner.get_run_state(sim.simulation_id)
            if run_state:
                sim_dict["current_round"] = run_state.current_round
                sim_dict["runner_status"] = run_state.runner_status.value
                # Vom Benutzer festgelegte total_rounds verwenden, sonst empfohlene Runden
                sim_dict["total_rounds"] = run_state.total_rounds if run_state.total_rounds > 0 else recommended_rounds
            else:
                sim_dict["current_round"] = 0
                sim_dict["runner_status"] = "idle"
                sim_dict["total_rounds"] = recommended_rounds

            # Zugehörige Projektdateiliste abrufen (maximal 3 Einträge)
            project = ProjectManager.get_project(sim.project_id)
            if project and hasattr(project, 'files') and project.files:
                sim_dict["files"] = [
                    {"filename": f.get("filename", "Unbekannte Datei")}
                    for f in project.files[:3]
                ]
            else:
                sim_dict["files"] = []

            # Zugehörige report_id abrufen (neuesten Bericht für diese Simulation finden)
            sim_dict["report_id"] = _get_report_id_for_simulation(sim.simulation_id)

            # Versionsnummer hinzufügen
            sim_dict["version"] = "v1.0.2"

            # Datum formatieren
            try:
                created_date = sim_dict.get("created_at", "")[:10]
                sim_dict["created_date"] = created_date
            except:
                sim_dict["created_date"] = ""

            enriched_simulations.append(sim_dict)

        return jsonify({
            "success": True,
            "data": enriched_simulations,
            "count": len(enriched_simulations)
        })

    except Exception as e:
        logger.error(f"Fehler beim Abrufen der historischen Simulationen: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/profiles', methods=['GET'])
def get_simulation_profiles(simulation_id: str):
    """
    Agent-Profile der Simulation abrufen

    Abfrageparameter:
        platform: Plattformtyp (reddit/twitter, Standard reddit)
    """
    try:
        platform = request.args.get('platform', 'reddit')

        manager = SimulationManager()
        profiles = manager.get_profiles(simulation_id, platform=platform)

        return jsonify({
            "success": True,
            "data": {
                "platform": platform,
                "count": len(profiles),
                "profiles": profiles
            }
        })

    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 404

    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Profile: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/profiles/realtime', methods=['GET'])
def get_simulation_profiles_realtime(simulation_id: str):
    """
    Agent-Profile der Simulation in Echtzeit abrufen (zur Ansicht während der Generierung).

    Unterschied zum /profiles-Endpunkt:
    - Liest Datei direkt, umgeht SimulationManager
    - Für Echtzeitansicht während der Generierung
    - Gibt zusätzliche Metadaten zurück (wie Dateiänderungszeit, ob Generierung läuft usw.)

    Abfrageparameter:
        platform: Plattformtyp (reddit/twitter, Standard reddit)

    Rückgabe:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "platform": "reddit",
                "count": 15,
                "total_expected": 93,  // Erwartete Gesamtzahl (falls vorhanden)
                "is_generating": true,  // Wird gerade generiert
                "file_exists": true,
                "file_modified_at": "2025-12-04T18:20:00",
                "profiles": [...]
            }
        }
    """
    import json
    import csv
    from datetime import datetime

    try:
        platform = request.args.get('platform', 'reddit')

        # Simulationsverzeichnis abrufen
        sim_dir = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)

        if not os.path.exists(sim_dir):
            return jsonify({
                "success": False,
                "error": f"Simulation existiert nicht: {simulation_id}"
            }), 404

        # Dateipfad bestimmen
        if platform == "reddit":
            profiles_file = os.path.join(sim_dir, "reddit_profiles.json")
        else:
            profiles_file = os.path.join(sim_dir, "twitter_profiles.csv")

        # Prüfen ob Dateien existieren
        file_exists = os.path.exists(profiles_file)
        profiles = []
        file_modified_at = None

        if file_exists:
            # Dateiänderungszeit abrufen
            file_stat = os.stat(profiles_file)
            file_modified_at = datetime.fromtimestamp(file_stat.st_mtime).isoformat()

            try:
                if platform == "reddit":
                    with open(profiles_file, 'r', encoding='utf-8') as f:
                        profiles = json.load(f)
                else:
                    with open(profiles_file, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        profiles = list(reader)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Fehler beim Lesen der Profildatei: {e}")
                profiles = []

        # Prüfen ob Generierung läuft (über status-Feld in state.json)
        is_generating = False
        total_expected = None

        state_file = os.path.join(sim_dir, "state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
                    status = state_data.get("status", "")
                    is_generating = status == "preparing"
                    total_expected = state_data.get("entities_count")
            except Exception:
                pass

        return jsonify({
            "success": True,
            "data": {
                "simulation_id": simulation_id,
                "platform": platform,
                "count": len(profiles),
                "total_expected": total_expected,
                "is_generating": is_generating,
                "file_exists": file_exists,
                "file_modified_at": file_modified_at,
                "profiles": profiles
            }
        })

    except Exception as e:
        logger.error(f"Fehler beim Echtzeit-Abrufen der Profile: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/config/realtime', methods=['GET'])
def get_simulation_config_realtime(simulation_id: str):
    """
    Simulationskonfiguration in Echtzeit abrufen (zur Ansicht während der Generierung).

    Unterschied zum /config-Endpunkt:
    - Liest Datei direkt, umgeht SimulationManager
    - Für Echtzeitansicht während der Generierung
    - Gibt zusätzliche Metadaten zurück (wie Dateiänderungszeit, ob Generierung läuft usw.)
    - Gibt Teilinformationen zurück auch wenn Konfiguration nicht vollständig generiert

    Rückgabe:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "file_exists": true,
                "file_modified_at": "2025-12-04T18:20:00",
                "is_generating": true,  // Wird gerade generiert
                "generation_stage": "generating_config",  // Aktuelle Generierungsphase
                "config": {...}  // Konfigurationsinhalt (falls vorhanden)
            }
        }
    """
    import json
    from datetime import datetime

    try:
        # Simulationsverzeichnis abrufen
        sim_dir = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)

        if not os.path.exists(sim_dir):
            return jsonify({
                "success": False,
                "error": f"Simulation existiert nicht: {simulation_id}"
            }), 404

        # Konfigurationsdateipfad
        config_file = os.path.join(sim_dir, "simulation_config.json")

        # Prüfen ob Dateien existieren
        file_exists = os.path.exists(config_file)
        config = None
        file_modified_at = None

        if file_exists:
            # Dateiänderungszeit abrufen
            file_stat = os.stat(config_file)
            file_modified_at = datetime.fromtimestamp(file_stat.st_mtime).isoformat()

            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Fehler beim Lesen der Konfigurationsdatei: {e}")
                config = None

        # Prüfen ob Generierung läuft (über status-Feld in state.json)
        is_generating = False
        generation_stage = None
        config_generated = False

        state_file = os.path.join(sim_dir, "state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
                    status = state_data.get("status", "")
                    is_generating = status == "preparing"
                    config_generated = state_data.get("config_generated", False)

                    # Aktuelle Phase beurteilen
                    if is_generating:
                        if state_data.get("profiles_generated", False):
                            generation_stage = "generating_config"
                        else:
                            generation_stage = "generating_profiles"
                    elif status == "ready":
                        generation_stage = "completed"
            except Exception:
                pass

        # Rückgabedaten erstellen
        response_data = {
            "simulation_id": simulation_id,
            "file_exists": file_exists,
            "file_modified_at": file_modified_at,
            "is_generating": is_generating,
            "generation_stage": generation_stage,
            "config_generated": config_generated,
            "config": config
        }

        # Wenn Konfiguration existiert, Schlüsselstatistiken extrahieren
        if config:
            response_data["summary"] = {
                "total_agents": len(config.get("agent_configs", [])),
                "simulation_hours": config.get("time_config", {}).get("total_simulation_hours"),
                "initial_posts_count": len(config.get("event_config", {}).get("initial_posts", [])),
                "hot_topics_count": len(config.get("event_config", {}).get("hot_topics", [])),
                "has_twitter_config": "twitter_config" in config,
                "has_reddit_config": "reddit_config" in config,
                "generated_at": config.get("generated_at"),
                "llm_model": config.get("llm_model")
            }

        return jsonify({
            "success": True,
            "data": response_data
        })

    except Exception as e:
        logger.error(f"Fehler beim Echtzeit-Abrufen der Konfiguration: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/config', methods=['GET'])
def get_simulation_config(simulation_id: str):
    """
    Simulationskonfiguration abrufen (mit LLM-Intelligenz generiert).

    Rückgabe:
        - time_config: Zeitkonfiguration (Simulationsdauer, Startzeit, Endzeit usw.)
        - agent_configs: Aktivitätskonfiguration für jeden Agent (Verhaltensmuster, Interaktionsstile usw.)
        - event_config: Ereigniskonfiguration (Initialbeiträge, Ereignissequenzen usw.)
        - platform_configs: Plattformkonfiguration
        - generation_reasoning: LLM-Konfigurationsschluss-Erklärung
    """
    try:
        manager = SimulationManager()
        config = manager.get_simulation_config(simulation_id)

        if not config:
            return jsonify({
                "success": False,
                "error": f"Simulationskonfiguration existiert nicht. Bitte rufen Sie zuerst /prepare auf"
            }), 404

        return jsonify({
            "success": True,
            "data": config
        })

    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Konfiguration: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/config/download', methods=['GET'])
def download_simulation_config(simulation_id: str):
    """Simulationskonfigurationsdatei herunterladen"""
    try:
        manager = SimulationManager()
        sim_dir = manager._get_simulation_dir(simulation_id)
        config_path = os.path.join(sim_dir, "simulation_config.json")

        if not os.path.exists(config_path):
            return jsonify({
                "success": False,
                "error": "Konfigurationsdatei existiert nicht. Bitte rufen Sie zuerst /prepare auf"
            }), 404

        return send_file(
            config_path,
            as_attachment=True,
            download_name="simulation_config.json"
        )

    except Exception as e:
        logger.error(f"Fehler beim Herunterladen der Konfiguration: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/script/<script_name>/download', methods=['GET'])
def download_simulation_script(script_name: str):
    """
    Simulationsausführungsskript herunterladen (allgemeine Skripte aus backend/scripts/)

    script_name mögliche Werte:
        - run_twitter_simulation.py
        - run_reddit_simulation.py
        - run_parallel_simulation.py
        - action_logger.py
    """
    try:
        # Skripte befinden sich im backend/scripts/-Verzeichnis
        scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts'))

        # Skriptnamen verifizieren
        allowed_scripts = [
            "run_twitter_simulation.py",
            "run_reddit_simulation.py",
            "run_parallel_simulation.py",
            "action_logger.py"
        ]

        if script_name not in allowed_scripts:
            return jsonify({
                "success": False,
                "error": f"Unbekanntes Skript: {script_name}, Optionen: {allowed_scripts}"
            }), 400

        script_path = os.path.join(scripts_dir, script_name)

        if not os.path.exists(script_path):
            return jsonify({
                "success": False,
                "error": f"Skriptdatei existiert nicht: {script_name}"
            }), 404

        return send_file(
            script_path,
            as_attachment=True,
            download_name=script_name
        )

    except Exception as e:
        logger.error(f"Fehler beim Herunterladen des Skripts: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Profilgenerierungs-Schnittstelle (eigenständige Verwendung) ==============

@simulation_bp.route('/generate-profiles', methods=['POST'])
def generate_profiles():
    """
    OASIS-Agent-Profile direkt aus dem Wissensgraph generieren (ohne Simulation zu erstellen)

    Anfrage (JSON):
        {
            "graph_id": "mirofish_xxxx",     // Erforderlich
            "entity_types": ["Student"],      // Optional, deutsche Typnamen
            "use_llm": true,                  // Optional
            "platform": "reddit"              // Optional
        }
    """
    try:
        data = request.get_json() or {}

        graph_id = data.get('graph_id')
        if not graph_id:
            return jsonify({
                "success": False,
                "error": "Bitte geben Sie graph_id an"
            }), 400

        entity_types = data.get('entity_types')
        use_llm = data.get('use_llm', True)
        platform = data.get('platform', 'reddit')

        storage = current_app.extensions.get('neo4j_storage')
        if not storage:
            raise ValueError("GraphStorage nicht initialisiert")
        reader = EntityReader(storage)
        filtered = reader.filter_defined_entities(
            graph_id=graph_id,
            defined_entity_types=entity_types,
            enrich_with_edges=True
        )

        if filtered.filtered_count == 0:
            return jsonify({
                "success": False,
                "error": "Keine übereinstimmenden Entitäten gefunden"
            }), 400

        generator = OasisProfileGenerator()
        profiles = generator.generate_profiles_from_entities(
            entities=filtered.entities,
            use_llm=use_llm
        )

        if platform == "reddit":
            profiles_data = [p.to_reddit_format() for p in profiles]
        elif platform == "twitter":
            profiles_data = [p.to_twitter_format() for p in profiles]
        else:
            profiles_data = [p.to_dict() for p in profiles]

        return jsonify({
            "success": True,
            "data": {
                "platform": platform,
                "entity_types": list(filtered.entity_types),
                "count": len(profiles_data),
                "profiles": profiles_data
            }
        })

    except Exception as e:
        logger.error(f"Fehler beim Generieren der Profile: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Simulationsausführungssteuerungs-Schnittstelle ==============

@simulation_bp.route('/start', methods=['POST'])
def start_simulation():
    """
    Simulation starten

    Anfrage (JSON):
        {
            "simulation_id": "sim_xxxx",          // Erforderlich, Simulations-ID
            "platform": "parallel",                // Optional: twitter / reddit / parallel (Standard)
            "max_rounds": 100,                     // Optional: Maximale Simulationsrunden, Standard unbegrenzt
            "enable_graph_memory_update": false,   // Optional: Ob Wissensgraph-Speicheraktualisierungen für Agents aktiviert werden sollen
            "force": false                         // Optional: Erzwungener Neustart (laufende Simulation stoppen und Laufzeitdateien bereinigen)
        }

    Über den force-Parameter:
        - Nach Aktivierung: wenn Simulation läuft oder abgeschlossen ist, Laufzeitlogs bereinigen
        - Bereinigung umfasst: run_state.json, actions.jsonl, simulation.log usw.
        - Konfigurationsdateien (simulation_config.json) und Profildateien werden nicht bereinigt
        - Für Szenarien die Simulation erneut ausführen müssen

    Über enable_graph_memory_update:
        - Nach Aktivierung aktualisieren alle Agents in der Simulation den Wissensgraph mit ihren Aktionen (Posts, Kommentare, Follows usw.)
        - Dies ermöglicht dem Wissensgraph, die Simulation zu "erinnern", verbessert Kontextverständnis und KI-Entscheidungsfindung
        - Erfordert dass das zugehörige Projekt eine gültige graph_id hat
        - Verwendet Batch-Aktualisierungsmechanismus zur Reduzierung des API-Overheads

    Rückgabe:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "runner_status": "running",
                "process_pid": 12345,
                "twitter_running": true,
                "reddit_running": true,
                "started_at": "2025-12-01T10:00:00",
                "graph_memory_update_enabled": true,  // Ob Wissensgraph-Speicheraktualisierung aktiviert
                "force_restarted": true               // Ob erzwungener Neustart
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get('simulation_id')
        if not simulation_id:
            return jsonify({
                "success": False,
                "error": "Bitte geben Sie simulation_id an"
            }), 400

        platform = data.get('platform', 'parallel')
        max_rounds = data.get('max_rounds')  # Optional: Maximale Simulationsrunden
        enable_graph_memory_update = data.get('enable_graph_memory_update', False)  # Optional: Ob Wissensgraph-Speicheraktualisierung aktiviert werden soll
        force = data.get('force', False)  # Optional: Erzwungener Neustart

        # max_rounds-Parameter verifizieren
        if max_rounds is not None:
            try:
                max_rounds = int(max_rounds)
                if max_rounds <= 0:
                    return jsonify({
                        "success": False,
                        "error": "max_rounds muss eine positive Ganzzahl sein"
                    }), 400
            except (ValueError, TypeError):
                return jsonify({
                    "success": False,
                    "error": "max_rounds muss eine gültige Ganzzahl sein"
                }), 400

        if platform not in ['twitter', 'reddit', 'parallel']:
            return jsonify({
                "success": False,
                "error": f"Ungültiger Plattformtyp: {platform}, Optionen: twitter/reddit/parallel"
            }), 400

        # Prüfen ob Simulation bereit ist
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)

        if not state:
            return jsonify({
                "success": False,
                "error": f"Simulation existiert nicht: {simulation_id}"
            }), 404

        force_restarted = False

        # Status intelligent behandeln: wenn Vorbereitungsarbeit abgeschlossen, Status auf ready zurücksetzen
        if state.status != SimulationStatus.READY:
            # Prüfen ob Vorbereitung abgeschlossen
            is_prepared, prepare_info = _check_simulation_prepared(simulation_id)

            if is_prepared:
                # Vorbereitungsarbeit abgeschlossen, prüfen ob Simulation nicht bereits läuft
                if state.status == SimulationStatus.RUNNING:
                    # Prüfen ob Simulationsprozess wirklich läuft
                    run_state = SimulationRunner.get_run_state(simulation_id)
                    if run_state and run_state.runner_status.value == "running":
                        # Prozess läuft tatsächlich
                        if force:
                            # Erzwungener Modus: Laufende Simulation stoppen
                            logger.info(f"Erzwungener Modus: Stoppe laufende Simulation {simulation_id}")
                            try:
                                SimulationRunner.stop_simulation(simulation_id)
                            except Exception as e:
                                logger.warning(f"Warnung beim Stoppen der Simulation: {str(e)}")
                        else:
                            return jsonify({
                                "success": False,
                                "error": f"Simulation läuft. Bitte rufen Sie zuerst /stop auf oder verwenden Sie force=true für erzwungenen Neustart."
                            }), 400

                # Wenn erzwungener Modus, Laufzeitlogs bereinigen
                if force:
                    logger.info(f"Erzwungener Modus: Bereinige Simulations-Laufzeitdateien für {simulation_id}")
                    cleanup_result = SimulationRunner.cleanup_simulation_logs(simulation_id)
                    if not cleanup_result.get("success"):
                        logger.warning(f"Warnung bei Log-Bereinigung: {cleanup_result.get('errors')}")
                    force_restarted = True

                # Prozess existiert nicht oder ist beendet, Status auf ready zurücksetzen
                logger.info(f"Simulation {simulation_id} Vorbereitung abgeschlossen, setze Status auf ready zurück (vorheriger Status: {state.status.value})")
                state.status = SimulationStatus.READY
                manager._save_simulation_state(state)
            else:
                # Vorbereitung nicht abgeschlossen
                return jsonify({
                    "success": False,
                    "error": f"Simulation nicht bereit. Aktueller Status: {state.status.value}. Bitte rufen Sie zuerst /prepare auf"
                }), 400

        # Wissensgraph-ID abrufen (für Wissensgraph-Speicheraktualisierung)
        graph_id = None
        if enable_graph_memory_update:
            # Aus Simulationsstatus oder Projekt graph_id abrufen
            graph_id = state.graph_id
            if not graph_id:
                # Versuche aus Projekt abzurufen
                project = ProjectManager.get_project(state.project_id)
                if project:
                    graph_id = project.graph_id

            if not graph_id:
                return jsonify({
                    "success": False,
                    "error": "Aktivierung der Wissensgraph-Speicheraktualisierung erfordert eine gültige graph_id, bitte stellen Sie sicher, dass der Projektgraph erstellt wurde"
                }), 400

            logger.info(f"Wissensgraph-Speicheraktualisierung aktiviert: simulation_id={simulation_id}, graph_id={graph_id}")

        # Simulation starten
        run_state = SimulationRunner.start_simulation(
            simulation_id=simulation_id,
            platform=platform,
            max_rounds=max_rounds,
            enable_graph_memory_update=enable_graph_memory_update,
            graph_id=graph_id
        )

        # Simulationsstatus aktualisieren
        state.status = SimulationStatus.RUNNING
        manager._save_simulation_state(state)

        response_data = run_state.to_dict()
        if max_rounds:
            response_data['max_rounds_applied'] = max_rounds
        response_data['graph_memory_update_enabled'] = enable_graph_memory_update
        response_data['force_restarted'] = force_restarted
        if enable_graph_memory_update:
            response_data['graph_id'] = graph_id

        return jsonify({
            "success": True,
            "data": response_data
        })

    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

    except Exception as e:
        logger.error(f"Fehler beim Starten der Simulation: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/stop', methods=['POST'])
def stop_simulation():
    """
    Simulation stoppen

    Anfrage (JSON):
        {
            "simulation_id": "sim_xxxx"  // Erforderlich, Simulations-ID
        }

    Rückgabe:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "runner_status": "stopped",
                "completed_at": "2025-12-01T12:00:00"
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get('simulation_id')
        if not simulation_id:
            return jsonify({
                "success": False,
                "error": "Bitte geben Sie simulation_id an"
            }), 400

        run_state = SimulationRunner.stop_simulation(simulation_id)

        # Simulationsstatus aktualisieren
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)
        if state:
            state.status = SimulationStatus.PAUSED
            manager._save_simulation_state(state)

        return jsonify({
            "success": True,
            "data": run_state.to_dict()
        })

    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

    except Exception as e:
        logger.error(f"Fehler beim Stoppen der Simulation: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Echtzeit-Statusüberwachungs-Schnittstelle ==============

@simulation_bp.route('/<simulation_id>/run-status', methods=['GET'])
def get_run_status(simulation_id: str):
    """
    Echtzeit-Laufstatus der Simulation abrufen (für Frontend-Polling)
    """
    try:
        run_state = SimulationRunner.get_run_state(simulation_id)

        if not run_state:
            return jsonify({
                "success": True,
                "data": {
                    "simulation_id": simulation_id,
                    "runner_status": "idle",
                    "current_round": 0,
                    "total_rounds": 0,
                    "progress_percent": 0,
                    "twitter_actions_count": 0,
                    "reddit_actions_count": 0,
                    "total_actions_count": 0,
                }
            })

        return jsonify({
            "success": True,
            "data": run_state.to_dict()
        })

    except Exception as e:
        logger.error(f"Fehler beim Abrufen des Laufstatus: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/run-status/detail', methods=['GET'])
def get_run_status_detail(simulation_id: str):
    """
    Detaillierten Laufstatus der Simulation abrufen (einschließlich aller Aktionen)

    Für Frontend zur Anzeige von Echtzeitdynamiken

    Abfrageparameter:
        platform: Plattformfilter (twitter/reddit, optional)
    """
    try:
        run_state = SimulationRunner.get_run_state(simulation_id)
        platform_filter = request.args.get('platform')

        if not run_state:
            return jsonify({
                "success": True,
                "data": {
                    "simulation_id": simulation_id,
                    "runner_status": "idle",
                    "all_actions": [],
                    "twitter_actions": [],
                    "reddit_actions": []
                }
            })

        # Vollständige Aktionsliste abrufen
        all_actions = SimulationRunner.get_all_actions(
            simulation_id=simulation_id,
            platform=platform_filter
        )

        # Aktionen nach Plattform abrufen
        twitter_actions = SimulationRunner.get_all_actions(
            simulation_id=simulation_id,
            platform="twitter"
        ) if not platform_filter or platform_filter == "twitter" else []

        reddit_actions = SimulationRunner.get_all_actions(
            simulation_id=simulation_id,
            platform="reddit"
        ) if not platform_filter or platform_filter == "reddit" else []

        # Aktuelle Rundenaktionen abrufen (recent_actions zeigt nur neueste Runde)
        current_round = run_state.current_round
        recent_actions = SimulationRunner.get_all_actions(
            simulation_id=simulation_id,
            platform=platform_filter,
            round_num=current_round
        ) if current_round > 0 else []

        # Grundlegende Statusinformationen abrufen
        result = run_state.to_dict()
        result["all_actions"] = [a.to_dict() for a in all_actions]
        result["twitter_actions"] = [a.to_dict() for a in twitter_actions]
        result["reddit_actions"] = [a.to_dict() for a in reddit_actions]
        result["rounds_count"] = len(run_state.rounds)
        # recent_actions zeigt nur Inhalt der neuesten Runde beider Plattformen
        result["recent_actions"] = [a.to_dict() for a in recent_actions]

        return jsonify({
            "success": True,
            "data": result
        })

    except Exception as e:
        logger.error(f"Fehler beim Abrufen des detaillierten Status: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/actions', methods=['GET'])
def get_simulation_actions(simulation_id: str):
    """
    Agent-Aktionshistorie der Simulation abrufen

    Abfrageparameter:
        limit: Rückgabemenge (Standard 100)
        offset: Versatz (Standard 0)
        platform: Plattformfilter (twitter/reddit)
        agent_id: Agent-ID-Filter
        round_num: Rundenfilter
    """
    try:
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        platform = request.args.get('platform')
        agent_id = request.args.get('agent_id', type=int)
        round_num = request.args.get('round_num', type=int)

        actions = SimulationRunner.get_actions(
            simulation_id=simulation_id,
            limit=limit,
            offset=offset,
            platform=platform,
            agent_id=agent_id,
            round_num=round_num
        )

        return jsonify({
            "success": True,
            "data": {
                "count": len(actions),
                "actions": [a.to_dict() for a in actions]
            }
        })

    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Aktionshistorie: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/timeline', methods=['GET'])
def get_simulation_timeline(simulation_id: str):
    """
    Simulationszeitachse abrufen (nach Runden zusammengefasst)

    Für Frontend zur Anzeige von Fortschrittsbalken und Zeitachsenansicht

    Abfrageparameter:
        start_round: Startrunde (Standard 0)
        end_round: Endrunde (Standard alle)

    Gibt Zusammenfassungsinformationen pro Runde zurück
    """
    try:
        start_round = request.args.get('start_round', 0, type=int)
        end_round = request.args.get('end_round', type=int)

        timeline = SimulationRunner.get_timeline(
            simulation_id=simulation_id,
            start_round=start_round,
            end_round=end_round
        )

        return jsonify({
            "success": True,
            "data": {
                "rounds_count": len(timeline),
                "timeline": timeline
            }
        })

    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Zeitachse: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/agent-stats', methods=['GET'])
def get_agent_stats(simulation_id: str):
    """
    Statistiken für jeden Agent abrufen

    Für Frontend-Anzeige von Agent-Aktivitätsranking und Statistiken.
    """
    try:
        stats = SimulationRunner.get_agent_stats(simulation_id)

        return jsonify({
            "success": True,
            "data": {
                "agents_count": len(stats),
                "stats": stats
            }
        })

    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Agent-Statistiken: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Datenbankabfrage-Schnittstelle ==============

@simulation_bp.route('/<simulation_id>/posts', methods=['GET'])
def get_simulation_posts(simulation_id: str):
    """
    Beiträge in der Simulation abrufen

    Abfrageparameter:
        platform: Plattformtyp (twitter/reddit)
        limit: Rückgabemenge (Standard 50)
        offset: Versatz

    Gibt Beitragsliste zurück (aus SQLite-Datenbank gelesen)
    """
    try:
        platform = request.args.get('platform', 'reddit')
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)

        sim_dir = os.path.join(
            os.path.dirname(__file__),
            f'../../uploads/simulations/{simulation_id}'
        )

        db_file = f"{platform}_simulation.db"
        db_path = os.path.join(sim_dir, db_file)

        if not os.path.exists(db_path):
            return jsonify({
                "success": True,
                "data": {
                    "platform": platform,
                    "count": 0,
                    "posts": [],
                    "message": "Datenbank existiert nicht, Simulation wurde möglicherweise noch nicht ausgeführt"
                }
            })

        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT * FROM post
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))

            posts = [dict(row) for row in cursor.fetchall()]

            cursor.execute("SELECT COUNT(*) FROM post")
            total = cursor.fetchone()[0]

        except sqlite3.OperationalError:
            posts = []
            total = 0

        conn.close()

        return jsonify({
            "success": True,
            "data": {
                "platform": platform,
                "total": total,
                "count": len(posts),
                "posts": posts
            }
        })

    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Beiträge: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/comments', methods=['GET'])
def get_simulation_comments(simulation_id: str):
    """
    Kommentare in der Simulation abrufen (nur Reddit)

    Abfrageparameter:
        post_id: Beitrags-ID filtern (optional)
        limit: Rückgabemenge
        offset: Versatz
    """
    try:
        post_id = request.args.get('post_id')
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)

        sim_dir = os.path.join(
            os.path.dirname(__file__),
            f'../../uploads/simulations/{simulation_id}'
        )

        db_path = os.path.join(sim_dir, "reddit_simulation.db")

        if not os.path.exists(db_path):
            return jsonify({
                "success": True,
                "data": {
                    "count": 0,
                    "comments": []
                }
            })

        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            if post_id:
                cursor.execute("""
                    SELECT * FROM comment
                    WHERE post_id = ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """, (post_id, limit, offset))
            else:
                cursor.execute("""
                    SELECT * FROM comment
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))

            comments = [dict(row) for row in cursor.fetchall()]

        except sqlite3.OperationalError:
            comments = []

        conn.close()

        return jsonify({
            "success": True,
            "data": {
                "count": len(comments),
                "comments": comments
            }
        })

    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Kommentare: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Interview-Schnittstelle ==============

@simulation_bp.route('/interview', methods=['POST'])
def interview_agent():
    """
    Einzelnen Agent interviewen

    Hinweis: Diese Funktion erfordert, dass die Simulation in einem laufenden oder abgeschlossenen Zustand ist (Simulation starten und auf Fortschritt warten).

    Anfrage (JSON):
        {
            "simulation_id": "sim_xxxx",       // Erforderlich, Simulations-ID
            "agent_id": 0,                     // Erforderlich, Agent-ID
            "prompt": "Was denken Sie darüber?",  // Erforderlich, Interviewfrage
            "platform": "twitter",             // Optional, angegebene Plattform (twitter/reddit)
                                               // Wenn nicht angegeben: Beide Plattformen bei Dualplattform-Simulationen
            "timeout": 60                      // Optional, Timeout in Sekunden, Standard 60
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get('simulation_id')
        agent_id = data.get('agent_id')
        prompt = data.get('prompt')
        platform = data.get('platform')  # Optional: twitter/reddit/None
        timeout = data.get('timeout', 60)

        if not simulation_id:
            return jsonify({
                "success": False,
                "error": "Bitte geben Sie simulation_id an"
            }), 400

        if agent_id is None:
            return jsonify({
                "success": False,
                "error": "Bitte geben Sie agent_id an"
            }), 400

        if not prompt:
            return jsonify({
                "success": False,
                "error": "Bitte geben Sie prompt (Interviewfrage) an"
            }), 400

        # platform-Parameter verifizieren
        if platform and platform not in ("twitter", "reddit"):
            return jsonify({
                "success": False,
                "error": "platform-Parameter kann nur 'twitter' oder 'reddit' sein"
            }), 400

        # Umgebungsstatus prüfen
        if not SimulationRunner.check_env_alive(simulation_id):
            return jsonify({
                "success": False,
                "error": "Simulationsumgebung läuft nicht oder ist geschlossen. Bitte stellen Sie sicher, dass die Simulation gestartet ist und warten Sie auf Fortschritt."
            }), 400

        # Prompt optimieren, Präfix hinzufügen um Agent-Tool-Aufrufe zu vermeiden
        optimized_prompt = optimize_interview_prompt(prompt)

        result = SimulationRunner.interview_agent(
            simulation_id=simulation_id,
            agent_id=agent_id,
            prompt=optimized_prompt,
            platform=platform,
            timeout=timeout
        )

        return jsonify({
            "success": result.get("success", False),
            "data": result
        })

    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

    except TimeoutError as e:
        return jsonify({
            "success": False,
            "error": f"Timeout beim Warten auf Interview-Antwort: {str(e)}"
        }), 504

    except Exception as e:
        logger.error(f"Interview fehlgeschlagen: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/interview/batch', methods=['POST'])
def interview_agents_batch():
    """
    Mehrere Agents im Batch interviewen

    Hinweis: Diese Funktion erfordert, dass die Simulation in einem laufenden oder abgeschlossenen Zustand ist.

    Anfrage (JSON):
        {
            "simulation_id": "sim_xxxx",       // Erforderlich, Simulations-ID
            "interviews": [                    // Erforderlich, Interviewliste
                {
                    "agent_id": 0,
                    "prompt": "Was denken Sie über A?",
                    "platform": "twitter"      // Optional, diesen Agent auf angegebener Plattform interviewen
                },
                {
                    "agent_id": 1,
                    "prompt": "Was denken Sie über B?"  // Ohne Plattformangabe wird Standardwert verwendet
                }
            ],
            "platform": "reddit",              // Optional, Standardplattform (wird durch Plattform jedes Elements überschrieben)
                                               // Wenn nicht angegeben: Beide Plattformen bei Dualplattform-Simulationen
            "timeout": 120                     // Optional, Timeout in Sekunden, Standard 120
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get('simulation_id')
        interviews = data.get('interviews')
        platform = data.get('platform')  # Optional: twitter/reddit/None
        timeout = data.get('timeout', 120)

        if not simulation_id:
            return jsonify({
                "success": False,
                "error": "Bitte geben Sie simulation_id an"
            }), 400

        if not interviews or not isinstance(interviews, list):
            return jsonify({
                "success": False,
                "error": "Bitte geben Sie interviews (Interviewliste) an"
            }), 400

        # platform-Parameter verifizieren
        if platform and platform not in ("twitter", "reddit"):
            return jsonify({
                "success": False,
                "error": "platform-Parameter kann nur 'twitter' oder 'reddit' sein"
            }), 400

        # Jedes Interview-Element verifizieren
        for i, interview in enumerate(interviews):
            if 'agent_id' not in interview:
                return jsonify({
                    "success": False,
                    "error": f"Interviewliste Element {i+1} fehlt agent_id"
                }), 400
            if 'prompt' not in interview:
                return jsonify({
                    "success": False,
                    "error": f"Interviewliste Element {i+1} fehlt prompt"
                }), 400
            # platform jedes Elements verifizieren (falls vorhanden)
            item_platform = interview.get('platform')
            if item_platform and item_platform not in ("twitter", "reddit"):
                return jsonify({
                    "success": False,
                    "error": f"Interviewliste Element {i+1}: platform muss 'twitter' oder 'reddit' sein"
                }), 400

        # Umgebungsstatus prüfen
        if not SimulationRunner.check_env_alive(simulation_id):
            return jsonify({
                "success": False,
                "error": "Simulationsumgebung läuft nicht oder ist geschlossen. Bitte stellen Sie sicher, dass die Simulation gestartet ist und warten Sie auf Fortschritt."
            }), 400

        # Prompt jedes Interview-Elements optimieren, Präfix hinzufügen um Agent-Tool-Aufrufe zu vermeiden
        optimized_interviews = []
        for interview in interviews:
            optimized_interview = interview.copy()
            optimized_interview['prompt'] = optimize_interview_prompt(interview.get('prompt', ''))
            optimized_interviews.append(optimized_interview)

        result = SimulationRunner.interview_agents_batch(
            simulation_id=simulation_id,
            interviews=optimized_interviews,
            platform=platform,
            timeout=timeout
        )

        return jsonify({
            "success": result.get("success", False),
            "data": result
        })

    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

    except TimeoutError as e:
        return jsonify({
            "success": False,
            "error": f"Timeout beim Warten auf Batch-Interview-Antwort: {str(e)}"
        }), 504

    except Exception as e:
        logger.error(f"Batch-Interview fehlgeschlagen: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/interview/all', methods=['POST'])
def interview_all_agents():
    """
    Globales Interview - Alle Agents mit derselben Frage interviewen

    Hinweis: Diese Funktion erfordert, dass die Simulation in einem laufenden oder abgeschlossenen Zustand ist.

    Anfrage (JSON):
        {
            "simulation_id": "sim_xxxx",            // Erforderlich, Simulations-ID
            "prompt": "Wie ist Ihre Gesamteinschätzung dazu?",  // Erforderlich, Interviewfrage
            "platform": "reddit",                   // Optional, angegebene Plattform (twitter/reddit)
                                                    // Wenn nicht angegeben: Beide Plattformen bei Dualplattform-Simulationen
            "timeout": 180                          // Optional, Timeout in Sekunden, Standard 180
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get('simulation_id')
        prompt = data.get('prompt')
        platform = data.get('platform')  # Optional: twitter/reddit/None
        timeout = data.get('timeout', 180)

        if not simulation_id:
            return jsonify({
                "success": False,
                "error": "Bitte geben Sie simulation_id an"
            }), 400

        if not prompt:
            return jsonify({
                "success": False,
                "error": "Bitte geben Sie prompt (Interviewfrage) an"
            }), 400

        # platform-Parameter verifizieren
        if platform and platform not in ("twitter", "reddit"):
            return jsonify({
                "success": False,
                "error": "platform-Parameter kann nur 'twitter' oder 'reddit' sein"
            }), 400

        # Umgebungsstatus prüfen
        if not SimulationRunner.check_env_alive(simulation_id):
            return jsonify({
                "success": False,
                "error": "Simulationsumgebung läuft nicht oder ist geschlossen. Bitte stellen Sie sicher, dass die Simulation gestartet ist und warten Sie auf Fortschritt."
            }), 400

        # Prompt optimieren, Präfix hinzufügen um Agent-Tool-Aufrufe zu vermeiden
        optimized_prompt = optimize_interview_prompt(prompt)

        result = SimulationRunner.interview_all_agents(
            simulation_id=simulation_id,
            prompt=optimized_prompt,
            platform=platform,
            timeout=timeout
        )

        return jsonify({
            "success": result.get("success", False),
            "data": result
        })

    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

    except TimeoutError as e:
        return jsonify({
            "success": False,
            "error": f"Timeout beim Warten auf globale Interview-Antwort: {str(e)}"
        }), 504

    except Exception as e:
        logger.error(f"Globales Interview fehlgeschlagen: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/interview/history', methods=['POST'])
def get_interview_history():
    """
    Interview-Verlaufsdatensätze abrufen

    Alle Interview-Datensätze aus der Simulationsdatenbank lesen

    Anfrage (JSON):
        {
            "simulation_id": "sim_xxxx",  // Erforderlich, Simulations-ID
            "platform": "reddit",          // Optional, Plattformtyp (reddit/twitter)
                                           // Wenn nicht angegeben, gesamten Verlauf beider Plattformen zurückgeben
            "agent_id": 0,                 // Optional, nur Interview-Verlauf für diesen Agent abrufen
            "limit": 100                   // Optional, Rückgabemenge, Standard 100
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get('simulation_id')
        platform = data.get('platform')  # Wenn nicht angegeben, Verlauf beider Plattformen zurückgeben
        agent_id = data.get('agent_id')
        limit = data.get('limit', 100)

        if not simulation_id:
            return jsonify({
                "success": False,
                "error": "Bitte geben Sie simulation_id an"
            }), 400

        history = SimulationRunner.get_interview_history(
            simulation_id=simulation_id,
            platform=platform,
            agent_id=agent_id,
            limit=limit
        )

        return jsonify({
            "success": True,
            "data": {
                "count": len(history),
                "history": history
            }
        })

    except Exception as e:
        logger.error(f"Fehler beim Abrufen des Interview-Verlaufs: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/env-status', methods=['POST'])
def get_env_status():
    """
    Simulationsumgebungsstatus abrufen

    Prüfen ob Simulationsumgebung aktiv ist (Interview-Anfragen empfangen kann).

    Anfrage (JSON):
        {
            "simulation_id": "sim_xxxx"  // Erforderlich, Simulations-ID
        }

    Rückgabe:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "env_alive": true,
                "twitter_available": true,
                "reddit_available": true,
                "message": "Umgebung läuft, bereit für Interview-Anfragen"
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get('simulation_id')

        if not simulation_id:
            return jsonify({
                "success": False,
                "error": "Bitte geben Sie simulation_id an"
            }), 400

        env_alive = SimulationRunner.check_env_alive(simulation_id)

        # Detailliertere Statusinformationen abrufen
        env_status = SimulationRunner.get_env_status_detail(simulation_id)

        if env_alive:
            message = "Umgebung läuft, bereit für Interview-Anfragen"
        else:
            message = "Umgebung läuft nicht oder ist geschlossen"

        return jsonify({
            "success": True,
            "data": {
                "simulation_id": simulation_id,
                "env_alive": env_alive,
                "twitter_available": env_status.get("twitter_available", False),
                "reddit_available": env_status.get("reddit_available", False),
                "message": message
            }
        })

    except Exception as e:
        logger.error(f"Fehler beim Abrufen des Umgebungsstatus: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/close-env', methods=['POST'])
def close_simulation_env():
    """
    Simulationsumgebung schließen

    Sendet Umgebungsschließungsbefehl an die Simulation zum ordnungsgemäßen Beenden und wartet auf Abschluss.

    Hinweis: Dies unterscheidet sich von /stop. /stop beendet die Simulation abrupt.
    Diese Schnittstelle lässt die Simulation die Umgebung ordnungsgemäß schließen und beenden.

    Anfrage (JSON):
        {
            "simulation_id": "sim_xxxx",  // Erforderlich, Simulations-ID
            "timeout": 30                  // Optional, Timeout in Sekunden, Standard 30
        }

    Rückgabe:
        {
            "success": true,
            "data": {
                "message": "Umgebungsschließungsbefehl gesendet",
                "result": {...},
                "timestamp": "2025-12-08T10:00:01"
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get('simulation_id')
        timeout = data.get('timeout', 30)

        if not simulation_id:
            return jsonify({
                "success": False,
                "error": "Bitte geben Sie simulation_id an"
            }), 400

        result = SimulationRunner.close_simulation_env(
            simulation_id=simulation_id,
            timeout=timeout
        )

        # Simulationsstatus aktualisieren
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)
        if state:
            state.status = SimulationStatus.COMPLETED
            manager._save_simulation_state(state)

        return jsonify({
            "success": result.get("success", False),
            "data": result
        })

    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

    except Exception as e:
        logger.error(f"Fehler beim Schließen der Umgebung: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500
