"""
OASIS Simulations-Runner
Führt Simulationen im Hintergrund aus und zeichnet Aktionen für jeden Agent auf, unterstützt Echtzeit-Statusüberwachung
"""

import os
import sys
import json
import time
import asyncio
import threading
import subprocess
import signal
import atexit
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import Queue

from ..config import Config
from ..utils.logger import get_logger
from .graph_memory_updater import GraphMemoryManager
from .simulation_ipc import SimulationIPCClient, CommandType, IPCResponse

logger = get_logger('mirofish.simulation_runner')

# Flag, ob Bereinigungsfunktion registriert ist
_cleanup_registered = False

# Plattformerkennung
IS_WINDOWS = sys.platform == 'win32'


class RunnerStatus(str, Enum):
    """Runner-Status"""
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentAction:
    """Agent-Aktionsdatensatz"""
    round_num: int
    timestamp: str
    platform: str  # twitter / reddit
    agent_id: int
    agent_name: str
    action_type: str  # CREATE_POST, LIKE_POST, usw.
    action_args: Dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_num": self.round_num,
            "timestamp": self.timestamp,
            "platform": self.platform,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "action_type": self.action_type,
            "action_args": self.action_args,
            "result": self.result,
            "success": self.success,
        }


@dataclass
class RoundSummary:
    """Rundenzusammenfassung"""
    round_num: int
    start_time: str
    end_time: Optional[str] = None
    simulated_hour: int = 0
    twitter_actions: int = 0
    reddit_actions: int = 0
    active_agents: List[int] = field(default_factory=list)
    actions: List[AgentAction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_num": self.round_num,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "simulated_hour": self.simulated_hour,
            "twitter_actions": self.twitter_actions,
            "reddit_actions": self.reddit_actions,
            "active_agents": self.active_agents,
            "actions_count": len(self.actions),
            "actions": [a.to_dict() for a in self.actions],
        }


@dataclass
class SimulationRunState:
    """Simulations-Laufzustand (Echtzeit)"""
    simulation_id: str
    runner_status: RunnerStatus = RunnerStatus.IDLE

    # Fortschrittsinformationen
    current_round: int = 0
    total_rounds: int = 0
    simulated_hours: int = 0
    total_simulation_hours: int = 0

    # Plattformunabhängige Runden und simulierte Zeit (für Dualplattform-Parallelanzeige)
    twitter_current_round: int = 0
    reddit_current_round: int = 0
    twitter_simulated_hours: int = 0
    reddit_simulated_hours: int = 0

    # Plattformstatus
    twitter_running: bool = False
    reddit_running: bool = False
    twitter_actions_count: int = 0
    reddit_actions_count: int = 0

    # Plattform-Abschlussstatus (erkannt über simulation_end-Events in actions.jsonl)
    twitter_completed: bool = False
    reddit_completed: bool = False

    # Rundenzusammenfassung
    rounds: List[RoundSummary] = field(default_factory=list)

    # Letzte Aktionen (für Echtzeit-Anzeige im Frontend)
    recent_actions: List[AgentAction] = field(default_factory=list)
    max_recent_actions: int = 50

    # Zeitstempel
    started_at: Optional[str] = None
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    # Fehlermeldung
    error: Optional[str] = None

    # Prozess-ID (zum Stoppen)
    process_pid: Optional[int] = None

    def add_action(self, action: AgentAction):
        """Aktion zur Liste der letzten Aktionen hinzufügen"""
        self.recent_actions.insert(0, action)
        if len(self.recent_actions) > self.max_recent_actions:
            self.recent_actions = self.recent_actions[:self.max_recent_actions]

        if action.platform == "twitter":
            self.twitter_actions_count += 1
        else:
            self.reddit_actions_count += 1

        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "simulation_id": self.simulation_id,
            "runner_status": self.runner_status.value,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "simulated_hours": self.simulated_hours,
            "total_simulation_hours": self.total_simulation_hours,
            "progress_percent": round(self.current_round / max(self.total_rounds, 1) * 100, 1),
            # Plattformunabhängige Runden und Zeit
            "twitter_current_round": self.twitter_current_round,
            "reddit_current_round": self.reddit_current_round,
            "twitter_simulated_hours": self.twitter_simulated_hours,
            "reddit_simulated_hours": self.reddit_simulated_hours,
            "twitter_running": self.twitter_running,
            "reddit_running": self.reddit_running,
            "twitter_completed": self.twitter_completed,
            "reddit_completed": self.reddit_completed,
            "twitter_actions_count": self.twitter_actions_count,
            "reddit_actions_count": self.reddit_actions_count,
            "total_actions_count": self.twitter_actions_count + self.reddit_actions_count,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "process_pid": self.process_pid,
        }

    def to_detail_dict(self) -> Dict[str, Any]:
        """Details mit letzten Aktionen"""
        result = self.to_dict()
        result["recent_actions"] = [a.to_dict() for a in self.recent_actions]
        result["rounds_count"] = len(self.rounds)
        return result


class SimulationRunner:
    """
    Simulations-Runner

    Zuständig für:
    1. Ausführung von OASIS-Simulationen in Hintergrundprozessen
    2. Parsen von Ausführungslogs und Aufzeichnen von Aktionen für jeden Agent
    3. Bereitstellung von Echtzeit-Statusabfrageschnittstellen
    4. Unterstützung von Pause-/Stopp-/Fortsetzen-Operationen
    """

    # Speicherverzeichnis für Laufzustand
    RUN_STATE_DIR = os.path.join(
        os.path.dirname(__file__),
        '../../uploads/simulations'
    )

    # Skriptverzeichnis
    SCRIPTS_DIR = os.path.join(
        os.path.dirname(__file__),
        '../../scripts'
    )

    # Laufzustand im Speicher
    _run_states: Dict[str, SimulationRunState] = {}
    _processes: Dict[str, subprocess.Popen] = {}
    _action_queues: Dict[str, Queue] = {}
    _monitor_threads: Dict[str, threading.Thread] = {}
    _stdout_files: Dict[str, Any] = {}  # Stdout-Dateihandles speichern
    _stderr_files: Dict[str, Any] = {}  # Stderr-Dateihandles speichern

    # Graph-Speicher-Aktualisierungskonfiguration
    _graph_memory_enabled: Dict[str, bool] = {}  # simulation_id -> enabled

    @classmethod
    def get_run_state(cls, simulation_id: str) -> Optional[SimulationRunState]:
        """Laufzustand abrufen"""
        if simulation_id in cls._run_states:
            return cls._run_states[simulation_id]

        # Versuche aus Datei zu laden
        state = cls._load_run_state(simulation_id)
        if state:
            cls._run_states[simulation_id] = state
        return state

    @classmethod
    def _load_run_state(cls, simulation_id: str) -> Optional[SimulationRunState]:
        """Laufzustand aus Datei laden"""
        state_file = os.path.join(cls.RUN_STATE_DIR, simulation_id, "run_state.json")
        if not os.path.exists(state_file):
            return None

        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            state = SimulationRunState(
                simulation_id=simulation_id,
                runner_status=RunnerStatus(data.get("runner_status", "idle")),
                current_round=data.get("current_round", 0),
                total_rounds=data.get("total_rounds", 0),
                simulated_hours=data.get("simulated_hours", 0),
                total_simulation_hours=data.get("total_simulation_hours", 0),
                # Plattformunabhängige Runden und Zeit
                twitter_current_round=data.get("twitter_current_round", 0),
                reddit_current_round=data.get("reddit_current_round", 0),
                twitter_simulated_hours=data.get("twitter_simulated_hours", 0),
                reddit_simulated_hours=data.get("reddit_simulated_hours", 0),
                twitter_running=data.get("twitter_running", False),
                reddit_running=data.get("reddit_running", False),
                twitter_completed=data.get("twitter_completed", False),
                reddit_completed=data.get("reddit_completed", False),
                twitter_actions_count=data.get("twitter_actions_count", 0),
                reddit_actions_count=data.get("reddit_actions_count", 0),
                started_at=data.get("started_at"),
                updated_at=data.get("updated_at", datetime.now().isoformat()),
                completed_at=data.get("completed_at"),
                error=data.get("error"),
                process_pid=data.get("process_pid"),
            )

            # Letzte Aktionen laden
            actions_data = data.get("recent_actions", [])
            for a in actions_data:
                state.recent_actions.append(AgentAction(
                    round_num=a.get("round_num", 0),
                    timestamp=a.get("timestamp", ""),
                    platform=a.get("platform", ""),
                    agent_id=a.get("agent_id", 0),
                    agent_name=a.get("agent_name", ""),
                    action_type=a.get("action_type", ""),
                    action_args=a.get("action_args", {}),
                    result=a.get("result"),
                    success=a.get("success", True),
                ))

            return state
        except Exception as e:
            logger.error(f"Fehler beim Laden des Laufzustands: {str(e)}")
            return None

    @classmethod
    def _save_run_state(cls, state: SimulationRunState):
        """Laufzustand in Datei speichern"""
        sim_dir = os.path.join(cls.RUN_STATE_DIR, state.simulation_id)
        os.makedirs(sim_dir, exist_ok=True)
        state_file = os.path.join(sim_dir, "run_state.json")

        data = state.to_detail_dict()

        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        cls._run_states[state.simulation_id] = state

    @classmethod
    def start_simulation(
        cls,
        simulation_id: str,
        platform: str = "parallel",  # twitter / reddit / parallel
        max_rounds: int = None,  # Maximale Simulationsrunden (optional, zum Kürzen langer Simulationen)
        enable_graph_memory_update: bool = False,  # Ob Aktivitäten in den Graph aktualisiert werden sollen
        graph_id: str = None,  # Graph-ID (erforderlich bei aktivierten Graph-Updates)
        storage: 'GraphStorage' = None  # GraphStorage-Instanz (erforderlich bei enable_graph_memory_update)
    ) -> SimulationRunState:
        """
        Simulation starten

        Args:
            simulation_id: Simulations-ID
            platform: Auszuführende Plattform (twitter/reddit/parallel)
            max_rounds: Maximale Simulationsrunden (optional, zum Kürzen langer Simulationen)
            enable_graph_memory_update: Ob Agent-Aktivitäten dynamisch in den Graph aktualisiert werden sollen
            graph_id: Graph-ID (erforderlich bei aktivierten Graph-Updates)

        Returns:
            SimulationRunState
        """
        # Prüfen ob bereits läuft
        existing = cls.get_run_state(simulation_id)
        if existing and existing.runner_status in [RunnerStatus.RUNNING, RunnerStatus.STARTING]:
            raise ValueError(f"Simulation läuft bereits: {simulation_id}")

        # Simulationskonfiguration laden
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        config_path = os.path.join(sim_dir, "simulation_config.json")

        if not os.path.exists(config_path):
            raise ValueError(f"Simulationskonfiguration existiert nicht, bitte zuerst /prepare aufrufen")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Laufzustand initialisieren
        time_config = config.get("time_config", {})
        total_hours = time_config.get("total_simulation_hours", 72)
        minutes_per_round = time_config.get("minutes_per_round", 30)
        total_rounds = int(total_hours * 60 / minutes_per_round)

        # Wenn max_rounds angegeben, kürzen
        if max_rounds is not None and max_rounds > 0:
            original_rounds = total_rounds
            total_rounds = min(total_rounds, max_rounds)
            if total_rounds < original_rounds:
                logger.info(f"Runden gekürzt: {original_rounds} -> {total_rounds} (max_rounds={max_rounds})")

        state = SimulationRunState(
            simulation_id=simulation_id,
            runner_status=RunnerStatus.STARTING,
            total_rounds=total_rounds,
            total_simulation_hours=total_hours,
            started_at=datetime.now().isoformat(),
        )

        cls._save_run_state(state)

        # Wenn Graph-Speicher-Aktualisierung aktiviert, Updater erstellen
        if enable_graph_memory_update:
            if not graph_id:
                raise ValueError("graph_id muss angegeben werden, wenn Graph-Speicher-Aktualisierung aktiviert ist")

            try:
                if not storage:
                    raise ValueError("storage (GraphStorage) muss angegeben werden, wenn Graph-Speicher-Aktualisierung aktiviert ist")
                GraphMemoryManager.create_updater(simulation_id, graph_id, storage)
                cls._graph_memory_enabled[simulation_id] = True
                logger.info(f"Graph-Speicher-Aktualisierung aktiviert: simulation_id={simulation_id}, graph_id={graph_id}")
            except Exception as e:
                logger.error(f"Fehler beim Erstellen des Graph-Speicher-Updaters: {e}")
                cls._graph_memory_enabled[simulation_id] = False
        else:
            cls._graph_memory_enabled[simulation_id] = False

        # Bestimmen welches Skript ausgeführt werden soll (Skripte befinden sich im backend/scripts/-Verzeichnis)
        if platform == "twitter":
            script_name = "run_twitter_simulation.py"
            state.twitter_running = True
        elif platform == "reddit":
            script_name = "run_reddit_simulation.py"
            state.reddit_running = True
        else:
            script_name = "run_parallel_simulation.py"
            state.twitter_running = True
            state.reddit_running = True

        script_path = os.path.join(cls.SCRIPTS_DIR, script_name)

        if not os.path.exists(script_path):
            raise ValueError(f"Skript existiert nicht: {script_path}")

        # Aktionswarteschlange erstellen
        action_queue = Queue()
        cls._action_queues[simulation_id] = action_queue

        # Simulationsprozess starten
        try:
            # Ausführungsbefehl mit vollständigen Pfaden erstellen
            # Neue Log-Struktur:
            #   twitter/actions.jsonl - Twitter-Aktionslog
            #   reddit/actions.jsonl  - Reddit-Aktionslog
            #   simulation.log        - Hauptprozess-Log

            cmd = [
                sys.executable,  # Python-Interpreter
                script_path,
                "--config", config_path,  # Vollständigen Konfigurationsdateipfad verwenden
            ]

            # Wenn max_rounds angegeben, zu Kommandozeilenargumenten hinzufügen
            if max_rounds is not None and max_rounds > 0:
                cmd.extend(["--max-rounds", str(max_rounds)])

            # Hauptlogdatei erstellen, um stdout/stderr-Pipe-Pufferüberlauf zu vermeiden
            main_log_path = os.path.join(sim_dir, "simulation.log")
            main_log_file = open(main_log_path, 'w', encoding='utf-8')

            # Unterprozess-Umgebungsvariablen setzen, um UTF-8-Kodierung unter Windows sicherzustellen
            # Dies behebt Probleme mit Drittanbieter-Bibliotheken (wie OASIS), die beim Lesen von Dateien keine Kodierung angeben
            env = os.environ.copy()
            env['PYTHONUTF8'] = '1'  # Python 3.7+ Unterstützung, alle open() verwenden standardmäßig UTF-8
            env['PYTHONIOENCODING'] = 'utf-8'  # Sicherstellen, dass stdout/stderr UTF-8 verwenden

            # Arbeitsverzeichnis auf Simulationsverzeichnis setzen (Datenbankdateien etc. werden hier erstellt)
            # start_new_session=True verwenden, um neue Prozessgruppe zu erstellen, damit alle Kindprozesse über os.killpg beendet werden können
            process = subprocess.Popen(
                cmd,
                cwd=sim_dir,
                stdout=main_log_file,
                stderr=subprocess.STDOUT,  # stderr auch in dieselbe Datei schreiben
                text=True,
                encoding='utf-8',  # Kodierung explizit angeben
                bufsize=1,
                env=env,  # Umgebungsvariablen mit UTF-8-Einstellungen übergeben
                start_new_session=True,  # Neue Prozessgruppe erstellen, sicherstellen dass alle zugehörigen Prozesse beim Serverherunterfahren beendet werden
            )

            # Dateihandle zum späteren Schließen speichern
            cls._stdout_files[simulation_id] = main_log_file
            cls._stderr_files[simulation_id] = None  # Kein separates stderr mehr nötig

            state.process_pid = process.pid
            state.runner_status = RunnerStatus.RUNNING
            cls._processes[simulation_id] = process
            cls._save_run_state(state)

            # Überwachungs-Thread starten
            monitor_thread = threading.Thread(
                target=cls._monitor_simulation,
                args=(simulation_id,),
                daemon=True
            )
            monitor_thread.start()
            cls._monitor_threads[simulation_id] = monitor_thread

            logger.info(f"Simulation erfolgreich gestartet: {simulation_id}, pid={process.pid}, platform={platform}")

        except Exception as e:
            state.runner_status = RunnerStatus.FAILED
            state.error = str(e)
            cls._save_run_state(state)
            raise

        return state

    @classmethod
    def _monitor_simulation(cls, simulation_id: str):
        """Simulationsprozess überwachen und Aktionslogs parsen"""
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)

        # Neue Log-Struktur: plattformspezifische Aktionslogs
        twitter_actions_log = os.path.join(sim_dir, "twitter", "actions.jsonl")
        reddit_actions_log = os.path.join(sim_dir, "reddit", "actions.jsonl")

        process = cls._processes.get(simulation_id)
        state = cls.get_run_state(simulation_id)

        if not process or not state:
            return

        twitter_position = 0
        reddit_position = 0

        try:
            while process.poll() is None:  # Prozess läuft noch
                # Twitter-Aktionslog lesen
                if os.path.exists(twitter_actions_log):
                    twitter_position = cls._read_action_log(
                        twitter_actions_log, twitter_position, state, "twitter"
                    )

                # Reddit-Aktionslog lesen
                if os.path.exists(reddit_actions_log):
                    reddit_position = cls._read_action_log(
                        reddit_actions_log, reddit_position, state, "reddit"
                    )

                # Status aktualisieren
                cls._save_run_state(state)
                time.sleep(2)

            # Nach Prozessende Logs ein letztes Mal lesen
            if os.path.exists(twitter_actions_log):
                cls._read_action_log(twitter_actions_log, twitter_position, state, "twitter")
            if os.path.exists(reddit_actions_log):
                cls._read_action_log(reddit_actions_log, reddit_position, state, "reddit")

            # Prozess beendet
            exit_code = process.returncode

            if exit_code == 0:
                state.runner_status = RunnerStatus.COMPLETED
                state.completed_at = datetime.now().isoformat()
                logger.info(f"Simulation abgeschlossen: {simulation_id}")
            else:
                state.runner_status = RunnerStatus.FAILED
                # Fehlerinformationen aus Hauptlogdatei lesen
                main_log_path = os.path.join(sim_dir, "simulation.log")
                error_info = ""
                try:
                    if os.path.exists(main_log_path):
                        with open(main_log_path, 'r', encoding='utf-8') as f:
                            error_info = f.read()[-2000:]  # Letzte 2000 Zeichen nehmen
                except Exception:
                    pass
                state.error = f"Prozess-Exit-Code: {exit_code}, Fehler: {error_info}"
                logger.error(f"Simulation fehlgeschlagen: {simulation_id}, error={state.error}")

            state.twitter_running = False
            state.reddit_running = False
            cls._save_run_state(state)

        except Exception as e:
            logger.error(f"Überwachungs-Thread-Ausnahme: {simulation_id}, error={str(e)}")
            state.runner_status = RunnerStatus.FAILED
            state.error = str(e)
            cls._save_run_state(state)

        finally:
            # Graph-Speicher-Updater stoppen
            if cls._graph_memory_enabled.get(simulation_id, False):
                try:
                    GraphMemoryManager.stop_updater(simulation_id)
                    logger.info(f"Graph-Speicher-Aktualisierung gestoppt: simulation_id={simulation_id}")
                except Exception as e:
                    logger.error(f"Fehler beim Stoppen des Graph-Speicher-Updaters: {e}")
                cls._graph_memory_enabled.pop(simulation_id, None)

            # Prozessressourcen bereinigen
            cls._processes.pop(simulation_id, None)
            cls._action_queues.pop(simulation_id, None)

            # Logdatei-Handle schließen
            if simulation_id in cls._stdout_files:
                try:
                    cls._stdout_files[simulation_id].close()
                except Exception:
                    pass
                cls._stdout_files.pop(simulation_id, None)
            if simulation_id in cls._stderr_files and cls._stderr_files[simulation_id]:
                try:
                    cls._stderr_files[simulation_id].close()
                except Exception:
                    pass
                cls._stderr_files.pop(simulation_id, None)

    @classmethod
    def _read_action_log(
        cls,
        log_path: str,
        position: int,
        state: SimulationRunState,
        platform: str
    ) -> int:
        """
        Aktionslog-Datei lesen

        Args:
            log_path: Aktionslog-Dateipfad
            position: Letzte Leseposition
            state: Laufzustand-Objekt
            platform: Plattformname (twitter/reddit)

        Returns:
            Neue Leseposition
        """
        # Prüfen ob Graph-Speicher-Aktualisierung aktiviert ist
        graph_memory_enabled = cls._graph_memory_enabled.get(state.simulation_id, False)
        graph_updater = None
        if graph_memory_enabled:
            graph_updater = GraphMemoryManager.get_updater(state.simulation_id)

        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                f.seek(position)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            action_data = json.loads(line)

                            # Event-Typ-Einträge verarbeiten
                            if "event_type" in action_data:
                                event_type = action_data.get("event_type")

                                # simulation_end-Event erkennen, Plattform als abgeschlossen markieren
                                if event_type == "simulation_end":
                                    if platform == "twitter":
                                        state.twitter_completed = True
                                        state.twitter_running = False
                                        logger.info(f"Twitter-Simulation abgeschlossen: {state.simulation_id}, total_rounds={action_data.get('total_rounds')}, total_actions={action_data.get('total_actions')}")
                                    elif platform == "reddit":
                                        state.reddit_completed = True
                                        state.reddit_running = False
                                        logger.info(f"Reddit-Simulation abgeschlossen: {state.simulation_id}, total_rounds={action_data.get('total_rounds')}, total_actions={action_data.get('total_actions')}")

                                    # Prüfen ob alle aktivierten Plattformen abgeschlossen sind
                                    all_completed = cls._check_all_platforms_completed(state)
                                    if all_completed:
                                        state.runner_status = RunnerStatus.COMPLETED
                                        state.completed_at = datetime.now().isoformat()
                                        logger.info(f"Alle Plattform-Simulationen abgeschlossen: {state.simulation_id}")

                                # Rundeninformationen aktualisieren (aus round_end-Event)
                                elif event_type == "round_end":
                                    round_num = action_data.get("round", 0)
                                    simulated_hours = action_data.get("simulated_hours", 0)

                                    # Plattformunabhängige Runden und Zeit aktualisieren
                                    if platform == "twitter":
                                        if round_num > state.twitter_current_round:
                                            state.twitter_current_round = round_num
                                        state.twitter_simulated_hours = simulated_hours
                                    elif platform == "reddit":
                                        if round_num > state.reddit_current_round:
                                            state.reddit_current_round = round_num
                                        state.reddit_simulated_hours = simulated_hours

                                    # Gesamtrunden nehmen Maximum beider Plattformen
                                    if round_num > state.current_round:
                                        state.current_round = round_num
                                    # Gesamtzeit nimmt Maximum beider Plattformen
                                    state.simulated_hours = max(state.twitter_simulated_hours, state.reddit_simulated_hours)

                                continue

                            action = AgentAction(
                                round_num=action_data.get("round", 0),
                                timestamp=action_data.get("timestamp", datetime.now().isoformat()),
                                platform=platform,
                                agent_id=action_data.get("agent_id", 0),
                                agent_name=action_data.get("agent_name", ""),
                                action_type=action_data.get("action_type", ""),
                                action_args=action_data.get("action_args", {}),
                                result=action_data.get("result"),
                                success=action_data.get("success", True),
                            )
                            state.add_action(action)

                            # Runden aktualisieren
                            if action.round_num and action.round_num > state.current_round:
                                state.current_round = action.round_num

                            # Wenn Graph-Speicher-Aktualisierung aktiviert, Aktivität an den Graph senden
                            if graph_updater:
                                graph_updater.add_activity_from_dict(action_data, platform)

                        except json.JSONDecodeError:
                            pass
                return f.tell()
        except Exception as e:
            logger.warning(f"Fehler beim Lesen des Aktionslogs: {log_path}, error={e}")
            return position

    @classmethod
    def _check_all_platforms_completed(cls, state: SimulationRunState) -> bool:
        """
        Prüfen ob alle aktivierten Plattformen die Simulation abgeschlossen haben

        Beurteilt ob eine Plattform aktiviert ist, indem geprüft wird, ob die entsprechende actions.jsonl-Datei existiert

        Returns:
            True wenn alle aktivierten Plattformen abgeschlossen sind
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, state.simulation_id)
        twitter_log = os.path.join(sim_dir, "twitter", "actions.jsonl")
        reddit_log = os.path.join(sim_dir, "reddit", "actions.jsonl")

        # Prüfen welche Plattformen aktiviert sind (über Dateiexistenz)
        twitter_enabled = os.path.exists(twitter_log)
        reddit_enabled = os.path.exists(reddit_log)

        # Wenn Plattform aktiviert aber nicht abgeschlossen, False zurückgeben
        if twitter_enabled and not state.twitter_completed:
            return False
        if reddit_enabled and not state.reddit_completed:
            return False

        # Mindestens eine Plattform ist aktiviert und abgeschlossen
        return twitter_enabled or reddit_enabled

    @classmethod
    def _terminate_process(cls, process: subprocess.Popen, simulation_id: str, timeout: int = 10):
        """
        Plattformübergreifend Prozess und seine Kindprozesse beenden

        Args:
            process: Zu beendender Prozess
            simulation_id: Simulations-ID (für Logging)
            timeout: Timeout für Prozessbeendigung (Sekunden)
        """
        if IS_WINDOWS:
            # Windows: taskkill-Befehl zum Beenden des Prozessbaums verwenden
            # /F = erzwungene Beendigung, /T = Prozessbaum beenden (einschließlich Kindprozesse)
            logger.info(f"Beende Prozessbaum (Windows): simulation={simulation_id}, pid={process.pid}")
            try:
                # Zuerst sanfte Beendigung versuchen
                subprocess.run(
                    ['taskkill', '/PID', str(process.pid), '/T'],
                    capture_output=True,
                    timeout=5
                )
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    # Erzwungene Beendigung
                    logger.warning(f"Prozess reagiert nicht, erzwinge Beendigung: {simulation_id}")
                    subprocess.run(
                        ['taskkill', '/F', '/PID', str(process.pid), '/T'],
                        capture_output=True,
                        timeout=5
                    )
                    process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"taskkill fehlgeschlagen, versuche terminate: {e}")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        else:
            # Unix: Prozessgruppen-Beendigung verwenden
            # Da start_new_session=True, entspricht Prozessgruppen-ID der Hauptprozess-PID
            pgid = os.getpgid(process.pid)
            logger.info(f"Beende Prozessgruppe (Unix): simulation={simulation_id}, pgid={pgid}")

            # Zuerst SIGTERM an die gesamte Prozessgruppe senden
            os.killpg(pgid, signal.SIGTERM)

            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Falls nach Timeout noch nicht beendet, SIGKILL erzwingen
                logger.warning(f"Prozessgruppe reagiert nicht auf SIGTERM, erzwinge Beendigung: {simulation_id}")
                os.killpg(pgid, signal.SIGKILL)
                process.wait(timeout=5)

    @classmethod
    def stop_simulation(cls, simulation_id: str) -> SimulationRunState:
        """Simulation stoppen"""
        state = cls.get_run_state(simulation_id)
        if not state:
            raise ValueError(f"Simulation existiert nicht: {simulation_id}")

        if state.runner_status not in [RunnerStatus.RUNNING, RunnerStatus.PAUSED]:
            raise ValueError(f"Simulation läuft nicht: {simulation_id}, status={state.runner_status}")

        state.runner_status = RunnerStatus.STOPPING
        cls._save_run_state(state)

        # Prozess beenden
        process = cls._processes.get(simulation_id)
        if process and process.poll() is None:
            try:
                cls._terminate_process(process, simulation_id)
            except ProcessLookupError:
                # Prozess existiert nicht mehr
                pass
            except Exception as e:
                logger.error(f"Fehler beim Beenden der Prozessgruppe: {simulation_id}, error={e}")
                # Fallback auf direkte Prozessbeendigung
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except Exception:
                    process.kill()

        state.runner_status = RunnerStatus.STOPPED
        state.twitter_running = False
        state.reddit_running = False
        state.completed_at = datetime.now().isoformat()
        cls._save_run_state(state)

        # Graph-Speicher-Updater stoppen
        if cls._graph_memory_enabled.get(simulation_id, False):
            try:
                GraphMemoryManager.stop_updater(simulation_id)
                logger.info(f"Graph-Speicher-Aktualisierung gestoppt: simulation_id={simulation_id}")
            except Exception as e:
                logger.error(f"Fehler beim Stoppen des Graph-Speicher-Updaters: {e}")
            cls._graph_memory_enabled.pop(simulation_id, None)

        logger.info(f"Simulation gestoppt: {simulation_id}")
        return state

    @classmethod
    def _read_actions_from_file(
        cls,
        file_path: str,
        default_platform: Optional[str] = None,
        platform_filter: Optional[str] = None,
        agent_id: Optional[int] = None,
        round_num: Optional[int] = None
    ) -> List[AgentAction]:
        """
        Aktionen aus einzelner Aktionsdatei lesen

        Args:
            file_path: Aktionslog-Dateipfad
            default_platform: Standardplattform (verwendet wenn Aktionsdatensatz kein Plattformfeld hat)
            platform_filter: Plattformfilter
            agent_id: Agent-ID-Filter
            round_num: Rundenfilter
        """
        if not os.path.exists(file_path):
            return []

        actions = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    # Nicht-Aktionsdatensätze überspringen (wie simulation_start, round_start, round_end Events)
                    if "event_type" in data:
                        continue

                    # Datensätze ohne agent_id überspringen (keine Agent-Aktionen)
                    if "agent_id" not in data:
                        continue

                    # Plattform abrufen: bevorzugt Plattform im Datensatz, sonst Standardplattform
                    record_platform = data.get("platform") or default_platform or ""

                    # Filtern
                    if platform_filter and record_platform != platform_filter:
                        continue
                    if agent_id is not None and data.get("agent_id") != agent_id:
                        continue
                    if round_num is not None and data.get("round") != round_num:
                        continue

                    actions.append(AgentAction(
                        round_num=data.get("round", 0),
                        timestamp=data.get("timestamp", ""),
                        platform=record_platform,
                        agent_id=data.get("agent_id", 0),
                        agent_name=data.get("agent_name", ""),
                        action_type=data.get("action_type", ""),
                        action_args=data.get("action_args", {}),
                        result=data.get("result"),
                        success=data.get("success", True),
                    ))

                except json.JSONDecodeError:
                    continue

        return actions

    @classmethod
    def get_all_actions(
        cls,
        simulation_id: str,
        platform: Optional[str] = None,
        agent_id: Optional[int] = None,
        round_num: Optional[int] = None
    ) -> List[AgentAction]:
        """
        Vollständige Aktionshistorie aller Plattformen abrufen (ohne Seitenlimit)

        Args:
            simulation_id: Simulations-ID
            platform: Plattformfilter (twitter/reddit)
            agent_id: Agent-Filter
            round_num: Rundenfilter

        Returns:
            Vollständige Aktionsliste (nach Zeitstempel sortiert, neueste zuerst)
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        actions = []

        # Twitter-Aktionsdatei lesen (Plattform automatisch auf twitter basierend auf Dateipfad setzen)
        twitter_actions_log = os.path.join(sim_dir, "twitter", "actions.jsonl")
        if not platform or platform == "twitter":
            actions.extend(cls._read_actions_from_file(
                twitter_actions_log,
                default_platform="twitter",  # Plattformfeld automatisch ausfüllen
                platform_filter=platform,
                agent_id=agent_id,
                round_num=round_num
            ))

        # Reddit-Aktionsdatei lesen (Plattform automatisch auf reddit basierend auf Dateipfad setzen)
        reddit_actions_log = os.path.join(sim_dir, "reddit", "actions.jsonl")
        if not platform or platform == "reddit":
            actions.extend(cls._read_actions_from_file(
                reddit_actions_log,
                default_platform="reddit",  # Plattformfeld automatisch ausfüllen
                platform_filter=platform,
                agent_id=agent_id,
                round_num=round_num
            ))

        # Wenn plattformspezifische Dateien nicht existieren, versuche altes Einzeldateiformat zu lesen
        if not actions:
            actions_log = os.path.join(sim_dir, "actions.jsonl")
            actions = cls._read_actions_from_file(
                actions_log,
                default_platform=None,  # Dateien im alten Format sollten ein Plattformfeld haben
                platform_filter=platform,
                agent_id=agent_id,
                round_num=round_num
            )

        # Nach Zeitstempel sortieren (neueste zuerst)
        actions.sort(key=lambda x: x.timestamp, reverse=True)

        return actions

    @classmethod
    def get_actions(
        cls,
        simulation_id: str,
        limit: int = 100,
        offset: int = 0,
        platform: Optional[str] = None,
        agent_id: Optional[int] = None,
        round_num: Optional[int] = None
    ) -> List[AgentAction]:
        """
        Aktionshistorie abrufen (mit Seitentrennung)

        Args:
            simulation_id: Simulations-ID
            limit: Rückgabemengenlimit
            offset: Versatz
            platform: Plattformfilter
            agent_id: Agent-Filter
            round_num: Rundenfilter

        Returns:
            Aktionsliste
        """
        actions = cls.get_all_actions(
            simulation_id=simulation_id,
            platform=platform,
            agent_id=agent_id,
            round_num=round_num
        )

        # Seitentrennung
        return actions[offset:offset + limit]

    @classmethod
    def get_timeline(
        cls,
        simulation_id: str,
        start_round: int = 0,
        end_round: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Simulationszeitachse abrufen (nach Runden zusammengefasst)

        Args:
            simulation_id: Simulations-ID
            start_round: Startrunde
            end_round: Endrunde

        Returns:
            Zusammenfassungsinformationen für jede Runde
        """
        actions = cls.get_actions(simulation_id, limit=10000)

        # Nach Runden gruppieren
        rounds: Dict[int, Dict[str, Any]] = {}

        for action in actions:
            round_num = action.round_num

            if round_num < start_round:
                continue
            if end_round is not None and round_num > end_round:
                continue

            if round_num not in rounds:
                rounds[round_num] = {
                    "round_num": round_num,
                    "twitter_actions": 0,
                    "reddit_actions": 0,
                    "active_agents": set(),
                    "action_types": {},
                    "first_action_time": action.timestamp,
                    "last_action_time": action.timestamp,
                }

            r = rounds[round_num]

            if action.platform == "twitter":
                r["twitter_actions"] += 1
            else:
                r["reddit_actions"] += 1

            r["active_agents"].add(action.agent_id)
            r["action_types"][action.action_type] = r["action_types"].get(action.action_type, 0) + 1
            r["last_action_time"] = action.timestamp

        # In Liste umwandeln
        result = []
        for round_num in sorted(rounds.keys()):
            r = rounds[round_num]
            result.append({
                "round_num": round_num,
                "twitter_actions": r["twitter_actions"],
                "reddit_actions": r["reddit_actions"],
                "total_actions": r["twitter_actions"] + r["reddit_actions"],
                "active_agents_count": len(r["active_agents"]),
                "active_agents": list(r["active_agents"]),
                "action_types": r["action_types"],
                "first_action_time": r["first_action_time"],
                "last_action_time": r["last_action_time"],
            })

        return result

    @classmethod
    def get_agent_stats(cls, simulation_id: str) -> List[Dict[str, Any]]:
        """
        Statistiken für jeden Agent abrufen

        Returns:
            Agent-Statistikliste
        """
        actions = cls.get_actions(simulation_id, limit=10000)

        agent_stats: Dict[int, Dict[str, Any]] = {}

        for action in actions:
            agent_id = action.agent_id

            if agent_id not in agent_stats:
                agent_stats[agent_id] = {
                    "agent_id": agent_id,
                    "agent_name": action.agent_name,
                    "total_actions": 0,
                    "twitter_actions": 0,
                    "reddit_actions": 0,
                    "action_types": {},
                    "first_action_time": action.timestamp,
                    "last_action_time": action.timestamp,
                }

            stats = agent_stats[agent_id]
            stats["total_actions"] += 1

            if action.platform == "twitter":
                stats["twitter_actions"] += 1
            else:
                stats["reddit_actions"] += 1

            stats["action_types"][action.action_type] = stats["action_types"].get(action.action_type, 0) + 1
            stats["last_action_time"] = action.timestamp

        # Nach Gesamtaktionen sortieren
        result = sorted(agent_stats.values(), key=lambda x: x["total_actions"], reverse=True)

        return result

    @classmethod
    def cleanup_simulation_logs(cls, simulation_id: str) -> Dict[str, Any]:
        """
        Simulations-Ausführungslogs bereinigen (für erzwungenen Neustart)

        Löscht folgende Dateien:
        - run_state.json
        - twitter/actions.jsonl
        - reddit/actions.jsonl
        - simulation.log
        - stdout.log / stderr.log
        - twitter_simulation.db (Simulationsdatenbank)
        - reddit_simulation.db (Simulationsdatenbank)
        - env_status.json (Umgebungsstatus)

        Hinweis: Löscht keine Konfigurationsdateien (simulation_config.json) und Profildateien

        Args:
            simulation_id: Simulations-ID

        Returns:
            Bereinigungsergebnis-Informationen
        """
        import shutil

        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)

        if not os.path.exists(sim_dir):
            return {"success": True, "message": "Simulationsverzeichnis existiert nicht, keine Bereinigung nötig"}

        cleaned_files = []
        errors = []

        # Zu löschende Dateien (einschließlich Datenbankdateien)
        files_to_delete = [
            "run_state.json",
            "simulation.log",
            "stdout.log",
            "stderr.log",
            "twitter_simulation.db",  # Twitter-Plattform-Datenbank
            "reddit_simulation.db",   # Reddit-Plattform-Datenbank
            "env_status.json",        # Umgebungsstatusdatei
        ]

        # Zu löschende Verzeichnisse (enthalten Aktionslogs)
        dirs_to_clean = ["twitter", "reddit"]

        # Dateien löschen
        for filename in files_to_delete:
            file_path = os.path.join(sim_dir, filename)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    cleaned_files.append(filename)
                except Exception as e:
                    errors.append(f"Fehler beim Löschen von {filename}: {str(e)}")

        # Aktionslogs in Plattformverzeichnissen bereinigen
        for dir_name in dirs_to_clean:
            dir_path = os.path.join(sim_dir, dir_name)
            if os.path.exists(dir_path):
                actions_file = os.path.join(dir_path, "actions.jsonl")
                if os.path.exists(actions_file):
                    try:
                        os.remove(actions_file)
                        cleaned_files.append(f"{dir_name}/actions.jsonl")
                    except Exception as e:
                        errors.append(f"Fehler beim Löschen von {dir_name}/actions.jsonl: {str(e)}")

        # Laufzustand im Speicher bereinigen
        if simulation_id in cls._run_states:
            del cls._run_states[simulation_id]

        logger.info(f"Bereinigung der Simulationslogs abgeschlossen: {simulation_id}, gelöschte Dateien: {cleaned_files}")

        return {
            "success": len(errors) == 0,
            "cleaned_files": cleaned_files,
            "errors": errors if errors else None
        }

    # Flag zur Vermeidung doppelter Bereinigung
    _cleanup_done = False

    @classmethod
    def cleanup_all_simulations(cls):
        """
        Alle laufenden Simulationsprozesse bereinigen

        Wird beim Herunterfahren des Servers aufgerufen, stellt sicher dass alle Kindprozesse beendet werden
        """
        # Doppelte Bereinigung verhindern
        if cls._cleanup_done:
            return
        cls._cleanup_done = True

        # Prüfen ob es Inhalte zu bereinigen gibt (vermeidet nutzlose Log-Ausgaben bei leerem Prozess)
        has_processes = bool(cls._processes)
        has_updaters = bool(cls._graph_memory_enabled)

        if not has_processes and not has_updaters:
            return  # Nichts zu bereinigen, leise zurückkehren

        logger.info("Bereinige alle Simulationsprozesse...")

        # Zuerst alle Graph-Speicher-Updater stoppen (stop_all gibt intern Logs aus)
        try:
            GraphMemoryManager.stop_all()
        except Exception as e:
            logger.error(f"Fehler beim Stoppen des Graph-Speicher-Updaters: {e}")
        cls._graph_memory_enabled.clear()

        # Dict kopieren, um Änderung während Iteration zu vermeiden
        processes = list(cls._processes.items())

        for simulation_id, process in processes:
            try:
                if process.poll() is None:  # Prozess läuft noch
                    logger.info(f"Beende Simulationsprozess: {simulation_id}, pid={process.pid}")

                    try:
                        # Plattformübergreifende Prozessbeendigung verwenden
                        cls._terminate_process(process, simulation_id, timeout=5)
                    except (ProcessLookupError, OSError):
                        # Prozess existiert möglicherweise nicht mehr, direkte Beendigung versuchen
                        try:
                            process.terminate()
                            process.wait(timeout=3)
                        except Exception:
                            process.kill()

                    # run_state.json aktualisieren
                    state = cls.get_run_state(simulation_id)
                    if state:
                        state.runner_status = RunnerStatus.STOPPED
                        state.twitter_running = False
                        state.reddit_running = False
                        state.completed_at = datetime.now().isoformat()
                        state.error = "Server heruntergefahren, Simulation beendet"
                        cls._save_run_state(state)

                    # Auch state.json aktualisieren, Status auf stopped setzen
                    try:
                        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
                        state_file = os.path.join(sim_dir, "state.json")
                        logger.info(f"Versuche state.json zu aktualisieren: {state_file}")
                        if os.path.exists(state_file):
                            with open(state_file, 'r', encoding='utf-8') as f:
                                state_data = json.load(f)
                            state_data['status'] = 'stopped'
                            state_data['updated_at'] = datetime.now().isoformat()
                            with open(state_file, 'w', encoding='utf-8') as f:
                                json.dump(state_data, f, indent=2, ensure_ascii=False)
                            logger.info(f"state.json-Status auf stopped aktualisiert: {simulation_id}")
                        else:
                            logger.warning(f"state.json existiert nicht: {state_file}")
                    except Exception as state_err:
                        logger.warning(f"Fehler beim Aktualisieren von state.json: {simulation_id}, error={state_err}")

            except Exception as e:
                logger.error(f"Fehler beim Bereinigen des Prozesses: {simulation_id}, error={e}")

        # Dateihandles bereinigen
        for simulation_id, file_handle in list(cls._stdout_files.items()):
            try:
                if file_handle:
                    file_handle.close()
            except Exception:
                pass
        cls._stdout_files.clear()

        for simulation_id, file_handle in list(cls._stderr_files.items()):
            try:
                if file_handle:
                    file_handle.close()
            except Exception:
                pass
        cls._stderr_files.clear()

        # Zustand im Speicher bereinigen
        cls._processes.clear()
        cls._action_queues.clear()

        logger.info("Bereinigung der Simulationsprozesse abgeschlossen")

    @classmethod
    def register_cleanup(cls):
        """
        Bereinigungsfunktion registrieren

        Wird beim Start der Flask-App aufgerufen, stellt sicher dass alle Simulationsprozesse beim Herunterfahren des Servers bereinigt werden
        """
        global _cleanup_registered

        if _cleanup_registered:
            return

        # Im Flask-Debug-Modus Bereinigung nur im Reloader-Kindprozess registrieren (Prozess der die App tatsächlich ausführt)
        # WERKZEUG_RUN_MAIN=true zeigt an, dass es ein Reloader-Kindprozess ist
        # Wenn nicht im Debug-Modus, gibt es keine solche Umgebungsvariable, muss trotzdem registriert werden
        is_reloader_process = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
        is_debug_mode = os.environ.get('FLASK_DEBUG') == '1' or os.environ.get('WERKZEUG_RUN_MAIN') is not None

        # Im Debug-Modus nur im Reloader-Kindprozess registrieren; im Nicht-Debug-Modus immer registrieren
        if is_debug_mode and not is_reloader_process:
            _cleanup_registered = True  # Als registriert markieren, Kindprozess daran hindern erneut zu versuchen
            return

        # Originalen Signal-Handler speichern
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)
        # SIGHUP existiert nur auf Unix-Systemen (macOS/Linux), nicht auf Windows
        original_sighup = None
        has_sighup = hasattr(signal, 'SIGHUP')
        if has_sighup:
            original_sighup = signal.getsignal(signal.SIGHUP)

        def cleanup_handler(signum=None, frame=None):
            """Signal-Handler: Zuerst Simulationsprozesse bereinigen, dann originalen Handler aufrufen"""
            # Nur Logs ausgeben wenn Prozesse zu bereinigen sind
            if cls._processes or cls._graph_memory_enabled:
                logger.info(f"Signal {signum} empfangen, starte Bereinigung...")
            cls.cleanup_all_simulations()

            # Originalen Signal-Handler aufrufen, Flask normal beenden lassen
            if signum == signal.SIGINT and callable(original_sigint):
                original_sigint(signum, frame)
            elif signum == signal.SIGTERM and callable(original_sigterm):
                original_sigterm(signum, frame)
            elif has_sighup and signum == signal.SIGHUP:
                # SIGHUP: Wird gesendet wenn Terminal geschlossen wird
                if callable(original_sighup):
                    original_sighup(signum, frame)
                else:
                    # Standardverhalten: normal beenden
                    sys.exit(0)
            else:
                # Wenn originaler Handler nicht aufrufbar (wie SIG_DFL), Standardverhalten verwenden
                raise KeyboardInterrupt

        # atexit-Handler registrieren (als Fallback)
        atexit.register(cls.cleanup_all_simulations)

        # Signal-Handler registrieren (nur im Haupt-Thread)
        try:
            # SIGTERM: Standard-Signal für kill-Befehl
            signal.signal(signal.SIGTERM, cleanup_handler)
            # SIGINT: Ctrl+C
            signal.signal(signal.SIGINT, cleanup_handler)
            # SIGHUP: Terminal geschlossen (nur Unix)
            if has_sighup:
                signal.signal(signal.SIGHUP, cleanup_handler)
        except ValueError:
            # Nicht im Haupt-Thread, kann nur atexit verwenden
            logger.warning("Signal-Handler kann nicht registriert werden (nicht im Haupt-Thread), verwende nur atexit")

        _cleanup_registered = True

    @classmethod
    def get_running_simulations(cls) -> List[str]:
        """
        Liste aller laufenden Simulations-IDs abrufen
        """
        running = []
        for sim_id, process in cls._processes.items():
            if process.poll() is None:
                running.append(sim_id)
        return running

    # ============== Interview-Funktionalität ==============

    @classmethod
    def check_env_alive(cls, simulation_id: str) -> bool:
        """
        Prüfen ob Simulationsumgebung aktiv ist (Interview-Befehle empfangen kann)

        Args:
            simulation_id: Simulations-ID

        Returns:
            True bedeutet Umgebung ist aktiv, False bedeutet Umgebung ist geschlossen
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        if not os.path.exists(sim_dir):
            return False

        ipc_client = SimulationIPCClient(sim_dir)
        return ipc_client.check_env_alive()

    @classmethod
    def get_env_status_detail(cls, simulation_id: str) -> Dict[str, Any]:
        """
        Detaillierte Statusinformationen der Simulationsumgebung abrufen

        Args:
            simulation_id: Simulations-ID

        Returns:
            Statusdetails-Dict, enthält status, twitter_available, reddit_available, timestamp
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        status_file = os.path.join(sim_dir, "env_status.json")

        default_status = {
            "status": "stopped",
            "twitter_available": False,
            "reddit_available": False,
            "timestamp": None
        }

        if not os.path.exists(status_file):
            return default_status

        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                status = json.load(f)
            return {
                "status": status.get("status", "stopped"),
                "twitter_available": status.get("twitter_available", False),
                "reddit_available": status.get("reddit_available", False),
                "timestamp": status.get("timestamp")
            }
        except (json.JSONDecodeError, OSError):
            return default_status

    @classmethod
    def interview_agent(
        cls,
        simulation_id: str,
        agent_id: int,
        prompt: str,
        platform: str = None,
        timeout: float = 60.0
    ) -> Dict[str, Any]:
        """
        Einzelnen Agent interviewen

        Args:
            simulation_id: Simulations-ID
            agent_id: Agent-ID
            prompt: Interviewfrage
            platform: Plattform angeben (optional)
                - "twitter": nur Twitter-Plattform interviewen
                - "reddit": nur Reddit-Plattform interviewen
                - None: beide Plattformen gleichzeitig in Dualplattform-Simulationen interviewen, integrierte Ergebnisse zurückgeben
            timeout: Timeout (Sekunden)

        Returns:
            Interview-Ergebnis-Dict

        Raises:
            ValueError: Simulation existiert nicht oder Umgebung läuft nicht
            TimeoutError: Timeout beim Warten auf Antwort
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        if not os.path.exists(sim_dir):
            raise ValueError(f"Simulation existiert nicht: {simulation_id}")

        ipc_client = SimulationIPCClient(sim_dir)

        if not ipc_client.check_env_alive():
            raise ValueError(f"Simulationsumgebung läuft nicht oder ist geschlossen, Interview kann nicht ausgeführt werden: {simulation_id}")

        logger.info(f"Sende Interview-Befehl: simulation_id={simulation_id}, agent_id={agent_id}, platform={platform}")

        response = ipc_client.send_interview(
            agent_id=agent_id,
            prompt=prompt,
            platform=platform,
            timeout=timeout
        )

        if response.status.value == "completed":
            return {
                "success": True,
                "agent_id": agent_id,
                "prompt": prompt,
                "result": response.result,
                "timestamp": response.timestamp
            }
        else:
            return {
                "success": False,
                "agent_id": agent_id,
                "prompt": prompt,
                "error": response.error,
                "timestamp": response.timestamp
            }

    @classmethod
    def interview_agents_batch(
        cls,
        simulation_id: str,
        interviews: List[Dict[str, Any]],
        platform: str = None,
        timeout: float = 120.0
    ) -> Dict[str, Any]:
        """
        Mehrere Agents im Batch interviewen

        Args:
            simulation_id: Simulations-ID
            interviews: Liste von Interviews, jedes Element enthält {"agent_id": int, "prompt": str, "platform": str(optional)}
            platform: Standardplattform (optional, wird durch die Plattform jedes Interview-Elements überschrieben)
                - "twitter": standardmäßig nur Twitter-Plattform interviewen
                - "reddit": standardmäßig nur Reddit-Plattform interviewen
                - None: jeden Agent auf beiden Plattformen gleichzeitig in Dualplattform-Simulationen interviewen
            timeout: Timeout (Sekunden)

        Returns:
            Batch-Interview-Ergebnis-Dict

        Raises:
            ValueError: Simulation existiert nicht oder Umgebung läuft nicht
            TimeoutError: Timeout beim Warten auf Antwort
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        if not os.path.exists(sim_dir):
            raise ValueError(f"Simulation existiert nicht: {simulation_id}")

        ipc_client = SimulationIPCClient(sim_dir)

        if not ipc_client.check_env_alive():
            raise ValueError(f"Simulationsumgebung läuft nicht oder ist geschlossen, Interview kann nicht ausgeführt werden: {simulation_id}")

        logger.info(f"Sende Batch-Interview-Befehl: simulation_id={simulation_id}, count={len(interviews)}, platform={platform}")

        response = ipc_client.send_batch_interview(
            interviews=interviews,
            platform=platform,
            timeout=timeout
        )

        if response.status.value == "completed":
            return {
                "success": True,
                "interviews_count": len(interviews),
                "result": response.result,
                "timestamp": response.timestamp
            }
        else:
            return {
                "success": False,
                "interviews_count": len(interviews),
                "error": response.error,
                "timestamp": response.timestamp
            }

    @classmethod
    def interview_all_agents(
        cls,
        simulation_id: str,
        prompt: str,
        platform: str = None,
        timeout: float = 180.0
    ) -> Dict[str, Any]:
        """
        Alle Agents interviewen (globales Interview)

        Alle Agents in der Simulation mit derselben Frage interviewen

        Args:
            simulation_id: Simulations-ID
            prompt: Interviewfrage (alle Agents verwenden dieselbe Frage)
            platform: Plattform angeben (optional)
                - "twitter": nur Twitter-Plattform interviewen
                - "reddit": nur Reddit-Plattform interviewen
                - None: jeden Agent auf beiden Plattformen gleichzeitig in Dualplattform-Simulationen interviewen
            timeout: Timeout (Sekunden)

        Returns:
            Globales Interview-Ergebnis-Dict
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        if not os.path.exists(sim_dir):
            raise ValueError(f"Simulation existiert nicht: {simulation_id}")

        # Alle Agent-Informationen aus Konfigurationsdatei abrufen
        config_path = os.path.join(sim_dir, "simulation_config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"Simulationskonfiguration existiert nicht: {simulation_id}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        agent_configs = config.get("agent_configs", [])
        if not agent_configs:
            raise ValueError(f"Keine Agents in der Simulationskonfiguration: {simulation_id}")

        # Batch-Interview-Liste erstellen
        interviews = []
        for agent_config in agent_configs:
            agent_id = agent_config.get("agent_id")
            if agent_id is not None:
                interviews.append({
                    "agent_id": agent_id,
                    "prompt": prompt
                })

        logger.info(f"Sende globalen Interview-Befehl: simulation_id={simulation_id}, agent_count={len(interviews)}, platform={platform}")

        return cls.interview_agents_batch(
            simulation_id=simulation_id,
            interviews=interviews,
            platform=platform,
            timeout=timeout
        )

    @classmethod
    def close_simulation_env(
        cls,
        simulation_id: str,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Simulationsumgebung schließen (nicht den Simulationsprozess stoppen)

        Sendet Umgebungsschließungsbefehl an die Simulation, um den Befehlswartmodus ordnungsgemäß zu beenden

        Args:
            simulation_id: Simulations-ID
            timeout: Timeout (Sekunden)

        Returns:
            Operationsergebnis-Dict
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        if not os.path.exists(sim_dir):
            raise ValueError(f"Simulation existiert nicht: {simulation_id}")

        ipc_client = SimulationIPCClient(sim_dir)

        if not ipc_client.check_env_alive():
            return {
                "success": True,
                "message": "Umgebung bereits geschlossen"
            }

        logger.info(f"Sende Umgebungsschließungsbefehl: simulation_id={simulation_id}")

        try:
            response = ipc_client.send_close_env(timeout=timeout)

            return {
                "success": response.status.value == "completed",
                "message": "Umgebungsschließungsbefehl gesendet",
                "result": response.result,
                "timestamp": response.timestamp
            }
        except TimeoutError:
            # Timeout möglicherweise weil Umgebung gerade geschlossen wird
            return {
                "success": True,
                "message": "Umgebungsschließungsbefehl gesendet (Timeout beim Warten auf Antwort, Umgebung wird möglicherweise geschlossen)"
            }

    @classmethod
    def _get_interview_history_from_db(
        cls,
        db_path: str,
        platform_name: str,
        agent_id: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Interview-Verlauf aus einzelner Datenbank abrufen"""
        import sqlite3

        if not os.path.exists(db_path):
            return []

        results = []

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            if agent_id is not None:
                cursor.execute("""
                    SELECT user_id, info, created_at
                    FROM trace
                    WHERE action = 'interview' AND user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (agent_id, limit))
            else:
                cursor.execute("""
                    SELECT user_id, info, created_at
                    FROM trace
                    WHERE action = 'interview'
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))

            for user_id, info_json, created_at in cursor.fetchall():
                try:
                    info = json.loads(info_json) if info_json else {}
                except json.JSONDecodeError:
                    info = {"raw": info_json}

                results.append({
                    "agent_id": user_id,
                    "response": info.get("response", info),
                    "prompt": info.get("prompt", ""),
                    "timestamp": created_at,
                    "platform": platform_name
                })

            conn.close()

        except Exception as e:
            logger.error(f"Fehler beim Lesen des Interview-Verlaufs ({platform_name}): {e}")

        return results

    @classmethod
    def get_interview_history(
        cls,
        simulation_id: str,
        platform: str = None,
        agent_id: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Interview-Verlaufsdatensätze abrufen (aus Datenbank lesen)

        Args:
            simulation_id: Simulations-ID
            platform: Plattformtyp (reddit/twitter/None)
                - "reddit": nur Reddit-Plattform-Verlauf abrufen
                - "twitter": nur Twitter-Plattform-Verlauf abrufen
                - None: gesamten Verlauf beider Plattformen abrufen
            agent_id: Agent-ID angeben (optional, nur Verlauf für diesen Agent abrufen)
            limit: Rückgabemengenlimit pro Plattform

        Returns:
            Interview-Verlaufsdatensatz-Liste
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)

        results = []

        # Abzufragende Plattformen bestimmen
        if platform in ("reddit", "twitter"):
            platforms = [platform]
        else:
            # Wenn Plattform nicht angegeben, beide Plattformen abfragen
            platforms = ["twitter", "reddit"]

        for p in platforms:
            db_path = os.path.join(sim_dir, f"{p}_simulation.db")
            platform_results = cls._get_interview_history_from_db(
                db_path=db_path,
                platform_name=p,
                agent_id=agent_id,
                limit=limit
            )
            results.extend(platform_results)

        # Nach Zeit absteigend sortieren
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        # Wenn mehrere Plattformen abgefragt, Gesamtanzahl begrenzen
        if len(platforms) > 1 and len(results) > limit:
            results = results[:limit]

        return results
