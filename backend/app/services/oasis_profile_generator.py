"""
OASIS Agent-Profil-Generator
Entitäten aus dem Wissensgraph in das vom OASIS-Simulationsplattform benötigte Agent-Profil-Format konvertieren

Optimierungsverbesserungen:
1. Wissensgraph-Abruffunktion aufrufen, um Node-Informationen anzureichern
2. Prompts optimieren, um sehr detaillierte Personas zu generieren
3. Zwischen individuellen Entitäten und abstrakten Gruppenentitäten unterscheiden
"""

import json
import random
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from openai import OpenAI

from ..config import Config
from ..utils.logger import get_logger
from .entity_reader import EntityNode
from ..storage import GraphStorage

logger = get_logger('mirofish.oasis_profile')


@dataclass
class OasisAgentProfile:
    """OASIS Agent-Profil-Datenstruktur"""
    # Gemeinsame Felder
    user_id: int
    user_name: str
    name: str
    bio: str
    persona: str

    # Optionale Felder - Reddit-Stil
    karma: int = 1000

    # Optionale Felder - Twitter-Stil
    friend_count: int = 100
    follower_count: int = 150
    statuses_count: int = 500

    # Zusätzliche Persona-Informationen
    age: Optional[int] = None
    gender: Optional[str] = None
    mbti: Optional[str] = None
    country: Optional[str] = None
    profession: Optional[str] = None
    interested_topics: List[str] = field(default_factory=list)

    # Quellentitäts-Informationen
    source_entity_uuid: Optional[str] = None
    source_entity_type: Optional[str] = None

    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))

    def to_reddit_format(self) -> Dict[str, Any]:
        """In Reddit-Plattform-Format konvertieren"""
        profile = {
            "user_id": self.user_id,
            "username": self.user_name,  # OASIS-Bibliothek erfordert Feldname als username (ohne Unterstrich)
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "karma": self.karma,
            "created_at": self.created_at,
        }

        # Zusätzliche Persona-Informationen hinzufügen (falls verfügbar)
        if self.age:
            profile["age"] = self.age
        if self.gender:
            profile["gender"] = self.gender
        if self.mbti:
            profile["mbti"] = self.mbti
        if self.country:
            profile["country"] = self.country
        if self.profession:
            profile["profession"] = self.profession
        if self.interested_topics:
            profile["interested_topics"] = self.interested_topics

        return profile

    def to_twitter_format(self) -> Dict[str, Any]:
        """In Twitter-Plattform-Format konvertieren"""
        profile = {
            "user_id": self.user_id,
            "username": self.user_name,  # OASIS-Bibliothek erfordert Feldname als username (ohne Unterstrich)
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "friend_count": self.friend_count,
            "follower_count": self.follower_count,
            "statuses_count": self.statuses_count,
            "created_at": self.created_at,
        }

        # Zusätzliche Persona-Informationen hinzufügen
        if self.age:
            profile["age"] = self.age
        if self.gender:
            profile["gender"] = self.gender
        if self.mbti:
            profile["mbti"] = self.mbti
        if self.country:
            profile["country"] = self.country
        if self.profession:
            profile["profession"] = self.profession
        if self.interested_topics:
            profile["interested_topics"] = self.interested_topics

        return profile

    def to_dict(self) -> Dict[str, Any]:
        """In vollständiges Wörterbuch-Format konvertieren"""
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "karma": self.karma,
            "friend_count": self.friend_count,
            "follower_count": self.follower_count,
            "statuses_count": self.statuses_count,
            "age": self.age,
            "gender": self.gender,
            "mbti": self.mbti,
            "country": self.country,
            "profession": self.profession,
            "interested_topics": self.interested_topics,
            "source_entity_uuid": self.source_entity_uuid,
            "source_entity_type": self.source_entity_type,
            "created_at": self.created_at,
        }


class OasisProfileGenerator:
    """
    OASIS-Profil-Generator

    Entitäten aus dem Wissensgraph in das für die OASIS-Simulation benötigte Agent-Profil konvertieren

    Optimierungsfunktionen:
    1. Wissensgraph-Abruffunktion aufrufen, um reichhaltigeren Kontext zu erhalten
    2. Sehr detaillierte Personas generieren (einschließlich Grundinformationen, Berufserfahrung, Persönlichkeitsmerkmale, Social-Media-Verhalten usw.)
    3. Zwischen individuellen Entitäten und abstrakten Gruppenentitäten unterscheiden
    """

    # MBTI-Typenliste
    MBTI_TYPES = [
        "INTJ", "INTP", "ENTJ", "ENTP",
        "INFJ", "INFP", "ENFJ", "ENFP",
        "ISTJ", "ISFJ", "ESTJ", "ESFJ",
        "ISTP", "ISFP", "ESTP", "ESFP"
    ]

    # Gängige Länderliste
    COUNTRIES = [
        "Deutschland", "Österreich", "Schweiz", "USA", "Großbritannien",
        "Frankreich", "Japan", "Kanada", "Australien", "Niederlande"
    ]

    # Entitätstypen für Einzelpersonen (benötigen spezifische Personas)
    INDIVIDUAL_ENTITY_TYPES = [
        "student", "alumni", "professor", "person", "publicfigure",
        "expert", "faculty", "official", "journalist", "activist"
    ]

    # Gruppen-/Institutionstypen (benötigen repräsentative Gruppen-Personas)
    GROUP_ENTITY_TYPES = [
        "university", "governmentagency", "organization", "ngo",
        "mediaoutlet", "company", "institution", "group", "community"
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        storage: Optional[GraphStorage] = None,
        graph_id: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model_name = model_name or Config.LLM_MODEL_NAME

        if not self.api_key:
            raise ValueError("LLM_API_KEY nicht konfiguriert")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        # GraphStorage für hybride Suchanreicherung
        self.storage = storage
        self.graph_id = graph_id

    def generate_profile_from_entity(
        self,
        entity: EntityNode,
        user_id: int,
        use_llm: bool = True
    ) -> OasisAgentProfile:
        """
        OASIS Agent-Profil aus Wissensgraph-Entität generieren

        Args:
            entity: Wissensgraph-Entitäts-Node
            user_id: Benutzer-ID (für OASIS)
            use_llm: Ob LLM zur Generierung detaillierter Persona verwendet werden soll

        Returns:
            OasisAgentProfile
        """
        entity_type = entity.get_entity_type() or "Entity"

        # Grundinformationen
        name = entity.name
        user_name = self._generate_username(name)

        # Kontextinformationen erstellen
        context = self._build_entity_context(entity)

        if use_llm:
            # LLM zur Generierung detaillierter Persona verwenden
            profile_data = self._generate_profile_with_llm(
                entity_name=name,
                entity_type=entity_type,
                entity_summary=entity.summary,
                entity_attributes=entity.attributes,
                context=context
            )
        else:
            # Regeln zur Generierung grundlegender Persona verwenden
            profile_data = self._generate_profile_rule_based(
                entity_name=name,
                entity_type=entity_type,
                entity_summary=entity.summary,
                entity_attributes=entity.attributes
            )

        return OasisAgentProfile(
            user_id=user_id,
            user_name=user_name,
            name=name,
            bio=profile_data.get("bio", f"{entity_type}: {name}"),
            persona=profile_data.get("persona", entity.summary or f"Ein(e) {entity_type} namens {name}."),
            karma=profile_data.get("karma", random.randint(500, 5000)),
            friend_count=profile_data.get("friend_count", random.randint(50, 500)),
            follower_count=profile_data.get("follower_count", random.randint(100, 1000)),
            statuses_count=profile_data.get("statuses_count", random.randint(100, 2000)),
            age=profile_data.get("age"),
            gender=profile_data.get("gender"),
            mbti=profile_data.get("mbti"),
            country=profile_data.get("country"),
            profession=profile_data.get("profession"),
            interested_topics=profile_data.get("interested_topics", []),
            source_entity_uuid=entity.uuid,
            source_entity_type=entity_type,
        )

    def _generate_username(self, name: str) -> str:
        """Benutzernamen generieren"""
        # Sonderzeichen entfernen, in Kleinbuchstaben konvertieren
        username = name.lower().replace(" ", "_")
        username = ''.join(c for c in username if c.isalnum() or c == '_')

        # Zufälliges Suffix hinzufügen, um Duplikate zu vermeiden
        suffix = random.randint(100, 999)
        return f"{username}_{suffix}"

    def _search_graph_for_entity(self, entity: EntityNode) -> Dict[str, Any]:
        """
        GraphStorage-Hybridsuche verwenden, um reichhaltige Informationen zur Entität zu erhalten

        Verwendet storage.search() (hybrides Vektor- + BM25) für sowohl Edges als auch Nodes.

        Args:
            entity: Entitäts-Node-Objekt

        Returns:
            Wörterbuch mit facts, node_summaries, context
        """
        if not self.storage:
            return {"facts": [], "node_summaries": [], "context": ""}

        entity_name = entity.name

        results = {
            "facts": [],
            "node_summaries": [],
            "context": ""
        }

        if not self.graph_id:
            logger.debug(f"Wissensgraph-Suche übersprungen: graph_id nicht gesetzt")
            return results

        comprehensive_query = f"Alle Informationen, Aktivitäten, Ereignisse, Beziehungen und Hintergründe über {entity_name}"

        try:
            # Edges durchsuchen (Fakten)
            edge_results = self.storage.search(
                graph_id=self.graph_id,
                query=comprehensive_query,
                limit=30,
                scope="edges"
            )

            all_facts = set()
            if isinstance(edge_results, dict) and 'edges' in edge_results:
                for edge in edge_results['edges']:
                    fact = edge.get('fact', '')
                    if fact:
                        all_facts.add(fact)
            results["facts"] = list(all_facts)

            # Nodes durchsuchen (Entitätszusammenfassungen)
            node_results = self.storage.search(
                graph_id=self.graph_id,
                query=comprehensive_query,
                limit=20,
                scope="nodes"
            )

            all_summaries = set()
            if isinstance(node_results, dict) and 'nodes' in node_results:
                for node in node_results['nodes']:
                    summary = node.get('summary', '')
                    if summary:
                        all_summaries.add(summary)
                    name = node.get('name', '')
                    if name and name != entity_name:
                        all_summaries.add(f"Verwandte Entität: {name}")
            results["node_summaries"] = list(all_summaries)

            # Kombinierten Kontext erstellen
            context_parts = []
            if results["facts"]:
                context_parts.append("Fakteninformationen:\n" + "\n".join(f"- {f}" for f in results["facts"][:20]))
            if results["node_summaries"]:
                context_parts.append("Verwandte Entitäten:\n" + "\n".join(f"- {s}" for s in results["node_summaries"][:10]))
            results["context"] = "\n\n".join(context_parts)

            logger.info(f"Wissensgraph-Hybridsuche abgeschlossen: {entity_name}, {len(results['facts'])} Fakten abgerufen, {len(results['node_summaries'])} verwandte Nodes")

        except Exception as e:
            logger.warning(f"Wissensgraph-Suche fehlgeschlagen ({entity_name}): {e}")

        return results

    def _build_entity_context(self, entity: EntityNode) -> str:
        """
        Vollständige Kontextinformationen für die Entität erstellen

        Beinhaltet:
        1. Edge-Informationen der Entität selbst (Fakten)
        2. Detaillierte Informationen zugehöriger Nodes
        3. Durch Wissensgraph-Hybridsuche abgerufene reichhaltige Informationen
        """
        context_parts = []

        # 1. Entitätsattribut-Informationen hinzufügen
        if entity.attributes:
            attrs = []
            for key, value in entity.attributes.items():
                if value and str(value).strip():
                    attrs.append(f"- {key}: {value}")
            if attrs:
                context_parts.append("### Entitätsattribute\n" + "\n".join(attrs))

        # 2. Verwandte Edge-Informationen hinzufügen (Fakten/Beziehungen)
        existing_facts = set()
        if entity.related_edges:
            relationships = []
            for edge in entity.related_edges:  # Keine Mengenbegrenzung
                fact = edge.get("fact", "")
                edge_name = edge.get("edge_name", "")
                direction = edge.get("direction", "")

                if fact:
                    relationships.append(f"- {fact}")
                    existing_facts.add(fact)
                elif edge_name:
                    if direction == "outgoing":
                        relationships.append(f"- {entity.name} --[{edge_name}]--> (Verwandte Entität)")
                    else:
                        relationships.append(f"- (Verwandte Entität) --[{edge_name}]--> {entity.name}")

            if relationships:
                context_parts.append("### Verwandte Fakten und Beziehungen\n" + "\n".join(relationships))

        # 3. Detaillierte Informationen verwandter Nodes hinzufügen
        if entity.related_nodes:
            related_info = []
            for node in entity.related_nodes:  # Keine Mengenbegrenzung
                node_name = node.get("name", "")
                node_labels = node.get("labels", [])
                node_summary = node.get("summary", "")

                # Standard-Labels herausfiltern
                custom_labels = [l for l in node_labels if l not in ["Entity", "Node"]]
                label_str = f" ({', '.join(custom_labels)})" if custom_labels else ""

                if node_summary:
                    related_info.append(f"- **{node_name}**{label_str}: {node_summary}")
                else:
                    related_info.append(f"- **{node_name}**{label_str}")

            if related_info:
                context_parts.append("### Verwandte Entitätsinformationen\n" + "\n".join(related_info))

        # 4. Wissensgraph-Hybridsuche für reichhaltigere Informationen verwenden
        graph_results = self._search_graph_for_entity(entity)

        if graph_results.get("facts"):
            # Deduplizierung: vorhandene Fakten ausschließen
            new_facts = [f for f in graph_results["facts"] if f not in existing_facts]
            if new_facts:
                context_parts.append("### Aus dem Wissensgraph abgerufene Fakten\n" + "\n".join(f"- {f}" for f in new_facts[:15]))

        if graph_results.get("node_summaries"):
            context_parts.append("### Aus dem Wissensgraph abgerufene verwandte Nodes\n" + "\n".join(f"- {s}" for s in graph_results["node_summaries"][:10]))

        return "\n\n".join(context_parts)

    def _is_individual_entity(self, entity_type: str) -> bool:
        """Bestimmen, ob die Entität ein individueller Typ ist"""
        return entity_type.lower() in self.INDIVIDUAL_ENTITY_TYPES

    def _is_group_entity(self, entity_type: str) -> bool:
        """Bestimmen, ob die Entität ein Gruppen-/Institutionstyp ist"""
        return entity_type.lower() in self.GROUP_ENTITY_TYPES

    def _generate_profile_with_llm(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str
    ) -> Dict[str, Any]:
        """
        LLM zur Generierung einer sehr detaillierten Persona verwenden

        Basierend auf dem Entitätstyp:
        - Individuelle Entitäten: spezifische Charakterprofile generieren
        - Gruppen-/Institutionsentitäten: repräsentative Kontoprofile generieren
        """

        is_individual = self._is_individual_entity(entity_type)

        if is_individual:
            prompt = self._build_individual_persona_prompt(
                entity_name, entity_type, entity_summary, entity_attributes, context
            )
        else:
            prompt = self._build_group_persona_prompt(
                entity_name, entity_type, entity_summary, entity_attributes, context
            )

        # Mehrfach versuchen, bis erfolgreich oder maximale Wiederholungsversuche erreicht
        max_attempts = 3
        last_error = None

        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt(is_individual)},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7 - (attempt * 0.1)  # Temperatur bei jedem Wiederholungsversuch senken
                    # max_tokens nicht setzen, LLM frei generieren lassen
                )

                content = response.choices[0].message.content

                # Prüfen, ob die Ausgabe abgeschnitten wurde (finish_reason ist nicht 'stop')
                finish_reason = response.choices[0].finish_reason
                if finish_reason == 'length':
                    logger.warning(f"LLM-Ausgabe abgeschnitten (Versuch {attempt+1}), Reparatur wird versucht...")
                    content = self._fix_truncated_json(content)

                # JSON parsen versuchen
                try:
                    result = json.loads(content)

                    # Pflichtfelder validieren
                    if "bio" not in result or not result["bio"]:
                        result["bio"] = entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}"
                    if "persona" not in result or not result["persona"]:
                        result["persona"] = entity_summary or f"{entity_name} ist ein(e) {entity_type}."

                    return result

                except json.JSONDecodeError as je:
                    logger.warning(f"JSON-Parsing fehlgeschlagen (Versuch {attempt+1}): {str(je)[:80]}")

                    # JSON-Reparatur versuchen
                    result = self._try_fix_json(content, entity_name, entity_type, entity_summary)
                    if result.get("_fixed"):
                        del result["_fixed"]
                        return result

                    last_error = je

            except Exception as e:
                logger.warning(f"LLM-Aufruf fehlgeschlagen (Versuch {attempt+1}): {str(e)[:80]}")
                last_error = e
                import time
                time.sleep(1 * (attempt + 1))  # Exponentielles Backoff

        logger.warning(f"LLM-Persona-Generierung fehlgeschlagen ({max_attempts} Versuche): {last_error}, regelbasierte Generierung wird verwendet")
        return self._generate_profile_rule_based(
            entity_name, entity_type, entity_summary, entity_attributes
        )

    def _fix_truncated_json(self, content: str) -> str:
        """Abgeschnittenes JSON reparieren (Ausgabe durch max_tokens-Limit abgeschnitten)"""
        import re

        # Wenn JSON abgeschnitten ist, versuchen es zu schließen
        content = content.strip()

        # Nicht geschlossene Klammern zählen
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')

        # Auf nicht geschlossene Zeichenketten prüfen
        # Einfache Prüfung: wenn letztes Zeichen nicht Komma oder schließende Klammer ist, könnte die Zeichenkette abgeschnitten sein
        if content and content[-1] not in '",}]':
            # Zeichenkette schließen versuchen
            content += '"'

        # Klammern schließen
        content += ']' * open_brackets
        content += '}' * open_braces

        return content

    def _try_fix_json(self, content: str, entity_name: str, entity_type: str, entity_summary: str = "") -> Dict[str, Any]:
        """Beschädigtes JSON reparieren versuchen"""
        import re

        # 1. Zuerst abgeschnittenen Fall reparieren versuchen
        content = self._fix_truncated_json(content)

        # 2. JSON-Teil extrahieren versuchen
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group()

            # 3. Zeilenumbruchprobleme in Zeichenketten behandeln
            # Alle Zeichenkettenwerte finden und Zeilenumbrüche ersetzen
            def fix_string_newlines(match):
                s = match.group(0)
                # Tatsächliche Zeilenumbrüche in der Zeichenkette durch Leerzeichen ersetzen
                s = s.replace('\n', ' ').replace('\r', ' ')
                # Überschüssige Leerzeichen ersetzen
                s = re.sub(r'\s+', ' ', s)
                return s

            # JSON-Zeichenkettenwerte abgleichen
            json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string_newlines, json_str)

            # 4. Parsen versuchen
            try:
                result = json.loads(json_str)
                result["_fixed"] = True
                return result
            except json.JSONDecodeError as e:
                # 5. Wenn immer noch fehlgeschlagen, aggressivere Reparatur versuchen
                try:
                    # Alle Steuerzeichen entfernen
                    json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
                    # Alle aufeinanderfolgenden Leerzeichen ersetzen
                    json_str = re.sub(r'\s+', ' ', json_str)
                    result = json.loads(json_str)
                    result["_fixed"] = True
                    return result
                except:
                    pass

        # 6. Teilinformationen aus dem Inhalt extrahieren versuchen
        bio_match = re.search(r'"bio"\s*:\s*"([^"]*)"', content)
        persona_match = re.search(r'"persona"\s*:\s*"([^"]*)', content)  # Könnte abgeschnitten sein

        bio = bio_match.group(1) if bio_match else (entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}")
        persona = persona_match.group(1) if persona_match else (entity_summary or f"{entity_name} ist ein(e) {entity_type}.")

        # Wenn sinnvoller Inhalt extrahiert wurde, als repariert markieren
        if bio_match or persona_match:
            logger.info(f"Teilinformationen aus beschädigtem JSON extrahiert")
            return {
                "bio": bio,
                "persona": persona,
                "_fixed": True
            }

        # 7. Vollständig fehlgeschlagen, Grundstruktur zurückgeben
        logger.warning(f"JSON-Reparatur fehlgeschlagen, Grundstruktur wird zurückgegeben")
        return {
            "bio": entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}",
            "persona": entity_summary or f"{entity_name} ist ein(e) {entity_type}."
        }

    def _get_system_prompt(self, is_individual: bool) -> str:
        """System-Prompt abrufen"""
        base_prompt = "Sie sind ein Experte für die Generierung von Social-Media-Benutzerprofilen. Generieren Sie detaillierte, realistische Personas für die Meinungssimulation, die die bestehende Realität maximal widerspiegeln. Sie müssen gültiges JSON-Format zurückgeben, wobei alle Zeichenkettenwerte keine unescapierten Zeilenumbrüche enthalten dürfen. Verwenden Sie Deutsch."
        return base_prompt

    def _build_individual_persona_prompt(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str
    ) -> str:
        """Detaillierten Persona-Prompt für individuelle Entitäten erstellen"""

        attrs_str = json.dumps(entity_attributes, ensure_ascii=False) if entity_attributes else "Keine"
        context_str = context[:3000] if context else "Kein zusätzlicher Kontext"

        return f"""Generieren Sie eine detaillierte Social-Media-Benutzerpersona für die Entität, die die bestehende Realität maximal widerspiegelt.

Entitätsname: {entity_name}
Entitätstyp: {entity_type}
Entitätszusammenfassung: {entity_summary}
Entitätsattribute: {attrs_str}

Kontextinformationen:
{context_str}

Bitte generieren Sie JSON mit den folgenden Feldern:

1. bio: Social-Media-Biografie, 200 Zeichen
2. persona: Detaillierte Persona-Beschreibung (2000 Wörter reiner Text), muss enthalten:
   - Grundinformationen (Alter, Beruf, Bildungshintergrund, Standort)
   - Persönlicher Hintergrund (wichtige Erfahrungen, Ereignisverbindungen, soziale Beziehungen)
   - Persönlichkeitsmerkmale (MBTI-Typ, Kernpersönlichkeit, emotionaler Ausdruck)
   - Social-Media-Verhalten (Beitragsfrequenz, Inhaltspräferenzen, Interaktionsstil, Sprachmerkmale)
   - Positionen und Ansichten (Einstellungen zu Themen, Inhalte die provozieren/berühren können)
   - Einzigartige Merkmale (Redewendungen, besondere Erfahrungen, persönliche Interessen)
   - Persönliche Erinnerungen (wichtiger Teil der Persona, stellen Sie die Verbindung dieser Person zu Ereignissen und ihre bestehenden Handlungen/Reaktionen bei Ereignissen vor)
3. age: Alter als Zahl (muss eine Ganzzahl sein)
4. gender: Geschlecht, muss auf Englisch sein: "male" oder "female"
5. mbti: MBTI-Typ (z. B. INTJ, ENFP)
6. country: Land (verwenden Sie Deutsch, z. B. "Deutschland")
7. profession: Beruf
8. interested_topics: Array von Interessensthemen

Wichtig:
- Alle Feldwerte müssen Zeichenketten oder Zahlen sein, verwenden Sie keine Zeilenumbrüche
- persona muss eine zusammenhängende Textbeschreibung sein
- Verwenden Sie Deutsch
- Inhalt muss mit den Entitätsinformationen übereinstimmen
- age muss eine gültige Ganzzahl sein, gender muss "male" oder "female" sein
"""

    def _build_group_persona_prompt(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str
    ) -> str:
        """Detaillierten Persona-Prompt für Gruppen-/Institutionsentitäten erstellen"""

        attrs_str = json.dumps(entity_attributes, ensure_ascii=False) if entity_attributes else "Keine"
        context_str = context[:3000] if context else "Kein zusätzlicher Kontext"

        return f"""Generieren Sie ein detailliertes Social-Media-Kontoprofil für eine institutionelle/Gruppenentität, das die bestehende Realität maximal widerspiegelt.

Entitätsname: {entity_name}
Entitätstyp: {entity_type}
Entitätszusammenfassung: {entity_summary}
Entitätsattribute: {attrs_str}

Kontextinformationen:
{context_str}

Bitte generieren Sie JSON mit den folgenden Feldern:

1. bio: Offizielles Konto-Bio, 200 Zeichen, professionell und angemessen
2. persona: Detaillierte Kontoprofil-Beschreibung (2000 Wörter reiner Text), muss enthalten:
   - Grundlegende Institutionsinformationen (offizieller Name, Organisationsnatur, Gründungshintergrund, Hauptfunktionen)
   - Kontopositionierung (Kontotyp, Zielgruppe, Kernfunktionen)
   - Sprechstil (Sprachmerkmale, häufige Ausdrücke, Tabuthemen)
   - Veröffentlichungsmerkmale (Inhaltstypen, Veröffentlichungsfrequenz, aktive Zeiträume)
   - Position und Haltung (offizielle Haltung zu Kernthemen, Umgang mit Kontroversen)
   - Besondere Hinweise (vertretene Gruppenprofile, betriebliche Gewohnheiten)
   - Institutionelle Erinnerungen (wichtiger Teil der institutionellen Persona, stellen Sie die Verbindung dieser Institution zu Ereignissen und ihre bestehenden Handlungen/Reaktionen bei Ereignissen vor)
3. age: Fest auf 30 (virtuelles Alter des institutionellen Kontos)
4. gender: Fest auf "other" (institutionelles Konto verwendet other zur Kennzeichnung als Nicht-Einzelperson)
5. mbti: MBTI-Typ zur Beschreibung des Kontostils, z. B. ISTJ steht für streng konservativ
6. country: Land (verwenden Sie Deutsch, z. B. "Deutschland")
7. profession: Beschreibung der institutionellen Funktion
8. interested_topics: Array von Schwerpunktbereichen

Wichtig:
- Alle Feldwerte müssen Zeichenketten oder Zahlen sein, keine Null-Werte erlaubt
- persona muss eine zusammenhängende Textbeschreibung sein, verwenden Sie keine Zeilenumbrüche
- Verwenden Sie Deutsch
- age muss die Ganzzahl 30 sein, gender muss die Zeichenkette "other" sein
- Institutionelle Kontosprache muss mit ihrer Identitätspositionierung übereinstimmen"""

    def _generate_profile_rule_based(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Grundlegende Persona regelbasiert generieren"""

        # Verschiedene Personas basierend auf Entitätstyp generieren
        entity_type_lower = entity_type.lower()

        if entity_type_lower in ["student", "alumni"]:
            return {
                "bio": f"{entity_type} mit Interessen in Wissenschaft und sozialen Themen.",
                "persona": f"{entity_name} ist ein(e) {entity_type.lower()}, der/die aktiv an akademischen und sozialen Diskussionen teilnimmt. Er/Sie teilt gerne Perspektiven und vernetzt sich mit Gleichgesinnten.",
                "age": random.randint(18, 30),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(self.MBTI_TYPES),
                "country": random.choice(self.COUNTRIES),
                "profession": "Student/Studentin",
                "interested_topics": ["Bildung", "Soziale Themen", "Technologie"],
            }

        elif entity_type_lower in ["publicfigure", "expert", "faculty"]:
            return {
                "bio": f"Experte und Meinungsführer in seinem/ihrem Fachgebiet.",
                "persona": f"{entity_name} ist ein(e) anerkannte(r) {entity_type.lower()}, der/die Einblicke und Meinungen zu wichtigen Themen teilt. Er/Sie ist bekannt für Expertise und Einfluss im öffentlichen Diskurs.",
                "age": random.randint(35, 60),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(["ENTJ", "INTJ", "ENTP", "INTP"]),
                "country": random.choice(self.COUNTRIES),
                "profession": entity_attributes.get("occupation", "Experte"),
                "interested_topics": ["Politik", "Wirtschaft", "Kultur & Gesellschaft"],
            }

        elif entity_type_lower in ["mediaoutlet", "socialmediaplatform"]:
            return {
                "bio": f"Offizielles Konto von {entity_name}. Nachrichten und Aktualisierungen.",
                "persona": f"{entity_name} ist eine Medienentität, die Nachrichten berichtet und den öffentlichen Diskurs fördert. Das Konto teilt aktuelle Nachrichten und interagiert mit dem Publikum über aktuelle Ereignisse.",
                "age": 30,  # Institutionelles virtuelles Alter
                "gender": "other",  # Institutionell verwendet other
                "mbti": "ISTJ",  # Institutioneller Stil: streng konservativ
                "country": "Deutschland",
                "profession": "Medien",
                "interested_topics": ["Allgemeine Nachrichten", "Aktuelle Ereignisse", "Öffentliche Angelegenheiten"],
            }

        elif entity_type_lower in ["university", "governmentagency", "ngo", "organization"]:
            return {
                "bio": f"Offizielles Konto von {entity_name}.",
                "persona": f"{entity_name} ist eine institutionelle Entität, die offizielle Positionen und Ankündigungen kommuniziert und mit Interessengruppen zu relevanten Themen interagiert.",
                "age": 30,  # Institutionelles virtuelles Alter
                "gender": "other",  # Institutionell verwendet other
                "mbti": "ISTJ",  # Institutioneller Stil: streng konservativ
                "country": "Deutschland",
                "profession": entity_type,
                "interested_topics": ["Öffentliche Politik", "Gemeinschaft", "Offizielle Ankündigungen"],
            }

        else:
            # Standard-Persona
            return {
                "bio": entity_summary[:150] if entity_summary else f"{entity_type}: {entity_name}",
                "persona": entity_summary or f"{entity_name} ist ein(e) {entity_type.lower()}, der/die an sozialen Diskussionen teilnimmt.",
                "age": random.randint(25, 50),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(self.MBTI_TYPES),
                "country": random.choice(self.COUNTRIES),
                "profession": entity_type,
                "interested_topics": ["Allgemeines", "Soziale Themen"],
            }

    def set_graph_id(self, graph_id: str):
        """Wissensgraph-ID für die Wissensgraph-Suche setzen"""
        self.graph_id = graph_id

    def generate_profiles_from_entities(
        self,
        entities: List[EntityNode],
        use_llm: bool = True,
        progress_callback: Optional[callable] = None,
        graph_id: Optional[str] = None,
        parallel_count: int = 5,
        realtime_output_path: Optional[str] = None,
        output_platform: str = "reddit"
    ) -> List[OasisAgentProfile]:
        """
        Agent-Profile in Stapeln aus Entitäten generieren (unterstützt parallele Generierung)

        Args:
            entities: Entitätsliste
            use_llm: Ob LLM zur Generierung detaillierter Personas verwendet werden soll
            progress_callback: Fortschritts-Callback-Funktion (aktuell, gesamt, nachricht)
            graph_id: Wissensgraph-ID für Wissensgraph-Suche zum Erhalt reichhaltigeren Kontexts
            parallel_count: Anzahl paralleler Generierungen, Standard 5
            realtime_output_path: Echtzeit-Ausgabedateipfad (wenn angegeben, nach jeder Generierung schreiben)
            output_platform: Ausgabeplattform-Format ("reddit" oder "twitter")

        Returns:
            Liste von Agent-Profilen
        """
        import concurrent.futures
        from threading import Lock

        # graph_id für Wissensgraph-Suche setzen
        if graph_id:
            self.graph_id = graph_id

        total = len(entities)
        profiles = [None] * total  # Liste vorab zuweisen, um Reihenfolge beizubehalten
        completed_count = [0]  # Liste verwenden für Änderung im Closure
        lock = Lock()

        # Hilfsfunktion für Echtzeit-Dateischreiben
        def save_profiles_realtime():
            """Generierte Profile in Echtzeit in Datei speichern"""
            if not realtime_output_path:
                return

            with lock:
                # Generierte Profile filtern
                existing_profiles = [p for p in profiles if p is not None]
                if not existing_profiles:
                    return

                try:
                    if output_platform == "reddit":
                        # Reddit JSON-Format
                        profiles_data = [p.to_reddit_format() for p in existing_profiles]
                        with open(realtime_output_path, 'w', encoding='utf-8') as f:
                            json.dump(profiles_data, f, ensure_ascii=False, indent=2)
                    else:
                        # Twitter CSV-Format
                        import csv
                        profiles_data = [p.to_twitter_format() for p in existing_profiles]
                        if profiles_data:
                            fieldnames = list(profiles_data[0].keys())
                            with open(realtime_output_path, 'w', encoding='utf-8', newline='') as f:
                                writer = csv.DictWriter(f, fieldnames=fieldnames)
                                writer.writeheader()
                                writer.writerows(profiles_data)
                except Exception as e:
                    logger.warning(f"Echtzeit-Profilspeicherung fehlgeschlagen: {e}")

        def generate_single_profile(idx: int, entity: EntityNode) -> tuple:
            """Worker-Funktion zur Generierung eines einzelnen Profils"""
            entity_type = entity.get_entity_type() or "Entity"

            try:
                profile = self.generate_profile_from_entity(
                    entity=entity,
                    user_id=idx,
                    use_llm=use_llm
                )

                # Generierte Persona in Echtzeit auf der Konsole und im Log ausgeben
                self._print_generated_profile(entity.name, entity_type, profile)

                return idx, profile, None

            except Exception as e:
                logger.error(f"Persona-Generierung für Entität {entity.name} fehlgeschlagen: {str(e)}")
                # Fallback-Profil erstellen
                fallback_profile = OasisAgentProfile(
                    user_id=idx,
                    user_name=self._generate_username(entity.name),
                    name=entity.name,
                    bio=f"{entity_type}: {entity.name}",
                    persona=entity.summary or f"Ein Teilnehmer an sozialen Diskussionen.",
                    source_entity_uuid=entity.uuid,
                    source_entity_type=entity_type,
                )
                return idx, fallback_profile, str(e)

        logger.info(f"Parallele Generierung von {total} Agent-Personas wird gestartet (parallele Anzahl: {parallel_count})...")
        print(f"\n{'='*60}")
        print(f"Agent-Persona-Generierung wird gestartet - {total} Entitäten insgesamt, parallele Anzahl: {parallel_count}")
        print(f"{'='*60}\n")

        # Thread-Pool für parallele Ausführung verwenden
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_count) as executor:
            # Alle Aufgaben einreichen
            future_to_entity = {
                executor.submit(generate_single_profile, idx, entity): (idx, entity)
                for idx, entity in enumerate(entities)
            }

            # Ergebnisse sammeln
            for future in concurrent.futures.as_completed(future_to_entity):
                idx, entity = future_to_entity[future]
                entity_type = entity.get_entity_type() or "Entity"

                try:
                    result_idx, profile, error = future.result()
                    profiles[result_idx] = profile

                    with lock:
                        completed_count[0] += 1
                        current = completed_count[0]

                    # Echtzeit-Dateischreiben
                    save_profiles_realtime()

                    if progress_callback:
                        progress_callback(
                            current,
                            total,
                            f"Abgeschlossen {current}/{total}: {entity.name} ({entity_type})"
                        )

                    if error:
                        logger.warning(f"[{current}/{total}] {entity.name} verwendet Fallback-Persona: {error}")
                    else:
                        logger.info(f"[{current}/{total}] Persona erfolgreich generiert: {entity.name} ({entity_type})")

                except Exception as e:
                    logger.error(f"Ausnahme bei der Verarbeitung von Entität {entity.name} aufgetreten: {str(e)}")
                    with lock:
                        completed_count[0] += 1
                    profiles[idx] = OasisAgentProfile(
                        user_id=idx,
                        user_name=self._generate_username(entity.name),
                        name=entity.name,
                        bio=f"{entity_type}: {entity.name}",
                        persona=entity.summary or "Ein Teilnehmer an sozialen Diskussionen.",
                        source_entity_uuid=entity.uuid,
                        source_entity_type=entity_type,
                    )
                    # Echtzeit-Dateischreiben (auch für Fallback-Personas)
                    save_profiles_realtime()

        print(f"\n{'='*60}")
        print(f"Persona-Generierung abgeschlossen! {len([p for p in profiles if p])} Agents generiert")
        print(f"{'='*60}\n")

        return profiles

    def _print_generated_profile(self, entity_name: str, entity_type: str, profile: OasisAgentProfile):
        """Generierte Persona in Echtzeit auf der Konsole ausgeben (vollständiger Inhalt, nicht gekürzt)"""
        separator = "-" * 70

        # Vollständigen Ausgabeinhalt erstellen (nicht gekürzt)
        topics_str = ', '.join(profile.interested_topics) if profile.interested_topics else 'Keine'

        output_lines = [
            f"\n{separator}",
            f"[Generiert] {entity_name} ({entity_type})",
            f"{separator}",
            f"Benutzername: {profile.user_name}",
            f"",
            f"[Biografie]",
            f"{profile.bio}",
            f"",
            f"[Detaillierte Persona]",
            f"{profile.persona}",
            f"",
            f"[Grundattribute]",
            f"Alter: {profile.age} | Geschlecht: {profile.gender} | MBTI: {profile.mbti}",
            f"Beruf: {profile.profession} | Land: {profile.country}",
            f"Interessensthemen: {topics_str}",
            separator
        ]

        output = "\n".join(output_lines)

        # Nur auf der Konsole ausgeben (Duplizierung vermeiden, Logger gibt keinen vollständigen Inhalt mehr aus)
        print(output)

    def save_profiles(
        self,
        profiles: List[OasisAgentProfile],
        file_path: str,
        platform: str = "reddit"
    ):
        """
        Profile in Datei speichern (korrektes Format basierend auf Plattform wählen)

        OASIS-Plattform-Formatanforderungen:
        - Twitter: CSV-Format
        - Reddit: JSON-Format

        Args:
            profiles: Profilliste
            file_path: Dateipfad
            platform: Plattformtyp ("reddit" oder "twitter")
        """
        if platform == "twitter":
            self._save_twitter_csv(profiles, file_path)
        else:
            self._save_reddit_json(profiles, file_path)

    def _save_twitter_csv(self, profiles: List[OasisAgentProfile], file_path: str):
        """
        Twitter-Profil als CSV-Format speichern (konform mit offiziellen OASIS-Anforderungen)

        OASIS Twitter benötigte CSV-Felder:
        - user_id: Benutzer-ID (beginnend bei 0 basierend auf CSV-Reihenfolge)
        - name: Klarname des Benutzers
        - username: Benutzername im System
        - user_char: Detaillierte Persona-Beschreibung (wird in LLM-System-Prompt injiziert, steuert Agent-Verhalten)
        - description: Kurze öffentliche Biografie (auf der Profilseite angezeigt)

        Unterschied user_char vs description:
        - user_char: Interne Verwendung, LLM-System-Prompt, bestimmt wie der Agent denkt und handelt
        - description: Externe Anzeige, für andere Benutzer sichtbar
        """
        import csv

        # Sicherstellen, dass die Dateierweiterung .csv ist
        if not file_path.endswith('.csv'):
            file_path = file_path.replace('.json', '.csv')

        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # OASIS-benötigten Header schreiben
            headers = ['user_id', 'name', 'username', 'user_char', 'description']
            writer.writerow(headers)

            # Datenzeilen schreiben
            for idx, profile in enumerate(profiles):
                # user_char: Vollständige Persona (bio + persona) für LLM-System-Prompt
                user_char = profile.bio
                if profile.persona and profile.persona != profile.bio:
                    user_char = f"{profile.bio} {profile.persona}"
                # Zeilenumbrüche behandeln (durch Leerzeichen ersetzen in CSV)
                user_char = user_char.replace('\n', ' ').replace('\r', ' ')

                # description: Kurze Biografie für externe Anzeige
                description = profile.bio.replace('\n', ' ').replace('\r', ' ')

                row = [
                    idx,                    # user_id: Fortlaufende ID beginnend bei 0
                    profile.name,           # name: Klarname
                    profile.user_name,      # username: Benutzername
                    user_char,              # user_char: Vollständige Persona (interne LLM-Verwendung)
                    description             # description: Kurze Biografie (externe Anzeige)
                ]
                writer.writerow(row)

        logger.info(f"{len(profiles)} Twitter-Profile in {file_path} gespeichert (OASIS CSV-Format)")

    def _normalize_gender(self, gender: Optional[str]) -> str:
        """
        Geschlechtsfeld in das von OASIS benötigte englische Format normalisieren

        OASIS erfordert: male, female, other
        """
        if not gender:
            return "other"

        gender_lower = gender.lower().strip()

        # Geschlechtszuordnung
        gender_map = {
            "male": "male",
            "female": "female",
            "other": "other",
        }

        return gender_map.get(gender_lower, "other")

    def _save_reddit_json(self, profiles: List[OasisAgentProfile], file_path: str):
        """
        Reddit-Profil als JSON-Format speichern

        Format konsistent mit to_reddit_format() verwenden, um sicherzustellen, dass OASIS korrekt lesen kann.
        Muss user_id-Feld enthalten, das ist der Schlüssel für OASIS agent_graph.get_agent() Zuordnung!

        Benötigte Felder:
        - user_id: Benutzer-ID (Ganzzahl, zum Abgleich mit poster_agent_id in initial_posts)
        - username: Benutzername
        - name: Anzeigename
        - bio: Biografie
        - persona: Detaillierte Persona
        - age: Alter (Ganzzahl)
        - gender: "male", "female" oder "other"
        - mbti: MBTI-Typ
        - country: Land
        """
        data = []
        for idx, profile in enumerate(profiles):
            # Format konsistent mit to_reddit_format() verwenden
            item = {
                "user_id": profile.user_id if profile.user_id is not None else idx,  # Schlüssel: muss user_id enthalten
                "username": profile.user_name,
                "name": profile.name,
                "bio": profile.bio[:150] if profile.bio else f"{profile.name}",
                "persona": profile.persona or f"{profile.name} ist ein Teilnehmer an sozialen Diskussionen.",
                "karma": profile.karma if profile.karma else 1000,
                "created_at": profile.created_at,
                # OASIS-benötigte Felder - sicherstellen, dass alle Standardwerte haben
                "age": profile.age if profile.age else 30,
                "gender": self._normalize_gender(profile.gender),
                "mbti": profile.mbti if profile.mbti else "ISTJ",
                "country": profile.country if profile.country else "Deutschland",
            }

            # Optionale Felder
            if profile.profession:
                item["profession"] = profile.profession
            if profile.interested_topics:
                item["interested_topics"] = profile.interested_topics

            data.append(item)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"{len(profiles)} Reddit-Profile in {file_path} gespeichert (JSON-Format, enthält user_id-Feld)")

    # Alten Methodennamen als Alias für Abwärtskompatibilität beibehalten
    def save_profiles_to_json(
        self,
        profiles: List[OasisAgentProfile],
        file_path: str,
        platform: str = "reddit"
    ):
        """[Veraltet] Bitte verwenden Sie die Methode save_profiles()"""
        logger.warning("save_profiles_to_json ist veraltet, bitte verwenden Sie die Methode save_profiles")
        self.save_profiles(profiles, file_path, platform)
