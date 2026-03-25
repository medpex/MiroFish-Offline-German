"""
Ontologie-Generierungsdienst
Schnittstelle 1: Textinhalt analysieren und Entitäts- sowie Beziehungstypdefinitionen generieren, die für die soziale Simulation geeignet sind
"""

import json
from typing import Dict, Any, List, Optional
from ..utils.llm_client import LLMClient


# System-Prompt für die Ontologie-Generierung
ONTOLOGY_SYSTEM_PROMPT = """Sie sind ein professioneller Experte für Wissensgraph-Ontologie-Design. Ihre Aufgabe ist es, gegebene Textinhalte und Simulationsanforderungen zu analysieren und Entitätstypen sowie Beziehungstypen zu entwerfen, die für die **Simulation von Meinungsbildung in sozialen Medien** geeignet sind.

**Wichtig: Sie müssen gültige JSON-Format-Daten ausgeben, geben Sie nichts anderes aus.**

## Kernaufgabe – Hintergrund

Wir erstellen ein **System zur Simulation von Meinungsbildung in sozialen Medien**. In diesem System:
- Jede Entität ist ein „Konto" oder „Subjekt", das in sozialen Medien Meinungen äußern, interagieren und Informationen verbreiten kann
- Entitäten beeinflussen sich gegenseitig, teilen Beiträge, kommentieren und antworten
- Wir müssen die Reaktionen verschiedener Parteien bei Meinungsereignissen und Informationsverbreitungswege simulieren

Daher **müssen Entitäten reale Akteure sein, die in sozialen Medien Meinungen äußern und interagieren können**:

**Zulässig**:
- Bestimmte Einzelpersonen (öffentliche Persönlichkeiten, Interessenvertreter, Meinungsführer, Experten, gewöhnliche Menschen)
- Unternehmen und Firmen (einschließlich ihrer offiziellen Konten)
- Organisationen (Universitäten, Verbände, NGOs, Gewerkschaften usw.)
- Regierungsbehörden und Aufsichtsbehörden
- Medieninstitutionen (Zeitungen, Fernsehsender, Influencer-Medien, Websites)
- Social-Media-Plattformen selbst
- Spezifische Gruppenvertreter (wie Alumni-Vereinigungen, Fangruppen, Interessengruppen usw.)

**Nicht zulässig**:
- Abstrakte Konzepte (wie „öffentliche Meinung", „Emotion", „Trend")
- Themen/Gegenstände (wie „akademische Integrität", „Bildungsreform")
- Ansichten/Haltungen (wie „Befürworter", „Gegner")

## Ausgabeformat

Bitte geben Sie JSON im folgenden Format aus:

```json
{
    "entity_types": [
        {
            "name": "Entitätstypname (Englisch, PascalCase)",
            "description": "Kurzbeschreibung (Englisch, maximal 100 Zeichen)",
            "attributes": [
                {
                    "name": "Attributname (Englisch, snake_case)",
                    "type": "text",
                    "description": "Attributbeschreibung"
                }
            ],
            "examples": ["Beispielentität 1", "Beispielentität 2"]
        }
    ],
    "edge_types": [
        {
            "name": "Beziehungstypname (Englisch, UPPER_SNAKE_CASE)",
            "description": "Kurzbeschreibung (Englisch, maximal 100 Zeichen)",
            "source_targets": [
                {"source": "Quellentitätstyp", "target": "Zielentitätstyp"}
            ],
            "attributes": []
        }
    ],
    "analysis_summary": "Kurze Analyse und Erläuterung des Textinhalts"
}
```

## Designrichtlinien (Äußerst wichtig!)

### 1. Entitätstyp-Design – Strikt zu befolgen

**Mengenanforderung: Es müssen genau 10 Entitätstypen vorhanden sein**

**Hierarchische Strukturanforderung (muss sowohl spezifische Typen als auch Fallback-Typen enthalten)**:

Ihre 10 Entitätstypen müssen die folgende Hierarchie umfassen:

A. **Fallback-Typen (müssen enthalten sein, an den letzten 2 Positionen der Liste)**:
   - `Person`: Fallback-Typ für jede natürliche Person. Wenn eine Person nicht zu anderen spezifischeren Personentypen passt, verwenden Sie diesen.
   - `Organization`: Fallback-Typ für jede Organisation. Wenn eine Organisation nicht zu anderen spezifischeren Organisationstypen passt, verwenden Sie diesen.

B. **Spezifische Typen (8, basierend auf dem Textinhalt entworfen)**:
   - Entwerfen Sie spezifischere Typen für die im Text vorkommenden Hauptfiguren
   - Beispiel: Wenn der Text akademische Ereignisse betrifft, können `Student`, `Professor`, `University` verwendet werden
   - Beispiel: Wenn der Text geschäftliche Ereignisse betrifft, können `Company`, `CEO`, `Employee` verwendet werden

**Warum Fallback-Typen benötigt werden**:
- Im Text erscheinen verschiedene Personen, wie „Lehrer", „zufällige Person", „ein Internetnutzer"
- Wenn kein spezifischer Typ passt, sollten sie als `Person` klassifiziert werden
- Ebenso sollten kleine Organisationen und temporäre Gruppen als `Organization` klassifiziert werden

**Designprinzipien für spezifische Typen**:
- Identifizieren Sie häufig vorkommende oder Schlüsselrollentypen aus dem Text
- Jeder spezifische Typ sollte klare Grenzen haben, Überschneidungen vermeiden
- Die Beschreibung muss den Unterschied zwischen diesem Typ und dem Fallback-Typ klar erklären

### 2. Beziehungstyp-Design

- Menge: 6-10
- Beziehungen sollten reale Verbindungen in Social-Media-Interaktionen widerspiegeln
- Stellen Sie sicher, dass die Beziehungs-source_targets Ihre definierten Entitätstypen abdecken

### 3. Attribut-Design

- 1-3 Schlüsselattribute pro Entitätstyp
- **Hinweis**: Attributnamen dürfen nicht `name`, `uuid`, `group_id`, `created_at`, `summary` verwenden (dies sind vom System reservierte Wörter)
- Empfohlen: `full_name`, `title`, `role`, `position`, `location`, `description` usw.

## Entitätstyp-Referenz

**Einzelpersonentypen (spezifisch)**:
- Student: Student/Studentin
- Professor: Professor/Wissenschaftler
- Journalist: Journalist/Journalistin
- Celebrity: Prominente/Internet-Prominenz
- Executive: Führungskraft
- Official: Regierungsbeamter
- Lawyer: Rechtsanwalt/Rechtsanwältin
- Doctor: Arzt/Ärztin

**Einzelpersonentypen (Fallback)**:
- Person: Jede natürliche Person (verwenden, wenn sie nicht zu anderen spezifischen Typen passt)

**Organisationstypen (spezifisch)**:
- University: Universität
- Company: Unternehmen/Firma
- GovernmentAgency: Regierungsbehörde
- MediaOutlet: Medieninstitution
- Hospital: Krankenhaus
- School: Grund-/Sekundarschule
- NGO: Nichtregierungsorganisation

**Organisationstypen (Fallback)**:
- Organization: Jede Organisation (verwenden, wenn sie nicht zu anderen spezifischen Typen passt)

## Beziehungstyp-Referenz

- WORKS_FOR: Arbeitet für
- STUDIES_AT: Studiert an
- AFFILIATED_WITH: Zugehörig zu
- REPRESENTS: Vertritt
- REGULATES: Reguliert
- REPORTS_ON: Berichtet über
- COMMENTS_ON: Kommentiert
- RESPONDS_TO: Antwortet auf
- SUPPORTS: Unterstützt
- OPPOSES: Widerspricht
- COLLABORATES_WITH: Arbeitet zusammen mit
- COMPETES_WITH: Konkurriert mit
"""


class OntologyGenerator:
    """
    Ontologie-Generator
    Textinhalt analysieren und Entitäts- sowie Beziehungstypdefinitionen generieren
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    def generate(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ontologie-Definition generieren

        Args:
            document_texts: Liste der Dokumenttexte
            simulation_requirement: Beschreibung der Simulationsanforderungen
            additional_context: Zusätzlicher Kontext

        Returns:
            Ontologie-Definition (entity_types, edge_types usw.)
        """
        # Benutzernachricht erstellen
        user_message = self._build_user_message(
            document_texts,
            simulation_requirement,
            additional_context
        )

        messages = [
            {"role": "system", "content": ONTOLOGY_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]

        # LLM aufrufen
        result = self.llm_client.chat_json(
            messages=messages,
            temperature=0.3,
            max_tokens=4096
        )

        # Validieren und nachbearbeiten
        result = self._validate_and_process(result)

        return result

    # Maximale Textlänge für LLM (50.000 Zeichen)
    MAX_TEXT_LENGTH_FOR_LLM = 50000

    def _build_user_message(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str]
    ) -> str:
        """Benutzernachricht erstellen"""

        # Texte zusammenführen
        combined_text = "\n\n---\n\n".join(document_texts)
        original_length = len(combined_text)

        # Wenn der Text 50.000 Zeichen überschreitet, kürzen (betrifft nur LLM-Eingabe, nicht die Graph-Konstruktion)
        if len(combined_text) > self.MAX_TEXT_LENGTH_FOR_LLM:
            combined_text = combined_text[:self.MAX_TEXT_LENGTH_FOR_LLM]
            combined_text += f"\n\n...(Originaltext hat {original_length} Zeichen, die ersten {self.MAX_TEXT_LENGTH_FOR_LLM} Zeichen wurden für die Ontologie-Analyse extrahiert)..."

        message = f"""## Simulationsanforderungen

{simulation_requirement}

## Dokumentinhalt

{combined_text}
"""

        if additional_context:
            message += f"""
## Zusätzliche Erläuterung

{additional_context}
"""

        message += """
Entwerfen Sie basierend auf dem obigen Inhalt Entitätstypen und Beziehungstypen, die für die Simulation der Meinungsbildung in sozialen Medien geeignet sind.

**Zu befolgende Regeln**:
1. Es müssen genau 10 Entitätstypen ausgegeben werden
2. Die letzten 2 müssen Fallback-Typen sein: Person (Einzelperson-Fallback) und Organization (Organisation-Fallback)
3. Die ersten 8 sind spezifische Typen, die basierend auf dem Textinhalt entworfen werden
4. Alle Entitätstypen müssen reale Subjekte sein, die Meinungen äußern können, keine abstrakten Konzepte
5. Attributnamen dürfen keine reservierten Wörter wie name, uuid, group_id verwenden, verwenden Sie stattdessen full_name, org_name usw.
"""

        return message

    def _validate_and_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Ergebnis validieren und nachbearbeiten"""

        # Sicherstellen, dass notwendige Felder vorhanden sind
        if "entity_types" not in result:
            result["entity_types"] = []
        if "edge_types" not in result:
            result["edge_types"] = []
        if "analysis_summary" not in result:
            result["analysis_summary"] = ""

        # Entitätstypen validieren
        for entity in result["entity_types"]:
            if "attributes" not in entity:
                entity["attributes"] = []
            if "examples" not in entity:
                entity["examples"] = []
            # Sicherstellen, dass die Beschreibung 100 Zeichen nicht überschreitet
            if len(entity.get("description", "")) > 100:
                entity["description"] = entity["description"][:97] + "..."

        # Beziehungstypen validieren
        for edge in result["edge_types"]:
            if "source_targets" not in edge:
                edge["source_targets"] = []
            if "attributes" not in edge:
                edge["attributes"] = []
            if len(edge.get("description", "")) > 100:
                edge["description"] = edge["description"][:97] + "..."

        # Zep-API-Limit: maximal 10 benutzerdefinierte Entitätstypen, maximal 10 benutzerdefinierte Edge-Typen
        MAX_ENTITY_TYPES = 10
        MAX_EDGE_TYPES = 10

        # Fallback-Typdefinitionen
        person_fallback = {
            "name": "Person",
            "description": "Any individual person not fitting other specific person types.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Vollständiger Name der Person"},
                {"name": "role", "type": "text", "description": "Rolle oder Beruf"}
            ],
            "examples": ["gewöhnlicher Bürger", "anonymer Internetnutzer"]
        }

        organization_fallback = {
            "name": "Organization",
            "description": "Any organization not fitting other specific organization types.",
            "attributes": [
                {"name": "org_name", "type": "text", "description": "Name der Organisation"},
                {"name": "org_type", "type": "text", "description": "Art der Organisation"}
            ],
            "examples": ["Kleinunternehmen", "Gemeinschaftsgruppe"]
        }

        # Prüfen, ob Fallback-Typen bereits vorhanden sind
        entity_names = {e["name"] for e in result["entity_types"]}
        has_person = "Person" in entity_names
        has_organization = "Organization" in entity_names

        # Hinzuzufügende Fallback-Typen
        fallbacks_to_add = []
        if not has_person:
            fallbacks_to_add.append(person_fallback)
        if not has_organization:
            fallbacks_to_add.append(organization_fallback)

        if fallbacks_to_add:
            current_count = len(result["entity_types"])
            needed_slots = len(fallbacks_to_add)

            # Wenn das Hinzufügen 10 überschreiten würde, müssen einige vorhandene Typen entfernt werden
            if current_count + needed_slots > MAX_ENTITY_TYPES:
                # Berechnen, wie viele entfernt werden müssen
                to_remove = current_count + needed_slots - MAX_ENTITY_TYPES
                # Vom Ende entfernen (wichtigere spezifische Typen vorne behalten)
                result["entity_types"] = result["entity_types"][:-to_remove]

            # Fallback-Typen hinzufügen
            result["entity_types"].extend(fallbacks_to_add)

        # Abschließende Prüfung, um sicherzustellen, dass Limits nicht überschritten werden (defensive Programmierung)
        if len(result["entity_types"]) > MAX_ENTITY_TYPES:
            result["entity_types"] = result["entity_types"][:MAX_ENTITY_TYPES]

        if len(result["edge_types"]) > MAX_EDGE_TYPES:
            result["edge_types"] = result["edge_types"][:MAX_EDGE_TYPES]

        return result

    def generate_python_code(self, ontology: Dict[str, Any]) -> str:
        """
        [VERALTET] Ontologie-Definition in Zep-Format-Pydantic-Code konvertieren.
        Wird in MiroFish-Offline nicht verwendet (Ontologie wird als JSON in Neo4j gespeichert).
        Nur als Referenz beibehalten.
        """
        code_lines = [
            '"""',
            'Benutzerdefinierte Entitätstyp-Definitionen',
            'Automatisch generiert von MiroFish für die Simulation der Meinungsbildung in sozialen Medien',
            '"""',
            '',
            'from pydantic import Field',
            'from zep_cloud.external_clients.ontology import EntityModel, EntityText, EdgeModel',
            '',
            '',
            '# ============== Entitätstyp-Definitionen ==============',
            '',
        ]

        # Entitätstypen generieren
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            desc = entity.get("description", f"A {name} entity.")

            code_lines.append(f'class {name}(EntityModel):')
            code_lines.append(f'    """{desc}"""')

            attrs = entity.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')

            code_lines.append('')
            code_lines.append('')

        code_lines.append('# ============== Beziehungstyp-Definitionen ==============')
        code_lines.append('')

        # Beziehungstypen generieren
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            # In PascalCase-Klassennamen konvertieren
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            desc = edge.get("description", f"A {name} relationship.")

            code_lines.append(f'class {class_name}(EdgeModel):')
            code_lines.append(f'    """{desc}"""')

            attrs = edge.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')

            code_lines.append('')
            code_lines.append('')

        # Typ-Wörterbücher generieren
        code_lines.append('# ============== Typ-Konfiguration ==============')
        code_lines.append('')
        code_lines.append('ENTITY_TYPES = {')
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            code_lines.append(f'    "{name}": {name},')
        code_lines.append('}')
        code_lines.append('')
        code_lines.append('EDGE_TYPES = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            code_lines.append(f'    "{name}": {class_name},')
        code_lines.append('}')
        code_lines.append('')

        # source_targets-Zuordnung für Edges generieren
        code_lines.append('EDGE_SOURCE_TARGETS = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            source_targets = edge.get("source_targets", [])
            if source_targets:
                st_list = ', '.join([
                    f'{{"source": "{st.get("source", "Entity")}", "target": "{st.get("target", "Entity")}"}}'
                    for st in source_targets
                ])
                code_lines.append(f'    "{name}": [{st_list}],')
        code_lines.append('}')

        return '\n'.join(code_lines)
