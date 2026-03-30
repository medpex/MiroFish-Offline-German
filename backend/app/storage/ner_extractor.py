"""
NER/RE-Extraktor — Entitäts- und Beziehungsextraktion mittels lokalem LLM

Ersetzt die integrierte NER/RE-Pipeline von Zep Cloud.
Verwendet LLMClient.chat_json() mit einem strukturierten Prompt zur Extraktion
von Entitäten und Beziehungen aus Textabschnitten, gesteuert durch die Graph-Ontologie.
"""

import logging
from typing import Dict, Any, List, Optional

from ..utils.llm_client import LLMClient

logger = logging.getLogger('mirofish.ner_extractor')

# System-Prompt-Vorlage für die Erkennung benannter Entitäten und Beziehungsextraktion
_SYSTEM_PROMPT = """Sie sind ein System zur Erkennung benannter Entitäten und Beziehungsextraktion.
Gegeben einen Text und eine Ontologie (Entitätstypen + Beziehungstypen), extrahieren Sie alle Entitäten und Beziehungen.

ONTOLOGIE:
{ontology_description}

REGELN:
1. Extrahieren Sie nur Entitätstypen und Beziehungstypen, die in der Ontologie definiert sind.
2. Normalisieren Sie Entitätsnamen: Entfernen Sie Leerzeichen, verwenden Sie die kanonische Form (z. B. "Max Mustermann" statt "mustermann max").
3. Jede Entität muss enthalten: Name, Typ (aus der Ontologie) und optionale Attribute.
4. Jede Beziehung muss enthalten: Name der Quellentität, Name der Zielentität, Typ (aus der Ontologie) und einen Faktensatz, der die Beziehung beschreibt.
5. Wenn keine Entitäten oder Beziehungen gefunden werden, geben Sie leere Listen zurück.
6. Seien Sie präzise — extrahieren Sie nur das, was explizit im Text angegeben oder stark impliziert ist.

Geben Sie NUR gültiges JSON in diesem exakten Format zurück:
{{
  "entities": [
    {{"name": "...", "type": "...", "attributes": {{"key": "value"}}}}
  ],
  "relations": [
    {{"source": "...", "target": "...", "type": "...", "fact": "..."}}
  ]
}}"""

_USER_PROMPT = """Extrahieren Sie Entitäten und Beziehungen aus dem folgenden Text:

{text}"""


class NERExtractor:
    """Entitäten und Beziehungen aus Text mittels lokalem LLM extrahieren."""

    def __init__(self, llm_client: Optional[LLMClient] = None, max_retries: int = 2):
        self.llm = llm_client or LLMClient()
        self.max_retries = max_retries

    def extract(self, text: str, ontology: Dict[str, Any]) -> Dict[str, Any]:
        """
        Entitäten und Beziehungen aus Text extrahieren, gesteuert durch die Ontologie.

        Args:
            text: Eingabetextabschnitt
            ontology: Dict mit 'entity_types' und 'relation_types' aus dem Graph

        Returns:
            Dict mit Listen 'entities' und 'relations':
            {
                "entities": [{"name": str, "type": str, "attributes": dict}],
                "relations": [{"source": str, "target": str, "type": str, "fact": str}]
            }
        """
        if not text or not text.strip():
            return {"entities": [], "relations": []}

        ontology_desc = self._format_ontology(ontology)
        system_msg = _SYSTEM_PROMPT.format(ontology_description=ontology_desc)
        user_msg = _USER_PROMPT.format(text=text.strip())

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                result = self.llm.chat_json(
                    messages=messages,
                    temperature=0.1,  # Niedrige Temperatur für Extraktionspräzision
                    max_tokens=4096,
                )
                return self._validate_and_clean(result, ontology)

            except ValueError as e:
                last_error = e
                logger.warning(
                    f"NER-Extraktion fehlgeschlagen (Versuch {attempt + 1}): ungültiges JSON — {e}"
                )
            except Exception as e:
                last_error = e
                logger.error(f"NER-Extraktionsfehler: {e}")
                if attempt >= self.max_retries:
                    break

        logger.error(
            f"NER-Extraktion nach {self.max_retries + 1} Versuchen fehlgeschlagen: {last_error}"
        )
        return {"entities": [], "relations": []}

    def _format_ontology(self, ontology: Dict[str, Any]) -> str:
        """Ontologie-Dict in lesbaren Text für den LLM-Prompt formatieren."""
        parts = []

        entity_types = ontology.get("entity_types", [])
        if entity_types:
            parts.append("Entitätstypen:")
            for et in entity_types:
                if isinstance(et, dict):
                    name = et.get("name", str(et))
                    desc = et.get("description", "")
                    attrs = et.get("attributes", [])
                    line = f"  - {name}"
                    if desc:
                        line += f": {desc}"
                    if attrs:
                        attr_names = [a.get("name", str(a)) if isinstance(a, dict) else str(a) for a in attrs]
                        line += f" (Attribute: {', '.join(attr_names)})"
                    parts.append(line)
                else:
                    parts.append(f"  - {et}")

        relation_types = ontology.get("relation_types", ontology.get("edge_types", []))
        if relation_types:
            parts.append("\nBeziehungstypen:")
            for rt in relation_types:
                if isinstance(rt, dict):
                    name = rt.get("name", str(rt))
                    desc = rt.get("description", "")
                    source_targets = rt.get("source_targets", [])
                    line = f"  - {name}"
                    if desc:
                        line += f": {desc}"
                    if source_targets:
                        st_strs = [f"{st.get('source', '?')} → {st.get('target', '?')}" for st in source_targets]
                        line += f" ({', '.join(st_strs)})"
                    parts.append(line)
                else:
                    parts.append(f"  - {rt}")

        if not parts:
            parts.append("Keine spezifische Ontologie definiert. Extrahieren Sie alle Entitäten und Beziehungen, die Sie finden.")

        return "\n".join(parts)

    def _validate_and_clean(
        self, result: Dict[str, Any], ontology: Dict[str, Any]
    ) -> Dict[str, Any]:
        """LLM-Ausgabe validieren und normalisieren."""
        entities = result.get("entities", [])
        relations = result.get("relations", [])

        # Gültige Typnamen aus der Ontologie ermitteln
        valid_entity_types = set()
        for et in ontology.get("entity_types", []):
            if isinstance(et, dict):
                valid_entity_types.add(et.get("name", "").strip())
            else:
                valid_entity_types.add(str(et).strip())

        valid_relation_types = set()
        for rt in ontology.get("relation_types", ontology.get("edge_types", [])):
            if isinstance(rt, dict):
                valid_relation_types.add(rt.get("name", "").strip())
            else:
                valid_relation_types.add(str(rt).strip())

        # Entitäten bereinigen
        cleaned_entities = []
        seen_names = set()
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            name = str(entity.get("name", "")).strip()
            etype = str(entity.get("type", "Entität")).strip()
            if not name:
                continue

            # Deduplizierung nach normalisiertem Namen
            name_lower = name.lower()
            if name_lower in seen_names:
                continue
            seen_names.add(name_lower)

            # Wenn die Ontologie Typen hat, warnen, aber Entitäten mit unbekannten Typen behalten
            if valid_entity_types and etype not in valid_entity_types:
                logger.debug(f"Entität '{name}' hat Typ '{etype}', der nicht in der Ontologie ist, wird dennoch beibehalten")

            cleaned_entities.append({
                "name": name,
                "type": etype,
                "attributes": entity.get("attributes", {}),
            })

        # Beziehungen bereinigen
        cleaned_relations = []
        entity_names_lower = {e["name"].lower() for e in cleaned_entities}
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            source = str(relation.get("source", "")).strip()
            target = str(relation.get("target", "")).strip()
            rtype = str(relation.get("type", "RELATED_TO")).strip()
            fact = str(relation.get("fact", "")).strip()

            if not source or not target:
                continue

            # Sicherstellen, dass Quell- und Zielentitäten existieren
            # (sie könnten fehlen, wenn das LLM eine Beziehung ohne die Entität halluziniert hat)
            if source.lower() not in entity_names_lower:
                cleaned_entities.append({
                    "name": source,
                    "type": "Entität",
                    "attributes": {},
                })
                entity_names_lower.add(source.lower())

            if target.lower() not in entity_names_lower:
                cleaned_entities.append({
                    "name": target,
                    "type": "Entität",
                    "attributes": {},
                })
                entity_names_lower.add(target.lower())

            cleaned_relations.append({
                "source": source,
                "target": target,
                "type": rtype,
                "fact": fact or f"{source} {rtype} {target}",
            })

        return {
            "entities": cleaned_entities,
            "relations": cleaned_relations,
        }
