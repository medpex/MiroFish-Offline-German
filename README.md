<div align="center">

<img src="./static/image/mirofish-offline-banner.png" alt="MiroFish Offline" width="100%"/>

# MiroFish-Offline (Deutsche Version)

**Komplett auf Deutsch übersetzt. Multi-Agenten-Simulation für öffentliche Meinung & Marktstimmung. Läuft 100% lokal mit Ollama + Neo4j.**

*Eine Multi-Agenten-Schwarmintelligenz-Engine zur Simulation von öffentlicher Meinung, Marktstimmung und sozialer Dynamik. Läuft komplett auf deiner eigenen Hardware — keine Cloud-APIs nötig.*

[![GitHub Stars](https://img.shields.io/github/stars/medpex/MiroFish-Offline?style=flat-square&color=DAA520)](https://github.com/medpex/MiroFish-Offline/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/medpex/MiroFish-Offline?style=flat-square)](https://github.com/medpex/MiroFish-Offline/network)
[![Docker](https://img.shields.io/badge/Docker-Build-2496ED?style=flat-square&logo=docker&logoColor=white)](https://hub.docker.com/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue?style=flat-square)](./LICENSE)

</div>

## Was ist das?

MiroFish ist eine Multi-Agenten-Simulations-Engine: Lade ein beliebiges Dokument hoch (Pressemitteilung, Strategiepapier, Marktanalyse) und das System generiert hunderte KI-Agenten mit einzigartigen Persönlichkeiten, die die öffentliche Reaktion in sozialen Medien simulieren. Posts, Argumente, Meinungsverschiebungen — Stunde für Stunde.

**Dieser Fork ist die vollständig deutsche Version** — UI, alle LLM-Prompts, Fehlermeldungen und Zeitzonen (MEZ) sind auf Deutsch lokalisiert.

## Workflow

1. **Graph erstellen** — Extrahiert Entitäten (Personen, Unternehmen, Ereignisse) und Beziehungen aus deinem Dokument. Erstellt einen Wissensgraph mit individuellem und kollektivem Gedächtnis über Neo4j.
2. **Umgebung einrichten** — Generiert hunderte Agenten-Persönlichkeiten, jeweils mit eigenem Charakter, Meinungstendenz, Reaktionsgeschwindigkeit, Einflussgrad und Erinnerung an vergangene Ereignisse.
3. **Simulation** — Agenten interagieren auf simulierten Social-Media-Plattformen: posten, antworten, diskutieren, ändern Meinungen. Das System verfolgt Stimmungsentwicklung, Themenverbreitung und Einflussdynamiken in Echtzeit.
4. **Bericht** — Ein Report-Agent analysiert die Simulationsergebnisse, befragt eine Fokusgruppe von Agenten, durchsucht den Wissensgraph nach Belegen und erstellt eine strukturierte Analyse.
5. **Interaktion** — Chatte mit jedem Agenten aus der simulierten Welt. Frage sie, warum sie gepostet haben, was sie gepostet haben. Vollständiges Gedächtnis und Persönlichkeit bleiben erhalten.

## Screenshot

<div align="center">
<img src="./static/image/mirofish-offline-screenshot.jpg" alt="MiroFish Offline" width="100%"/>
</div>

## Schnellstart

### Voraussetzungen

- Docker & Docker Compose (empfohlen), **oder**
- Python 3.11+, Node.js 18+, Neo4j 5.15+, Ollama

### Option A: Docker (am einfachsten)

```bash
git clone https://github.com/medpex/MiroFish-Offline.git
cd MiroFish-Offline
cp .env.example .env

# Alle Services starten (Neo4j, Ollama, MiroFish)
docker compose up -d

# Benötigte Modelle in Ollama laden
docker exec mirofish-ollama ollama pull qwen2.5:7b
docker exec mirofish-ollama ollama pull nomic-embed-text
```

Öffne `http://localhost:3000` — fertig.

### Option B: Manuell

**1. Neo4j starten**

```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/mirofish \
  neo4j:5.18-community
```

**2. Ollama starten & Modelle laden**

```bash
ollama serve &
ollama pull qwen2.5:7b         # LLM (oder qwen2.5:14b für mehr Qualität)
ollama pull nomic-embed-text    # Embeddings (768d)
```

**3. Backend konfigurieren & starten**

```bash
cp .env.example .env
# .env anpassen falls Neo4j/Ollama auf anderen Ports laufen

cd backend
pip install -r requirements.txt
python run.py
```

**4. Frontend starten**

```bash
cd frontend
npm install
npm run dev
```

Öffne `http://localhost:3000`.

## Konfiguration

Alle Einstellungen befinden sich in `.env` (kopieren von `.env.example`):

```bash
# LLM — zeigt auf lokalen Ollama (OpenAI-kompatible API)
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL_NAME=qwen2.5:7b

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=mirofish

# Embeddings
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_BASE_URL=http://localhost:11434
```

Funktioniert mit jeder OpenAI-kompatiblen API — tausche Ollama gegen Claude, GPT oder einen anderen Anbieter, indem du `LLM_BASE_URL` und `LLM_API_KEY` änderst.

## Architektur

```
┌─────────────────────────────────────────┐
│              Flask API                   │
│  graph.py  simulation.py  report.py     │
└──────────────┬──────────────────────────┘
               │ app.extensions['neo4j_storage']
┌──────────────▼──────────────────────────┐
│           Service-Schicht                │
│  EntityReader  GraphToolsService         │
│  GraphMemoryUpdater  ReportAgent         │
└──────────────┬──────────────────────────┘
               │ storage: GraphStorage
┌──────────────▼──────────────────────────┐
│         GraphStorage (abstrakt)          │
│              │                            │
│    ┌─────────▼─────────┐                │
│    │   Neo4jStorage     │                │
│    │  ┌───────────────┐ │                │
│    │  │ EmbeddingService│ ← Ollama       │
│    │  │ NERExtractor   │ ← Ollama LLM   │
│    │  │ SearchService  │ ← Hybridsuche   │
│    │  └───────────────┘ │                │
│    └───────────────────┘                │
└─────────────────────────────────────────┘
               │
        ┌──────▼──────┐
        │  Neo4j CE   │
        │  5.18       │
        └─────────────┘
```

**Wichtige Design-Entscheidungen:**

- `GraphStorage` ist ein abstraktes Interface — Neo4j kann durch jede andere Graph-DB ersetzt werden
- Dependency Injection über Flask `app.extensions` — keine globalen Singletons
- Hybridsuche: 0,7 × Vektor-Ähnlichkeit + 0,3 × BM25-Schlüsselwortsuche
- Synchrone NER/RE-Extraktion über lokales LLM
- Alle Original-Tools (InsightForge, Panorama, Agenten-Befragungen) erhalten

## Hardware-Anforderungen

| Komponente | Minimum | Empfohlen |
|---|---|---|
| RAM | 16 GB | 32 GB |
| VRAM (GPU) | 6 GB (7b Modell) | 16+ GB (14b Modell) |
| Festplatte | 20 GB | 50 GB |
| CPU | 4 Kerne | 8+ Kerne |

CPU-Modus funktioniert, ist aber deutlich langsamer für LLM-Inferenz. Für leichtere Setups verwende `qwen2.5:7b`.

## Anwendungsfälle

- **PR-Krisentests** — Simuliere die öffentliche Reaktion auf eine Pressemitteilung vor der Veröffentlichung
- **Marktstimmungsanalyse** — Füttere Finanznachrichten ein und beobachte die simulierte Marktstimmung
- **Politikfolgenabschätzung** — Teste Gesetzesentwürfe gegen simulierte öffentliche Reaktionen
- **KMU-Marktanalyse** — Teste die Positionierung deines Unternehmens in einem simulierten Marktumfeld

## Lizenz

AGPL-3.0 — siehe [LICENSE](./LICENSE).

## Änderungen in diesem Fork

- Komplette UI-Übersetzung auf Deutsch (14 Vue-Dateien, 400+ Strings)
- Alle LLM-Prompts auf Deutsch (Ontologie, NER, Simulation, Report-Generierung)
- Alle Backend-Fehlermeldungen und Log-Nachrichten auf Deutsch
- Zeitzonen von Peking auf MEZ (Mitteleuropäische Zeit) umgestellt
- Datumsformate auf de-DE angepasst
- Standard-Land auf Deutschland geändert
