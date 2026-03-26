# Indent Discovery V2 (Conversation-Driven Intent/FAQ/Blueprint Extraction)

Streamlit app to upload conversation transcripts and run a staged pipeline:
Normalize → Atoms → FAQs → Intents/Flows → Blueprint.

## Process Overview (Upload → Atoms → Clusters → Intents/Flows)
1. Conversation upload
Upload a `.json`, `.jsonl`, or `.txt` file from the sidebar (`Upload conversations`). The file is saved under `workspace/<project_id>/input/`.
`.txt` uploads now also support the MyOperator-style batch export format (headers + `Recording #...`, `URL: ...`, `Timestamp: ...` blocks) and will split into separate conversations automatically.
Then click `Normalize conversations` to parse and clean the raw records into `workspace/<project_id>/input/normalized_conversations.jsonl`.

2. Atoms extraction
Click `Extract atoms`. This calls the LLM (requires `OPENAI_API_KEY`) to extract `flow_candidates`, `faq_candidates`, and `blueprint_signals` from each normalized conversation.
Output is written to `workspace/<project_id>/stage_atoms/atoms.jsonl`.

3. Clustering intents
Click `Generate embeddings` in the sidebar (requires `OPENAI_API_KEY`), then open the **Clusters** tab.
Flow keys are embedded and clustered using the similarity threshold and the **Cluster on intent only** setting (intent vs intent+scenario).
Clusters are saved to `workspace/<project_id>/stage_cluster/flow_clusters.json`.

4. Generating intents/flows
Click `Build Intents/Flows`. This reads atoms + clusters and compiles the intent/flow catalogue.
Output is written to `workspace/<project_id>/stage_catalogue/intent_flow_catalogue.json`.

## Prompts & Output Formats
Below are the default prompts and required output schemas used by each LLM-powered step.
These prompts are editable from the UI (sidebar) where applicable.

### Atoms Extraction
**System prompt**
```text
You extract FAQ candidates, flow candidates, and blueprint signals from customer-agent conversations. Be faithful. Do not invent facts or pricing. Always respond in English.
```

**User template (excerpt)**
```text
Input is a normalized conversation JSON with messages_for_llm[] in order.
Return ONLY valid JSON matching the Atom Schema.
...
Atom schema:
{
  "conversation_id": "string",
  "faq_candidates": [
    {
      "q_raw": "string",
      "q_clean": "string",
      "a_candidates": ["string"],
      "evidence": { "q_turn": [int,int], "a_turn": [int,int] }
    }
  ],
  "flow_candidates": [
    {
      "intent_candidate": "string",
      "scenario_candidate": "string",
      "flow_steps_candidate": ["string"],
      "resolution": { "status": "resolved|pending|unresolved|unknown", "summary": "string" },
      "evidence": { "turn_span": [int,int] }
    }
  ],
  "blueprint_signals": {
    "policy": ["string"],
    "tone": ["string"],
    "escalations": ["string"]
  }
}
Return ONLY valid JSON matching this schema. Do not add extra fields.
```

**Required output format**
```json
{
  "conversation_id": "string",
  "faq_candidates": [
    {
      "q_raw": "string",
      "q_clean": "string",
      "a_candidates": ["string"],
      "evidence": { "q_turn": [0, 1], "a_turn": [2, 3] }
    }
  ],
  "flow_candidates": [
    {
      "intent_candidate": "string",
      "scenario_candidate": "string",
      "flow_steps_candidate": ["string"],
      "resolution": { "status": "resolved", "summary": "string" },
      "evidence": { "turn_span": [0, 3] }
    }
  ],
  "blueprint_signals": {
    "policy": ["string"],
    "tone": ["string"],
    "escalations": ["string"]
  }
}
```

### FAQ Generation
**System prompt**
```text
You consolidate similar FAQ questions and draft a KB-style answer using provided candidates. Always respond in English.
```

**User template (excerpt)**
```text
Given:
- question variants (q_raw and q_clean)
- answer candidates list (grounded)
Output ONLY JSON:
{
  'faq_id': string,
  'canonical_question': string,
  'question_variants': string[],
  'draft_answer': string,
  'needs_verification': boolean,
  'value_score': { '0_to_5': number, 'reason': string }
}
```

**Required output format**
```json
{
  "faq_id": "string",
  "canonical_question": "string",
  "question_variants": ["string"],
  "draft_answer": "string",
  "needs_verification": true,
  "value_score": { "0_to_5": 2.5, "reason": "string" }
}
```

### Intents/Flows
**System prompt**
```text
You standardize intents and create bot flow templates. Always respond in English.
```

**User template (excerpt)**
```text
Given multiple flow_candidates (intent_candidate, scenario_candidate, flow_steps_candidate, resolution),
produce ONLY JSON:
{
  'intent_name': string,
  'definition': string,
  'scenarios': [
    {
      'scenario_name': string,
      'frequency': number,
      'flow_template': [
        { 'type': 'ask|inform|action|handover', 'text': string }
      ],
      'required_fields': string[],
      'handover_rules': string[],
      'example_conversation_ids': string[]
    }
  ]
}
```

**Required output format**
```json
{
  "intent_name": "string",
  "definition": "string",
  "scenarios": [
    {
      "scenario_name": "string",
      "frequency": 3,
      "flow_template": [
        { "type": "ask", "text": "string" }
      ],
      "required_fields": ["string"],
      "handover_rules": ["string"],
      "example_conversation_ids": ["string"]
    }
  ]
}
```

### Blueprint
**System prompt**
```text
You are an analyst who produces a business blueprint summary in JSON only.
```

**User template (excerpt)**
```text
Given the base blueprint and evidence, return JSON only with keys:
tone_summary (list of {tone, count}), policy_summary (list of {policy, count, examples}),
escalation_patterns (list of {pattern, count, examples}), gaps (list of {faq_id, reason}).
Do not add keys. Use the same schema.
```

**Required output format**
```json
{
  "tone_summary": [{ "tone": "string", "count": 3 }],
  "policy_summary": [{ "policy": "string", "count": 2, "examples": ["string"] }],
  "escalation_patterns": [{ "pattern": "string", "count": 1, "examples": ["string"] }],
  "gaps": [{ "faq_id": "string", "reason": "string" }]
}
```

### Final Prompt (System Prompt Generation)
**System prompt**
```text
You are a senior prompt engineer. Return only the system prompt text.
```

**User template (excerpt)**
```text
You are given an intent JSON describing an automation-ready intent and its scenarios.
Write a SINGLE system prompt for a production chatbot that can automate this intent end-to-end.
...
Output ONLY the system prompt text. No JSON, no markdown.

Intent JSON:
{intent_json}
```

**Required output format**
```text
<plain system prompt text>
```

### Testing (Judge Scoring)
**System prompt**
```text
You are a strict evaluator. Return JSON only.
```

**User template (excerpt)**
```text
Evaluate the chatbot using the intent specification and transcripts.
Score each category from 0 to 5 (5 is best).
...
Return ONLY JSON in this schema:
{
  "scores": {
    "task_completion": 0,
    "required_fields": 0,
    "flow_adherence": 0,
    "response_quality": 0,
    "escalation_handling": 0
  },
  "rationales": {
    "task_completion": "...",
    "required_fields": "...",
    "flow_adherence": "...",
    "response_quality": "...",
    "escalation_handling": "..."
  },
  "overall_notes": "..."
}
```

**Required output format**
```json
{
  "scores": {
    "task_completion": 4,
    "required_fields": 3,
    "flow_adherence": 4,
    "response_quality": 4,
    "escalation_handling": 5
  },
  "rationales": {
    "task_completion": "string",
    "required_fields": "string",
    "flow_adherence": "string",
    "response_quality": "string",
    "escalation_handling": "string"
  },
  "overall_notes": "string"
}
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
```

## Run
```bash
streamlit run streamlit_app.py
```

## Notes
- Artifacts are stored under `workspace/<project_id>/`.
- LLM calls are cached under `workspace/<project_id>/cache/`.
- Embeddings fall back to deterministic hashing if no API key is provided.

## Parlant Integration (Optional)
Indent Discovery V2 can publish intents/flows and blueprint rules to a Parlant server and run the Chatbot Preview in Parlant mode.

### Parlant Setup (macOS/Linux)
```bash
pip install parlant
export EMCIE_API_KEY="your_key_here"
python parlant_server/main.py --port 8800
```

### Parlant Setup (macOS/Linux, OpenAI mode)
```bash
pip install parlant
export OPENAI_API_KEY="your_key_here"
python parlant_server/main.py --openai --port 8800
```

### Parlant Setup (Windows PowerShell)
```powershell
pip install parlant
$env:EMCIE_API_KEY="your_key_here"
python parlant_server\main.py --port 8800
```

### Parlant Setup (Windows PowerShell, OpenAI mode)
```powershell
pip install parlant
$env:OPENAI_API_KEY="your_key_here"
python parlant_server\main.py --openai --port 8800
```

### Verify Parlant
- UI should be reachable at `http://localhost:8800` (or the port you chose)
- In the Streamlit sidebar, set `PARLANT_BASE_URL` to `http://localhost:8800`

### Enable Parlant in the App
1) Open the Streamlit sidebar.
2) Enable **Parlant** and set the base URL.
3) Click **Test connection**.
4) Click **Sync journeys now** (or **Publish to Parlant** in the stages list).
5) In **Chatbot (Preview)**, switch Runtime mode to **Parlant**.

### What Gets Published
- Journeys: one per intent scenario with steps mapped from `flow_template`.
- Guidelines: global tone/policy/escalation + journey-specific required fields/handover rules.
- Tools: action steps mapped to the existing tool runner.

### Troubleshooting
- If Parlant is unreachable, the UI falls back to Native mode.
- Sync logs are stored at `workspace/<project_id>/parlant_sync.log`.
- A sync manifest is stored at `workspace/<project_id>/parlant_sync_manifest.json`.
