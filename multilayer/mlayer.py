"""
Log Template Extractor — Hybrid Pipeline
=========================================
Layers:
  L1  Raw log ingestion (file / list / stream)
  L2  Regex pre-filter  — masks volatile tokens before DRAIN sees them
  L3  DRAIN3            — streaming parse-tree clustering
  L4  Template merge    — edit-distance dedup + low-count anomaly flagging
  L5  LLM fallback      — OpenAI call for low-confidence / new templates
  L6  Structured output — dict / JSON ready for ES, Loki, ClickHouse, etc.

Install:
    pip install drain3 openai python-Levenshtein
"""

from __future__ import annotations

import json
import re
import os
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional

# ── DRAIN3 ────────────────────────────────────────────────────────────────────
try:
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig
except ImportError:
    raise ImportError("Install drain3:  pip install drain3")

# ── Levenshtein (optional, for L4 merge) ─────────────────────────────────────
try:
    from Levenshtein import distance as lev_distance
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False

# ── OpenAI (optional, for L5 LLM fallback) ───────────────────────────────────
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LogEvent:
    raw: str                          # original log line
    masked: str                       # after regex pre-filter
    template_id: str                  # cluster ID from DRAIN
    template: str                     # e.g. "User <*> logged in from <*>"
    variables: list[str]              # extracted dynamic values
    confidence: float                 # 0.0–1.0  (DRAIN similarity score)
    source: str = "drain"             # "drain" | "llm_fallback"
    is_anomaly: bool = False          # flagged by L4
    merged_into: Optional[str] = None # if this template was merged

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# L2 — Regex pre-filter
# ─────────────────────────────────────────────────────────────────────────────

# Each tuple: (compiled_pattern, replacement_token)
DEFAULT_MASKS: list[tuple[re.Pattern, str]] = [
    # ISO-8601 / common log timestamps
    (re.compile(
        r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?"),
     "<TIMESTAMP>"),
    # epoch timestamps (10- or 13-digit numbers)
    (re.compile(r"\b\d{10,13}\b"), "<EPOCH>"),
    # IPv4 (with optional port)
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b"), "<IP>"),
    # IPv6
    (re.compile(
        r"\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b"), "<IPV6>"),
    # UUIDs
    (re.compile(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "<UUID>"),
    # hex strings (≥6 chars, prefixed or standalone)
    (re.compile(r"\b0x[0-9a-fA-F]{4,}\b"), "<HEX>"),
    (re.compile(r"\b[0-9a-fA-F]{8,}\b"), "<HEX>"),
    # file paths
    (re.compile(r"(?:/[\w.\-]+){2,}"), "<PATH>"),
    # URLs
    (re.compile(r"https?://\S+"), "<URL>"),
    # email addresses
    (re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}"), "<EMAIL>"),
    # standalone integers (optional — comment out if too aggressive)
    (re.compile(r"\b\d{2,}\b"), "<NUM>"),
]


class RegexPreFilter:
    """
    L2 — Apply masking rules to a raw log line before feeding to DRAIN.
    Add your own custom rules via `extra_masks`.
    """

    def __init__(self, extra_masks: list[tuple[str, str]] | None = None):
        self.masks = list(DEFAULT_MASKS)
        if extra_masks:
            for pattern, token in extra_masks:
                self.masks.append((re.compile(pattern), token))

    def mask(self, line: str) -> str:
        for pattern, token in self.masks:
            line = pattern.sub(token, line)
        return line


# ─────────────────────────────────────────────────────────────────────────────
# L3 — DRAIN3 streaming extractor
# ─────────────────────────────────────────────────────────────────────────────

def _build_drain_config(
    sim_th: float = 0.4,
    depth: int = 4,
    max_children: int = 100,
) -> TemplateMinerConfig:
    cfg = TemplateMinerConfig()
    cfg.drain_sim_th = sim_th          # similarity threshold (0–1)
    cfg.drain_depth = depth            # parse-tree depth
    cfg.drain_max_children = max_children
    cfg.drain_extra_delimiters = []
    cfg.parametrize_numeric_tokens = True
    return cfg


class DrainExtractor:
    """
    L3 — Thin wrapper around DRAIN3's TemplateMiner.
    Tracks per-template occurrence counts for L4 anomaly gating.
    """

    def __init__(self, sim_th: float = 0.4, depth: int = 4):
        cfg = _build_drain_config(sim_th=sim_th, depth=depth)
        self.miner = TemplateMiner(config=cfg)
        self.counts: dict[str, int] = defaultdict(int)

    def process(self, masked_line: str) -> tuple[str, str, float]:
        """
        Returns (template_id, template_string, confidence).
        confidence derived from change_type: none=1.0, cluster_created=0.0.
        """
        result = self.miner.add_log_message(masked_line)
        # DRAIN3 returns a flat dict with keys:
        # change_type, cluster_id, cluster_size, template_mined, cluster_count
        template_id = str(result["cluster_id"])
        template_str = result["template_mined"]
        change_type = result.get("change_type", "none")
        cluster_size = result.get("cluster_size", 1)

        confidence_map = {
            "none": 1.0,
            "update": min(0.9, cluster_size / (cluster_size + 1)),
            "cluster_created": 0.1,
        }
        confidence = confidence_map.get(change_type, 0.5)
        self.counts[template_id] += 1
        return template_id, template_str, confidence

    def extract_variables(self, masked_line: str, template: str) -> list[str]:
        """
        Align masked log tokens against template tokens to pull variable slots.
        """
        log_tokens = masked_line.split()
        tmpl_tokens = template.split()
        variables = []
        for l_tok, t_tok in zip(log_tokens, tmpl_tokens):
            if t_tok == "<*>":
                variables.append(l_tok)
        return variables

    @property
    def templates(self) -> list[dict]:
        """All discovered templates as a list of dicts."""
        return [
            {
                "id": str(c.cluster_id),
                "template": c.get_template(),
                "count": c.size,
            }
            for c in self.miner.drain.id_to_cluster.values()
        ]


# ─────────────────────────────────────────────────────────────────────────────
# L4 — Template merge + anomaly gate
# ─────────────────────────────────────────────────────────────────────────────

class TemplateMerger:
    """
    L4 — Post-process DRAIN output:
      • Merge templates that differ by ≤ edit_distance_threshold tokens
      • Flag templates with occurrence count < anomaly_count_threshold
    """

    def __init__(
        self,
        edit_distance_threshold: int = 2,
        anomaly_count_threshold: int = 3,
    ):
        self.edit_dist_th = edit_distance_threshold
        self.anomaly_th = anomaly_count_threshold
        self._merge_map: dict[str, str] = {}  # old_id → canonical_id

    def _token_edit_distance(self, a: str, b: str) -> int:
        """Token-level edit distance between two template strings."""
        if HAS_LEVENSHTEIN:
            return lev_distance(a, b)
        # fallback: count differing tokens
        ta, tb = a.split(), b.split()
        if len(ta) != len(tb):
            return abs(len(ta) - len(tb)) + sum(x != y for x, y in zip(ta, tb))
        return sum(x != y for x, y in zip(ta, tb))

    def build_merge_map(self, templates: list[dict]) -> dict[str, str]:
        """
        Cluster templates by edit distance. Returns {template_id → canonical_id}.
        """
        canonical: dict[str, str] = {}  # id → canonical id
        tmpl_list = list(templates)

        for i, ti in enumerate(tmpl_list):
            if ti["id"] in canonical:
                continue
            canonical[ti["id"]] = ti["id"]
            for tj in tmpl_list[i + 1:]:
                if tj["id"] in canonical:
                    continue
                dist = self._token_edit_distance(ti["template"], tj["template"])
                if dist <= self.edit_dist_th:
                    canonical[tj["id"]] = ti["id"]  # merge tj → ti

        self._merge_map = canonical
        return canonical

    def resolve(self, template_id: str) -> str:
        """Return the canonical template ID for a given ID."""
        return self._merge_map.get(template_id, template_id)

    def is_anomaly(self, count: int) -> bool:
        return count < self.anomaly_th


# ─────────────────────────────────────────────────────────────────────────────
# L5 — LLM fallback
# ─────────────────────────────────────────────────────────────────────────────

LLM_PROMPT = """\
You are a log parsing expert. Given a raw log line, extract a log template by \
replacing all dynamic/variable parts (IDs, IPs, numbers, names, timestamps, \
file paths, hex strings) with the placeholder <*>. Keep static keywords, \
action verbs, and structural words unchanged.

Return ONLY the template string. No explanation. No markdown.

Log line:
{log_line}"""


class LLMFallback:
    """
    L5 — Call an LLM for log lines where DRAIN is uncertain.
    Results are cached so each unique masked line is only called once.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ):
        self.threshold = confidence_threshold
        self.model = model
        self._cache: dict[str, str] = {}

        if not HAS_OPENAI:
            print("[LLMFallback] openai not installed — fallback disabled.")
            self._client = None
        else:
            key = api_key or os.getenv("OPENAI_API_KEY")
            self._client = OpenAI(api_key=key) if key else None
            if not self._client:
                print("[LLMFallback] No API key found — fallback disabled.")

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def should_fallback(self, confidence: float) -> bool:
        return self.enabled and confidence < self.threshold

    def extract_template(self, raw_line: str) -> str | None:
        if not self.enabled:
            return None
        if raw_line in self._cache:
            return self._cache[raw_line]
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": LLM_PROMPT.format(log_line=raw_line)}
                ],
                temperature=0,
                max_tokens=200,
            )
            template = response.choices[0].message.content.strip()
            self._cache[raw_line] = template
            return template
        except Exception as exc:
            print(f"[LLMFallback] API error: {exc}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class LogTemplateExtractor:
    """
    Full hybrid pipeline:
      L2 Regex pre-filter → L3 DRAIN3 → L4 Merge/anomaly → L5 LLM fallback

    Usage
    -----
    extractor = LogTemplateExtractor()

    # Process a single line
    event = extractor.process_line("2024-03-25 ERROR User 42 login failed from 10.0.0.1")

    # Process a list of lines
    events = extractor.process_lines(lines)

    # Process a log file
    events = extractor.process_file("app.log")

    # After processing, get all discovered templates
    templates = extractor.get_templates()
    """

    def __init__(
        self,
        # L2
        extra_masks: list[tuple[str, str]] | None = None,
        # L3
        sim_th: float = 0.4,
        depth: int = 4,
        # L4
        edit_distance_threshold: int = 2,
        anomaly_count_threshold: int = 3,
        # L5
        llm_confidence_threshold: float = 0.3,
        llm_model: str = "gpt-4o-mini",
        openai_api_key: str | None = None,
        enable_llm: bool = False,
    ):
        self.prefilter = RegexPreFilter(extra_masks=extra_masks)
        self.drain = DrainExtractor(sim_th=sim_th, depth=depth)
        self.merger = TemplateMerger(
            edit_distance_threshold=edit_distance_threshold,
            anomaly_count_threshold=anomaly_count_threshold,
        )
        self.llm = LLMFallback(
            confidence_threshold=llm_confidence_threshold,
            model=llm_model,
            api_key=openai_api_key,
        ) if enable_llm else None

        self._events: list[LogEvent] = []

    # ── core ──────────────────────────────────────────────────────────────────

    def process_line(self, raw_line: str) -> LogEvent:
        raw_line = raw_line.rstrip("\n")

        # L2 — mask volatile tokens
        masked = self.prefilter.mask(raw_line)

        # L3 — DRAIN clustering
        template_id, template, confidence = self.drain.process(masked)
        variables = self.drain.extract_variables(masked, template)
        source = "drain"

        # L5 — LLM fallback for low-confidence lines
        if self.llm and self.llm.should_fallback(confidence):
            llm_template = self.llm.extract_template(raw_line)
            if llm_template:
                template = llm_template
                source = "llm_fallback"
                confidence = 1.0  # LLM output treated as authoritative

        event = LogEvent(
            raw=raw_line,
            masked=masked,
            template_id=template_id,
            template=template,
            variables=variables,
            confidence=confidence,
            source=source,
        )
        self._events.append(event)
        return event

    def process_lines(self, lines: list[str]) -> list[LogEvent]:
        events = [self.process_line(line) for line in lines]
        self._apply_merge_and_anomaly(events)
        return events

    def process_file(self, path: str, encoding: str = "utf-8") -> list[LogEvent]:
        with open(path, encoding=encoding, errors="replace") as fh:
            lines = fh.readlines()
        return self.process_lines(lines)

    # ── L4 post-pass ──────────────────────────────────────────────────────────

    def _apply_merge_and_anomaly(self, events: list[LogEvent]) -> None:
        """Run L4 merge + anomaly flagging after a batch is processed."""
        templates = self.drain.templates
        merge_map = self.merger.build_merge_map(templates)

        for ev in events:
            canonical = merge_map.get(ev.template_id, ev.template_id)
            if canonical != ev.template_id:
                ev.merged_into = canonical
            count = self.drain.counts.get(ev.template_id, 0)
            ev.is_anomaly = self.merger.is_anomaly(count)

    # ── output helpers ─────────────────────────────────────────────────────────

    def get_templates(self) -> list[dict]:
        """Return all unique templates discovered so far."""
        return self.drain.templates

    def get_events(self) -> list[LogEvent]:
        return self._events

    def summary(self) -> dict:
        templates = self.get_templates()
        anomalies = [e for e in self._events if e.is_anomaly]
        llm_used = [e for e in self._events if e.source == "llm_fallback"]
        return {
            "total_lines": len(self._events),
            "unique_templates": len(templates),
            "anomaly_events": len(anomalies),
            "llm_fallback_events": len(llm_used),
            "templates": templates,
        }

    def to_jsonl(self, path: str) -> None:
        """Write all events as JSON Lines."""
        with open(path, "w") as fh:
            for ev in self._events:
                fh.write(json.dumps(ev.to_dict()) + "\n")
        print(f"Written {len(self._events)} events → {path}")

    def to_txt(self, path: str) -> None:
        """Write discovered templates to a human-readable .txt file."""
        summary = self.summary()
        anomalies = sum(1 for e in self._events if e.is_anomaly)
        llm_calls = sum(1 for e in self._events if e.source == "llm_fallback")

        with open(path, "w") as fh:
            fh.write("Log Template Extraction Results\n")
            fh.write("================================\n")
            fh.write(f"Total lines : {summary['total_lines']}\n")
            fh.write(f"Templates   : {summary['unique_templates']}\n")
            fh.write(f"Anomalies   : {anomalies}\n")
            fh.write(f"LLM calls   : {llm_calls}\n")
            fh.write("================================\n\n")

            for t in summary["templates"]:
                fh.write(f"[T-{t['id']}]  (occurrences: {t['count']})\n")
                fh.write(f"       {t['template']}\n\n")

            fh.write("================================\n")
            fh.write("Placeholder legend\n")
            fh.write("------------------\n")
            fh.write("<TIMESTAMP>  — date/time token (ISO-8601, epoch, etc.)\n")
            fh.write("<IP>         — IPv4 or IPv6 address\n")
            fh.write("<NUM>        — standalone integer or numeric value\n")
            fh.write("<PATH>       — file system path\n")
            fh.write("<HEX>        — hexadecimal string\n")
            fh.write("<UUID>       — UUID value\n")
            fh.write("<URL>        — HTTP/HTTPS URL\n")
            fh.write("<EMAIL>      — email address\n")
            fh.write("<*>          — any other dynamic / variable token\n")
            fh.write("================================\n")

        print(f"Written {summary['unique_templates']} templates → {path}")



# ─────────────────────────────────────────────────────────────────────────────
# Quick demo — run directly:  python log_template_extractor.py
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_LOGS = [
    "2024-03-25T10:43:02Z INFO  User 1001 logged in from 192.168.1.10",
    "2024-03-25T10:43:05Z INFO  User 2045 logged in from 10.0.0.5",
    "2024-03-25T10:43:07Z ERROR Failed to connect to db-host-01:5432 after 3 retries",
    "2024-03-25T10:43:09Z ERROR Failed to connect to db-host-02:5432 after 5 retries",
    "2024-03-25T10:43:11Z WARN  Disk usage at 87% on /dev/sda1",
    "2024-03-25T10:43:13Z WARN  Disk usage at 92% on /dev/sdb2",
    "2024-03-25T10:43:15Z INFO  Request GET /api/v1/users/42 completed in 120ms status=200",
    "2024-03-25T10:43:17Z INFO  Request POST /api/v1/orders/789 completed in 340ms status=201",
    "2024-03-25T10:43:19Z INFO  Cache miss for key session:abc123def456",
    "2024-03-25T10:43:20Z INFO  Cache hit for key session:xyz789abc012",
    "2024-03-25T10:43:21Z ERROR Unexpected exception in worker-3: NullPointerException",
    "2024-03-25T10:43:22Z ERROR Unexpected exception in worker-7: IndexOutOfBoundsException",
    "2024-03-25T10:43:23Z INFO  Job 550e8400-e29b-41d4-a716-446655440000 completed in 4.2s",
    "2024-03-25T10:43:24Z INFO  Job 6ba7b810-9dad-11d1-80b4-00c04fd430c8 completed in 1.8s",
]


if __name__ == "__main__":
    print("=" * 60)
    print("Log Template Extractor — demo run")
    print("=" * 60)

    extractor = LogTemplateExtractor(
        sim_th=0.4,
        depth=4,
        edit_distance_threshold=2,
        anomaly_count_threshold=2,
        enable_llm=False,   # flip to True and set OPENAI_API_KEY for L5
    )

    events = extractor.process_lines(SAMPLE_LOGS)

    print(f"\n{'─'*60}")
    print(f"{'RAW LOG':<52} → TEMPLATE")
    print(f"{'─'*60}")
    for ev in events:
        raw_short = ev.raw[:50].ljust(50)
        flag = " [ANOMALY]" if ev.is_anomaly else ""
        merged = f" [merged→{ev.merged_into}]" if ev.merged_into else ""
        print(f"{raw_short}  {ev.template}{flag}{merged}")

    print(f"\n{'─'*60}")
    print("DISCOVERED TEMPLATES")
    print(f"{'─'*60}")
    for t in extractor.get_templates():
        print(f"  [{t['id']:>3}] (n={t['count']:>3})  {t['template']}")

    summary = extractor.summary()
    print(f"\nSummary: {summary['total_lines']} lines → "
          f"{summary['unique_templates']} templates | "
          f"{summary['anomaly_events']} anomalies | "
          f"{summary['llm_fallback_events']} LLM calls")

    extractor.to_txt("template.txt")