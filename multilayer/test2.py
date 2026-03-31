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
"""

from __future__ import annotations

import json
import re
import os
import sys
import csv  # <-- Added for CSV export
from collections import defaultdict
from dataclasses import dataclass, asdict
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
    raw: str
    masked: str
    template_id: str
    template: str
    variables: list[str]
    confidence: float
    source: str = "drain"
    is_anomaly: bool = False
    merged_into: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# L2 — Regex pre-filter
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MASKS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?"), "<TIMESTAMP>"),
    (re.compile(r"\b\d{10,13}\b"), "<EPOCH>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b"), "<IP>"),
    (re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b"), "<IPV6>"),
    (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "<UUID>"),
    (re.compile(r"\b0x[0-9a-fA-F]{4,}\b"), "<HEX>"),
    (re.compile(r"\b[0-9a-fA-F]{8,}\b"), "<HEX>"),
    (re.compile(r"(?:/[\w.\-]+){2,}"), "<PATH>"),
    (re.compile(r"https?://\S+"), "<URL>"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}"), "<EMAIL>"),
    (re.compile(r"\b\d{2,}\b"), "<NUM>"),
]


class RegexPreFilter:
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

def _build_drain_config(sim_th: float = 0.4, depth: int = 4, max_children: int = 100) -> TemplateMinerConfig:
    cfg = TemplateMinerConfig()
    cfg.drain_sim_th = sim_th
    cfg.drain_depth = depth
    cfg.drain_max_children = max_children
    cfg.drain_extra_delimiters = []
    cfg.parametrize_numeric_tokens = True
    return cfg


class DrainExtractor:
    def __init__(self, sim_th: float = 0.4, depth: int = 4):
        cfg = _build_drain_config(sim_th=sim_th, depth=depth)
        self.miner = TemplateMiner(config=cfg)
        self.counts: dict[str, int] = defaultdict(int)

    def process(self, masked_line: str) -> tuple[str, str, float]:
        result = self.miner.add_log_message(masked_line)
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
        log_tokens = masked_line.split()
        tmpl_tokens = template.split()
        variables = []
        for l_tok, t_tok in zip(log_tokens, tmpl_tokens):
            if t_tok == "<*>":
                variables.append(l_tok)
        return variables

    @property
    def templates(self) -> list[dict]:
        return [
            {"id": str(c.cluster_id), "template": c.get_template(), "count": c.size}
            for c in self.miner.drain.id_to_cluster.values()
        ]


# ─────────────────────────────────────────────────────────────────────────────
# L4 — Template merge + anomaly gate
# ─────────────────────────────────────────────────────────────────────────────

class TemplateMerger:
    def __init__(self, edit_distance_threshold: int = 2, anomaly_count_threshold: int = 3):
        self.edit_dist_th = edit_distance_threshold
        self.anomaly_th = anomaly_count_threshold
        self._merge_map: dict[str, str] = {}

    def _token_edit_distance(self, a: str, b: str) -> int:
        if HAS_LEVENSHTEIN:
            return lev_distance(a, b)
        ta, tb = a.split(), b.split()
        if len(ta) != len(tb):
            return abs(len(ta) - len(tb)) + sum(x != y for x, y in zip(ta, tb))
        return sum(x != y for x, y in zip(ta, tb))

    def build_merge_map(self, templates: list[dict]) -> dict[str, str]:
        canonical: dict[str, str] = {}
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
                    canonical[tj["id"]] = ti["id"]

        self._merge_map = canonical
        return canonical

    def resolve(self, template_id: str) -> str:
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
    def __init__(self, confidence_threshold: float = 0.3, model: str = "gpt-4o-mini", api_key: str | None = None):
        self.threshold = confidence_threshold
        self.model = model
        self._cache: dict[str, str] = {}

        if not HAS_OPENAI:
            self._client = None
        else:
            key = api_key or os.getenv("OPENAI_API_KEY")
            self._client = OpenAI(api_key=key) if key else None

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
                messages=[{"role": "user", "content": LLM_PROMPT.format(log_line=raw_line)}],
                temperature=0,
                max_tokens=200,
            )
            template = response.choices[0].message.content.strip()
            self._cache[raw_line] = template
            return template
        except Exception as exc:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class LogTemplateExtractor:
    def __init__(
        self,
        extra_masks: list[tuple[str, str]] | None = None,
        sim_th: float = 0.4,
        depth: int = 4,
        edit_distance_threshold: int = 2,
        anomaly_count_threshold: int = 3,
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

    def process_line(self, raw_line: str) -> LogEvent:
        raw_line = raw_line.rstrip("\n")
        masked = self.prefilter.mask(raw_line)
        template_id, template, confidence = self.drain.process(masked)
        variables = self.drain.extract_variables(masked, template)
        source = "drain"

        if self.llm and self.llm.should_fallback(confidence):
            llm_template = self.llm.extract_template(raw_line)
            if llm_template:
                template = llm_template
                source = "llm_fallback"
                confidence = 1.0

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

    def _apply_merge_and_anomaly(self, events: list[LogEvent]) -> None:
        templates = self.drain.templates
        merge_map = self.merger.build_merge_map(templates)

        for ev in events:
            canonical = merge_map.get(ev.template_id, ev.template_id)
            if canonical != ev.template_id:
                ev.merged_into = canonical
            count = self.drain.counts.get(ev.template_id, 0)
            ev.is_anomaly = self.merger.is_anomaly(count)

    def get_templates(self) -> list[dict]:
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
        with open(path, "w", encoding="utf-8") as fh:
            for ev in self._events:
                fh.write(json.dumps(ev.to_dict()) + "\n")
        print(f"Written {len(self._events)} events → {path}")

    # ── NEW: CSV Export Method ───────────────────────────────────────────────
    def to_csv(self, path: str) -> None:
        """Write the unique discovered templates to a CSV file matching standard loghub format."""
        templates = self.get_templates()
        
        with open(path, mode="w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            # Write exactly two columns: EventId (formatted as E#) and EventTemplate
            for t in templates:
                event_id = f"E{t['id']}"
                writer.writerow([event_id, t['template']])
        
        print(f"Written {len(templates)} unique templates → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Command Line Execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python log_template_extractor.py <path_to_log_file>")
        sys.exit(1)

    log_file_path = sys.argv[1]

    print("=" * 60)
    print(f"Log Template Extractor — Processing: {log_file_path}")
    print("=" * 60)

    extractor = LogTemplateExtractor(
        sim_th=0.4,
        depth=4,
        edit_distance_threshold=2,
        anomaly_count_threshold=2,
        enable_llm=False,
    )

    try:
        events = extractor.process_file(log_file_path)
    except FileNotFoundError:
        print(f"Error: Could not find file '{log_file_path}'")
        sys.exit(1)
        
    summary = extractor.summary()
    print(f"Processed {summary['total_lines']} lines → Found {summary['unique_templates']} unique templates.")

    # 1. Save the full structured event logs
    jsonl_output = f"{log_file_path}.parsed.jsonl"
    extractor.to_jsonl(jsonl_output)

    # 2. Save the isolated template list (This matches your image!)
    csv_output = f"{log_file_path}_templates.csv"
    extractor.to_csv(csv_output)