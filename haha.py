#!/usr/bin/env python3
"""
Log Template Miner — Drain Algorithm (Production Grade)
=========================================================
Faithful implementation of Drain (He et al., IEEE ICWS 2017) with:
  - Correct fixed-depth parse tree routing
  - Length-aware cluster matching
  - Proper similarity metric (non-wildcard matching fraction)
  - Configurable pre-masking (only truly opaque tokens masked upfront)
  - Post-processing wildcard collapse
  - Rich 3-phase progress bar

Usage
-----
    python log_template_miner_drain.py -i <input.log> -o <output.csv>

Key Options
-----------
    --depth        Parse tree depth            (default: 4)
    --similarity   Merge threshold 0-1         (default: 0.5)
    --max-children Max children per inner node (default: 100)
    --min-count    Min occurrences to output   (default: 1)
    --rex          Extra regex pre-mask pattern (repeatable)
    --no-header-strip  Disable log-header stripping
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# ANSI colours
# ─────────────────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

WC = "<*>"   # wildcard sentinel


# ─────────────────────────────────────────────────────────────────────────────
# Pre-processing  (mask only tokens that are ALWAYS dynamic)
# ─────────────────────────────────────────────────────────────────────────────
# KEY DESIGN DECISION:
#   We mask as LITTLE as possible before Drain sees the line.
#   Structural keywords (PacketResponder, Exception, BLOCK*, allocateBlock…)
#   must stay intact so Drain can use them for tree routing.
#   We only mask things that can never be a useful routing token:
#   block IDs, IPs, hashes, paths, URLs.
#   Pure numbers are intentionally NOT pre-masked; Drain will wildcard them
#   naturally through the merge process.

_DEFAULT_REX: list[str] = [
    # HDFS block IDs  blk_123  blk_-456  blk_123_456
    r'\bblk_-?\d+(?:_\d+)?\b',
    # IPv4 [+port]
    r'\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b',
    # UUIDs
    r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}'
    r'-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b',
    # Long hex strings (hashes / IDs >= 8 hex chars)
    r'\b[0-9a-fA-F]{8,}\b',
    # Unix/Windows file paths
    r'(?:/[\w.\-_]+){2,}',
    r'[A-Za-z]:\\(?:[\w\s.\-_]+\\?)+',
    # URLs
    r'https?://\S+',
]

# Log-header regex — stripped so header tokens don't bleed into routing slots
_HEADER_RE = re.compile(
    r'^(?:'
    r'\d{6}\s+\d{6}\s+\d+\s+|'                                                         # HDFS:  081109 203518 148
    r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:\d{2})?\s*|'  # ISO ts
    r'\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\S+\s+|'                                  # syslog
    r'\[\d{2}[:/]\d{2}[:/]\d{2}(?:[.,]\d+)?\]\s*|'                                     # [HH:MM:SS]
    r'(?:INFO|WARN(?:ING)?|ERROR|DEBUG|FATAL|CRITICAL|TRACE)\s*:?\s*'                   # level
    r')+'
)


def build_rex(patterns: list[str]) -> re.Pattern:
    return re.compile('|'.join(patterns), re.IGNORECASE)


def preprocess(raw: str, rex: re.Pattern, strip_header: bool = True) -> list[str]:
    """Strip log header, apply pre-masking, return clean token list."""
    line = raw.strip()
    if not line:
        return []
    if strip_header:
        m = _HEADER_RE.match(line)
        line = line[m.end():].strip() if m else line
    if not line:
        return []
    masked = rex.sub(WC, line)
    # Collapse adjacent wildcards produced by masking
    masked = re.sub(r'(<\*>\s*){2,}', f'{WC} ', masked)
    return masked.split()


# ─────────────────────────────────────────────────────────────────────────────
# Drain data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LogCluster:
    """One event template cluster (lives at a tree leaf)."""
    id:     int
    tokens: list[str]
    count:  int = 1

    def template(self) -> str:
        """Collapse consecutive wildcards and return template string."""
        out: list[str] = []
        prev_wc = False
        for t in self.tokens:
            if t == WC:
                if not prev_wc:
                    out.append(WC)
                prev_wc = True
            else:
                out.append(t)
                prev_wc = False
        return ' '.join(out)


@dataclass
class Node:
    """Inner tree node."""
    children: dict[str, "Node"]  = field(default_factory=dict)
    clusters: list[LogCluster]   = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Drain parser
# ─────────────────────────────────────────────────────────────────────────────

class Drain:
    """
    Online Drain log parser.

    Tree layout (depth = D):
      Root
        [len]                  ← level 1  (length bucket)
          [token[0]]           ← level 2
            [token[1]]         ← level 3
              ...              ← levels 4…D-1
                [leaf Node]    ← level D  holds list[LogCluster]

    Routing rules
    -------------
    * Each inner level checks tokens[i] for position i = level-2.
    * If a node is full (children >= max_children), new keys fall into a <*> bucket.
    * Wildcard tokens in the incoming log line route through the <*> child.

    Matching rules
    --------------
    * Only clusters whose token-length equals the incoming line's length are compared.
    * Similarity = (# positions where both template and line share the same
                    non-wildcard value) / (# non-wildcard positions in template).
    * Best match wins if its similarity >= threshold.

    Update rule
    -----------
    * Positions that differ between template and matched line → <*>.
    """

    def __init__(
        self,
        depth:        int   = 4,
        similarity:   float = 0.5,
        max_children: int   = 100,
    ) -> None:
        self.depth        = max(depth, 2)
        self.similarity   = similarity
        self.max_children = max_children
        self.root         = Node()
        self._next_id     = 1
        self.id_to_cluster: dict[int, LogCluster] = {}

    # ── public ────────────────────────────────────────────────────────────────

    def add(self, tokens: list[str]) -> LogCluster:
        """Process one tokenised log line; return its cluster."""
        if not tokens:
            return self._new_cluster([])

        leaf    = self._route(tokens)
        cluster = self._best_match(leaf.clusters, tokens)

        if cluster is None:
            cluster = self._new_cluster(tokens)
            leaf.clusters.append(cluster)
        else:
            cluster.tokens = self._update(cluster.tokens, tokens)
            cluster.count += 1

        return cluster

    def sorted_clusters(self) -> list[LogCluster]:
        return sorted(self.id_to_cluster.values(), key=lambda c: -c.count)

    # ── tree routing ──────────────────────────────────────────────────────────

    def _route(self, tokens: list[str]) -> Node:
        node = self.root

        # Level 1 — length bucket
        node = self._child(node, str(len(tokens)))

        # Levels 2 … depth-1 — positional routing
        routing_levels = self.depth - 2
        for i in range(min(routing_levels, len(tokens))):
            tok = tokens[i] if tokens[i] != WC else WC

            if tok in node.children:
                node = node.children[tok]
            elif WC in node.children:
                # Already created a wildcard bucket at this level — use it
                node = node.children[WC]
            elif len(node.children) < self.max_children:
                node = self._child(node, tok)
            else:
                # Node is full: collapse to wildcard slot
                node = self._child(node, WC)

        return node

    @staticmethod
    def _child(node: Node, key: str) -> Node:
        if key not in node.children:
            node.children[key] = Node()
        return node.children[key]

    # ── cluster matching ─────────────────────────────────────────────────────

    def _best_match(
        self,
        candidates: list[LogCluster],
        tokens: list[str],
    ) -> Optional[LogCluster]:
        best: Optional[LogCluster] = None
        best_sim = -1.0

        for c in candidates:
            if len(c.tokens) != len(tokens):
                continue
            sim = self._sim(c.tokens, tokens)
            if sim > best_sim:
                best_sim = sim
                best = c

        return best if best_sim >= self.similarity else None

    @staticmethod
    def _sim(template: list[str], tokens: list[str]) -> float:
        """
        Fraction of template's non-wildcard positions that match the line.
        All-wildcard template → 1.0 (matches everything).
        """
        non_wc = [t for t in template if t != WC]
        if not non_wc:
            return 1.0
        matches = sum(
            1 for t, s in zip(template, tokens)
            if t == s and t != WC
        )
        return matches / len(non_wc)

    # ── template update ───────────────────────────────────────────────────────

    @staticmethod
    def _update(template: list[str], tokens: list[str]) -> list[str]:
        return [t if t == s else WC for t, s in zip(template, tokens)]

    # ── helpers ───────────────────────────────────────────────────────────────

    def _new_cluster(self, tokens: list[str]) -> LogCluster:
        c = LogCluster(id=self._next_id, tokens=tokens[:])
        self._next_id += 1
        self.id_to_cluster[c.id] = c
        return c


# ─────────────────────────────────────────────────────────────────────────────
# Progress bar
# ─────────────────────────────────────────────────────────────────────────────

def progress_bar(cur: int, total: int, label: str, width: int = 40) -> None:
    frac   = cur / total if total else 1.0
    filled = int(width * frac)
    bar    = '█' * filled + '░' * (width - filled)
    sys.stdout.write(
        f'\r{CYAN}{label}{RESET} [{GREEN}{bar}{RESET}] '
        f'{BOLD}{frac*100:5.1f}%{RESET}  ({cur:,}/{total:,})'
    )
    sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def count_lines(path: str) -> int:
    n = 0
    with open(path, 'rb') as f:
        for _ in f:
            n += 1
    return n


def run(
    input_path:   str,
    output_path:  str,
    depth:        int        = 4,
    similarity:   float      = 0.5,
    max_children: int        = 100,
    min_count:    int        = 1,
    encoding:     str        = 'utf-8',
    chunk:        int        = 1000,
    extra_rex:    list[str]  = None,
    strip_header: bool       = True,
) -> None:

    W = 58
    print(f"\n{BOLD}╔{'═'*W}╗{RESET}")
    print(f"{BOLD}║{'  Log Template Miner  ·  Drain Algorithm':^{W}}║{RESET}")
    print(f"{BOLD}╠{'═'*W}╣{RESET}")
    def row(label, val):
        s = f"  {label}: {val}"
        print(f"{BOLD}║{RESET}{s:<{W}}{BOLD}║{RESET}")
    row("Input",      input_path)
    row("Output",     output_path)
    row("Depth",      f"{depth}   Similarity: {similarity}   MaxChildren: {max_children}")
    row("MinCount",   min_count)
    print(f"{BOLD}╚{'═'*W}╝{RESET}\n")

    rex = build_rex(_DEFAULT_REX + (extra_rex or []))

    # ── Phase 1: count ────────────────────────────────────────────────────────
    print(f"{YELLOW}[1/3] Counting lines …{RESET}")
    t0 = time.time()
    total = count_lines(input_path)
    print(f"      {total:,} lines  ({time.time()-t0:.2f}s)\n")

    # ── Phase 2: parse ────────────────────────────────────────────────────────
    print(f"{YELLOW}[2/3] Parsing with Drain …{RESET}")
    t1 = time.time()
    parser  = Drain(depth=depth, similarity=similarity, max_children=max_children)
    skipped = 0

    with open(input_path, encoding=encoding, errors='replace') as fh:
        for idx, raw in enumerate(fh, 1):
            if idx % chunk == 0 or idx == total:
                progress_bar(idx, total, 'Drain     ')

            tokens = preprocess(raw, rex, strip_header)
            if not tokens:
                skipped += 1
                continue
            parser.add(tokens)

    clusters = [c for c in parser.sorted_clusters() if c.count >= min_count]
    print(
        f"\n      {total-skipped:,} lines parsed  ·  {skipped:,} empty skipped"
        f"  →  {len(clusters):,} templates  ({time.time()-t1:.2f}s)\n"
    )

    # ── Phase 3: write ────────────────────────────────────────────────────────
    print(f"{YELLOW}[3/3] Writing CSV …{RESET}")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['EventId', 'EventTemplate', 'Occurrences'])
        for i, c in enumerate(clusters, 1):
            w.writerow([f'E{i}', c.template(), c.count])

    elapsed = time.time() - t0
    print(f"\n{'─'*60}")
    print(f"{GREEN}{BOLD}✔  Done!{RESET}  {len(clusters):,} templates → {output_path}")
    print(f"   Total time : {elapsed:.2f}s")
    print(f"{'─'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Drain log template miner — He et al. ICWS 2017',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('-i', '--input',          required=True,
                   help='Input log file')
    p.add_argument('-o', '--output',         required=True,
                   help='Output CSV file')
    p.add_argument('--depth',        type=int,   default=4,
                   help='Tree depth (default 4)')
    p.add_argument('--similarity',   type=float, default=0.5,
                   help='Merge threshold 0-1 (default 0.5)')
    p.add_argument('--max-children', type=int,   default=100,
                   help='Max node children (default 100)')
    p.add_argument('--min-count',    type=int,   default=1,
                   help='Min occurrences to output (default 1)')
    p.add_argument('--encoding',     default='utf-8',
                   help='File encoding (default utf-8)')
    p.add_argument('--chunk-size',   type=int,   default=1000,
                   help='Progress update interval (default 1000)')
    p.add_argument('--rex',          action='append', default=[],
                   metavar='PATTERN',
                   help='Extra pre-mask regex (repeatable)')
    p.add_argument('--no-header-strip', action='store_true',
                   help='Disable automatic log-header stripping')
    return p.parse_args()


if __name__ == '__main__':
    args = cli()    
    run(
        input_path   = args.input,
        output_path  = args.output,
        depth        = args.depth,
        similarity   = args.similarity,
        max_children = args.max_children,
        min_count    = args.min_count,
        encoding     = args.encoding,
        chunk        = args.chunk_size,
        extra_rex    = args.rex,
        strip_header = not args.no_header_strip,
    )