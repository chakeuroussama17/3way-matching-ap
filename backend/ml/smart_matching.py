"""
Smart fuzzy matching: hybrid of the existing rule-based string score and
semantic embedding similarity, with a learned alias-boost table.

Final confidence
----------------
    confidence = RULE_WEIGHT * rule_similarity
               + EMBED_WEIGHT * embedding_similarity
               + alias_boost            (capped)

- ``rule_similarity``      : normalised token similarity (the existing style
  of matching, preserved and reused here).
- ``embedding_similarity`` : cosine similarity of ``all-MiniLM-L6-v2``
  embeddings **if** ``sentence-transformers`` is installed; otherwise it
  transparently falls back to the rule similarity so the endpoint always works.
- ``alias_boost``          : reward for pairs previously approved by a human,
  e.g. once *"Acme Corporation" → "ACME CORP"* is approved, that pattern is
  boosted on future invoices. Learned from the approved-matches log and
  refreshed monthly via :meth:`SmartMatcher.retrain`.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Optional

from ml.base import LOGS_DIR, ModelMetadata, get_logger, load_artifact, new_version, save_artifact

log = get_logger("smart_matching")

RULE_WEIGHT = 0.70
EMBED_WEIGHT = 0.30
ALIAS_BOOST_MAX = 0.15
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

APPROVED_LOG = LOGS_DIR / "approved_matches.jsonl"

# Legal-entity suffixes stripped during normalisation so "Acme Corporation"
# and "ACME CORP" collapse to the same canonical form.
_SUFFIXES = {
    "corp", "corporation", "inc", "incorporated", "llc", "ltd", "limited",
    "co", "company", "gmbh", "plc", "lp", "llp", "sa", "ag", "bv", "pty",
}

# ── Optional embedding backend (lazy, cached) ─────────────────────────────────
try:  # pragma: no cover - availability depends on the environment
    from sentence_transformers import SentenceTransformer  # type: ignore

    _ST_AVAILABLE = True
except Exception:  # noqa: BLE001
    SentenceTransformer = None  # type: ignore
    _ST_AVAILABLE = False

try:  # rapidfuzz gives a better token ratio than difflib when present
    from rapidfuzz.fuzz import token_sort_ratio as _rf_ratio  # type: ignore

    _RF_AVAILABLE = True
except Exception:  # noqa: BLE001
    _rf_ratio = None  # type: ignore
    _RF_AVAILABLE = False

_embedder = None


def _get_embedder():
    """Lazily load and cache the sentence-transformer model (process-wide)."""
    global _embedder
    if not _ST_AVAILABLE:
        return None
    if _embedder is None:
        log.info("Loading embedding model '%s' (first use may take a few seconds)...", EMBED_MODEL_NAME)
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation and legal suffixes, collapse whitespace."""
    text = re.sub(r"[^\w\s]", " ", (text or "").lower())
    tokens = [t for t in text.split() if t and t not in _SUFFIXES]
    return " ".join(tokens)


class SmartMatcher:
    """Hybrid rule + embedding matcher with a learned alias-boost table."""

    NAME = "smart_matching"

    def __init__(
        self,
        alias_table: Optional[dict[str, dict[str, int]]] = None,
        config: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        # alias_table[normalized_source][normalized_target] = approval_count
        self.alias_table: dict[str, dict[str, int]] = alias_table or {}
        self.config = config or {"rule_weight": RULE_WEIGHT, "embed_weight": EMBED_WEIGHT}
        self.metadata = metadata or {}

    @property
    def uses_embeddings(self) -> bool:
        return _ST_AVAILABLE

    # ── Similarity components ────────────────────────────────────────────────
    @staticmethod
    def _rule_similarity(a: str, b: str) -> float:
        na, nb = normalize_text(a), normalize_text(b)
        if not na or not nb:
            return 0.0
        if _RF_AVAILABLE:
            return _rf_ratio(na, nb) / 100.0
        return SequenceMatcher(None, na, nb).ratio()

    def _embedding_similarity(self, a: str, b: str) -> tuple[float, str]:
        model = _get_embedder()
        if model is None:
            # Graceful fallback: reuse the rule similarity as the semantic proxy.
            return self._rule_similarity(a, b), "fuzzy_fallback"
        va = _embed_cached(a)
        vb = _embed_cached(b)
        # cosine similarity, remapped from [-1,1] to [0,1]
        import numpy as np

        cos = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
        return max(0.0, min(1.0, (cos + 1) / 2)), "embeddings"

    def _alias_boost(self, a: str, b: str) -> tuple[float, Optional[str]]:
        na, nb = normalize_text(a), normalize_text(b)
        targets = self.alias_table.get(na)
        if not targets:
            return 0.0, None
        count = targets.get(nb, 0)
        if count <= 0:
            return 0.0, None
        # Diminishing-returns boost: saturates toward ALIAS_BOOST_MAX.
        boost = ALIAS_BOOST_MAX * (1 - 1 / (1 + count))
        return boost, f"{na} -> {nb} (approved {count}x)"

    # ── Public API ────────────────────────────────────────────────────────────
    def match(self, po_text: str, invoice_text: str) -> dict[str, Any]:
        """Compute hybrid match confidence with a per-component explanation."""
        rule = self._rule_similarity(po_text, invoice_text)
        embed, method = self._embedding_similarity(po_text, invoice_text)
        boost, alias = self._alias_boost(po_text, invoice_text)

        rw = self.config.get("rule_weight", RULE_WEIGHT)
        ew = self.config.get("embed_weight", EMBED_WEIGHT)
        confidence = min(1.0, rw * rule + ew * embed + boost)

        return {
            "po_text": po_text,
            "invoice_text": invoice_text,
            "confidence": round(confidence, 4),
            "is_match": confidence >= 0.75,
            "components": {
                "rule_similarity": round(rule, 4),
                "embedding_similarity": round(embed, 4),
                "alias_boost": round(boost, 4),
            },
            "weights": {"rule": rw, "embedding": ew},
            "method": method,
            "matched_alias": alias,
            "explanation": _build_explanation(rule, embed, boost, alias, method),
        }

    def log_approved_match(self, po_text: str, invoice_text: str) -> None:
        """Persist a human-approved match so future retrains can learn it."""
        record = {"po_text": po_text, "invoice_text": invoice_text, "ts": new_version()}
        with APPROVED_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        # Update the in-memory table immediately for the current process.
        self._ingest_pair(po_text, invoice_text)

    def _ingest_pair(self, po_text: str, invoice_text: str) -> None:
        na, nb = normalize_text(po_text), normalize_text(invoice_text)
        if not na or not nb:
            return
        self.alias_table.setdefault(na, {})
        self.alias_table[na][nb] = self.alias_table[na].get(nb, 0) + 1

    # ── Training / retraining ────────────────────────────────────────────────
    def train(self, df=None, seed_aliases: bool = True) -> ModelMetadata:
        """Initialise the alias table.

        Seeds a few demonstrative aliases and folds in any pairs already in
        the approved-matches log. ``df`` is accepted for a uniform trainer
        interface but not required (matching learns from the approvals log).
        """
        if seed_aliases:
            for src, tgt in [
                ("Acme Corporation", "ACME CORP"),
                ("Globex Industries", "Globex Ind."),
                ("Initech LLC", "Initech"),
            ]:
                self._ingest_pair(src, tgt)
        self._load_approved_log()
        return self._make_meta()

    def retrain(self) -> ModelMetadata:
        """Rebuild the alias table from scratch off the approved-matches log."""
        self.alias_table = {}
        self._load_approved_log()
        log.info("Retrained alias table: %d source patterns", len(self.alias_table))
        return self._make_meta()

    def _load_approved_log(self) -> None:
        if not APPROVED_LOG.exists():
            return
        for line in APPROVED_LOG.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                self._ingest_pair(rec["po_text"], rec["invoice_text"])
            except (json.JSONDecodeError, KeyError):
                continue

    def _make_meta(self) -> ModelMetadata:
        n_pairs = sum(len(v) for v in self.alias_table.values())
        meta = ModelMetadata(
            name=self.NAME,
            version=new_version(),
            trained_at=new_version(),
            n_samples=n_pairs,
            feature_names=["rule_similarity", "embedding_similarity", "alias_boost"],
            metrics={"alias_sources": len(self.alias_table), "alias_pairs": n_pairs},
            extra={"uses_embeddings": _ST_AVAILABLE},
        )
        self.metadata = meta.to_dict()
        return meta

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self) -> None:
        meta = ModelMetadata(**{**_default_meta(self.NAME), **self.metadata})
        save_artifact(self.NAME, {"alias_table": self.alias_table, "config": self.config}, meta)

    @classmethod
    def load(cls) -> Optional["SmartMatcher"]:
        bundle = load_artifact(cls.NAME)
        if not bundle:
            return None
        p = bundle["payload"]
        return cls(alias_table=p.get("alias_table", {}), config=p.get("config"), metadata=bundle.get("metadata", {}))


@lru_cache(maxsize=2048)
def _embed_cached(text: str):
    """Cache embeddings per normalised text to keep repeat scoring fast."""
    model = _get_embedder()
    return model.encode(normalize_text(text), show_progress_bar=False)


def _build_explanation(rule: float, embed: float, boost: float, alias: Optional[str], method: str) -> str:
    parts = [f"rule similarity {rule:.0%}", f"semantic similarity {embed:.0%} ({method})"]
    if boost > 0:
        parts.append(f"learned alias boost +{boost:.0%} [{alias}]")
    return "; ".join(parts)


def _default_meta(name: str) -> dict:
    return {
        "name": name,
        "version": new_version(),
        "trained_at": new_version(),
        "n_samples": 0,
        "feature_names": [],
        "metrics": {},
        "extra": {},
    }


# ── Module singleton ──────────────────────────────────────────────────────────
_MODEL: Optional[SmartMatcher] = None


def get_smart_matcher(reload: bool = False) -> SmartMatcher:
    """Return the process-wide SmartMatcher.

    Unlike the other models this never returns ``None``: if no artifact
    exists yet it returns an unseeded matcher that still works on pure rule
    similarity, so ``/smart-match`` is always available.
    """
    global _MODEL
    if _MODEL is None or reload:
        _MODEL = SmartMatcher.load() or SmartMatcher()
    return _MODEL
