"""
Document Classifier Engine.
Performs multi-signal template scoring over extracted OCR text without relying on filenames.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from kyc_sorter.detector.templates import TemplateSignature, get_template_signatures

logger = logging.getLogger(__name__)

# Optional fast fuzzy matching via RapidFuzz
try:
    from rapidfuzz import fuzz as _rf_fuzz
    from rapidfuzz import process as _rf_process
    _HAS_RAPIDFUZZ = True
except ImportError:
    from difflib import SequenceMatcher
    _HAS_RAPIDFUZZ = False

# Regex for word tokenization & space-collapsed text
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

# Core identity document templates
_PRIMARY_IDENTITY_TEMPLATES = frozenset({
    "AADHAAR", "PAN", "VOTER_ID", "DRIVING_LICENCE", "PASSPORT",
    "BANK_STATEMENT", "BANK_PASSBOOK", "RATION_CARD", "GOVT_ID",
})

# Administrative templates prone to incidental matches
_BROAD_ADMIN_TEMPLATES = frozenset({
    "CKYC_CONFIRMATION", "IT_RETURNS", "REGISTRATION_CERTIFICATE",
    "CKYC_DECLARATION", "MUNICIPAL_TAX_RECEIPT",
})

# Admin form markers that distinguish checklist mentions from actual standalone documents
_ADMIN_FORM_MARKERS = frozenset({
    "cersai", "central kyc registry", "central kyc", "customer information form",
    "kyc number", "aadhaar vault", "registration and stamp", "certificate of encumbrance",
    "indian income tax return acknowledgement", "e-filing acknowledgement",
})

# Distinctive anchor signatures for detecting multiple KYC documents in a single image
_DOC_ANCHORS: Dict[str, Tuple[List[str], List[str]]] = {
    "AADHAAR": (
        ["uidai", "unique identification authority", "aadhaar", "aadhar", "help@uidai", "www.uidai", "1947", "uniqueidentificationauthorityofindia"],
        [r"\b\d{4}\s?\d{4}\s?\d{4}\b", r"\buid\s*[:/\-]?\s*\d"],
    ),
    "PAN": (
        ["income tax department", "permanent account number", "income tax dept", "aaykar vibhag"],
        [r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"],
    ),
    "VOTER_ID": (
        ["election commission of india", "elector photo identity card", "elector's name", "electoral photo identity card", "epic no", "epic"],
        [r"\b[A-Z]{3}\d{7}\b", r"\b[A-Z]{3}[0-9]{7}\b"],
    ),
    "PASSPORT": (
        ["republic of india passport", "government of india passport", "passeport", "passport no", "passport number", "ministry of external affairs", "republic of india"],
        [r"\b[A-PR-WYa-pr-wy][1-9]\d\s?\d{4}[1-9]\b", r"P<IND", r"\b[A-Z]\d{7}\b"],
    ),
    "DRIVING_LICENCE": (
        ["driving licence", "driving license", "motor vehicles department", "union of india driving licence", "form 7 rule 16"],
        [r"\b[A-Z]{2}[-\s]?\d{2}[-\s]?\d{4}[-\s]?\d{7}\b", r"\b[A-Z]{2}\d{13,14}\b"],
    ),
    "BANK_STATEMENT": (
        ["bank statement", "account statement", "statement of account", "savings bank passbook", "passbook", "pass book", "statement period", "opening balance", "closing balance", "ifsc", "micr"],
        [r"\b[A-Z]{4}0[A-Z0-9]{6}\b", r"\baccount\s*(type|number)\s*[:/]", r"\bcustomer\s*id\s*[:/]"],
    ),
    "ELECTRICITY_BILL": (
        ["electricity bill", "energy charges", "units consumed", "discom", "power distribution", "kwh"],
        [r"\bunits\s+consumed\b", r"\benergy\s+charges\b", r"\bmeter\s+no\.?\b", r"\belectricity\s+bill\b"],
    ),
    "GAS_BILL": (
        ["gas bill", "lpg", "lpg connection", "bharatgas", "indane", "hp gas", "type of cylinders", "lpgid"],
        [r"\blpgid\b", r"\blpg\b", r"\bgas\s+bill\b"],
    ),
    "PHONE_BILL": (
        ["phone bill", "telephone bill", "monthly bill for mobile", "mobile services", "airtel", "bsnl", "vodafone", "jio"],
        [r"\bmobile\s+number\s*[:\-]?\s*\d{10}\b", r"\bbill\s+number\b", r"\bcall\s+charges\b"],
    ),
    "RATION_CARD": (
        ["ration card", "household card", "fair price shop", "food and civil supplies", "head of household"],
        [r"\bration\s+card\s+number\b", r"\bhousehold\s+card\b", r"\bfp\s+shop\b"],
    ),
    "REGISTRATION_CERTIFICATE": (
        ["certificate of registration", "transport department", "chassis number", "engine number", "vehicle class", "registering authority"],
        [r"\bregn\s+no\.?\s*[:\-]?\s*[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{4}\b", r"\bchassis\s+number\b", r"\bengine\s+number\b"],
    ),
    "CKYC_CONFIRMATION": (
        ["central kyc registry", "central kyc", "cersai", "customer information form", "kyc number", "aadhaar vault"],
        [r"\bkyc\s+number\b", r"\bcersai\b", r"\bsearch\s+reference\s+id\b"],
    ),
    "CKYC_DECLARATION": (
        ["form no. 60", "form no.60", "form 60", "rule 114b", "form for declaration to be filed", "declaration under ckyc", "ckyc declaration", "kyc declaration"],
        [r"\bform\s+no\.?\s*60\b", r"\brule\s+114b\b", r"\bdeclaration\s+to\s+be\s+filed\b"],
    ),
    "MUNICIPAL_TAX_RECEIPT": (
        ["property tax receipt", "municipal tax receipt", "municipal corporation", "property tax bill", "nagar nigam", "holding tax receipt", "greater chennai corporation"],
        [r"\bproperty\s+tax\s+receipt\b", r"\bmunicipal\s+corporation\b", r"\bproperty\s+tax\b"],
    ),
    "GOVT_ID": (
        ["government identity card", "govt identity card", "central government employee", "state government employee", "service identity card", "official identity card", "employee photo identity card"],
        [r"\b(government|govt)\s+identity\s+card\b", r"\bidentity\s+card\s+no\.?\b"],
    ),
    "IT_RETURNS": (
        ["indian income tax return", "income tax return acknowledgement", "itr-1", "itr-2", "itr-3", "itr-4", "e-filing acknowledgement"],
        [r"\bITR[-\s]?\d\b", r"\be[-\s]?filing\s+acknowledgement\s+number\b"],
    ),
}

_IDENTITY_PREFERENCE_MARGIN = 1.5


@dataclass
class ClassificationResult:
    """Detailed result of KYC document classification."""
    document_type: str
    confidence: float
    status: str  # "IDENTIFIED", "UNKNOWN", "FAILED"
    matched_template: Optional[str] = None
    matched_signals: Dict[str, Any] = field(default_factory=dict)
    raw_score: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_type": self.document_type,
            "confidence": round(self.confidence, 4),
            "status": self.status,
            "matched_template": self.matched_template,
            "raw_score": round(self.raw_score, 2),
            "matched_signals": self.matched_signals,
            "error": self.error,
        }


class DocumentClassifier:
    """
    Multi-signal document classifier using template signatures.
    Supports strong keywords, regular keywords, regex patterns, negative keywords,
    mandatory keyword gates, and space-collapsed OCR matching.
    """

    def __init__(
        self,
        templates: Optional[Dict[str, TemplateSignature]] = None,
        confidence_threshold: float = 0.80,
        min_score: float = 3.0,
        fuzzy_threshold: float = 0.85,
    ):
        self.templates = templates if templates is not None else get_template_signatures()
        self.confidence_threshold = confidence_threshold
        self.min_score = min_score
        self.fuzzy_threshold = fuzzy_threshold

    def classify_text(self, ocr_text: str) -> ClassificationResult:
        """
        Classify document based solely on OCR text content.
        Never uses the filename as evidence.
        """
        if not ocr_text or not ocr_text.strip():
            return ClassificationResult(
                document_type="UNKNOWN",
                confidence=0.0,
                status="UNKNOWN",
                matched_template=None,
                raw_score=0.0,
                matched_signals={"reason": "No text detected in image"},
            )

        text_clean = ocr_text.lower()
        tokens = set(_TOKEN_RE.findall(text_clean))
        text_concat = _NON_ALNUM_RE.sub("", text_clean)

        # PASS 1: Exact matches + regex
        scores: Dict[str, float] = {}
        signals: Dict[str, Dict[str, Any]] = {}

        for tmpl_name, tmpl in self.templates.items():
            if not self._check_mandatory_keywords(tmpl, text_clean, tokens, text_concat):
                scores[tmpl_name] = 0.0
                signals[tmpl_name] = {"mandatory_passed": False}
                continue

            score, tmpl_signals = self._score_exact(tmpl, text_clean, tokens, text_concat)
            scores[tmpl_name] = score
            signals[tmpl_name] = tmpl_signals

        best_template = max(scores, key=scores.get) if scores else None
        best_score = scores.get(best_template, 0.0) if best_template else 0.0

        # PASS 2: Fuzzy fallback if no template cleared min_score
        if best_score < self.min_score:
            for tmpl_name, tmpl in self.templates.items():
                if not signals[tmpl_name].get("mandatory_passed", True):
                    continue
                fuzzy_delta, fuzzy_matches = self._score_fuzzy(tmpl, text_clean, tokens)
                scores[tmpl_name] += fuzzy_delta
                signals[tmpl_name]["fuzzy_matches"] = fuzzy_matches

            best_template = max(scores, key=scores.get) if scores else None
            best_score = scores.get(best_template, 0.0) if best_template else 0.0

        # Section 9: Multiple KYC Documents in One Image Detection
        # A single image must contain EXACTLY ONE recognized KYC document to be sent to KYC.
        # If one image contains two or more different KYC documents, the entire image MUST go to Unknown.
        qualifying_doc_types: Set[str] = set()

        for tmpl_name, tmpl in self.templates.items():
            if tmpl_name in ("BANK_PASSBOOK",):
                continue
            if not signals.get(tmpl_name, {}).get("mandatory_passed", True):
                continue

            tmpl_sig = signals.get(tmpl_name, {})
            tmpl_net = scores.get(tmpl_name, 0.0)
            pos_score = tmpl_sig.get("pos_score", tmpl_net)

            if pos_score >= self.min_score and tmpl_net >= 2.0 and self._check_template_anchor(tmpl_name, text_clean, tokens):
                neg_hits = tmpl_sig.get("negative_matches", [])
                has_admin_marker = any(m in neg_hit.lower() for neg_hit in neg_hits for m in _ADMIN_FORM_MARKERS)
                if not has_admin_marker:
                    qualifying_doc_types.add(tmpl.kyc_name)

        if len(qualifying_doc_types) > 1:
            logger.info(
                "Multiple KYC documents detected in single image: %s -> Routing to UNKNOWN",
                sorted(list(qualifying_doc_types)),
            )
            return ClassificationResult(
                document_type="UNKNOWN",
                confidence=0.0,
                status="UNKNOWN",
                matched_template=None,
                raw_score=0.0,
                matched_signals={
                    "reason": "Multiple KYC documents detected in single image",
                    "detected_documents": sorted(list(qualifying_doc_types)),
                },
            )

        if not best_template or best_score < self.min_score:
            return ClassificationResult(
                document_type="UNKNOWN",
                confidence=round(min(0.49, best_score / (self.min_score * 2.0)), 2) if best_score > 0 else 0.1,
                status="UNKNOWN",
                matched_template=best_template,
                raw_score=best_score,
                matched_signals=signals.get(best_template, {}) if best_template else {},
            )

        # Apply identity preference if broad admin template won by a narrow margin
        best_template, best_score = self._apply_identity_preference(best_template, best_score, scores)

        target_template = self.templates[best_template]
        best_signals = signals.get(best_template, {})

        calculated_confidence = self._compute_confidence(best_score, best_signals)

        if calculated_confidence < self.confidence_threshold:
            logger.info(
                "Document %s scored %.1f but confidence %.2f below threshold %.2f -> UNKNOWN",
                target_template.kyc_name,
                best_score,
                calculated_confidence,
                self.confidence_threshold,
            )
            return ClassificationResult(
                document_type="UNKNOWN",
                confidence=round(calculated_confidence, 4),
                status="UNKNOWN",
                matched_template=best_template,
                raw_score=best_score,
                matched_signals=best_signals,
            )

        return ClassificationResult(
            document_type=target_template.kyc_name,
            confidence=round(calculated_confidence, 4),
            status="IDENTIFIED",
            matched_template=best_template,
            raw_score=best_score,
            matched_signals=best_signals,
        )

    def _compute_confidence(self, score: float, signals: Dict[str, Any]) -> float:
        """Calculate confidence score [0.0 - 1.0] from raw score and matched signals."""
        if score < self.min_score:
            return round(min(0.49, score / (self.min_score * 2.0)), 2) if score > 0 else 0.1
        strong_count = len(signals.get("strong_matches", []))
        regex_count = len(signals.get("regex_matches", []))

        base_conf = 0.70 + min(0.20, (score - self.min_score) / 10.0)
        if strong_count >= 1:
            base_conf += 0.05
        if strong_count >= 2:
            base_conf += 0.03
        if regex_count >= 1:
            base_conf += 0.02
        return min(0.98, base_conf)

    def _check_template_anchor(self, tmpl_key: str, text_clean: str, tokens: Set[str]) -> bool:
        """Check if distinctive anchor keywords or patterns for this document type are present."""
        anchor_info = _DOC_ANCHORS.get(tmpl_key)
        if not anchor_info:
            return False
        anchor_kws, anchor_regexes = anchor_info
        for kw in anchor_kws:
            if self._exact_match(kw, text_clean, tokens):
                return True
        for pattern in anchor_regexes:
            try:
                if re.search(pattern, text_clean, re.IGNORECASE):
                    return True
            except re.error:
                pass
        return False

    def _check_mandatory_keywords(
        self,
        template: TemplateSignature,
        text_clean: str,
        tokens: Set[str],
        text_concat: str,
    ) -> bool:
        """Return True if template has no mandatory keywords or at least one is present."""
        if not template.mandatory_keywords:
            return True
        for kw in template.mandatory_keywords:
            if self._exact_match(kw.lower(), text_clean, tokens, text_concat):
                return True
        return False

    def _score_exact(
        self,
        template: TemplateSignature,
        text_clean: str,
        tokens: Set[str],
        text_concat: str,
    ) -> Tuple[float, Dict[str, Any]]:
        """Score template using exact keyword, concatenated text, and regex matches."""
        score = 0.0
        strong_hits: List[str] = []
        regular_hits: List[str] = []
        regex_hits: List[str] = []
        negative_hits: List[str] = []

        # Strong keywords: +3
        for kw in template.strong_keywords:
            kw_l = kw.lower()
            if self._exact_match(kw_l, text_clean, tokens, text_concat):
                score += 3.0
                strong_hits.append(kw)

        # Regular keywords: +1
        for kw in template.keywords:
            kw_l = kw.lower()
            if self._exact_match(kw_l, text_clean, tokens, text_concat):
                score += 1.0
                regular_hits.append(kw)

        # Regex patterns: +2
        for pattern in template.regex_patterns:
            try:
                if re.search(pattern, text_clean, re.IGNORECASE):
                    score += 2.0
                    regex_hits.append(pattern)
            except re.error:
                pass

        pos_score = score

        # Negative keywords: -2
        for kw in template.negative_keywords:
            kw_l = kw.lower()
            if self._exact_match(kw_l, text_clean, tokens, text_concat):
                score -= 2.0
                negative_hits.append(kw)

        signals = {
            "mandatory_passed": True,
            "pos_score": pos_score,
            "strong_matches": strong_hits,
            "regular_matches": regular_hits,
            "regex_matches": regex_hits,
            "negative_matches": negative_hits,
        }
        return max(0.0, score), signals

    def _score_fuzzy(
        self,
        template: TemplateSignature,
        text_clean: str,
        tokens: Set[str],
    ) -> Tuple[float, List[str]]:
        """Compute delta from fuzzy matches for keywords that did not match exactly."""
        delta = 0.0
        fuzzy_hits: List[str] = []

        for kw in template.strong_keywords:
            kw_l = kw.lower()
            if not self._exact_match(kw_l, text_clean, tokens):
                if self._fuzzy_match(kw_l, text_clean, tokens):
                    delta += 2.0
                    fuzzy_hits.append(f"strong:{kw}")

        for kw in template.keywords:
            kw_l = kw.lower()
            if not self._exact_match(kw_l, text_clean, tokens):
                if self._fuzzy_match(kw_l, text_clean, tokens):
                    delta += 0.5
                    fuzzy_hits.append(f"regular:{kw}")

        for kw in template.negative_keywords:
            kw_l = kw.lower()
            if not self._exact_match(kw_l, text_clean, tokens):
                if self._fuzzy_match(kw_l, text_clean, tokens):
                    delta -= 1.0
                    fuzzy_hits.append(f"negative:{kw}")

        return delta, fuzzy_hits

    @staticmethod
    def _exact_match(
        keyword: str,
        text: str,
        tokens: Set[str],
        text_concat: str = "",
    ) -> bool:
        """Check if keyword appears verbatim in token set, text, or concatenated string."""
        if " " not in keyword and "/" not in keyword and "." not in keyword and "-" not in keyword:
            return keyword in tokens
        if keyword in text:
            return True
        if text_concat:
            kw_concat = _NON_ALNUM_RE.sub("", keyword)
            if kw_concat and len(kw_concat) >= 6 and kw_concat in text_concat:
                return True
        return False

    def _fuzzy_match(self, keyword: str, text: str, tokens: Set[str]) -> bool:
        """Perform fuzzy match using RapidFuzz or SequenceMatcher."""
        if len(keyword) < 4:
            return False

        if _HAS_RAPIDFUZZ:
            best = _rf_process.extractOne(
                keyword,
                tokens,
                scorer=_rf_fuzz.ratio,
                score_cutoff=int(self.fuzzy_threshold * 100),
            )
            if best is not None:
                return True
            partial = _rf_fuzz.partial_ratio(keyword, text)
            return partial >= self.fuzzy_threshold * 100

        # difflib fallback
        from difflib import SequenceMatcher
        for w in tokens:
            if len(w) >= 3 and SequenceMatcher(None, keyword, w).ratio() >= self.fuzzy_threshold:
                return True
        return False

    def _apply_identity_preference(
        self,
        best_template: str,
        best_score: float,
        scores: Dict[str, float],
    ) -> Tuple[str, float]:
        """Prioritize core identity documents when broad administrative templates win by a narrow margin."""
        if best_template not in _BROAD_ADMIN_TEMPLATES:
            return best_template, best_score

        best_id_tmpl = None
        best_id_score = -1.0
        for name, score in scores.items():
            if name in _PRIMARY_IDENTITY_TEMPLATES and score > best_id_score:
                best_id_name = name
                best_id_score = score

        if (
            best_id_tmpl is not None
            and best_id_score >= self.min_score
            and (best_score - best_id_score) <= _IDENTITY_PREFERENCE_MARGIN
        ):
            logger.info(
                "Identity preference applied: Overriding %s (%.1f) with %s (%.1f)",
                best_template,
                best_score,
                best_id_name,
                best_id_score,
            )
            return best_id_name, best_id_score

        return best_template, best_score
