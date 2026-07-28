"""
Cross-source player-name normalization and similarity.

The problem this solves: our `players` table stores names as scraped from
Wikipedia ("İlkay Gündoğan", "Łukasz Fabiański", "Lionel Messi (c)"), while an
external rating source (FIFA/FC dataset) writes the same humans its own way
("Ilkay Gundogan", "L. Fabianski", ...). To attach a rating to a player we must
recognise that two differently-written strings are the same name — *without*
ever guessing a wrong match (the project's no-fabrication rule). This module is
the string half of that; the blocking/scoring/tiering logic (nationality + DOB +
club) lives in the matching algorithm that uses these functions.
"""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

# A handful of letters that Unicode NFKD does NOT split into base + combining
# mark, because they are treated as distinct letters, not accented forms.
# Without this map "Ødegaard" keeps its ø and never matches a source that wrote
# "Odegaard". Grown as real mismatches appear — same spirit as former_names /
# the planned club_aliases table, not a claim to be exhaustive.
_LETTER_FOLD = {
    "ø": "o", "Ø": "o",
    "ł": "l", "Ł": "l",
    "đ": "d", "Đ": "d",
    "ð": "d", "Ð": "d",
    "ß": "ss",
    "æ": "ae", "Æ": "ae",
    "œ": "oe", "Œ": "oe",
    "þ": "th", "Þ": "th",
    "ı": "i",  # Turkish dotless i (lowercase) — NFKD leaves it unchanged
}


def normalize_name(name: str) -> str:
    """
    Reduce a name to a diacritic-free, punctuation-free, lowercase token string
    so the same human written two ways compares equal enough to match.

    Steps, in order:
      1. Strip parenthetical markers — real rows carry "(c)" for captain and
         the occasional footnote; "Lionel Messi (c)" must become "lionel messi",
         not leave a spurious "c" token behind.
      2. "Last, First" -> "First Last" — some sources list surname-first.
      3. Unicode NFKD decomposition. NFKD splits an accented character into its
         base letter + a separate "combining" mark (ü -> u + ¨), so the mark can
         then be deleted, leaving the base letter — this is what collapses
         accented and unaccented spellings together.
      4. Delete combining marks (unicodedata.combining(c) != 0 for them).
      5. Fold the handful of letters NFKD leaves alone (ø, ł, ß, ...).
      6. Lowercase; drop apostrophes with NO gap so "N'Golo" -> "ngolo" (not
         "n golo"); turn any other non-alphanumeric run into a single space;
         trim.

    Known, accepted limitation: NFKD maps ü -> u, so a source using the German
    transliteration "Mueller" will NOT collapse to the same string as "Müller"
    (-> "muller"). Those cases fall to fuzzy similarity + the DOB/club block in
    the matcher, or to the manual review queue — never to a silent wrong match.

    Returns "" for empty/None input (callers treat "" as unmatchable, never a
    wildcard), rather than raising.
    """
    if not name:
        return ""

    # 1. drop "(c)" / "(captain)" / footnote parentheticals
    name = re.sub(r"\([^)]*\)", " ", name)

    # 2. surname-first -> forename-first (only the first comma is structural)
    if "," in name:
        surname, _, forename = name.partition(",")
        name = f"{forename} {surname}"

    # 3-4. decompose to base letters + combining marks, then drop the marks
    decomposed = unicodedata.normalize("NFKD", name)
    without_marks = "".join(c for c in decomposed if not unicodedata.combining(c))

    # 5. fold the non-decomposing letters (ø, ł, ß, æ, ...)
    folded = "".join(_LETTER_FOLD.get(c, c) for c in without_marks)

    # 6. lowercase; apostrophes vanish (no space); everything else non-alnum
    #    becomes a space; collapse and trim
    folded = folded.lower().replace("'", "").replace("’", "")  # ' and ’
    folded = re.sub(r"[^a-z0-9]+", " ", folded)
    return folded.strip()


def name_similarity(a: str, b: str) -> float:
    """
    Token-set similarity of two names, in [0, 1].

    Both inputs are normalized first (rapidfuzz does not strip diacritics on its
    own). rapidfuzz.fuzz.token_set_ratio compares the *set* of tokens, so it is
    robust to word order and to one name carrying extra tokens — "lionel messi"
    vs "lionel andres messi cuccitini" scores near 1.0 rather than being
    penalised for the extra forename/surnames, which is exactly what we want for
    Spanish/Portuguese two-surname names and middle names. token_set_ratio
    returns 0-100, so divide by 100.

    Returns 0.0 if either name normalizes to empty — an empty name is never
    "100% similar" to anything.
    """
    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:
        return 0.0
    return fuzz.token_set_ratio(na, nb) / 100.0
