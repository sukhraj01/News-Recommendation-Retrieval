"""Shared tokenization for BM25 indexing and query construction.

One function, used identically on both sides (article text at index time,
history-derived text at query time) — a mismatch here would silently break
term matching. Unicode-aware `\\w+` handles Danish (æ/ø/å) and English with
the same code path, per ADR-002's single-code-path goal.

Stopword removal (combined Danish+English list, since the same tokenizer
runs over both datasets) is not a cosmetic choice — it was added after
benchmarking showed it was load-bearing. EB-NeRD's per-user queries
concatenate very long histories (up to ~1,459 articles) as an unweighted
token multiset; without stopword removal, high-frequency function words
("er", "og", "i", "til"...) dominate the query's term-count mass and BM25's
own idf downweighting isn't enough to compensate, driving EB-NeRD-validation
recall@200 *below* the random baseline (200/11,777 corpus ≈ 1.7%; measured
2.53% pre-fix vs. 4.07% post-fix on a 1,500-impression sample — see
ADR-005's Benchmark Results). Deduplicating the query multiset instead of
removing stopwords was also tested and made recall worse, not better —
repeated terms are informative (topic recurrence in history), so the
multiset is kept; stopwords are filtered instead.
"""
import re

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Compact, standard stopword lists (function words only) — not exhaustive
# lemmatization/NLP treatment, just enough to stop the highest-document-
# frequency words from dominating query term-count mass. See module
# docstring for why this is here and the benchmark evidence behind it.
_DANISH_STOPWORDS = frozenset("""
ad af alle als andet andre at bare blev blive bliver da de dem den denne
der deres det dette dig din dine disse dog du eller en end ens er et for
fra ham han hans har havde have hende hendes her hos hun hvad hvem hvis
hvor hvordan hvorfor hvornar i ikke ind ingen intet jeg jer jeres kan kom
kommer kun kunne lad lidt lige man mange med meget men mig min mine mit
mod ned nej ni noget nogle nu nar ogsa om op os over pa selv si sig sin
sine sit skal skulle som sadan tag tak til ud under var vare ved vi vil
ville vor vores
""".split())

_ENGLISH_STOPWORDS = frozenset("""
a an the and or but if in on at to for of with by from is are was were
be been being this that these those it its as not no do does did have
has had can could will would should may might must i you he she we they
them his her their our your s t just
""".split())

STOPWORDS = _DANISH_STOPWORDS | _ENGLISH_STOPWORDS


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]
