"""Weekly arXiv paper tracker for ads & recommendation systems — v2.

What changed in v2 (per reader feedback):
- STRICT relevance gate: a paper must be *about* recommendation / ads /
  targeting / retrieval / ranking for recommendation. Technique words
  alone (retrieval, ranking, graph, transformer...) no longer qualify a
  paper — a domain anchor in the title, or repeated domain anchors in
  the abstract, is required.
- TOP 10 ONLY: each weekly run keeps at most 10 papers, ranked by a
  relevance score (domain fit first, then big-tech company, citations,
  recency). data/papers.json holds the current week's Top 10 only.
- CONCISE SUMMARIES: instead of the raw abstract, each paper gets
  "Innovation: <what is new, one sentence>. Why it matters: <effect on
  targeting / retrieval / ranking / recommendation>."
- LINK INTEGRITY: every kept paper is re-queried on arXiv by id and its
  stored title must match the arXiv title; mismatches are dropped.
  abs/pdf URLs are built only from the API-returned base id.

Runnable as:  python scripts/fetch_papers.py
Standard library + requests only.
"""

import json
import re
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
PAPERS_PATH = ROOT / "data" / "papers.json"
HIGHLIGHTS_PATH = ROOT / "data" / "highlights.json"
ARXIV_API = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/arXiv:"
TOP_N = 10

CATEGORIES = ["cs.IR", "cs.LG", "cs.CL", "cs.AI"]

# Domain anchors: the paper's *subject* must be one of these areas.
DOMAIN_TITLE = [
    "recommendation", "recommender", "recsys", "advertising", "advertisement",
    "ad ranking", "ad targeting", "ads ranking", "click-through",
    "click through rate", "ctr prediction", "conversion prediction",
    "sponsored search", "online advertising", "ad auction",
]
DOMAIN_ABSTRACT = DOMAIN_TITLE + [
    "recommendation system", "recommender system", "product recommendation",
    "news recommendation", "video recommendation", "music recommendation",
    "e-commerce", "online ads", "advertiser", "user-item", "user item interaction",
    "collaborative filtering", "candidate generation for recommendation",
]
# Technique signals: add score/topic tags, but can NEVER qualify alone.
TECHNIQUE = [
    "retrieval", "ranking", "targeting", "representation learning",
    "embedding", "sequential", "graph", "transformer", "foundation model",
    "large language model", "llm", "generative", "ctr", "cvr",
]

CANONICAL_TOPICS = [
    "Generative Retrieval", "Foundation Models", "Representation Learning",
    "Sequence Modeling", "User Understanding", "Graph Learning",
    "Retrieval", "Ranking", "Targeting", "LLM in RecSys",
    "CTR/CVR Prediction",
]
TOPIC_KEYWORDS = {
    "Generative Retrieval": ["generative retrieval", "generative retriever", "generative recommendation"],
    "Foundation Models": ["foundation model"],
    "Representation Learning": ["representation learning", "embedding learning"],
    "Sequence Modeling": ["sequential recommendation", "sequence model", "sequential model"],
    "User Understanding": ["user understanding", "user interest model", "user behavior model", "user behaviour model", "user modeling", "user modelling"],
    "Graph Learning": ["graph neural", "graph learning", "heterogeneous graph"],
    "Retrieval": ["retrieval", "candidate generation", "dense retrieval"],
    "Ranking": ["ranking", "learning to rank", "re-ranking", "reranking"],
    "Targeting": ["targeting", "audience targeting", "ad targeting"],
    "LLM in RecSys": ["llm-based recommendation", "llm for recommendation", "language model for recommendation", "large language model", "llm-based recommender"],
    "CTR/CVR Prediction": ["ctr", "cvr", "click-through", "conversion rate prediction", "conversion prediction"],
}
WHY_IT_MATTERS = {
    "Targeting": "better audience selection decides who sees an ad, which drives campaign reach and efficiency.",
    "Retrieval": "candidate generation sets the recall ceiling for everything downstream in the recommendation pipeline.",
    "Ranking": "better ordering of candidates directly lifts clicks, conversions, and user satisfaction.",
    "CTR/CVR Prediction": "more accurate estimates improve ranking, bidding, and budget allocation in ads.",
    "Generative Retrieval": "generating candidates directly could reshape the classic retrieve-then-rank pipeline.",
    "LLM in RecSys": "shows what LLMs add beyond classic ID-based models in production-style recsys.",
    "Foundation Models": "a shared foundation model could serve ranking, retrieval, and targeting from one backbone.",
    "Representation Learning": "better user/item embeddings improve every stage that consumes them.",
    "Sequence Modeling": "richer sequence models capture intent as it evolves over time.",
    "User Understanding": "better user models sharpen every downstream decision.",
    "Graph Learning": "graph structure surfaces user-item relations that flat features miss.",
}
WHY_DEFAULT = "relevant to how candidates are retrieved, ranked, and served to users."

BIG_TECH = [
    "Google", "Meta", "Amazon", "Microsoft", "Apple", "Netflix",
    "ByteDance", "Alibaba", "Tencent", "Baidu", "Kuaishou",
    "Pinterest", "LinkedIn", "Snap", "Uber", "Airbnb", "Spotify",
    "NVIDIA", "Huawei", "Meituan", "JD.com", "Salesforce", "Adobe",
]
ADS_KEYWORDS = [
    "advertis", "ad ranking", "ad targeting", "ad click", "sponsored",
    "auction", "advertiser", "ad creative",
]
SEARCH_QUERIES = [
    'all:"recommender system"', 'all:"recommendation system"',
    'all:"recommendation" AND all:"ranking"',
    'all:"recommendation" AND all:"retrieval"',
    'all:"advertising" AND all:"ranking"',
    'all:"click-through rate"', 'all:"ctr prediction"',
    'all:"sequential recommendation"', 'all:"generative retrieval" AND all:"recommendation"',
    'all:"llm" AND all:"recommendation"',
    'all:"targeting" AND all:"advertising"',
    'all:"collaborative filtering"',
]


def week_label(d=None):
    d = d or date.today()
    iso = d.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def base_id(arxiv_id):
    return re.sub(r"v\d+$", "", arxiv_id.strip())


def count_hits(text, phrases):
    return sum(1 for p in phrases if p in text)


def relevance(title, abstract):
    """Return (score, qualified). Domain anchor is mandatory."""
    t = title.lower()
    a = abstract.lower()
    title_hits = count_hits(t, DOMAIN_TITLE)
    abs_hits = count_hits(a, DOMAIN_ABSTRACT)
    qualified = title_hits >= 1 or abs_hits >= 2
    if not qualified:
        return 0, False
    tech = count_hits(t + " " + a, TECHNIQUE)
    score = title_hits * 6 + min(abs_hits, 6) * 2 + min(tech, 8)
    return score, True


def classify_topics(title, abstract):
    text = f"{title} {abstract}".lower()
    topics = [topic for topic in CANONICAL_TOPICS
              if any(kw in text for kw in TOPIC_KEYWORDS[topic])]
    return topics[:4]


def classify_ads(title, abstract):
    text = f"{title} {abstract}".lower()
    return any(kw in text for kw in ADS_KEYWORDS)


def detect_company(affiliations):
    text = " ".join(affiliations or [])
    for company in BIG_TECH:
        if re.search(rf"\b{re.escape(company)}\b", text, re.IGNORECASE):
            return company
    return None


def first_sentences(text, cues, max_len):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chosen = None
    for s in sentences:
        if any(c in s.lower() for c in cues):
            chosen = s
            break
    if not chosen and sentences:
        chosen = sentences[0]
    chosen = (chosen or "").strip()
    if len(chosen) > max_len:
        chosen = chosen[:max_len].rsplit(" ", 1)[0] + "…"
    return chosen


def concise_summary(abstract, topics):
    innovation = first_sentences(
        abstract,
        ["we propose", "we present", "we introduce", "this paper proposes",
         "this work proposes", "we develop", "we design", "novel"],
        200)
    why = WHY_IT_MATTERS.get(topics[0], WHY_DEFAULT) if topics else WHY_DEFAULT
    why = why[0].lower() + why[1:] if why else why
    return f"Innovation: {innovation} Why it matters: {why}"


def arxiv_fetch(params):
    resp = requests.get(ARXIV_API, params=params, timeout=60,
                        headers={"User-Agent": "research-paper-tracker/2.0"})
    resp.raise_for_status()
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)
    out = []
    for entry in root.findall("atom:entry", ns):
        raw_id = entry.findtext("atom:id", "", ns).rsplit("/", 1)[-1].strip()
        aid = base_id(raw_id)
        out.append({
            "id": aid,
            "title": " ".join(entry.findtext("atom:title", "", ns).split()),
            "authors": [x.findtext("atom:name", "", ns)
                        for x in entry.findall("atom:author", ns)],
            "summary_raw": " ".join(entry.findtext("atom:summary", "", ns).split()),
            "published": entry.findtext("atom:published", "", ns)[:10],
            "url": f"https://arxiv.org/abs/{aid}",
            "pdf": f"https://arxiv.org/pdf/{aid}",
        })
    return out


def arxiv_search(since, per_query=50):
    cat = " OR ".join(f"cat:{c}" for c in CATEGORIES)
    seen, out = set(), []
    for q in SEARCH_QUERIES:
        try:
            rows = arxiv_fetch({
                "search_query": f"({q}) AND ({cat})",
                "sortBy": "submittedDate", "sortOrder": "descending",
                "start": 0, "max_results": per_query,
            })
        except Exception as exc:
            print(f"  query failed ({q}): {exc}", file=sys.stderr)
            continue
        for r in rows:
            if r["published"] and r["published"] < since.isoformat():
                continue
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            out.append(r)
        time.sleep(0.3)
    return out


def verify_link(paper):
    """Re-query arXiv by id; stored title must match the live title."""
    try:
        rows = arxiv_fetch({"id_list": paper["id"]})
        if not rows:
            return False
        norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())
        return norm(rows[0]["title"]) == norm(paper["title"])
    except Exception as exc:
        print(f"  verify failed for {paper['id']}: {exc}", file=sys.stderr)
        return False


def enrich(paper):
    try:
        url = SEMANTIC_SCHOLAR_API + urllib.parse.quote(paper["id"])
        resp = requests.get(url, params={"fields": "citationCount,authors"},
                            timeout=30)
        if resp.status_code != 200:
            return paper
        data = resp.json()
        paper["citations"] = int(data.get("citationCount") or 0)
        affiliations = []
        for author in data.get("authors") or []:
            affiliations.extend(author.get("affiliations") or [])
        paper["company"] = detect_company(affiliations)
        time.sleep(0.2)
    except Exception as exc:
        print(f"  enrichment failed for {paper['id']}: {exc}", file=sys.stderr)
        paper.setdefault("citations", 0)
        paper.setdefault("company", None)
    return paper


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main():
    today = date.today()
    week = week_label(today)
    # Start with a 7-day window; widen to 14 only if too few qualify.
    found, qualified = [], []
    for days in (7, 14):
        since = today - timedelta(days=days)
        print(f"Fetching arXiv papers since {since} (week {week}, window {days}d)...")
        found = arxiv_search(since)
        qualified = []
        for raw in found:
            score, ok = relevance(raw["title"], raw["summary_raw"])
            if ok:
                raw["_score"] = score
                qualified.append(raw)
        print(f"  {len(found)} candidates, {len(qualified)} pass the relevance gate")
        if len(qualified) >= TOP_N:
            break

    # Rank: relevance, then big-tech, then citations (enriched later is
    # too slow for all) — enrich a shortlist of 25, then final-rank.
    qualified.sort(key=lambda p: p["_score"], reverse=True)
    shortlist = qualified[:25]
    for raw in shortlist:
        enrich(raw)
    shortlist.sort(key=lambda p: (p["_score"],
                                  4 if p.get("company") else 0,
                                  min(p.get("citations", 0), 20)),
                   reverse=True)

    kept = []
    for raw in shortlist:
        if len(kept) >= TOP_N:
            break
        if not verify_link(raw):
            print(f"  DROPPED (link/title mismatch): {raw['id']} {raw['title']}")
            continue
        topics = classify_topics(raw["title"], raw["summary_raw"])
        kept.append({
            "id": raw["id"], "title": raw["title"],
            "authors": raw["authors"], "company": raw.get("company"),
            "topics": topics,
            "ads": classify_ads(raw["title"], raw["summary_raw"]),
            "summary": concise_summary(raw["summary_raw"], topics),
            "citations": raw.get("citations", 0),
            "url": raw["url"], "pdf": raw["pdf"],
            "published": raw["published"], "week": week,
        })
        print(f"  KEPT [{raw['_score']}] {raw['id']} {raw['title'][:80]}")
    print(f"  {len(kept)} papers in this week's Top {TOP_N}")

    # papers.json = current week's Top 10 only (reading-list saves live
    # in the reader's browser, so older weeks are not lost to the user).
    save_json(PAPERS_PATH, kept)
    print(f"Wrote {PAPERS_PATH}")

    highlights = load_json(HIGHLIGHTS_PATH)
    highlights = [h for h in highlights if h.get("week") != week]
    if kept:
        top = kept[0]
        why = top["summary"].split("Why it matters: ")[-1] if "Why it matters: " in top["summary"] else WHY_DEFAULT
        company = f" from {top['company']}" if top.get("company") else ""
        highlights.append({
            "week": week, "paper_id": top["id"],
            "why": f"Top-ranked this week{company}: \"{top['title']}\". {why}",
        })
        print(f"Weekly Highlight for {week}: {top['id']}")
    save_json(HIGHLIGHTS_PATH, highlights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
