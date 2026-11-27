"""Weekly arXiv paper tracker for ads & recommendation systems.

Queries the arXiv API for the last 7 days in cs.IR, cs.LG and cs.CL,
enriches results via the Semantic Scholar Graph API, classifies each
paper with the fixed canonical topic taxonomy, detects big-tech
companies from affiliations, and writes/updates data/papers.json.

Also appends one Weekly Highlight per week to data/highlights.json
(never overwriting past entries).

Runnable as:  python scripts/fetch_papers.py
Safe to run repeatedly: papers are deduplicated by arXiv id, and a
week is only added to highlights.json once.

Standard library + requests only.
"""

import json
import re
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
PAPERS_PATH = ROOT / "data" / "papers.json"
HIGHLIGHTS_PATH = ROOT / "data" / "highlights.json"
ARXIV_API = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/arXiv:"

CATEGORIES = ["cs.IR", "cs.LG", "cs.CL"]
KEYWORDS = [
    "advertising", "recommendation", "targeting", "retrieval",
    "ranking", "representation learning", "LLM-based recommendation",
    "recommender system", "click-through rate", "ads ranking",
]

# Fixed canonical topic taxonomy -- use exactly these strings everywhere
# (data files, this script, and the index.html topic filter).
CANONICAL_TOPICS = [
    "Generative Retrieval", "Foundation Models", "Representation Learning",
    "Sequence Modeling", "User Understanding", "Graph Learning",
    "Retrieval", "Ranking", "Targeting", "LLM in RecSys",
    "CTR/CVR Prediction",
]

# Canonical topic -> keywords (matched against title + abstract, lowercased).
TOPIC_KEYWORDS = {
    "Generative Retrieval": ["generative retrieval", "generative retriever", "generative search"],
    "Foundation Models": ["foundation model", "large language model", "llm", "pretrained model", "pre-trained model"],
    "Representation Learning": ["representation learning", "embedding", "representation"],
    "Sequence Modeling": ["sequence model", "sequential recommendation", "sequence modelling", "transformer"],
    "User Understanding": ["user understanding", "user interest", "user behaviour", "user behavior", "user modelling", "user modeling", "user profile"],
    "Graph Learning": ["graph neural", "graph learning", "graph-based", "heterogeneous graph"],
    "Retrieval": ["retrieval", "candidate generation", "dense retrieval", "retriever"],
    "Ranking": ["ranking", "ranker", "learning to rank", "re-ranking", "reranking"],
    "Targeting": ["targeting", "audience", "ad targeting"],
    "LLM in RecSys": ["llm in recsys", "llm-based recommendation", "llm for recommendation", "language model for recommendation", "instruction-tuned"],
    "CTR/CVR Prediction": ["ctr", "cvr", "click-through", "click through", "conversion rate", "conversion prediction"],
}

BIG_TECH = [
    "Google", "Meta", "Amazon", "Microsoft", "Apple", "Netflix",
    "ByteDance", "Alibaba", "Tencent", "Baidu", "Kuaishou",
    "Pinterest", "LinkedIn", "Snap", "Uber", "Airbnb", "Spotify",
    "NVIDIA", "Huawei",
]

ADS_KEYWORDS = [
    "advertis", "ad ranking", "ad targeting", "ad click", "sponsored",
    "auction", "bid", "advertiser", "ad creative", "ads ",
]
NON_ADS_HINTS = ["recommendation", "recommender"]


def week_label(d=None):
    d = d or date.today()
    iso = d.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def classify_topics(title, abstract):
    text = f"{title} {abstract}".lower()
    topics = []
    for topic in CANONICAL_TOPICS:
        for kw in TOPIC_KEYWORDS.get(topic, []):
            if kw in text:
                topics.append(topic)
                break
    return topics or ["Ranking"]


def classify_ads(title, abstract):
    text = f"{title} {abstract}".lower()
    return any(kw in text for kw in ADS_KEYWORDS)


def detect_company(affiliations):
    text = " ".join(affiliations or [])
    for company in BIG_TECH:
        if re.search(rf"\b{re.escape(company)}\b", text, re.IGNORECASE):
            return company
    return None


def arxiv_search(since, max_results=100):
    query = " OR ".join(f'all:"{kw}"' for kw in KEYWORDS)
    cat = " OR ".join(f"cat:{c}" for c in CATEGORIES)
    params = {
        "search_query": f"({query}) AND ({cat})",
        "sortBy": "submittedDate", "sortOrder": "descending",
        "start": 0, "max_results": max_results,
    }
    resp = requests.get(ARXIV_API, params=params, timeout=60,
                        headers={"User-Agent": "research-paper-tracker/1.0"})
    resp.raise_for_status()
    ns = {"atom": "http://www.w3.org/2005/Atom",
          "arxiv": "http://arxiv.org/schemas/atom"}
    root = ET.fromstring(resp.text)
    out = []
    for entry in root.findall("atom:entry", ns):
        arxiv_id = entry.findtext("atom:id", "", ns).rsplit("/", 1)[-1].strip()
        published = entry.findtext("atom:published", "", ns)[:10]
        if published and published < since.isoformat():
            continue
        links = {l.get("title") or l.get("rel"): l.get("href")
                 for l in entry.findall("atom:link", ns)}
        out.append({
            "id": arxiv_id,
            "title": " ".join(entry.findtext("atom:title", "", ns).split()),
            "authors": [a.findtext("atom:name", "", ns)
                        for a in entry.findall("atom:author", ns)],
            "summary": " ".join(entry.findtext("atom:summary", "", ns).split()),
            "published": published,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf": f"https://arxiv.org/pdf/{arxiv_id}",
        })
    return out


def enrich(paper):
    """Add citation count and affiliations via Semantic Scholar."""
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
        time.sleep(0.2)  # be polite to the API
    except Exception as exc:  # enrichment is best-effort; never block the run
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


def pick_highlight(new_papers, week):
    """Pick this week's highlight: prefer big-tech papers, then citations."""
    if not new_papers:
        return None
    ranked = sorted(new_papers,
                    key=lambda p: (1 if p.get("company") else 0,
                                   p.get("citations", 0)), reverse=True)
    top = ranked[0]
    topics = ", ".join(top.get("topics") or [])
    company = f" from {top['company']}" if top.get("company") else ""
    why = (f"This week's standout{company} on {topics or 'recommendation'}. "
           f"\"{top['title']}\" is the paper most likely to change how "
           f"ads and recommendation systems are built this week.")
    return {"week": week, "paper_id": top["id"], "why": why}


def main():
    today = date.today()
    since = today - timedelta(days=7)
    week = week_label(today)
    print(f"Fetching arXiv papers since {since} (week {week})...")
    try:
        found = arxiv_search(since)
    except Exception as exc:
        print(f"arXiv query failed: {exc}", file=sys.stderr)
        return 1
    print(f"  {len(found)} candidates from arXiv")

    existing = load_json(PAPERS_PATH)
    seen = {p.get("id") for p in existing}
    new_papers = []
    for raw in found:
        if raw["id"] in seen:
            continue
        text = f"{raw['title']} {raw['summary']}"
        paper = {
            "id": raw["id"], "title": raw["title"],
            "authors": raw["authors"], "company": None,
            "topics": classify_topics(raw["title"], raw["summary"]),
            "ads": classify_ads(raw["title"], raw["summary"]),
            "summary": raw["summary"], "citations": 0,
            "url": raw["url"], "pdf": raw["pdf"],
            "published": raw["published"], "week": week,
        }
        enrich(paper)
        new_papers.append(paper)
        seen.add(paper["id"])
    print(f"  {len(new_papers)} new papers")

    if new_papers:
        save_json(PAPERS_PATH, existing + new_papers)
        print(f"Updated {PAPERS_PATH}")
    else:
        print("No new papers; data/papers.json unchanged.")

    # Highlights: append-only -- never overwrite past entries, one per week.
    highlights = load_json(HIGHLIGHTS_PATH)
    if any(h.get("week") == week for h in highlights):
        print(f"Highlight for {week} already exists; highlights unchanged.")
    elif new_papers:
        entry = pick_highlight(new_papers, week)
        highlights.append(entry)
        save_json(HIGHLIGHTS_PATH, highlights)
        print(f"Appended Weekly Highlight for {week}: {entry['paper_id']}")
    else:
        print("No new papers, so no new highlight this week.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
