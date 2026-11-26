# research-paper-tracker

Weekly arXiv paper tracker for ads & recommendation systems — big tech papers, most-cited classics, and industry blog watch.

Open `index.html` (published via GitHub Pages) to browse the Weekly Digest, Most Cited papers, Industry Blogs, the Highlights Archive, and your Reading List.

## How it works

### Weekly Action

`.github/workflows/weekly-digest.yml` runs every Monday at 14:00 UTC, and can also be run manually (Actions → Weekly Digest → Run workflow). It sets up Python, installs `requests`, runs `python scripts/fetch_papers.py`, and commits any changed data files back to the `main` branch.

`scripts/fetch_papers.py` (standard library + `requests` only) queries the arXiv API for the last 7 days in `cs.IR`, `cs.LG` and `cs.CL` using keywords for advertising, recommendation, targeting, retrieval, ranking, representation learning, and LLM-based recommendation. It enriches results via the Semantic Scholar Graph API (citation count and affiliations), detects big-tech companies by matching affiliation text against a built-in list (Google, Meta, Amazon, Microsoft, Apple, Netflix, ByteDance, Alibaba, Tencent, Baidu, Kuaishou, Pinterest, LinkedIn, Snap, Uber, Airbnb, Spotify, NVIDIA, Huawei), classifies ads vs non-ads by keyword rules on title + abstract, and classifies topics using the fixed canonical topic taxonomy. It is safe to run repeatedly: papers are deduplicated by arXiv id.

### Data files

- `data/papers.json` — this week's papers, updated by the weekly Action.
- `data/classics.json` — well-known, highly cited papers in recommendation/ads (for example Wide & Deep, DeepFM, and Deep Neural Networks for YouTube Recommendations).
- `data/blogs.json` — industry blog posts from big-tech engineering and research blogs.
- `data/highlights.json` — past Weekly Highlights, one entry per week (`week`, `paper_id`, and a `why` note explaining why it matters). The fetch script appends a new entry each week and never overwrites past entries.

### Topics

Topic filtering uses a fixed canonical taxonomy, with multiple topics per paper and a multi-select topic filter in `index.html`. The canonical topics are: Generative Retrieval, Foundation Models, Representation Learning, Sequence Modeling, User Understanding, Graph Learning, Retrieval, Ranking, Targeting, LLM in RecSys, CTR/CVR Prediction. Use exactly these strings in the data files, in the fetch script's topic classification, and in the UI filter list.

### Weekly Highlight

Each week one paper is featured prominently at the top of the Weekly Digest, with a short "why it matters" note. Past highlights are preserved in `data/highlights.json` and listed by week in the Highlights Archive tab in `index.html`; every highlight can be saved to the reading list like any other paper.

### Enabling GitHub Pages

Settings → Pages → Source: Deploy from a branch → Branch: `main`, Folder: `/ (root)` → Save. The site is then served from the repository root (`index.html`), at `https://wguo123.github.io/research-paper-tracker/`.

### Reading list

Click **Save** on any paper card (Weekly Digest, Most Cited, or Highlights Archive) to add it to **My Reading List**. Saved papers are stored in your browser's `localStorage`, so they persist across visits on the same browser but are not shared between browsers or devices. In the Reading List you can mark a paper read/unread or remove it; clicking **Save** again on a saved paper also removes it.
