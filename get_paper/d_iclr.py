import argparse
import os
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

import openreview


def parse_args():
    p = argparse.ArgumentParser(description="Download ICLR PDFs by year (accepted only).")
    p.add_argument("--start", type=int, default=2021, help="Start year (default: 2021).")
    p.add_argument("--end", type=int, default=2025, help="End year inclusive (default: 2025).")
    p.add_argument("-o", "--outdir", type=str, default="~/hpc-share/revolutionizing-higher-ed-rankings/Downloads/ICLR", help="Output root directory (default: ./ICLR).")
    p.add_argument("-j", "--jobs", type=int, default=8, help="Concurrent download workers (default: 8).")
    p.add_argument("--username", type=str, default=os.getenv("OPENREVIEW_USERNAME"), help="OpenReview username (optional).")
    p.add_argument("--password", type=str, default=os.getenv("OPENREVIEW_PASSWORD"), help="OpenReview password (optional).")
    p.add_argument("--dry-run", action="store_true", help="List targets without downloading.")
    return p.parse_args()

# ---------------- Clients ----------------
def make_clients(username=None, password=None):
    # API2 client (new venues)
    c2 = openreview.api.OpenReviewClient(
        baseurl="https://api2.openreview.net",
        username=username if username else None,
        password=password if username and password else None,
    )
    # API1 client (legacy venues)
    c1 = openreview.Client(
        baseurl="https://api.openreview.net",
        username=username if username else None,
        password=password if username and password else None,
    )
    return c1, c2

# ---------------- Utils ----------------
_slug_pat = re.compile(r"[^-\w\u4e00-\u9fff]+", flags=re.UNICODE)
def slugify(title: str, maxlen: int = 120) -> str:
    if not title:
        return "untitled"
    title = title.strip().replace("—", "-").replace("–", "-").replace("/", "-")
    title = re.sub(r"\s+", " ", title).replace(" ", "_")
    title = _slug_pat.sub("", title)
    return title[:maxlen].strip("_") or "untitled"

def title_from_note(note):
    """Robustly extract title across API1/2 shapes."""
    try:
        c = getattr(note, "content", None)
        if not isinstance(c, dict):
            return ""
        t = c.get("title")
        if isinstance(t, dict):
            # API2 常见: {"value": "..."}
            return t.get("value") or ""
        elif isinstance(t, str):
            return t
        else:
            return ""
    except Exception:
        return ""

def plan_filename(root: Path, year: int, note) -> Path:
    """Safe filename planning even if number/title missing."""
    title = slugify(title_from_note(note)) or "untitled"
    num = getattr(note, "number", None)
    nid = getattr(note, "id", "") or "noid"
    name = f"{num:04d}_{title}_[{nid}].pdf" if isinstance(num, int) else f"{title}_[{nid}].pdf"
    return (root / str(year) / name)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# def plan_filename(root: Path, year: int, note) -> Path:
#     title = slugify(title_from_note(note))
#     num = getattr(note, "number", None)
#     name = f"{num:04d}_{title}_[{note.id}].pdf" if isinstance(num, int) else f"{title}_[{note.id}].pdf"
#     return root / str(year) / name

# ---------------- API version detection ----------------
def is_api2_for_year(client_v2, year: int) -> bool:
    venue_id = f"ICLR.cc/{year}/Conference"
    try:
        g = client_v2.get_group(venue_id)
        return bool(getattr(g, "domain", None))  # API2 has domain; API1 returns None
    except Exception:
        return False

# ---------------- List accepted notes ----------------
def list_accepted_api2(client_v2, year: int):
    venue_id = f"ICLR.cc/{year}/Conference"
    # Filter accepted papers by venueid (API2)
    try:
        notes = client_v2.get_all_notes(content={"venueid": venue_id}, select="id,number,content.title,content.pdf")
    except Exception:
        notes = client_v2.get_all_notes(content={"venueid": venue_id})
    return notes or []

def list_accepted_api1(client_v1, year: int):
    inv = f"ICLR.cc/{year}/Conference/-/Blind_Submission"
    subs = client_v1.get_all_notes(invitation=inv, details="directReplies,original")
    accepted = []
    for s in subs:
        replies = (s.details or {}).get("directReplies", []) or []
        decisions = [r for r in replies if str(r.get("invitation", "")).endswith("Decision")]
        is_accept = any("Accept" in str((d.get("content", {}) or {}).get("decision", "")) for d in decisions)
        if not is_accept:
            continue

        orig_json = (s.details or {}).get("original") or {}
        try:
            if orig_json:
                accepted.append(openreview.Note.from_json(orig_json))
            else:
                accepted.append(s)  # graceful fallback
        except Exception:
            accepted.append(s)      # still fallback if from_json fails
    return accepted

class DownloadError(Exception): 
    pass

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=30),
       retry=retry_if_exception_type(DownloadError), reraise=True)
def fetch_pdf_api2(client_v2, note):
    try:
        return client_v2.get_attachment(field_name="pdf", id=note.id)
    except Exception as e:
        raise DownloadError(e)

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=30),
       retry=retry_if_exception_type(DownloadError), reraise=True)

def fetch_pdf_api1(client_v1, note):
    try:
        data = client_v1.get_attachment(getattr(note, "id", None), "pdf")
        if data:
            return data
    except Exception:
        pass

    forum_id = getattr(note, "forum", None)
    if forum_id:
        try:
            data = client_v1.get_attachment(forum_id, "pdf")
            if data:
                return data
        except Exception:
            pass

    try:
        return client_v1.get_pdf(getattr(note, "id", None))
    except Exception as e:
        raise DownloadError(e)

def download_one(fetch_fn, note, outpath: Path, dry_run=False):
    if outpath.exists():
        return "skip_exists"
    if dry_run:
        return "dry_run"
    pdf_bytes = fetch_fn(note)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "wb") as f:
        f.write(pdf_bytes)
    time.sleep(0.05)
    return "ok"

def main():
    args = parse_args()
    if args.start > args.end:
        print("Start year must be <= end year.", file=sys.stderr); sys.exit(2)

    client_v1, client_v2 = make_clients(args.username, args.password)
    root = Path(args.outdir) / "ICLR"

    for year in range(args.start, args.end + 1):
        api2 = is_api2_for_year(client_v2, year)
        notes = list_accepted_api2(client_v2, year) if api2 else list_accepted_api1(client_v1, year)

        if not notes:
            print(f"[{year}] No accepted notes found (API{2 if api2 else 1}).")
            continue

        ensure_dir(root / str(year))
        tasks = [(note, plan_filename(root, year, note)) for note in notes]

        status = {"ok": 0, "skip_exists": 0, "dry_run": 0, "fail": 0}
        fetch = (lambda n: fetch_pdf_api2(client_v2, n)) if api2 else (lambda n: fetch_pdf_api1(client_v1, n))

        with ThreadPoolExecutor(max_workers=max(args.jobs, 1)) as ex:
            futs = [ex.submit(download_one, fetch, n, p, args.dry_run) for (n, p) in tasks]
            for (task, fut) in tqdm(zip(tasks, futs), total=len(tasks), desc=f"ICLR {year} (API{2 if api2 else 1})"):
                (n, p) = task
                try:
                    r = fut.result(); status[r] += 1
                except Exception as e:
                    status["fail"] += 1
                    print(f"[{year}] FAIL id={n.id} -> {e}", file=sys.stderr)

if __name__ == "__main__":
    main()