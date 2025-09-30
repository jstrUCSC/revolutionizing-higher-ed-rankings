import re
import os
import sys
import time
import math
import argparse
import itertools
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://aclanthology.org"

CANDIDATE_VOL_SUFFIXES = [
    "acl-long",
    "acl-short",
    "acl-main",       
    "acl-demo",
    "acl-demos",
    "acl-industry",
]

FINDINGS_SUFFIX = "findings-acl"
SRW_SUFFIX = "acl-srw"

ARTICLE_PDF_RE = re.compile(
    r"^https?://aclanthology\.org/\d{4}\.(?:acl-[a-z-]+|findings-acl)\.\d+\.pdf$"
)

FULL_PDF_RE = re.compile(
    r"^https?://aclanthology\.org/\d{4}\.(?:acl-[a-z-]+|findings-acl)\.pdf$"
)

def make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    })
    return s

def get_ok(url, session, timeout=20):
    for i in range(5):
        try:
            resp = session.get(url, timeout=timeout)
            # 2xx
            if 200 <= resp.status_code < 300:
                return resp
            if resp.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(1.5 * (i + 1))
    return None

def volume_exists(year, suffix, session):
    url = f"{BASE}/volumes/{year}.{suffix}/"
    resp = get_ok(url, session)
    return (resp is not None), url, resp

def extract_article_pdf_links(volume_html, base_url):
    soup = BeautifulSoup(volume_html, "html.parser")
    links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/"):
            href = urljoin(BASE, href)
        elif href.startswith("http") is False:
            href = urljoin(base_url, href)

        if ARTICLE_PDF_RE.match(href):
            links.add(href)

    return sorted(links)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sanitize_filename(name: str) -> str:
    return name

def download_one(url, outdir: Path, session, max_retries=4):
    fname = url.split("/")[-1]
    save_path = outdir / sanitize_filename(fname)

    if save_path.exists() and save_path.stat().st_size > 0:
        return ("skip", url, save_path)

    tmp_path = save_path.with_suffix(save_path.suffix + ".part")

    for attempt in range(max_retries):
        try:
            with session.get(url, stream=True, timeout=60) as r:
                if r.status_code != 200:
                    time.sleep(1.2 * (attempt + 1))
                    continue
                total = int(r.headers.get("Content-Length", 0))
                done = 0
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            done += len(chunk)
                if total and done < total * 0.8:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                tmp_path.replace(save_path)
                return ("ok", url, save_path)
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))

    return ("fail", url, save_path)

def collect_volume_links_for_year(year, include_findings, include_srw, session):
    found = []
    for suffix in CANDIDATE_VOL_SUFFIXES:
        exists, vol_url, resp = volume_exists(year, suffix, session)
        if exists and resp is not None:
            pdfs = extract_article_pdf_links(resp.text, vol_url)
            if pdfs:
                found.append((suffix, vol_url, pdfs))

    if include_findings:
        exists, vol_url, resp = volume_exists(year, FINDINGS_SUFFIX, session)
        if exists and resp is not None:
            pdfs = extract_article_pdf_links(resp.text, vol_url)
            if pdfs:
                found.append((FINDINGS_SUFFIX, vol_url, pdfs))

    if include_srw:
        exists, vol_url, resp = volume_exists(year, SRW_SUFFIX, session)
        if exists and resp is not None:
            pdfs = extract_article_pdf_links(resp.text, vol_url)
            if pdfs:
                found.append((SRW_SUFFIX, vol_url, pdfs))

    return found

def main():
    parser = argparse.ArgumentParser(description="Download ACL PDFs by year.")
    parser.add_argument("--from-year", type=int, default=2020, help="Start year (default: 2020)")
    parser.add_argument("--to-year", type=int, default=0, help="End year inclusive (default: current year)")
    parser.add_argument("-o", "--outdir", type=str, default="./ACL", help="Output root directory (default: ./ACL)")
    parser.add_argument("-j", "--jobs", type=int, default=6, help="Concurrent downloads (default: 8)")
    parser.add_argument("--dry-run", action="store_true", help="Only list would-be downloads.")
    args = parser.parse_args()

    from datetime import datetime
    current_year = datetime.now().year
    start_year = max(2021, int(args.from_year))
    end_year = int(args.to_year) if args.to_year else current_year

    include_findings = ("findings" in args.include)
    include_srw = ("srw" in args.include)

    root = Path(args.outdir).resolve()
    ensure_dir(root)

    session = make_session()

    total_links = 0
    plan = []  # list of (year, url)

    for year in range(start_year, end_year + 1):
        volinfo = collect_volume_links_for_year(year, include_findings, include_srw, session)
        if not volinfo:
            print(f"[{year}] no volumes found (or none matched).")
            continue

        year_dir = root / str(year)
        ensure_dir(year_dir)

        year_urls = list(itertools.chain.from_iterable(pdfs for _, _, pdfs in volinfo))

        year_urls = sorted(set(year_urls))
        total_links += len(year_urls)
        for u in year_urls:
            plan.append((year, u))

        vols = ", ".join(s for s, _, _ in volinfo)
        print(f"[{year}] volumes: {vols} | papers: {len(year_urls)}")

    if args.dry_run:
        print(f"\n[DRY RUN] Total planned: {len(plan)} files")
        for y, u in plan[:20]:
            print(f"  {y} -> {u}")
        if len(plan) > 20:
            print(f"  ... and {len(plan)-20} more")
        return

    print(f"\nStart downloading {len(plan)} PDFs with {args.jobs} workers ...")
    successes = 0
    skips = 0
    failures = 0

    def task(item):
        y, url = item
        target_dir = root / str(y)
        ensure_dir(target_dir)
        time.sleep(0.15)
        return download_one(url, target_dir, session)

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = [ex.submit(task, it) for it in plan]
        for fut in as_completed(futures):
            status, url, path = fut.result()
            if status == "ok":
                successes += 1
            elif status == "skip":
                skips += 1
            else:
                failures += 1

            done = successes + skips + failures
            if done % 25 == 0 or status == "fail":
                print(f"[{done}/{len(plan)}] ok={successes} skip={skips} fail={failures}")

    print(f"\nDone. ok={successes}, skip={skips}, fail={failures}, saved under {root}")

if __name__ == "__main__":
    main()



#   python d_acl.py --from-year 2020 --to-year 2025 -o ~/hpc-share/revolutionizing-higher-ed-rankings/Downloads/ACL -j 4 --include findings
