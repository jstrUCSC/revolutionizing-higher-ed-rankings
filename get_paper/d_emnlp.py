import argparse
import os
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

BASE = "https://aclanthology.org"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

def create_session(retries: int = 5, backoff: float = 1.5) -> requests.Session:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=100)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(HEADERS)
    return session

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def event_url(year: int) -> str:
    return f"{BASE}/events/emnlp-{year}/"

def volume_url(slug: str) -> str:
    # slug: e.g., "2024.emnlp-main"
    return f"{BASE}/volumes/{slug}/"

def get_default_slugs_for_year(year: int, include_main=True, include_findings=False, include_demos=False):
    slugs = set()
    if include_main:
        slugs.add(f"{year}.emnlp-main")
    if include_findings:
        slugs.add(f"{year}.findings-emnlp")
    if include_demos:
        slugs.add(f"{year}.emnlp-demos")
    return slugs

def parse_all_volume_slugs_from_event(session: requests.Session, year: int) -> set:
    url = event_url(year)
    r = session.get(url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    slugs = set()
    for a in soup.select('a[href^="/volumes/"]'):
        href = a.get("href", "").strip()
        # e.g. /volumes/2024.emnlp-main/
        slug = href.split("/volumes/")[-1].strip("/")
        if slug.startswith(f"{year}."):
            slugs.add(slug)
    return slugs

def parse_pdfs_from_volume(session: requests.Session, slug: str) -> list:
    url = volume_url(slug)
    r = session.get(url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    pdfs = set()
    for a in soup.select('a[href$=".pdf"]'):
        href = a.get("href", "").strip()
        if not href:
            continue
        full = urljoin(BASE, href)
        basename = os.path.basename(urlparse(full).path)
        if basename.lower() == f"{slug}.pdf".lower():
            continue
        if urlparse(full).netloc.endswith("aclanthology.org"):
            pdfs.add(full)
    return sorted(pdfs)

def plan_for_year(session: requests.Session, year: int, outdir: Path, include_findings: bool, include_demos: bool, all_volumes: bool):

    if all_volumes:
        slugs = parse_all_volume_slugs_from_event(session, year)
    else:
        slugs = get_default_slugs_for_year(
            year,
            include_main=True,
            include_findings=include_findings,
            include_demos=include_demos,
        )

    pdf_urls = []
    for slug in sorted(slugs):
        try:
            urls = parse_pdfs_from_volume(session, slug)
            pdf_urls.extend(urls)
        except Exception as e:
            print(f"[WARN] Failed to parse volume {slug}: {e}")

        time.sleep(0.2)

    tasks = []
    year_dir = outdir / str(year)
    ensure_dir(year_dir)
    for u in pdf_urls:
        fname = os.path.basename(urlparse(u).path)
        dest = year_dir / fname
        tasks.append((u, dest))

    uniq = {}
    for u, d in tasks:
        uniq.setdefault(d.name, (u, d))
    return list(uniq.values())  # [(url, dest_path), ...]

def download_one(session: requests.Session,
                 url: str,
                 dest: Path,
                 timeout: int = 60,
                 overwrite: bool = False,
                 sleep_per_req: float = 0.0) -> bool:
    if dest.exists() and dest.stat().st_size > 0 and not overwrite:
        return True
    if sleep_per_req > 0:
        time.sleep(sleep_per_req)
    with session.get(url, timeout=timeout, stream=True) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 128):
                if chunk:
                    f.write(chunk)
        tmp.rename(dest)
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Download EMNLP PDFs (ACL Anthology) by year, organized into folders."
    )
    parser.add_argument("-o", "--outdir", type=str, default="./EMNLP", help="Output root directory (default: ./EMNLP)")
    parser.add_argument("-y", "--years", nargs="+", type=int, required=True, help="Years to download, e.g., -y 2020 2021 2022")
    parser.add_argument("-j", "--jobs", type=int, default=8, help="Concurrent downloads (default: 8)")
    parser.add_argument("--all-volumes", action="store_true", help="Download ALL volumes listed on events/emnlp-YYYY (includes workshops etc.)")
    parser.add_argument("--retries", type=int, default=5, help="HTTP retries (default: 5)")
    parser.add_argument("--timeout", type=int, default=60, help="Per-request timeout seconds (default: 60)")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds before each PDF GET (soft rate limit, default: 0)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be downloaded and exit")

    args = parser.parse_args()
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    session = create_session(retries=args.retries)

    grand_plan = []  # [(url, dest)]
    for year in sorted(args.years):
        year_plan = plan_for_year(
            session=session,
            year=year,
            outdir=outdir,
            include_findings=args.include_findings,
            include_demos=args.include_demos,
            all_volumes=args.all_volumes,
        )
        print(f"[Plan] {year}: {len(year_plan)} files")
        grand_plan.extend(year_plan)

    if args.dry_run:
        for url, dest in grand_plan[:20]:
            print(f"  {dest}  <-- {url}")
        if len(grand_plan) > 20:
            print(f"  ... and {len(grand_plan)-20} more")
        return

    total = len(grand_plan)
    if total == 0:
        print("[Info] Nothing to download. Bye.")
        return

    pbar = tqdm(total=total, desc="Downloading", unit="file") if tqdm else None
    ok = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = []
        for url, dest in grand_plan:
            futures.append(
                ex.submit(
                    download_one,
                    session,
                    url,
                    dest,
                    timeout=args.timeout,
                    overwrite=args.overwrite,
                    sleep_per_req=args.sleep,
                )
            )
        for fut in as_completed(futures):
            try:
                res = fut.result()
                ok += 1 if res else 0
            except Exception as e:
                print(f"[ERR] {e}")
            finally:
                if pbar:
                    pbar.update(1)
    if pbar:
        pbar.close()
    print(f"[Done] Downloaded/verified: {ok}/{total}")

if __name__ == "__main__":
    main()


# python d_emnlp.py -o ~/hpc-share/revolutionizing-higher-ed-rankings/Downloads/EMNLP --years 2020 2021 2022 2023 2024 --jobs 8