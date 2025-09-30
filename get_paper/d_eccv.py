import argparse
import time
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

import re


ECVA_INDEX = "https://www.ecva.net/papers.php"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

def fetch_pdf_links(year: int, with_supp: bool = False, debug=False):
    pages = [
        f"https://www.ecva.net/papers.php?conf=eccv{year}",
        "https://www.ecva.net/papers.php",
    ]
    pdfs, sups = set(), set()
    pat   = re.compile(r'href=["\']([^"\']*eccv_%d[^"\']+\.pdf)["\']' % year, re.I)

    for url in pages:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            html = resp.text
        except Exception:
            continue

        if debug:
            print(f"[debug] {url} bytes={len(html)}")

        for link in pat.findall(html):
            full = urljoin(url, link)
            (sups if full.endswith("-supp.pdf") else pdfs).add(full)

        if pdfs:
            break

    pdfs = sorted(pdfs)
    sups = sorted(sups) if with_supp else []
    if debug:
        print(f"[debug] found {len(pdfs)} pdf, {len(sups)} supp")

    return pdfs, sups
# def fetch_pdf_links(year: int, with_supp: bool = False):

#     candidate_pages = [
#         f"https://www.ecva.net/papers.php?conf=eccv{year}",
#         "https://www.ecva.net/papers.php",
#     ]

#     pdfs, sups = set(), set()
#     prefix = f"/papers/eccv_{year}/" 

#     for url in candidate_pages:
#         try:
#             html = requests.get(url, headers=HEADERS, timeout=60).text
#         except Exception:
#             continue

#         soup = BeautifulSoup(html, "html.parser")

#         for a in soup.select('a[href$=".pdf"]'):
#             href = a.get("href", "")
#             if prefix not in href:
#                 continue
#             link = urljoin(url, href)
#             (sups if link.endswith("-supp.pdf") else pdfs).add(link)

#         if pdfs:                      
#             break

#     pdfs = sorted(pdfs)
#     sups = sorted(sups) if with_supp else []

#     return pdfs, sups


def _safe_filename(url: str) -> str:
    name = url.split("/")[-1]
    return "".join(c for c in name if c.isalnum() or c in ("-", "_", ".")).strip()

def download_one(url: str, outdir: Path, session: requests.Session, max_retries=3):
    outdir.mkdir(parents=True, exist_ok=True)
    fname = _safe_filename(url)
    fpath = outdir / fname

    if fpath.exists() and fpath.stat().st_size > 0:
        return fname, "skip"

    for attempt in range(1, max_retries + 1):
        try:
            with session.get(url, headers=HEADERS, stream=True, timeout=120) as r:
                r.raise_for_status()
                tmp = fpath.with_suffix(fpath.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 15):
                        if chunk:
                            f.write(chunk)
                tmp.rename(fpath)
            return fname, "ok"
        except Exception:
            if attempt == max_retries:
                return fname, "fail"
            time.sleep(1.0 * attempt + random.uniform(0, 0.5))

def main():
    parser = argparse.ArgumentParser(description="ECCV PDF downloader")
    parser.add_argument("-y", "--years", nargs="*", type=int, default=[2020, 2022, 2024], help="default download year: 20, 22, 24")
    parser.add_argument("-o", "--outdir", type=str, default="ECCV", help="")
    parser.add_argument("-j", "--jobs", type=int, default=8, help="")
    args = parser.parse_args()

    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    for y in sorted(set(args.years)):
        print(f"\n==  ECCV {y} ==")
        pdf_links, supp_links = fetch_pdf_links(y, with_supp=args.with_supp, debug=args.debug)
        if not pdf_links:
            print(f"No ECCV")
            continue

        outdir_year = root / f"ECCV{y}"
        todo = [(url, outdir_year) for url in pdf_links]
        if args.with_supp and supp_links:
            outdir_supp = root / f"ECCV{y}_supp"
            todo += [(url, outdir_supp) for url in supp_links]

        results = []
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(download_one, url, odir, session) for url, odir in todo]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"Downloading {y}"):
                results.append(fut.result())

        ok = sum(1 for _, s in results if s == "ok")
        skip = sum(1 for _, s in results if s == "skip")
        fail = sum(1 for _, s in results if s == "fail")
        print(f"ECCV {y}: ok: {ok}, exist: {skip}, fail: {fail}")

if __name__ == "__main__":
    main()

# python d_eccv.py -o ~/ECCV -j 4 
# python d_eccv.py -y 2024 -j 4 -o ./ECCV