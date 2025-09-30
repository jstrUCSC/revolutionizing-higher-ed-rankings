# python d_neurips.py --years 3 --out ~/hpc-share/revolutionizing-higher-ed-rankings/Downloads/neurips --workers 4

import argparse, os, time, random, concurrent.futures as fut
import requests, bs4, tqdm, urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_INDEX = "https://papers.nips.cc"
SESSION = requests.Session()         
SESSION.headers["User-Agent"] = "Mozilla/5.0 (+github.com/yourname)"

def get_paper_links(year: int):
    idx_url = f"{BASE_INDEX}/paper_files/paper/{year}"
    idx_soup = bs4.BeautifulSoup(SESSION.get(idx_url, timeout=30).text,"html.parser")

    abs_urls = [BASE_INDEX + a["href"]
        for a in idx_soup.select("a[href^='/paper_files/paper/{}']".format(year))]

    pdf_links = []
    for abs_url in abs_urls:
        abs_soup = bs4.BeautifulSoup(SESSION.get(abs_url, timeout=30).text,"html.parser")
        pdf_a = abs_soup.find("a", string="Paper")
        if pdf_a and pdf_a.get("href", "").endswith(".pdf"):
            pdf_links.append(BASE_INDEX + pdf_a["href"])
    return pdf_links

def download_one(url: str, out_dir: str, tries=3):
    fname = url.split("/")[-1]        
    fpath = os.path.join(out_dir, fname)

    # tmp = fpath + ".part"
    # downloaded = os.path.getsize(tmp) if os.path.exists(tmp) else 0

    # for attempt in range(1, tries + 1):
    #     try:
    #         headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
    #         with SESSION.get(url, timeout=60, headers=headers, stream=True) as r:
    #             r.raise_for_status()
    #             mode = "ab" if downloaded else "wb"
    #             with open(tmp, mode) as f:
    #                 for chunk in r.iter_content(chunk_size=1 << 15):
    #                     if chunk:
    #                         f.write(chunk)
    #         os.rename(tmp, fpath)
    #         return
    #
    #     except Exception as e:
    #         if attempt == tries:
    #             print(f"[FAIL] {fname}: {e}")
    #         else:
    #             wait = 2 ** attempt
    #             time.sleep(wait)

    if os.path.exists(fpath):         
        return
    with SESSION.get(url, timeout=120, stream=True) as r:
        r.raise_for_status()
        with open(fpath, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 14):
                if chunk:
                    f.write(chunk)
    
    time.sleep(random.uniform(0.1, 0.3))

# def build_session(retries=5, backoff=0.5, pool_max=100):
#     s = requests.Session()
#     retry = Retry(
#         total=retries,
#         read=retries,
#         connect=retries,
#         backoff_factor=backoff,
#         status_forcelist=[500, 502, 503, 504],
#         allowed_methods=["GET"],
#         raise_on_status=False,
#     )
#     adapter = HTTPAdapter(max_retries=retry, pool_maxsize=pool_max)
#     s.mount("http://", adapter)
#     s.mount("https://", adapter)
#     s.headers["User-Agent"] = "Mozilla/5.0 (+your-email@example.com)"
#     return s

# SESSION = build_session()

def main(years, out_root, workers):
    current = 2025          
    targets = list(range(current - years, current))
    os.makedirs(out_root, exist_ok=True)

    for y in targets:
        print(f"\n=== {y} ===")
        out_dir = os.path.join(out_root, str(y))
        os.makedirs(out_dir, exist_ok=True)
        pdf_links = get_paper_links(y)
        print(f"{len(pdf_links)} papers found.")

        with fut.ThreadPoolExecutor(max_workers=workers) as pool:
            list(tqdm.tqdm(pool.map(lambda u: download_one(u, out_dir), pdf_links),
                           total=len(pdf_links),
                           desc=f"Downloading {y}"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=5,
                        help="years")
    parser.add_argument("--out", type=str, default="~/hpc-share/revolutionizing-higher-ed-rankings/Downloads",
                        help="output d")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel")
    args = parser.parse_args()
    main(args.years, args.out, args.workers)
