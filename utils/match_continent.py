import pandas as pd
import re, unicodedata
from pathlib import Path

UNI_PATH = Path("2_f.csv")
CI_PATH = Path("../CSrankings/country-info.csv")   # institution, countryabbrv, (region)
COUNTRIES_PATH = Path("../CSrankings/countries.csv")  # alpha_2, region, sub_region, ...

def strip_accents(s: str) -> str:
    if s is None: return ""
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))

def norm_uni(s: str) -> str:
    s = strip_accents(s).lower().strip()
    s = re.sub(r"\buniv\b\.?", "university", s)
    s = s.replace("&", " and ")
    s = re.sub(r"\binst\b\.?", "institute", s)
    s = re.sub(r"\bu of\b", "university of", s)
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_unknown(x) -> bool:
    s = "" if pd.isna(x) else str(x).strip().lower()
    return s in {"", "unknown", "na", "none", "null"}

def region_to_continent(region, sub_region):
    r = ("" if pd.isna(region) else str(region)).strip().lower()
    sr = ("" if pd.isna(sub_region) else str(sub_region)).strip().lower()
    if r == "asia": return "Asia"
    if r == "europe": return "Europe"
    if r == "africa": return "Africa"
    if r == "oceania": return "Australasia"
    if r in {"americas", "america"}:
        if "south america" in sr:
            return "South America"
        # Northern America / Central America / Caribbean 
        return "North America"
    if r == "antarctica": return "Antarctica"
    return "Unknown"

df = pd.read_csv(UNI_PATH)
ci = pd.read_csv(CI_PATH)
cc = pd.read_csv(COUNTRIES_PATH)

for need in ["University", "Continent"]:
    if need not in df.columns:
        if need == "Continent":
            df["Continent"] = "Unknown"
        else:
            raise RuntimeError(f"3cv_f.csv 缺少列: {need}")

ci_cols = {c.lower(): c for c in ci.columns}
inst_col = ci_cols.get("institution")
abbr_col = ci_cols.get("countryabbrv")
if not inst_col or not abbr_col:
    raise RuntimeError("country-info.csv: institution, countryabbrv")

cc_cols = {c.lower(): c for c in cc.columns}
alpha2_col = cc_cols.get("alpha_2") or cc_cols.get("alpha-2") or cc_cols.get("alpha2")
region_col = cc_cols.get("region")
sub_region_col = cc_cols.get("sub_region") or cc_cols.get("sub-region") or cc_cols.get("subregion")
if not (alpha2_col and region_col and sub_region_col):
    raise RuntimeError("countries.csv: alpha_2, region, sub_region")

# 3) institution -> alpha2
ci_tmp = ci[[inst_col, abbr_col]].dropna()
ci_tmp["__inst_norm"] = ci_tmp[inst_col].astype(str).map(norm_uni)
ci_tmp["__alpha2"] = ci_tmp[abbr_col].astype(str).str.upper().str.strip()
inst2alpha2 = dict(zip(ci_tmp["__inst_norm"], ci_tmp["__alpha2"]))

# 4) alpha2 -> (region, sub_region) -> Continent
cc_tmp = cc[[alpha2_col, region_col, sub_region_col]].dropna(subset=[alpha2_col])
cc_tmp["__alpha2"] = cc_tmp[alpha2_col].astype(str).str.upper().str.strip()
alpha2_to_cont = {
    row["__alpha2"]: region_to_continent(row[region_col], row[sub_region_col])
    for _, row in cc_tmp.iterrows()
}

df["__uni_norm"] = df["University"].astype(str).map(norm_uni)
mask = df["Continent"].apply(is_unknown)
df.loc[mask, "__alpha2"] = df.loc[mask, "__uni_norm"].map(inst2alpha2)
df.loc[mask, "Continent"] = df.loc[mask, "__alpha2"].map(alpha2_to_cont).fillna(df.loc[mask, "Continent"])

hard_overrides = {
    "Northwestern University": "North America",
    "Northeastern University": "North America",
    "Univ. of California - Berkeley": "North America",
    "University of California - Berkeley": "North America",
}
df["Continent"] = df.apply(lambda r: hard_overrides.get(r["University"], r["Continent"]), axis=1)

df.loc[df["Continent"].apply(is_unknown), "Continent"] = "North America"

df.drop(columns=[c for c in ["__uni_norm", "__alpha2"] if c in df.columns], inplace=True)


df.to_csv("../public/2_f.csv", index=False)
print("[Done] Continent filled using CSrankings polygons.")
