import pandas as pd
import re
import unicodedata
from pathlib import Path
from typing import Optional

UNI_PATH = Path("3_f.csv")
CI_PATH = Path("../CSrankings/country-info.csv")    # CSRankings: institution, region, countryabbrv
COUNTRIES_PATH = Path("../CSrankings/countries.csv")       # UN: name, alpha_2, alpha_3, region, sub_region, ...
OUT_PATH       = Path("../public/3_f_1.csv")               

ALLOW_US_FALLBACK = True
US_ALPHA2 = "US"

def strip_accents(s: str) -> str:
    if s is None:
        return ""
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))

def norm_df_uni(s: str) -> str:
    s = strip_accents(s).lower().strip()
    s = re.sub(r"\buniv\b\.?", "university", s)
    s = s.replace("&", " and ")
    s = re.sub(r"\binst\b\.?", "institute", s)
    s = re.sub(r"\bu of\b", "university of", s)
    s = re.sub(r"\([^)]*\)", " ", s) 
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def norm_ci_inst(s: str) -> str:
    s = strip_accents(s).lower().strip()
    s = re.sub(r"\buniv\b\.?", "university", s)
    s = s.replace("&", " and ")
    s = re.sub(r"\binst\b\.?", "institute", s)
    s = re.sub(r"\bu of\b", "university of", s)
    s = re.sub(r"[^a-z0-9()]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_unknown(x) -> bool:
    s = "" if pd.isna(x) else str(x).strip().lower()
    return s in {"", "unknown", "na", "none", "null"}

def region_to_continent(region, sub_region, intermediate_region=None) -> str:
    def norm(x):
        return ("" if pd.isna(x) else str(x)).strip().lower().replace("_", " ")
    r  = norm(region)
    sr = norm(sub_region)
    ir = norm(intermediate_region)

    if r == "asia": return "Asia"
    if r == "europe": return "Europe"
    if r == "africa": return "Africa"
    if r == "oceania": return "Australasia"

    if r in {"americas", "america"}:
        if "south america" in ir or sr == "south america":
            return "South America"
        return "North America"

    if r == "antarctica": return "Antarctica"
    return "Unknown"

df = pd.read_csv(UNI_PATH)
ci = pd.read_csv(CI_PATH, dtype=str)
cc = pd.read_csv(COUNTRIES_PATH, dtype=str)

if "University" not in df.columns:
    raise RuntimeError(f"{UNI_PATH} must contain column 'University'")
if "Continent" not in df.columns:
    df["Continent"] = "Unknown"
if "Country" not in df.columns:
    df["Country"] = "Unknown"

ci_cols = {c.lower(): c for c in ci.columns}
inst_col = ci_cols.get("institution")
abbr_col = ci_cols.get("countryabbrv")
if not inst_col or not abbr_col:
    raise RuntimeError("country-info.csv must contain: institution, countryabbrv")

cc_cols  = {c.lower(): c for c in cc.columns}
alpha2_c = cc_cols.get("alpha_2") or cc_cols.get("alpha-2") or cc_cols.get("alpha2")
alpha3_c = cc_cols.get("alpha_3") or cc_cols.get("alpha-3") or cc_cols.get("alpha3")
name_c   = cc_cols.get("name")
region_c = cc_cols.get("region")
subreg_c = cc_cols.get("sub_region") or cc_cols.get("sub-region") or cc_cols.get("subregion")
inter_c  = cc_cols.get("intermediate_region") or cc_cols.get("intermediate-region") or cc_cols.get("intermediateregion")
if not (alpha2_c and region_c and subreg_c):
    raise RuntimeError("countries.csv must contain: alpha_2, region, sub_region")

need_cols = [alpha2_c, region_c, subreg_c]
for c in [inter_c, name_c, alpha3_c]:
    if c: need_cols.append(c)

cc_all = cc[need_cols].dropna(subset=[alpha2_c]).copy()
cc_all["__alpha2"] = cc_all[alpha2_c].astype(str).str.upper().str.strip()

alpha2_to_cont = {}
for _, row in cc_all.iterrows():
    alpha2_to_cont[row["__alpha2"]] = region_to_continent(
        row.get(region_c), row.get(subreg_c), row.get(inter_c) if inter_c else None
    )

alpha2_to_name = {}
if name_c:
    for _, row in cc_all.iterrows():
        alpha2_to_name[row["__alpha2"]] = row[name_c]

alpha3_to_alpha2 = {}
if alpha3_c:
    cc_a3 = cc[[alpha2_c, alpha3_c]].dropna(subset=[alpha2_c, alpha3_c])
    for _, row in cc_a3.iterrows():
        alpha3_to_alpha2[str(row[alpha3_c]).upper().strip()] = str(row[alpha2_c]).upper().strip()

ci_tmp = ci[[inst_col, abbr_col]].dropna().copy()
ci_tmp["__inst_norm"] = ci_tmp[inst_col].map(norm_ci_inst)

def to_alpha2(abbr: Optional[str]) -> Optional[str]:
    if abbr is None: return None
    a = str(abbr).strip().upper()
    if not a: return None
    if a in alpha2_to_cont: return a
    if a in alpha3_to_alpha2: return alpha3_to_alpha2[a]
    return None

ci_tmp["__alpha2"] = ci_tmp[abbr_col].map(to_alpha2)
ci_tmp = ci_tmp.dropna(subset=["__alpha2"])
inst2alpha2 = dict(zip(ci_tmp["__inst_norm"], ci_tmp["__alpha2"]))

df["__uni_norm"] = df["University"].map(norm_df_uni)
df["__alpha2"]   = df["__uni_norm"].map(inst2alpha2)

if ALLOW_US_FALLBACK:
    mask_cont_unknown = df["Continent"].apply(is_unknown)
    need_us = mask_cont_unknown & df["__alpha2"].isna()
    df.loc[need_us, "__alpha2"] = US_ALPHA2

mask_cont   = df["Continent"].apply(is_unknown)
mask_country= df["Country"].apply(is_unknown)

df.loc[mask_cont,   "Continent"] = df.loc[mask_cont,   "__alpha2"].map(alpha2_to_cont).fillna(df.loc[mask_cont,   "Continent"])
df.loc[mask_country,"Country"]   = df.loc[mask_country,"__alpha2"].map(alpha2_to_name).fillna(df.loc[mask_country,"Country"])

unmatched = df[df["__alpha2"].isna()][["University"]].drop_duplicates().sort_values("University")
unmatched.to_csv(OUT_PATH.parent / "_unmatched_universities.csv", index=False)
print(f"[Info] unmatched universities (no alpha2): {len(unmatched)} -> _unmatched_universities.csv")

df.drop(columns=[c for c in ["__uni_norm","__alpha2"] if c in df.columns], inplace=True)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print(f"[Done] Continent filled using CSrankings polygons. Wrote to {OUT_PATH}")

