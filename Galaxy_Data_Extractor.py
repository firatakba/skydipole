"""Build the +/-1 handedness label file from a VizieR export of Galaxy Zoo 1 (J/MNRAS/410/166).

Input : vizier_votable.tsv  (all columns, no row limit, 893,212 rows) in the current directory
Output: real_galaxy_labels.csv  with columns ra, dec, spin
        spin = +1 if pcS >= threshold (clockwise, Z-wise), -1 if paS >= threshold (anticlockwise, S-wise)
Note  : exact duplicate positions are NOT removed here; the analysis scripts remove them
        (positions rounded to 1e-4 deg), which turns the 36,268 rows written here into N = 36,267.
"""
import io
import pandas as pd

file_path = "vizier_votable.tsv"   # name of the file downloaded from VizieR
threshold = 0.80                   # clean-sample threshold of Land et al. (2008); 0.50 gives a larger sample

print("Reading VizieR VOTable/CSV data...")

with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

# locate the line where the CSV header starts
start_idx = None
for idx, line in enumerate(lines):
    if line.startswith("_RAJ2000;") or "_RAJ2000;" in line:
        start_idx = idx
        break

if start_idx is None:
    raise ValueError("Data header (_RAJ2000) not found in the input file!")

# drop the XML trailer (CDATA closing tags)
clean_lines = []
for line in lines[start_idx:]:
    if "]]>" in line or "</DATA>" in line or "</VOTABLE>" in line:
        break
    clean_lines.append(line)

# skiprows=[1, 2] skips the two lines under the header (units row 'deg;deg;...' and the dashes row)
df = pd.read_csv(
    io.StringIO("".join(clean_lines)), sep=";", skiprows=[1, 2], low_memory=False
)
df.columns = df.columns.str.strip()

# numeric conversions
df["_RAJ2000"] = pd.to_numeric(df["_RAJ2000"], errors="coerce")
df["_DEJ2000"] = pd.to_numeric(df["_DEJ2000"], errors="coerce")
df["pcS"] = pd.to_numeric(df["pcS"], errors="coerce")
df["paS"] = pd.to_numeric(df["paS"], errors="coerce")

cw_mask = df["pcS"] >= threshold
acw_mask = df["paS"] >= threshold

clean_df = df[cw_mask | acw_mask].copy()

# spin labels: clockwise (pcS) -> +1, anticlockwise (paS) -> -1
clean_df["spin"] = 0
clean_df.loc[cw_mask, "spin"] = 1
clean_df.loc[acw_mask, "spin"] = -1

output_df = clean_df.rename(
    columns={"_RAJ2000": "ra", "_DEJ2000": "dec"}
)[["ra", "dec", "spin"]].dropna()

output_df.to_csv("real_galaxy_labels.csv", index=False)
print(
    f"Done! {len(output_df)} galaxies with clean labels"
    " written to 'real_galaxy_labels.csv'."
)
