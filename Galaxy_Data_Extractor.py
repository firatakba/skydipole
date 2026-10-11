import io
import pandas as pd

file_path = "vizier_votable.tsv"  # Dosya adınızı buraya yazın

print("VizieR VOTable/CSV verisi okunuyor...")

with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
  lines = f.readlines()

start_idx = None
for idx, line in enumerate(lines):
  if line.startswith("_RAJ2000;") or "_RAJ2000;" in line:
    start_idx = idx
    break

if start_idx is None:
  raise ValueError("Veri başlığı (_RAJ2000) dosya içinde bulunamadı!")

# XML altındaki verileri ve CDATA kapanışını temizleyelim
clean_lines = []
for line in lines[start_idx:]:
  if "]]>" in line or "</DATA>" in line or "</VOTABLE>" in line:
    break
  clean_lines.append(line)

df = pd.read_csv(
    io.StringIO("".join(clean_lines)), sep=";", skiprows=[1, 2], low_memory=False
)
df.columns = df.columns.str.strip()

# Sayısal dönüşümler
df["_RAJ2000"] = pd.to_numeric(df["_RAJ2000"], errors="coerce")
df["_DEJ2000"] = pd.to_numeric(df["_DEJ2000"], errors="coerce")
df["pcS"] = pd.to_numeric(df["pcS"], errors="coerce")
df["paS"] = pd.to_numeric(df["paS"], errors="coerce")

threshold = 0.80

cw_mask = df["pcS"] >= threshold
acw_mask = df["paS"] >= threshold

clean_df = df[cw_mask | acw_mask].copy()

# Spin etiketleri: Clockwise (pcS) -> +1, Anti-Clockwise (paS) -> -1
clean_df["spin"] = 0
clean_df.loc[cw_mask, "spin"] = 1
clean_df.loc[acw_mask, "spin"] = -1

output_df = clean_df.rename(
    columns={"_RAJ2000": "ra", "_DEJ2000": "dec"}
)[["ra", "dec", "spin"]].dropna()

output_df.to_csv("real_galaxy_labels.csv", index=False)
print(
    f"Başarılı! Toplam {len(output_df)} adet net etiketli galaksi"
    " 'real_galaxy_labels.csv' dosyasına yazıldı."
)
