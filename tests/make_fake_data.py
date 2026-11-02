#generates fake data with the same schema as the kaggle csvs, only for testing the pipeline
import sys
from pathlib import Path

import numpy as np
import pandas as pd

out = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/fake_data")
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
races = ["Bahrain Grand Prix", "Monaco Grand Prix", "British Grand Prix", "Italian Grand Prix", "Japanese Grand Prix",
         "Brazilian Grand Prix", "Dutch Grand Prix", "Canadian Grand Prix", "Pre-Season Testing"]
life = {"SOFT": 15, "MEDIUM": 25, "HARD": 35, "INTERMEDIATE": 20}
rows = []
for year in [2022, 2023, 2024]:
    for race in races:
        n_laps = int(rng.integers(50, 72))
        for d in range(20):
            drv = f"D{d:03d}"
            stint, tl, deg, pos = 1, 1, 0.0, d + 1
            cmp = rng.choice(["SOFT", "MEDIUM", "HARD"])
            target_len = max(5, int(rng.normal(life[cmp], 5)))
            pits = []
            for lap in range(1, n_laps + 1):
                pit_now = 1 if (pits and pits[-1] == lap) else 0
                if pit_now:
                    stint, tl, deg = stint + 1, 1, 0.0
                    cmp = rng.choice(["MEDIUM", "HARD"])
                    target_len = max(5, int(rng.normal(life[cmp], 5)))
                lt = 90 + 0.05 * tl * (2 if cmp == "SOFT" else 1) + rng.normal(0, 0.6)
                deg += 0.05 * tl
                pit_next = int(tl >= target_len and lap < n_laps - 3)
                if pit_next:
                    pits.append(lap + 1)
                newpos = int(np.clip(pos + rng.integers(-1, 2), 1, 20))
                rows.append([drv, cmp, race, year, pit_now, lap, stint, float(tl), pos, lt, 0.0, deg,
                             lap / n_laps, float(newpos - pos), float(pit_next)])
                pos = newpos
                tl += 1
cols = ["Driver", "Compound", "Race", "Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position", "LapTime (s)",
        "LapTime_Delta", "Cumulative_Degradation", "RaceProgress", "Position_Change", "PitNextLap"]
df = pd.DataFrame(rows, columns=cols)
df["LapTime_Delta"] = df.groupby(["Driver", "Race", "Year"])["LapTime (s)"].diff().fillna(0)
df = df.sample(frac=1, random_state=0).reset_index(drop=True)
df.insert(0, "id", np.arange(len(df)))
cut = int(len(df) * 0.7)
df.iloc[:cut].to_csv(out / "train.csv", index=False)
df.iloc[cut:].drop(columns="PitNextLap").to_csv(out / "test.csv", index=False)
print(len(df), "rows written to", out)
