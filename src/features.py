import numpy as np
import pandas as pd

target = "PitNextLap"
id_col = "id"
group = ["Driver", "Race", "Year"]
cat_cols = ["Driver", "Compound", "Race"]
seq_cols = ["TyreLife", "LapTime (s)", "Position", "Stint", "PitStop", "Cumulative_Degradation"]


def build_features(train, test):
    #train and test laps come from the same races, so sequences are built on both (target is never used)
    df = pd.concat([train.assign(_is_train=1), test.assign(_is_train=0)], ignore_index=True)
    df = df.sort_values(group + ["LapNumber"]).reset_index(drop=True)
    g = df.groupby(group, sort=False)

    #lap sequence (current and earlier laps only)
    df["lap_index"] = g.cumcount()
    df["laps_since_prev"] = g["LapNumber"].diff()
    for col in seq_cols:
        df[f"{col}_prev1"] = g[col].shift(1)
        df[f"{col}_prev2"] = g[col].shift(2)
    lt = "LapTime (s)"
    df["laptime_change"] = df[lt] - df[f"{lt}_prev1"]
    df["laptime_slope3"] = (df[lt] - df[f"{lt}_prev2"]) / 2
    df["laptime_roll3"] = g[lt].rolling(3, min_periods=1).mean().reset_index(level=group, drop=True)
    df["laptime_roll5"] = g[lt].rolling(5, min_periods=1).mean().reset_index(level=group, drop=True)
    df["laptime_vs_roll3"] = df[lt] - df["laptime_roll3"]
    df["laptime_vs_best"] = df[lt] - g[lt].cummin()
    df["deg_change"] = df["Cumulative_Degradation"] - df["Cumulative_Degradation_prev1"]
    df["position_lag_change"] = df["Position"] - df["Position_prev2"]
    df["driver_pits_so_far"] = g["PitStop"].cumsum()

    #stint pace
    gs = df.groupby(group + ["Stint"], sort=False)
    df["stint_laptime_mean"] = gs[lt].cumsum() / (gs.cumcount() + 1)
    df["laptime_vs_stint"] = df[lt] - df["stint_laptime_mean"]
    df["stint_best_laptime"] = gs[lt].cummin()

    #tyre context: typical stint length for this race and compound
    stint_len = df.groupby(group + ["Stint", "Compound"])["TyreLife"].max().rename("stint_len").reset_index()
    typical = stint_len.groupby(["Race", "Year", "Compound"])["stint_len"].median().rename("typical_stint_len")
    df = df.merge(typical, on=["Race", "Year", "Compound"], how="left")
    df["tyrelife_vs_typical"] = df["TyreLife"] / (df["typical_stint_len"] + 1)
    df["laps_to_typical"] = df["typical_stint_len"] - df["TyreLife"]

    #race context
    df["race_laps"] = (df["LapNumber"] / df["RaceProgress"].replace(0, np.nan)).round()
    df["laps_remaining"] = df["race_laps"] - df["LapNumber"]

    #field context: how many cars around the field have been pitting
    lap_pits = df.groupby(["Race", "Year", "LapNumber"])["PitStop"].sum().rename("field_pits_lap").reset_index()
    lap_pits = lap_pits.sort_values(["Race", "Year", "LapNumber"])
    gl = lap_pits.groupby(["Race", "Year"])["field_pits_lap"]
    lap_pits["field_pits_last3"] = gl.rolling(3, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
    lap_pits["field_pits_so_far"] = gl.cumsum()
    n_drivers = df.groupby(["Race", "Year"])["Driver"].nunique().rename("field_size")
    df = df.merge(lap_pits, on=["Race", "Year", "LapNumber"], how="left").merge(n_drivers, on=["Race", "Year"], how="left")
    df["field_pit_share"] = df["field_pits_so_far"] / df["field_size"]

    #ratios
    df["tyrelife_per_stint"] = df["TyreLife"] / (df["Stint"] + 1)
    df["deg_per_lap"] = df["Cumulative_Degradation"] / (df["TyreLife"] + 1)
    df["progress_x_tyrelife"] = df["RaceProgress"] * df["TyreLife"]

    for col in cat_cols:
        df[col] = df[col].astype("category")

    tr = df[df["_is_train"] == 1].drop(columns="_is_train").sort_values(id_col).reset_index(drop=True)
    te = df[df["_is_train"] == 0].drop(columns=["_is_train", target]).sort_values(id_col).reset_index(drop=True)
    return tr, te
