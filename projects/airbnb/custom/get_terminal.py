def get_terminal(joined):
    df = joined[joined["station_name"].isin(["東京", "新宿", "池袋", "渋谷", "上野"])]
    ddf = df.groupby("id")
    df = df.loc[ddf["dist_from_station"].idxmin(
    ), ["id", "station_name", "dist_from_station"]]
    return df.rename(columns={"station_name": "nearest_terminal", "dist_from_station": "dist_from_terminal"})


terminal_train = get_terminal(joined_train)
terminal_test = get_terminal(joined_test)

pd.concat([terminal_test, terminal_train]).reset_index(
    drop=True).to_csv("data/nearest_terminal.csv", index=False)
