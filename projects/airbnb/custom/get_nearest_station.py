from geopy.distance import geodesic
import pandas as pd


def cross_join(df_a, df_b):
    df_a = df_a.copy()
    df_b = df_b.copy()
    df_a['tmp'] = 1
    df_b['tmp'] = 1
    df = pd.merge(df_a, df_b, how='outer')
    return df.drop("tmp", axis=1)


def get_dist(row):
    hotel = (row["latitude"], row["longitude"])
    station = (row["sta_latitude"], row["sta_longitude"])
    # .km -> .m とすることでメートルでの距離も取得可能
    return geodesic(hotel, station).km


train = pd.read_csv("data/train_data.csv")
test = pd.read_csv("data/test_data.csv")
station = pd.read_csv("data/station_list.csv")

# 位置が同じだったり名前が同じだったりするものを削除する.
station = station.drop_duplicates(
    subset=["longitude", "latitude"]).drop_duplicates(subset=["station_name"]).reset_index(drop=True)
station = station.rename(
    columns={"longitude": "sta_longitude", "latitude": "sta_latitude"})


joined = cross_join(test, station)


joined["dist_from_station"] = joined.apply(get_dist, axis=1)
joined.to_csv("data/joined_dist_test.csv", index=False)


ddf = joined.groupby("id")
hoge = joined.loc[ddf['dist_from_station'].idxmin(), :].reset_index(drop=True)
hoge[["id", "station_name", "sta_longitude", "sta_latitude",
      "dist_from_station"]].to_csv("data/nearest_station_test.csv", index=False)
