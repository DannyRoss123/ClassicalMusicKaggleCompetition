import pandas as pd 
import numpy as np 
import seaborn as sns 
from pathlib import Path


# #load data 
# concert_csv = pd.read_csv(
#     r"C:\GitRepos\ClassicalMusicKaggleCompetition\for_students\concerts.csv")

# print(concert_csv.head())
# print(concert_csv.columns)

# #columns are season, concert name, set, who played, what they played, location. Just have season data not actual date 
# subscription_csv = pd.read_csv(
#     r"C:\GitRepos\ClassicalMusicKaggleCompetition\for_students\subscriptions.csv"
# )

# print(subscription_csv.head())
# print(subscription_csv.columns)
# #columns are account id, season, package -> full, quartet etc, location -> can match with other csv, section, price, subscription tier, multiple subs
# #doesnt have set or who so hard to link with subscription csv 

# account_df = pd.read_csv(
#     r"C:\GitRepos\ClassicalMusicKaggleCompetition\for_students\account.csv",
#     encoding="latin1",           
#     dtype={
#         "account.id": "string",
#         "shipping.zip.code": "string",
#         "billing.zip.code": "string",
#     },
#     parse_dates=["first.donated"],  
# )

# print(account_df.columns)
# print(account_df.head())
# #columns are -> account id, shipping zip code, billing zip code, shipping city, billing city, relationship
# #amount donated 2013 amount donated lifetime, no donations lifetime, first donated 

# zipcodes_df = pd.read_csv(
#     r"C:\GitRepos\ClassicalMusicKaggleCompetition\for_students\zipcodes.csv"
# )

# print(zipcodes_df.head())
# print(zipcodes_df.columns)
# print(zipcodes_df.shape)
# print(zipcodes_df.isna().sum())
# #zipcode, zipcodetype, city, state, locationtype, lat, long, location, decommisioned, taxreturnfiled, estimatedpopulation, totalwages

# tickets_df = pd.read_csv(
#     r"C:\GitRepos\ClassicalMusicKaggleCompetition\for_students\tickets_all.csv"
# )

# print(tickets_df.head())
# print(tickets_df.columns)
# #columns are account id, price level, no seats, marketing source, season, location, set, multile tickets

# #TRAIN CSV FILE, WHAT WE ACTUALLY ARE TRAINING OUR MODEL ON 
# train_df = pd.read_csv(
#     r"C:\GitRepos\ClassicalMusicKaggleCompetition\for_students\train.csv"
# )

# print(train_df.head())
# print(train_df.columns)

# #Just account ID and label 
#the training set containing target labels indicating whether the patrons have purchased a 2014-15 subscription or not

#Ok so lets think about this: we have a training set with id and label for if they bought a ticket or not .. we need to add features to this. We can pretty 
#easily use the tickets_df just join based on the account id - this will give us price level, no seats, marketing source, season, location, set. 
#we can also use account df to get amount donated, first donated etc to show donation history might have an effect on if they get seats or not
#number of times they have attended in the past 
#subscriptions -> type of subscription, for how long etc. 
#build model from these, then if we want to add stuff we can figure out a way to add in the zip code information, lets   now build this using better function structure

class DataAnalysis: 
    def __init__(self, path1, path2, path3, path4, path5, path6):
        self.path1 = path1 
        self.path2 = path2 
        self.path3 = path3
        self.path4 = path4
        self.path5 = path5
        self.path6 = path6



    def load_data(self):
        self.concert_df = pd.read_csv(self.path1)
        self.subscription_df = pd.read_csv(self.path2)
        self.account_df = pd.read_csv(
            self.path3,
            encoding="latin1",           
            dtype={
                "account.id": "string",
                "shipping.zip.code": "string",
                "billing.zip.code": "string",
            },
            parse_dates=["first.donated"],  
        )
        self.zipcodes_df = pd.read_csv(self.path4)
        self.tickets_df = pd.read_csv(self.path5)
        self.train_df = pd.read_csv(self.path6)

    def view_nas(self):
        print("Concert DF NAs:\n", self.concert_df.isna().sum())
        print("Subscription DF NAs:\n", self.subscription_df.isna().sum())
        print("Account DF NAs:\n", self.account_df.isna().sum())
        print("Zipcodes DF NAs:\n", self.zipcodes_df.isna().sum())
        print("Tickets DF NAs:\n", self.tickets_df.isna().sum())
        print("Train DF NAs:\n", self.train_df.isna().sum())

    def merge_data(self):
        #merge data 
        account_train_merge = pd.merge(
            self.train_df,
            self.account_df,
            how = 'left', # want to only use keys from train
            on = 'account.id'
        )

        #to merge subscriptions in we need to aggregate columns properly 
        subscription_summary = (self.subscription_df.groupby('account.id').agg(total_seats=('no.seats', 'sum'), num_seasons=('season', 'nunique'), last_season=('season', 'max'))
        .reset_index())

        print(subscription_summary.head())
        print(f"subscription summary shape: {subscription_summary.shape}")
        print(subscription_summary.isna().sum())

        account_sub_merge = pd.merge( 
            account_train_merge, 
            subscription_summary,
            how = 'left',
            on = 'account.id'
        )
        #account id, label (TARGET), shipping zip code, billing zip code, shipping city, billing city, relationship, amount donated 2013, 
        #amount donated lifetime, no donations, first donated, season, package, no seats, location, section, price level, 
        #subscription tier, multiple subs


        print(account_sub_merge.head())
        print(account_sub_merge.columns)
        print(account_sub_merge.shape)
        print(account_sub_merge.isna().sum())

        return account_sub_merge

    def drop_columns_figure_nas(self, account_sub_merge):
        #drop columns with a ton of nas 
        cleaned_df = account_sub_merge.copy()

        def _clean_zip(series):
            series = (
                series.astype("string")
                .str.strip()
                .str.extract(r"(\d{5})")[0]
            )
            return series

        cleaned_df["shipping_zip_clean"] = _clean_zip(cleaned_df["shipping.zip.code"])
        cleaned_df["billing_zip_clean"] = _clean_zip(cleaned_df["billing.zip.code"])

        cleaned_df["primary_zip"] = cleaned_df["shipping_zip_clean"].combine_first(
            cleaned_df["billing_zip_clean"]
        )
        cleaned_df["has_shipping_zip"] = cleaned_df["shipping_zip_clean"].notna().astype(int)
        cleaned_df["has_billing_zip"] = cleaned_df["billing_zip_clean"].notna().astype(int)
        cleaned_df["shipping_matches_billing"] = (
            cleaned_df["shipping_zip_clean"] == cleaned_df["billing_zip_clean"]
        ).fillna(False).astype(int)

        cleaned_df = cleaned_df.drop(columns=['shipping.zip.code', 'billing.zip.code', 'shipping.city', 'billing.city','relationship',])

        #for first donated, total,seats,num_seasons,last_season we can't just get rid of the nas 
        # flag the accounts that actually appeared in the subscription summary
        cleaned_df['has_subscription_history'] = (cleaned_df['total_seats'].notna().astype(int))

        # fill the missing totals with zero seats/seasons
        cleaned_df['total_seats'] = cleaned_df['total_seats'].fillna(0)
        cleaned_df['num_seasons'] = cleaned_df['num_seasons'].fillna(0)
        cleaned_df['last_season'] = cleaned_df['last_season'].fillna(0)

        cleaned_df['has_donated'] = (cleaned_df['first.donated'].notna().astype(int))
        cleaned_df['first.donated'] = cleaned_df['first.donated'].fillna(0)

        print(cleaned_df.isna().sum())
        print(cleaned_df.head())
        print(cleaned_df.columns)

        return cleaned_df

    def merge_zipcodes(self, cleaned_df):
        
        zipcode_df = self.zipcodes_df.rename(columns=str.lower).copy()
        zipcode_df["zipcode"] = (
            zipcode_df["zipcode"]
            .astype("string")
            .str.strip()
            .str.extract(r"(\d{5})")[0]
        )
        zipcode_df = zipcode_df.dropna(subset=["zipcode"]).drop_duplicates("zipcode")
        zipcode_indexed = zipcode_df.set_index("zipcode")

        shipping_features = zipcode_indexed.add_prefix("shipping_")
        shipping_features.index.name = "shipping_zip_clean"
        billing_features = zipcode_indexed.add_prefix("billing_")
        billing_features.index.name = "billing_zip_clean"

        merged = cleaned_df.join(shipping_features, on="shipping_zip_clean")
        merged = merged.join(billing_features, on="billing_zip_clean")

        merged["has_shipping_zip_info"] = merged["shipping_zipcodetype"].notna().astype(int)
        merged["has_billing_zip_info"] = merged["billing_zipcodetype"].notna().astype(int)

        shipping_cols = [col for col in merged.columns if col.startswith("shipping_")]
        billing_cols = [col for col in merged.columns if col.startswith("billing_")]

        for col in shipping_cols + billing_cols:
            if merged[col].dtype.kind in {"b", "i", "u", "f", "c"}:
                merged[col] = merged[col].fillna(0)
            else:
                merged[col] = merged[col].fillna("unknown")

        print(merged.shape)
        print(merged.head())
        print(merged.columns)
        print(merged.isna().sum())

        return merged

    def merge_tickets(self, cleaned_df):

        tickets_filtered = self.tickets_df.copy()
        #make the season numeric 
        tickets_filtered["season_start_year"] = pd.to_numeric(
            tickets_filtered["season"].str[:4], errors="coerce"
        )
        tickets_filtered = tickets_filtered[
            tickets_filtered["season_start_year"].isna()
            | (tickets_filtered["season_start_year"] < 2014)
        ]

        tickets_filtered["price_level_numeric"] = pd.to_numeric(
            tickets_filtered["price.level"], errors="coerce"
        )
        tickets_filtered["is_multiple_ticket"] = (
            tickets_filtered["multiple.tickets"].fillna("").str.lower().eq("yes").astype(int)
        )

        #create columns that oculd be useful 
        ticket_summary = (
            tickets_filtered.groupby("account.id")
            .agg(
                total_ticket_orders=("account.id", "size"),
                total_seats_ticket=("no.seats", "sum"),
                avg_seats_per_order=("no.seats", "mean"),
                ticket_season_count=("season", "nunique"),
                locations_visited=("location", "nunique"),
                sets_attended=("set", "nunique"),
                last_ticket_season=("season_start_year", "max"),
                max_price_level=("price_level_numeric", "max"),
                any_multiple_tickets=("is_multiple_ticket", "max"),
                share_multiple_tickets=("is_multiple_ticket", "mean"),
            )
            .reset_index()
        )

        merge_acc_tic_sub = pd.merge(cleaned_df, ticket_summary, on='account.id', how='left')

        merge_acc_tic_sub["has_ticket_history"] = (
            merge_acc_tic_sub["total_ticket_orders"].notna().astype(int)
        )

        ticket_fill_cols = [
            "total_ticket_orders",
            "total_seats_ticket",
            "avg_seats_per_order",
            "ticket_season_count",
            "locations_visited",
            "sets_attended",
            "last_ticket_season",
            "max_price_level",
            "any_multiple_tickets",
            "share_multiple_tickets",
        ]
        for col in ticket_fill_cols:
            if col in merge_acc_tic_sub.columns:
                merge_acc_tic_sub[col] = merge_acc_tic_sub[col].fillna(0)

        print(merge_acc_tic_sub.shape)
        print(merge_acc_tic_sub.head())
        print(merge_acc_tic_sub.columns)
        print(merge_acc_tic_sub.isna().sum())

        return merge_acc_tic_sub

    def merge_concerts(self, tickets_merged_df):
        
        tickets_with_metadata = self.tickets_df.copy()
        tickets_with_metadata["season_start_year"] = pd.to_numeric(
            tickets_with_metadata["season"].str[:4], errors="coerce"
        )
        tickets_with_metadata = tickets_with_metadata[
            tickets_with_metadata["season_start_year"].isna()
            | (tickets_with_metadata["season_start_year"] < 2014)
        ]

        concerts_columns = ["season", "set", "location", "concert.name", "who", "what"]
        concerts_trimmed = self.concert_df[concerts_columns].copy()

        tickets_with_metadata = pd.merge(
            tickets_with_metadata,
            concerts_trimmed,
            on=["season", "set", "location"],
            how="left",
        )

        tickets_with_metadata["has_concert_metadata"] = (
            tickets_with_metadata["concert.name"].notna().astype(int)
        )

        concert_summary = (
            tickets_with_metadata.groupby("account.id")
            .agg(
                concert_ticket_orders=("account.id", "size"),
                known_concert_orders=("has_concert_metadata", "sum"),
                unique_concert_names=("concert.name", lambda s: s.dropna().nunique()),
                unique_performers=("who", lambda s: s.dropna().nunique()),
                unique_programs=("what", lambda s: s.dropna().nunique()),
            )
            .reset_index()
        )

        concert_summary["share_known_concerts"] = (
            concert_summary["known_concert_orders"].div(
                concert_summary["concert_ticket_orders"].replace(0, np.nan)
            )
        ).fillna(0)

        concert_summary["has_concert_metadata"] = (
            concert_summary["known_concert_orders"] > 0
        ).astype(int)

        concert_summary = concert_summary.drop(columns=["concert_ticket_orders"])

        merged_with_concerts = pd.merge(
            tickets_merged_df,
            concert_summary,
            on="account.id",
            how="left",
        )

        concert_fill_cols = [
            "known_concert_orders",
            "unique_concert_names",
            "unique_performers",
            "unique_programs",
            "share_known_concerts",
            "has_concert_metadata",
        ]
        for col in concert_fill_cols:
            if col in merged_with_concerts.columns:
                merged_with_concerts[col] = merged_with_concerts[col].fillna(0)

        print(merged_with_concerts.shape)
        print(merged_with_concerts.head())
        print(merged_with_concerts.columns)
        print(merged_with_concerts.isna().sum())

        return merged_with_concerts

    def encode_features(self, dataset, training=True):
        """Encode categorical features so the dataset is modeling ready."""
        model_df = dataset.copy()

        categorical_cols = [
            col
            for col in model_df.select_dtypes(include=["object", "string"]).columns
            if col != "account.id"
        ]

        self.encoding_maps = {}
        for col in categorical_cols:
            freq_map = (
                model_df[col]
                .fillna("unknown")
                .value_counts(dropna=False, normalize=True)
                .to_dict()
            )
            model_df[f"{col}_freq"] = model_df[col].map(freq_map).fillna(0.0)
            self.encoding_maps[col] = freq_map

        model_df = model_df.drop(columns=categorical_cols)

        bool_cols = model_df.select_dtypes(include=["bool"]).columns
        for col in bool_cols:
            model_df[col] = model_df[col].astype(int)

        numeric_cols = model_df.select_dtypes(include=["float", "int"]).columns
        model_df[numeric_cols] = model_df[numeric_cols].fillna(0)

        if "label" in model_df.columns and training:
            label_series = model_df.pop("label")
        else:
            label_series = None

        account_series = model_df.pop("account.id")

        ordered_frames = [account_series]
        ordered_columns = ["account.id"]

        if label_series is not None:
            ordered_frames.append(label_series)
            ordered_columns.append("label")

        feature_columns = [col for col in model_df.columns if col not in ["account.id", "label"]]
        ordered_frames.append(model_df[feature_columns])
        ordered_columns.extend(feature_columns)

        model_df = pd.concat(ordered_frames, axis=1)
        model_df.columns = ordered_columns

        print(model_df.shape)
        print(model_df.head())
        print(model_df.columns)

        return model_df

def main(): 
    #paths to data files 
    concert_path = r"C:\GitRepos\ClassicalMusicKaggleCompetition\for_students\concerts.csv"
    subscription_path = r"C:\GitRepos\ClassicalMusicKaggleCompetition\for_students\subscriptions.csv"
    account_path = r"C:\GitRepos\ClassicalMusicKaggleCompetition\for_students\account.csv"
    zipcodes_path = r"C:\GitRepos\ClassicalMusicKaggleCompetition\for_students\zipcodes.csv"
    tickets_path = r"C:\GitRepos\ClassicalMusicKaggleCompetition\for_students\tickets_all.csv"
    train_path = r"C:\GitRepos\ClassicalMusicKaggleCompetition\for_students\train.csv"

    data_class = DataAnalysis(
        concert_path,
        subscription_path,
        account_path,
        zipcodes_path,
        tickets_path,
        train_path
    )

    data_class.load_data()
    data_class.view_nas()
    account_sub_merge = data_class.merge_data()
    cleaned_data = data_class.drop_columns_figure_nas(account_sub_merge)
    zipcode_enriched = data_class.merge_zipcodes(cleaned_data)
    tickets_enriched = data_class.merge_tickets(zipcode_enriched)
    final_dataset = data_class.merge_concerts(tickets_enriched)
    model_ready = data_class.encode_features(final_dataset)
    print(model_ready.head())
    output_path = Path("model_ready_features.csv")
    model_ready.to_csv(output_path, index=False)
    print(f"Saved model-ready features to {output_path.resolve()}")




if __name__ == "__main__":
    main()
