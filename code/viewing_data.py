import pandas as pd 
import numpy as np 
import seaborn as sns 

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
        cleaned_df = account_sub_merge.drop(columns=['shipping.zip.code', 'billing.zip.code', 'shipping.city', 'billing.city','relationship',])

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

        return cleaned_df

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
    data_class.drop_columns_figure_nas(account_sub_merge)



if __name__ == "__main__":
    main()