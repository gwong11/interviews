#!/usr/bin/python3

import pandas as pd
import re
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
pd.set_option("max_rows", None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

def convert_to_seconds(elapsed_time):
    elapsed_time_split = elapsed_time.split(':')
    if (len(elapsed_time_split) == 3):
        return (int(elapsed_time_split[0])*3600) + (int(elapsed_time_split[1])*60) + (int(elapsed_time_split[2]))
    elif (len(elapsed_time_split) == 2):
        return (int(elapsed_time_split[0])*60) + (int(elapsed_time_split[1]))
    else:
        return "N/A"

def get_division(age):
    if age >= 0 and age <= 14:
        return "A"
    elif age >=15 and age <= 19:
        return "B"
    elif age >=20 and age <= 29:
        return "C"
    elif age >=30 and age <= 39:
        return "D"
    elif age >=40 and age <= 49:
        return "E"
    elif age >=50 and age <= 59:
        return "F"
    elif age >=60 and age <= 69:
        return "G"
    elif age >=70 and age <= 79:
        return "H"
    elif age >=80 and age <= 89:
        return "I"
    elif age >=90 and age <= 99:
        return "J"
    elif age >=100 and age <= 110:
        return "K"
    
def new_describe(df):
    df1 = df.describe()
    df1.loc["range"] = df1.loc['max'] - df1.loc['min']
    df1.loc["10%"] = df.quantile(.1)

    return df1

def main(file_male, file_female):
    male_df = pd.read_csv(file_male, sep='\t', lineterminator='\r')
    female_df = pd.read_csv(file_female, sep='\t', lineterminator='\r')

    # Remove "newline in Place"
    male_df['Place'] = male_df['Place'].str.strip()
    female_df['Place'] = female_df['Place'].str.strip()

    # Count the number of empty values in each column
    print("---------Count of empty values in each column--------")
    print(male_df.isnull().sum())
    print()
    print(female_df.isnull().sum())
    print()

    # Check for empty values in the Num column
    print("---------Empty rows for the Num column--------")
    print(male_df[male_df['Num'].isnull()])
    print()
    print(female_df[female_df['Num'].isnull()])
    print()

    # Drop rows with at least 5 NAs
    male_df = male_df.dropna(thresh=5)
    female_df = female_df.dropna(thresh=5)

    # Convert Num to int
    male_df['Num'] = male_df['Num'].astype(int)
    female_df['Num'] = female_df['Num'].astype(int)

    # Count the number of empty values in each column after dropping rows with more than 5 NAs
    print("---------Count of empty values in each column after dropping row with more than 5 NAs--------")
    print(male_df.isnull().sum())
    print()
    print(female_df.isnull().sum())
    print()

    # Check for empty values in the Ag column
    print("---------Empty rows for the Ag column--------")
    print(male_df[male_df['Ag'].isnull()])
    print()
    print(female_df[female_df['Ag'].isnull()])
    print()

    # Drop rows with NAs in Ag
    male_df = male_df[pd.notnull(male_df['Ag'])]
    female_df = female_df[pd.notnull(female_df['Ag'])]

    # Convert Ag to int
    male_df['Ag'] = male_df['Ag'].astype(int)
    female_df['Ag'] = female_df['Ag'].astype(int)

    # Remove letter from Gun Time
    male_df['Gun Tim'] = male_df['Gun Tim'].str.replace('([a-zA-Z\s]*)', '', regex=True)
    female_df['Gun Tim'] = female_df['Gun Tim'].str.replace('([a-zA-Z\s]*)', '', regex=True)

    # Remove special characters from Net Tim
    male_df['Net Tim'] = male_df['Net Tim'].str.replace('([#*]*)', '', regex=True)
    female_df['Net Tim'] = female_df['Net Tim'].str.replace('([#*]*)', '', regex=True)

    # Convert time to seconds 
    male_df['Gun Tim(s)'] = male_df.apply(lambda x: convert_to_seconds(x['Gun Tim']),axis=1)
    female_df['Gun Tim(s)'] = female_df.apply(lambda x: convert_to_seconds(x['Gun Tim']),axis=1)
    male_df['Net Tim(s)'] = male_df.apply(lambda x: convert_to_seconds(x['Net Tim']),axis=1)
    female_df['Net Tim(s)'] = female_df.apply(lambda x: convert_to_seconds(x['Net Tim']),axis=1)
    male_df['Pace(s)'] = male_df.apply(lambda x: convert_to_seconds(x['Pace']),axis=1)
    female_df['Pace(s)'] = female_df.apply(lambda x: convert_to_seconds(x['Pace']),axis=1)
     
    # Difference between Gun Time and Net Time
    male_df['Diff Tim(s)'] = male_df.apply(lambda x: x['Gun Tim(s)'] - x['Net Tim(s)'],axis=1)
    female_df['Diff Tim(s)'] = female_df.apply(lambda x: x['Gun Tim(s)'] - x['Net Tim(s)'],axis=1)

    # Get Division
    male_df['Div'] = male_df.apply(lambda x: get_division(x['Ag']),axis=1)
    female_df['Div'] = female_df.apply(lambda x: get_division(x['Ag']),axis=1)

    #print(male_df[male_df['Net Tim'].str.contains('#')].head(20))
    #print(female_df[female_df['Net Tim'].str.contains('#')].head(20))

    print("---------First 10 rows for males and females--------")
    print(male_df.head(10))
    print()
    print(female_df.head(10))
    print()

    print("---------Descriptive Statistics--------")
    print(new_describe(male_df))
    print()
    print(new_describe(female_df))
    print()


    print("---------Performance of Chris Doe--------")
    print(male_df[male_df['Name'] == 'Chris Doe'])
    print()

    print("---------Mode Statistics--------")
    print()
    print("---------Ag--------")
    print(male_df['Ag'].mode().tolist())
    print(female_df['Ag'].mode().tolist())
    print()
    print("---------Gun Tim(s)--------")
    print(male_df['Gun Tim(s)'].mode().tolist())
    print(female_df['Gun Tim(s)'].mode().tolist())
    print()
    print("---------Net Tim(s)--------")
    print(male_df['Net Tim(s)'].mode().tolist())
    print(female_df['Net Tim(s)'].mode().tolist())
    print()
    print("---------Diff Tim(s)--------")
    print(male_df['Diff Tim(s)'].mode().tolist())
    print(female_df['Diff Tim(s)'].mode().tolist())
    print()
    print("---------Pace Tim(s)--------")
    print(male_df['Pace(s)'].mode().tolist())
    print(female_df['Pace(s)'].mode().tolist())
    print()

    print("---------Groupby--------")
    print(male_df.groupby("Ag")[["Net Tim(s)"]].count())
    print(female_df.groupby("Ag")[["Net Tim(s)"]].count())
    print()
    print(male_df.groupby("Div")[["Net Tim(s)"]].count())
    print(female_df.groupby("Div")[["Net Tim(s)"]].count())
    print()

    print("---------Plots--------")
    print()
    print("---------Plot Gun Tim(s), Net Tim(s), and Diff Tim(s)--------")
    male_df['Gun Tim(s)'].plot()
    male_df['Net Tim(s)'].plot(color='red')
    male_df['Diff Tim(s)'].plot(color='green')
    plt.show()

if __name__ == "__main__":
    filename_male = "MA_Exer_PikesPeak_Males.txt"
    filename_female = "MA_Exer_PikesPeak_Females.txt"
    main(filename_male, filename_female)