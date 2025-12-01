import pandas as pd
import numpy as np
import re
import dictionaries

#-------- Data Loading --------#
restaurants_existed_path = 'Data/ca_restaurants_clean.csv'
restaurants_for_lease_path = 'Data/CASitesForRent.xlsx'
ipums_usa_path = 'Data/usa_00026.csv'
california_geo_path = 'Data/California_Overlapping_Cities_and_Counties_and_Identifiers_with_Coastal_Buffers_-3513278014611499511.xlsx'
restaurants_for_lease_2_path = 'Data/crexi-full-cleaned.csv'
restaurants_for_lease_3_path = 'Data/CASitesForRent_with_averages.xlsx'

# Load the CSV file into a DataFrame
restaurants_existed_df = pd.read_csv(restaurants_existed_path)
restaurants_for_lease_df = pd.read_excel(restaurants_for_lease_path)
restaurants_for_lease_2_df = pd.read_csv(restaurants_for_lease_2_path)
restaurants_for_lease_3_df = pd.read_excel(restaurants_for_lease_3_path)
ipums_usa_df = pd.read_csv(ipums_usa_path)
california_geo_df = pd.read_excel(california_geo_path)

#-------- Data Cleaning --------#
def data_cleaning():
    def cleaning_site_level(restaurants_for_lease_df, restaurants_for_lease_3_df, california_geo_df):
        # Drop rows with 'Undisclosed Rate' in the last column (column 6, 0-indexed)
        restaurants_for_lease_df = restaurants_for_lease_df[restaurants_for_lease_df.iloc[:, 6] != 'Undisclosed Rate'].reset_index(drop=True)

        # Drop rows with no squarefeet data
        restaurants_for_lease_3_df = restaurants_for_lease_3_df.dropna(subset=['Min_SqFt']).reset_index(drop=True)

        # Extract city, state, zip from address1
        city_state_zip_pattern = r'^\s*([A-Za-z .\'-]+?),\s*([A-Z]{2})\s*(\d{5}(?:-\d{4})?)\s*$'
        restaurants_for_lease_df[['city', 'state', 'zip']] = (
            restaurants_for_lease_df['address1']
            .str.extract(city_state_zip_pattern, expand=True)
        )

        # From address_detail, first remove ", STATE ZIP" tail (vectorized)
        left = restaurants_for_lease_df['address_detail'].str.replace(
            r',\s*[A-Z]{2}\s*\d{5}(?:-\d{4})?\s*$',
            '',
            regex=True
        )

        # Then remove the trailing city (row-wise, because city is per-row)
        def drop_trailing_city(text, city):
            if pd.isna(text) or pd.isna(city):
                return text
            # remove a final " ... <city>" at the end
            pat = r'\s+' + re.escape(city) + r'\s*$'
            return re.sub(pat, '', text).strip()

        restaurants_for_lease_df['address line'] = [
            drop_trailing_city(l, c) for l, c in zip(left, restaurants_for_lease_df['city'])
        ]

        # Merge counties using city
        # Remove ' County' from the 'COUNTY' column
        california_geo_df['COUNTY'] = california_geo_df['COUNTY'].str.replace(' County', '', regex=False)
        california_geo_cities = california_geo_df[['COUNTY', 'PLACE_NAME']].rename(columns={'PLACE_NAME': 'city'})
        restaurants_for_lease_df = pd.merge(restaurants_for_lease_df, california_geo_cities, on='city', how='inner')

        restaurants_for_lease_df.drop_duplicates(inplace=True)

        restaurants_for_lease_3_df.rename(columns={'Address': 'full_address'}, inplace=True)
        restaurants_for_lease_df.rename(columns={'address_detail': 'full_address'}, inplace=True)

        final_restaurants_for_lease = pd.merge(restaurants_for_lease_df, restaurants_for_lease_3_df, on='full_address', how='inner')

        final_restaurants_for_lease.drop_duplicates(inplace=True)

        columns_to_keep_and_rename = {
            'address2': 'Name',
            'Name': 'details',
            'image_link	': 'image_link',
            'full_address': 'full_address',
            'Price': 'Price',
            'address line': 'address_line',
            'city': 'city',
            'state': 'state',
            'zip': 'zip',
            'COUNTY': 'County',
            'Min_SqFt': 'Min_SqFt',
            'Max_SqFt': 'Max_SqFt',
            'Avg_SqFt': 'Avg_SqFt',
            'monthly_Rate': 'monthly_Rate',
            'Total_Rent':  'Total_Rent'
        }

        # Filter columns that actually exist in the DataFrame
        existing_columns = {old_name: new_name for old_name, new_name in columns_to_keep_and_rename.items() if old_name in final_restaurants_for_lease.columns}

        # Select and rename columns
        final_restaurants_for_lease = final_restaurants_for_lease[list(existing_columns.keys())].rename(columns=existing_columns)

        return final_restaurants_for_lease
    
    def cleaning_county_level(ipums_usa_df):
        county_fips_to_name = dictionaries.county_fips_to_name
        ipums_usa_df["County_Name"] = ipums_usa_df["COUNTYFIP"].map(county_fips_to_name)

        # Drop abnomolies inctot
        ipums_usa_df = ipums_usa_df[ipums_usa_df['INCTOT'] > 20000]

        # Group by County_Name and calculate requested statistics
        county_stats_df = ipums_usa_df.groupby('County_Name').agg(
            Median_Age=('AGE', 'median'),
            Percentage_Hispanic=('HISPAN', lambda x: (x != 0).mean() * 100),
            Percentage_Asian=('RACASIAN', lambda x: (x == 2).mean() * 100),
            Percentage_White=('RACE', lambda x: (x == 1).mean() * 100),
            Percentage_Black=('RACE', lambda x: (x == 2).mean() * 100),
            Median_INCTOT=('INCTOT', 'median')
        ).reset_index()

        # Calculate yearly population per county and then the median yearly population
        yearly_population = ipums_usa_df.groupby(['County_Name', 'YEAR']).size().reset_index(name='Yearly_Population')
        median_yearly_population = yearly_population.groupby('County_Name')['Yearly_Population'].median().reset_index(name='Median_Yearly_Population')

        # Merge the median yearly population back to the county_stats_df
        county_stats_df = pd.merge(county_stats_df, median_yearly_population, on='County_Name')

        return county_stats_df
    
    def cleaning_zip_level(restaurants_existed_df):
        # Categorize restaurant types using ChatGPT
        asian_food_keywords = dictionaries.asian_food_keywords
        def assign_asian_country(restaurant_type):
            if pd.isna(restaurant_type):
                return "other"

            text = restaurant_type.lower()

            for country, keywords in asian_food_keywords.items():
                for kw in keywords:
                    if kw.lower() in text:
                        return country

            return "other"

        restaurants_existed_df['asian_country'] = restaurants_existed_df['restaurant_type'].apply(assign_asian_country)

        restaurants_existed_df = restaurants_existed_df[
            restaurants_existed_df['price_mid'].notna() &
            restaurants_existed_df['review_score'].notna() &
            restaurants_existed_df['num_ratings'].notna() &
            restaurants_existed_df['has_free_parking'].notna()
        ]

        # 1. Pivot table for restaurant type counts
        type_counts_df = (
            restaurants_existed_df
            .groupby(['zip', 'asian_country'])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        # 2. Your existing aggregated metrics
        zip_stats_df = restaurants_existed_df.groupby('zip').agg(
            Median_review_score=('review_score', 'median'),
            Median_rating_count=('num_ratings', 'median'),
            Median_price_mid=('price_mid', 'median'),
            Count_has_free_parking=('has_free_parking', lambda x: (x == True).sum()),
            Count_total_restaurant=('restaurant_name', 'count')
        ).reset_index()

        # 3. Merge them together
        zip_stats_df = zip_stats_df.merge(type_counts_df, on='zip', how='left')

        return zip_stats_df
    
    return cleaning_site_level(restaurants_for_lease_df, restaurants_for_lease_3_df, california_geo_df), cleaning_county_level(ipums_usa_df), cleaning_zip_level(restaurants_existed_df)

#-------- Data Scaling --------#
def scaling(county_stats_df, zip_stats_df, site_level_df, inputs):
    # -------- County Level Dummification -------- #
    def dummify_list(df, name, column, input):
        Q1, Q2 = df[column].quantile([0.25, 0.75])
        if input == 'high':
            result = df[column].apply(lambda x: 100 if x > Q2 else 0)
        elif input == 'medium':
            result = df[column].apply(lambda x: 100 if Q1 <= x <= Q2 else 0)
        else:
            result = df[column].apply(lambda x: 100 if x < Q1 else 0)
        return result

    county_stats_df['Dummy_Percentage_Asian'] = dummify_list(county_stats_df, 'County', 'Percentage_Asian', inputs['Percentage_Asian_input'])
    county_stats_df['Dummy_Median_INCTOT'] = dummify_list(county_stats_df, 'County', 'Median_INCTOT', inputs['Median_INCTOT_input'])
    county_stats_df['Dummy_Median_Yearly_Population'] = dummify_list(county_stats_df, 'County', 'Median_Yearly_Population', inputs['Median_Yearly_Population_input'])

    # -------- Zip Level Scaling -------- #
    base_cols = ['Count_total_restaurant', 'Median_price_mid', inputs['Restaurant_type_input']]
    scaled_cols = ["Scaled_" + col for col in base_cols]
    # Empty DataFrame to hold scaled columns
    scaled_data = pd.DataFrame(index=zip_stats_df.index)
    # Robust MinMax using quantiles (e.g., 10th and 90th percentiles)
    low_q = 0.10
    high_q = 0.90
    for col, scaled_col in zip(base_cols, scaled_cols):
        s = zip_stats_df[col]
        q_low = s.quantile(low_q)
        q_high = s.quantile(high_q)
        # Avoid division by zero if q_high == q_low
        if q_high == q_low:
            scaled = pd.Series(50, index=s.index)  # flat mid-score if no variation
        else:
            scaled = (s - q_low) / (q_high - q_low)
        # Clip to [0, 1] then scale to [0, 100]
        scaled = scaled.clip(lower=0, upper=1) * 100
        scaled_data[scaled_col] = scaled
    # Assign back to zip_stats_df
    zip_stats_df[scaled_cols] = scaled_data

    # -------- Site Level Scaling -------- #
    # Standard deviation for scaling
    sigma = site_level_df['Avg_SqFt'].std()

    site_level_df['Scaled_Avg_SqFt'] = 100*np.exp(
        -((site_level_df['Avg_SqFt'] - inputs['Avg_SqFt_input']) ** 2) 
        / (2 * sigma ** 2)
    )
    base_cols = ['Total_Rent']
    scaled_cols = ["Scaled_" + col for col in base_cols]

    # Empty DataFrame to hold scaled columns
    scaled_data = pd.DataFrame(index=site_level_df.index)

    for col, scaled_col in zip(base_cols, scaled_cols):
        s = site_level_df[col]

        q_low = s.quantile(low_q)
        q_high = s.quantile(high_q)

        # Avoid division by zero if q_high == q_low
        if q_high == q_low:
            scaled = pd.Series(50, index=s.index)  # flat mid-score if no variation
        else:
            scaled = (s - q_low) / (q_high - q_low)

        # Clip to [0, 1] then scale to [0, 100]
        scaled = scaled.clip(lower=0, upper=1) * 100

        scaled_data[scaled_col] = scaled

    # Assign back to final_restaurants_for_lease
    site_level_df[scaled_cols] = scaled_data

    return county_stats_df, zip_stats_df, site_level_df

#-------- Data Merging --------#
def merge_all_levels(county_stats_df, zip_stats_df, site_level_df):
    # Rename 'County_Name' in county_stats_df to 'COUNTY' for merging
    county_stats_df.rename(columns={'County_Name': 'County'}, inplace=True)

    # Merge restaurants_for_lease_df with county_stats_df on 'COUNTY'
    merged_df = pd.merge(site_level_df, county_stats_df, on='County', how='inner')

    # Ensure 'zip' columns have the same data type (e.g., int) before merging
    # Convert 'zip' in zip_stats_df to int, handling potential NaNs
    zip_stats_df['zip'] = zip_stats_df['zip'].dropna().astype(int)
    # Convert 'zip' in merged_df to int, handling potential NaNs
    merged_df['zip'] = merged_df['zip'].dropna().astype(int)

    # Merge the result with zip_stats_df on 'zip'
    final_merged_df = pd.merge(merged_df, zip_stats_df, on='zip', how='inner')
    return final_merged_df

def calculate_final_scores(weights, inputs, top_n=None):
    # Demo Weights (add up to 1)
    Percentage_Asian_Weight = 0.30
    Median_Yearly_Population_Weight = 0.40
    Median_INCTOT_Weight = 0.30

    # Comp Weights (add up to 1)
    Count_total_target_restaurant_Weight = 0.50  
    Count_total_restaurant_Weight = 0.30  
    Median_price_mid_Weight = 0.20

    # Site Weights (add up to 1)
    Total_Rent_Weight = 0.50  
    Avg_SqFt_Weight = 0.50

    site_level_df, county_level_df, zip_level_df = data_cleaning()
    county_level_df, zip_level_df, site_level_df = scaling(
        county_level_df, zip_level_df, site_level_df, inputs
    )
    final_merged_df = merge_all_levels(county_level_df, zip_level_df, site_level_df)

    final_merged_df["Demo_score"] = (
        final_merged_df["Dummy_Percentage_Asian"] * Percentage_Asian_Weight
        + final_merged_df["Dummy_Median_INCTOT"] * Median_INCTOT_Weight
        + final_merged_df["Dummy_Median_Yearly_Population"] * Median_Yearly_Population_Weight
    )

    # NOTE: need different quotes for the f-string here
    target_col = f"Scaled_{inputs['Restaurant_type_input']}"
    final_merged_df["Comp_score"] = (
        (100 - final_merged_df["Scaled_Count_total_restaurant"]) * Count_total_restaurant_Weight
        + (100 - final_merged_df[target_col]) * Count_total_target_restaurant_Weight
        + final_merged_df["Scaled_Median_price_mid"] * Median_price_mid_Weight
    )

    final_merged_df["Site_score"] = (
        final_merged_df["Scaled_Avg_SqFt"] * Avg_SqFt_Weight
        + (100 - final_merged_df["Scaled_Total_Rent"]) * Total_Rent_Weight
    )

    final_merged_df["fit_score"] = (
        final_merged_df["Demo_score"] * weights["Demo_weight"]
        + final_merged_df["Comp_score"] * weights["Comp_weight"]
        + final_merged_df["Site_score"] * weights["Site_weight"]
    )

    final_merged_df_sorted = (
        final_merged_df.sort_values(by="fit_score", ascending=False)
        .reset_index(drop=True)
    )

    # assign a rank (1 = best)
    final_merged_df_sorted["rank"] = final_merged_df_sorted.index + 1

    # Optionally cut to top N
    if top_n is not None:
        final_merged_df_sorted = final_merged_df_sorted.head(top_n).copy()

    return final_merged_df_sorted
