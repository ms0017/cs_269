import geopandas as gpd
import pandas as pd

def get_country_boundaries_and_stats(countries, stats_data):
    shapefile_path = r"./10m_cultural/10m_cultural/ne_10m_admin_0_countries.shp"
    world = gpd.read_file(shapefile_path)
    
    # Create DataFrame
    stats_df = pd.DataFrame.from_dict(stats_data, orient='index', columns=['Sites', 'Aircraft_Seen'])
    stats_df.index.name = 'Country'
    stats_df.reset_index(inplace=True)
    
    # Get African countries from the shapefile
    african_countries = world[world['SOVEREIGNT'].isin(countries)].copy()
    merged_data = pd.merge(african_countries, stats_df, how='inner', left_on='SOVEREIGNT', right_on='Country')
    merged_data['Aircraft_per_Site'] = (merged_data['Aircraft_Seen'] / merged_data['Sites']).round(2)
    
    return merged_data

def analyze_coverage(merged_data):
    """
    Analyze and print coverage statistics
    """
    total_sites = merged_data['Sites'].sum()
    total_aircraft = merged_data['Aircraft_Seen'].sum()
    
    print("\nCoverage Analysis:")
    print(f"Total monitoring sites: {total_sites}")
    print(f"Total aircraft tracked: {total_aircraft:,}")
    print("\nTop 5 countries by number of sites:")
    top_sites = merged_data.nlargest(5, 'Sites')[['Country', 'Sites']]
    print(top_sites)
    
    print("\nTop 5 countries by aircraft tracked:")
    top_aircraft = merged_data.nlargest(5, 'Aircraft_Seen')[['Country', 'Aircraft_Seen']]
    print(top_aircraft)
    
    print("\nTop 5 countries by aircraft per site:")
    top_efficiency = merged_data.nlargest(5, 'Aircraft_per_Site')[['Country', 'Aircraft_per_Site']]
    print(top_efficiency)

if __name__ == '__main__':
    try:
        stats_data = {
            'South Africa': [142, 289285],
            'Morocco': [9, 79784],
            'Kenya': [18, 52805],
            'Ghana': [6, 7803],
            'Zimbabwe': [5, 6182],
            'Cabo Verde': [4, 6912],
            'Togo': [3, 6565],
            'United Republic of Tanzania': [2, 4144],
            'Mozambique': [4, 4620],
            'Somalia': [2, 4902],
            'Senegal': [2, 4036],
            'Botswana': [2, 2616],
            'Benin': [2, 2687],
            'Burkina Faso': [2, 2657],
            'Angola': [7, 2703],
            'Zambia': [2, 1903],
            'Niger': [1, 2022],
            'Madagascar': [5, 1428],
            'Gabon': [3, 1860],
            'Seychelles': [4, 1594],
            'Mali': [2, 1339],
            'Mauritania': [1, 1300],
            'Rwanda': [1, 1064],
            'São Tomé and Príncipe': [1, 1123],
            'Malawi': [1, 948],
            'Uganda': [1, 822],
            'Liberia': [2, 597],
            'Guinea': [1, 556],
            'Cameroon': [1, 469],
            'eSwatini': [1, 5],
            'Republic of the Congo': [1, 0],
            'Nigeria': [11, 13736],
            'Ivory Coast': [9, 11080],
            'Namibia': [13, 9222],
            'Ethiopia': [2, 7321]
        }
        
        merged_data = get_country_boundaries_and_stats(stats_data.keys(), stats_data)
        analyze_coverage(merged_data)
        merged_data.to_csv('african_countries_flight_data.csv', index=False)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")