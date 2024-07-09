import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import csv
import re
import spacy
import altair as alt
from datetime import datetime
import plotly.express as px
import pydeck as pdk
from opencage.geocoder import OpenCageGeocode


api_key = '8861eb1a815746868fd864e939e4a9b8'


def get_country_coordinates(country_name, api_key):
    # Initialize the OpenCage geocoder with the provided API key
    geocoder = OpenCageGeocode(api_key)

    # Geocode the country name
    results = geocoder.geocode(country_name)

    # Check if results were found
    if results and len(results):
        location = results[0]['geometry']
        return (location['lat'], location['lng'])
    else:
        return None

def validate_coordinates(df, lat_col='Latitude', lon_col='Longitude'):
    """
    Validate and clean the Latitude and Longitude columns.
    Removes rows with invalid coordinates.
    """
    if lat_col in df.columns and lon_col in df.columns:
        df = df.dropna(subset=[lat_col, lon_col])
        df = df[(df[lat_col].between(-90, 90)) & (df[lon_col].between(-180, 180))]
    else:
        st.error("Latitude and Longitude columns are missing in the dataframe.")
        df = pd.DataFrame()  # return an empty dataframe if columns are missing
    return df

st.set_page_config(layout='wide', initial_sidebar_state='expanded')



# 1. as sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Filters",
        options=["Upload", "Consignee", "People", "Location"],
        icons=["upload","wallet", "person", "globe2"],
        menu_icon="cast",
        default_index=0,
        # orientation="horizontal"
    )

if selected == "Upload":
    st.title('Dmetrics Data Parsing and Visualization')
    st.subheader('Insert CSV file')

    uploaded_file = st.file_uploader('Choose a csv file', type=['csv'])

    if uploaded_file:


        st.markdown('---')
        # Read the CSV file
        d = pd.read_csv(uploaded_file)


        df = pd.DataFrame(d)

        # Display the parsed dataframe
        st.subheader("Parsed Data")
        st.session_state['df'] = df
        st.write(df)

    elif 'df' in st.session_state:
        st.header("Sorted Data")
        st.write(st.session_state['df'])

if selected == "Consignee":
    if 'df' in st.session_state:
        df = st.session_state['df']  # Retrieve DataFrame from session_state

        st.sidebar.subheader("Options")
        selected_consignees = st.sidebar.multiselect('Select Consignees', df['Consignee'].unique())

        if selected_consignees:
            time_filter = st.sidebar.checkbox("Filter by Time")

            if time_filter:
                selected_start_date = st.sidebar.date_input(
                    "Select a start date",
                    datetime.now().date()
                )
                selected_end_date = st.sidebar.date_input(
                    "Select an end date",
                    datetime.now().date()
                )

                if selected_start_date and selected_end_date:
                    # Convert the date inputs to datetime format and make them timezone-aware
                    start_date = pd.to_datetime(selected_start_date).tz_localize('UTC')
                    end_date = pd.to_datetime(selected_end_date).tz_localize('UTC')

                    # Convert 'Entry Time' column to datetime format if it's not already
                    if pd.api.types.is_string_dtype(df['Entry Time']):
                        df['Entry Time'] = pd.to_datetime(df['Entry Time'])

                    if df['Entry Time'].dt.tz is None:
                        df['Entry Time'] = df['Entry Time'].dt.tz_localize('UTC')

                    # Filter the DataFrame
                    filtered_data = df[
                        (df['Consignee'].isin(selected_consignees)) &
                        (df['Entry Time'] >= start_date) &
                        (df['Entry Time'] <= end_date)
                    ]
                else:
                    filtered_data = df[df['Consignee'].isin(selected_consignees)]
            else:
                filtered_data = df[df['Consignee'].isin(selected_consignees)]

            if not filtered_data.empty:

                col1, col2 = st.columns(2)

                with col1:
                    # Combine data for all selected consignees for the bar chart
                    combined_data = filtered_data.groupby('Commodities').size().reset_index(name='Frequency')

                    # Display combined bar chart for all selected consignees
                    st.subheader('Combined Commodities Distribution Bar Graph')
                    bar_chart_combined = alt.Chart(combined_data).mark_bar().encode(
                        x=alt.X('Commodities', sort='-y'),
                        y='Frequency'
                    ).properties(
                        title='Combined Commodities Distribution'
                    )
                    st.altair_chart(bar_chart_combined, use_container_width=True)
                with col2:
                    # Display separate pie charts for each consignee
                    for consignee in selected_consignees:
                        consignee_data = filtered_data[filtered_data['Consignee'] == consignee]

                        # Group data by commodity for the pie chart
                        commodity_distribution = consignee_data['Commodities'].value_counts().reset_index()
                        commodity_distribution.columns = ['Commodity', 'Frequency']

                        st.subheader(f'Commodities Distribution for Consignee: {consignee}')

                        # Create pie chart using Plotly
                        pie_chart = px.pie(commodity_distribution, names='Commodity', values='Frequency',
                                           title=f'Commodities Distribution for Consignee: {consignee}')
                        pie_chart.update_layout(
                            legend=dict(
                                orientation="h",
                                yanchor="top",
                                y=-0.2,  # position the legend below the chart
                                xanchor="center",
                                x=0.5
                            ),
                            margin=dict(t=50, b=100, l=50, r=50)  # Add top, bottom, left, and right margins
                        )
                        st.plotly_chart(pie_chart, use_container_width=True)

            else:
                st.write("No data available for the selected Consignees")
        else:
            st.write("Please select at least one Consignee")
    else:
        st.write("Please upload a CSV file in the 'Upload' section first.")

if selected == "People":
    if 'df' in st.session_state:

        st.header("Person Distribution in Data")

        df = st.session_state['df']  # Retrieve DataFrame from session_state

        # Process 'Persons' column to split names and flatten them
        all_persons = []
        for persons in df['Persons']:
            if persons != "Not Found":
                if isinstance(persons, str) and persons.startswith('[') and persons.endswith(']'):
                    names = eval(persons)
                    all_persons.extend(names)
                else:
                    all_persons.append(persons)

        unique_persons = list(set(all_persons))

        selected_persons = st.sidebar.multiselect('Select Persons', unique_persons)

        time_filter = st.sidebar.checkbox("Filter by Time")

        if time_filter:
            selected_start_date = st.sidebar.date_input(
                "Select a start date",
                datetime.now().date()
            )
            selected_end_date = st.sidebar.date_input(
                "Select an end date",
                datetime.now().date()
            )

            if selected_start_date and selected_end_date:
                # Convert the date inputs to pandas Timestamp
                start_date = pd.Timestamp(selected_start_date)
                end_date = pd.Timestamp(selected_end_date)

                # Ensure 'Entry Time' column is datetime and timezone-aware
                df['Entry Time'] = pd.to_datetime(df['Entry Time'], utc=True)

                # Filter the DataFrame by time
                filtered_data = df[
                    (df['Entry Time'] >= start_date.tz_localize('UTC')) &
                    (df['Entry Time'] <= end_date.tz_localize('UTC'))
                    ]
            else:
                filtered_data = df
        else:
            filtered_data = df

        if selected_persons:
            combined_consignee_counts = pd.DataFrame(columns=['Consignee', 'Count'])

            for person in selected_persons:
                # Filter rows where the selected person is mentioned
                person_data = filtered_data[filtered_data['Persons'].apply(
                    lambda x: person in eval(x) if isinstance(x, str) and x.startswith('[') and x.endswith(']') else x == person
                )]

                if not person_data.empty:
                    consignee_counts = person_data['Consignee'].value_counts().reset_index()
                    consignee_counts.columns = ['Consignee', 'Count']

                    # Aggregate counts for the combined bar chart
                    combined_consignee_counts = pd.concat([combined_consignee_counts, consignee_counts]) \
                        .groupby('Consignee', as_index=False)['Count'].sum()

            if not combined_consignee_counts.empty:
                col1, col2 = st.columns(2)

                with col1:
                    # Combined bar chart
                    bar_chart = alt.Chart(combined_consignee_counts).mark_bar().encode(
                        x='Consignee',
                        y='Count'
                    ).properties(
                        title=f'Distribution of {", ".join(selected_persons)} across Consignees',
                        width=100,
                        height=400
                    )

                    st.altair_chart(bar_chart, use_container_width=True)

                with col2:
                    # Combined table
                    st.header("Combined Consignees Distribution")
                    st.table(combined_consignee_counts)

            else:
                st.write("No data available for the selected persons")

            for person in selected_persons:
                # Display individual pie charts for each person
                st.subheader(f'{person} Distribution Pie Chart')

                # Filter rows where the selected person is mentioned
                person_data = filtered_data[filtered_data['Persons'].apply(
                    lambda x: person in eval(x) if isinstance(x, str) and x.startswith('[') and x.endswith(
                        ']') else x == person
                )]

                if not person_data.empty:
                    consignee_counts = person_data['Consignee'].value_counts().reset_index()
                    consignee_counts.columns = ['Consignee', 'Count']

                    # Create pie chart for the individual person
                    pie_chart = px.pie(consignee_counts, names='Consignee', values='Count',
                                       title=f'{person} Distribution')
                    pie_chart.update_layout(
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=-0.2,  # position the legend below the chart
                            xanchor="center",
                            x=0.5
                        ),
                        margin=dict(t=50, b=100, l=50, r=50)  # Add top, bottom, left, and right margins
                    )
                    st.plotly_chart(pie_chart, use_container_width=True)
                else:
                    st.write(f"No data available for {person}")
        else:
            st.write("Please select at least one person")

    else:
        st.write("Please upload a CSV file in the 'Upload' section first.")

if selected == "Location":
    if 'df' in st.session_state:
        df = st.session_state['df']
        st.header("Map Visualization")

        st.sidebar.subheader("Options")

        # Extract unique countries from 'Departure' and 'Arrival' columns
        departure_countries = df['Departure'].unique()
        arrival_countries = df['Arrival'].unique()

        # Combine and get unique countries
        unique_countries = sorted(set(departure_countries) | set(arrival_countries))

        # Sidebar multi-select for countries
        selected_countries = st.sidebar.multiselect("Select Country/Countries", unique_countries)

        if selected_countries:
            # Checkboxes for filter options: Imports and Exports
            show_imports = st.sidebar.checkbox("Show Imports")
            show_exports = st.sidebar.checkbox("Show Exports")

            if show_imports or show_exports:
                # Checkboxes for additional filters: Consignee, Persons, Time
                show_consignee = st.sidebar.checkbox("Filter by Consignee")
                show_persons = st.sidebar.checkbox("Filter by Persons")
                show_time = st.sidebar.checkbox("Filter by Time")

                try:
                    # Filter the DataFrame based on selected countries and additional filters
                    if show_imports and show_exports:
                        filtered_df = df[(df['Departure'].isin(selected_countries)) | (df['Arrival'].isin(selected_countries))]
                    elif show_imports:
                        filtered_df = df[df['Arrival'].isin(selected_countries)]
                    elif show_exports:
                        filtered_df = df[df['Departure'].isin(selected_countries)]
                    else:
                        st.write("Please select at least one option (Imports or Exports).")
                        st.stop()

                    # Apply additional filters
                    if show_consignee:
                        consignee = st.sidebar.text_input("Enter Consignee")
                        if consignee:
                            filtered_df = filtered_df[filtered_df['Consignee'].str.contains(consignee, case=False, na=False)]

                    if show_persons:
                        persons = st.sidebar.text_input("Enter Persons")
                        if persons:
                            filtered_df = filtered_df[filtered_df['Persons'].str.contains(persons, case=False, na=False)]

                    if show_time:
                        start_date = st.sidebar.date_input("Start Date")
                        end_date = st.sidebar.date_input("End Date")

                        if start_date and end_date:
                            # Convert 'Entry Time' column to datetime if it's not already
                            filtered_df['Entry Time'] = pd.to_datetime(filtered_df['Entry Time'])

                            # Convert user inputs to UTC timestamps
                            start_utc = pd.Timestamp(start_date).tz_localize('UTC')
                            end_utc = pd.Timestamp(end_date).tz_localize('UTC')

                            # Filter by date range
                            filtered_df = filtered_df[
                                (filtered_df['Entry Time'] >= start_utc) & (filtered_df['Entry Time'] <= end_utc)
                            ]

                    combined_df = filtered_df.copy()

                    # Check if coordinates are missing and retrieve them if necessary
                    if 'Latitude' not in combined_df.columns or 'Longitude' not in combined_df.columns:
                        for country in selected_countries:
                            coords = get_country_coordinates(country, api_key)
                            if coords:
                                combined_df.loc[combined_df['Departure'] == country, 'Latitude'] = coords[0]
                                combined_df.loc[combined_df['Departure'] == country, 'Longitude'] = coords[1]
                                combined_df.loc[combined_df['Arrival'] == country, 'Latitude'] = coords[0]
                                combined_df.loc[combined_df['Arrival'] == country, 'Longitude'] = coords[1]
                            else:
                                st.write(f"Coordinates for {country} could not be found.")

                        # Validate and clean the coordinates for combined data
                        combined_df = validate_coordinates(combined_df)

                    if not combined_df.empty:
                        # Calculate the stack height for each hexagon based on number of entries
                        combined_df['StackHeight'] = combined_df.groupby('Departure').cumcount() // 5  # Every 5 entries add a new stack

                        # Initialize lists to store import and export lines data
                        import_lines = []
                        export_lines = []

                        # Display individual metrics for each selected country
                        for country in selected_countries:
                            country_df = combined_df[(combined_df['Departure'] == country) | (combined_df['Arrival'] == country)]

                            most_found_commodity = country_df['Commodities'].mode().values[0] if not country_df['Commodities'].mode().empty else 'N/A'
                            commodity_occurrences = country_df['Commodities'].value_counts().max() if not country_df['Commodities'].value_counts().empty else 0
                            least_found_commodity = country_df['Commodities'].value_counts().idxmin() if not country_df['Commodities'].value_counts().empty else 'N/A'
                            least_commodity_occurrences = country_df['Commodities'].value_counts().min() if not country_df['Commodities'].value_counts().empty else 0
                            most_found_person = country_df['Persons'].mode().values[0] if not country_df['Persons'].mode().empty else 'N/A'
                            person_occurrences = country_df['Persons'].value_counts().max() if not country_df['Persons'].value_counts().empty else 0
                            least_found_person = country_df['Persons'].value_counts().idxmin() if not country_df['Persons'].value_counts().empty else 'N/A'
                            least_person_occurrences = country_df['Persons'].value_counts().min() if not country_df['Persons'].value_counts().empty else 0

                            # Clean up names for display
                            most_found_person_clean = most_found_person.replace('[','').replace(']','').replace("'", "")
                            least_found_person_clean = least_found_person.replace('[','').replace(']','').replace("'", "")

                            # Display metrics using style_metric_cards
                            with st.expander(f"Metrics - {country}", expanded=True):
                                st.markdown("<style>div.stMetric div.stMetricText {color: black;}</style>",
                                            unsafe_allow_html=True)

                                # Display table of unique names and their counts if names are found
                                unique_names_counts = country_df['Persons'].str.title().value_counts().reset_index()
                                unique_names_counts.columns = ['Name', 'Count']
                                unique_names_counts = unique_names_counts[
                                    (unique_names_counts['Name'] != 'Not Found') & (
                                                unique_names_counts['Name'] != 'N/A')]
                                unique_names_counts['Name'] = unique_names_counts['Name'].str.replace('[',
                                                                                                      '').str.replace(
                                    ']', '').str.replace("'", "")

                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric(label="Most Found Commodity",
                                              value=most_found_commodity.replace('[', '').replace(']', '').replace("'",
                                                                                                                   ""))
                                    st.metric(label="Occurrences (Most Commodity)", value=commodity_occurrences)
                                with col2:
                                    if not unique_names_counts.empty:
                                        most_found_person_clean = unique_names_counts.iloc[0]['Name']
                                        person_occurrences = unique_names_counts.iloc[0]['Count']
                                        st.metric(label="Most Found Person", value=most_found_person_clean)
                                        st.metric(label="Occurrences (Most Person)", value=person_occurrences)
                                    else:
                                        st.metric(label="Most Found Person", value="N/A")
                                        st.metric(label="Occurrences (Most Person)", value="N/A")
                                with col3:
                                    st.metric(label="Least Found Commodity",
                                              value=least_found_commodity.replace('[', '').replace(']', '').replace("'",
                                                                                                                    ""))
                                    st.metric(label="Occurrences (Least Commodity)", value=least_commodity_occurrences)
                                with col4:
                                    if not unique_names_counts.empty:
                                        least_found_person_clean = unique_names_counts.iloc[-1]['Name']
                                        least_person_occurrences = unique_names_counts.iloc[-1]['Count']
                                        st.metric(label="Least Found Persons", value=least_found_person_clean)
                                        st.metric(label="Occurrences (Least Person)", value=least_person_occurrences)
                                    else:
                                        st.metric(label="Least Found Person", value="N/A")
                                        st.metric(label="Occurrences (Least Person)", value="N/A")

                                if not unique_names_counts.empty:
                                    st.write("### Unique Names and Counts")
                                    st.table(unique_names_counts)

                            # Prepare import and export lines data
                            if show_imports:
                                imports = country_df[country_df['Arrival'] == country]['Departure'].unique()
                                for dest_country in imports:
                                    start_coords = get_country_coordinates(country, api_key)
                                    end_coords = get_country_coordinates(dest_country, api_key)
                                    if start_coords and end_coords:
                                        import_lines.append({
                                            'start_lon': start_coords[1],  # Swap longitude and latitude
                                            'start_lat': start_coords[0],
                                            'end_lon': end_coords[1],
                                            'end_lat': end_coords[0],
                                            'color': [0, 0, 255],  # Blue color for imports
                                        })

                            if show_exports:
                                exports = country_df[country_df['Departure'] == country]['Arrival'].unique()
                                for dest_country in exports:
                                    start_coords = get_country_coordinates(country, api_key)
                                    end_coords = get_country_coordinates(dest_country, api_key)
                                    if start_coords and end_coords:
                                        export_lines.append({
                                            'start_lon': start_coords[1],  # Swap longitude and latitude
                                            'start_lat': start_coords[0],
                                            'end_lon': end_coords[1],
                                            'end_lat': end_coords[0],
                                            'color': [0, 255, 0],  # Green color for exports
                                        })

                        # Map visualization using pydeck
                        st.pydeck_chart(pdk.Deck(
                            initial_view_state=pdk.ViewState(
                                latitude=combined_df['Latitude'].mean() if 'Latitude' in combined_df else 0,
                                longitude=combined_df['Longitude'].mean() if 'Longitude' in combined_df else 0,
                                zoom=3,  # Zoom out the map initially
                                pitch=40,
                            ),
                            layers=[
                                pdk.Layer(
                                    'HexagonLayer',
                                    data=combined_df,
                                    get_position='[Longitude, Latitude]',
                                    radius=100000,  # Radius of the hexagons
                                    elevation_scale=4,  # Scale the elevation for more prominent height differences
                                    elevation_range=[0, 3000],  # Elevation range for the hexagons
                                    get_fill_color='[255, StackHeight * 50, StackHeight * 50, 180]',  # Color based on stack height
                                    pickable=True,
                                    extruded=True,
                                ),
                                pdk.Layer(
                                    'ArcLayer',
                                    data=import_lines,
                                    get_source_position='[start_lon, start_lat]',
                                    get_target_position='[end_lon, end_lat]',
                                    get_source_color='color',
                                    get_target_color='color',
                                    pickable=True,
                                    auto_highlight=True,
                                    width_scale=2,  # Adjust width scale to control the width of the arcs
                                    width_min_pixels=2,
                                    parameters={
                                        'great_circle': False,  # Use straight lines instead of great circle arcs
                                        'controlPoints': 50,  # Adjust control points for arc curvature
                                    },
                                ),
                                pdk.Layer(
                                    'ArcLayer',
                                    data=export_lines,
                                    get_source_position='[start_lon, start_lat]',
                                    get_target_position='[end_lon, end_lat]',
                                    get_source_color='color',
                                    get_target_color='color',
                                    pickable=True,
                                    auto_highlight=True,
                                    width_scale=2,  # Adjust width scale to control the width of the arcs
                                    width_min_pixels=2,
                                    parameters={
                                        'great_circle': False,  # Use straight lines instead of great circle arcs
                                        'controlPoints': 50,  # Adjust control points for arc curvature
                                    },
                                ),
                            ],
                        ))
                        st.write(filtered_df)

                    else:
                        st.write("No entries fit the selected parameters.")

                except ValueError as e:
                    st.write("No entries fit the selected parameters.")
                    st.stop()

        else:
            st.write("Please select at least one country.")

    else:
        st.write("Please upload a CSV file first.")
