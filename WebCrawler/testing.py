import streamlit as st
import pandas as pd

# Load the CSV files
df = pd.read_csv('translated.csv')
bol_consignee_df = pd.read_csv("unique_consignee.csv")

# Extract and clean the unique entries from the Organizations column
unique_organizations = df['Organizations'].dropna().unique()
unique_people = df['Persons']
cleaned_organizations = [org for org in unique_organizations if org != 'None Found']
cleaned_people = [person for person in unique_people if person != 'None Found' and person != 'None']

# Create dataframes from the cleaned unique entries
unique_orgs_df = pd.DataFrame(cleaned_organizations, columns=['Unique Organizations'])
unique_people_df = pd.DataFrame(cleaned_people, columns=['Unique Persons'])

# Display the cleaned dataframes
st.write("Unique Organizations")
st.write(unique_orgs_df)
st.write("Unique Persons")
st.write(unique_people_df)

# Data for comparison
persons = [
    'Yanet Hurtado', 'Michel Mizrach', 'Ruben Medellin', 'Max Yang', 'Leonor Maria',
    'Joaquin Meneses', 'Mardia Avalos', 'Karla Venegas', 'Armendariz Ramirez', 'Liliana Escamilla',
    'Gabriel Aguilar', 'Patricia Vargas', 'Isabel Moreno', 'Marco Hernandez', 'Hilda Paredes',
    'Aracely Almaguer', 'Xochitl Torres', 'Ricardo Rojo', 'Nuria Flores', 'Andrea Sat',
    'Alejandra Fragoso', 'Ricardo Navarro', 'Manola Resendiz', 'Noemi Rodriguez', 'Gilberto Meza',
    'Alejandro Noh', 'David Anguiano', 'Paulina Duron', 'Jose Antonio', 'Ana Carolina',
    'Janeth Campos', 'Montserrat Quintanilla', 'Roberto Vaquera', 'Ernesto Olhagaray', 'Karla Sanchez',
    'Andreia Lima', 'Ramon Alvarado', 'Hugo Toledo', 'Susana Zesati', 'Mayra Camacho',
    'Suzie Souza', 'Thalia Arano', 'Claudia Ruiz', 'Mexico Laura', 'Laura Padilla',
    'Julio Castellanos', 'Antonio Torres', 'Cleber Santos', 'Pablo Palma', 'Victor Valdes',
    'Mayte Leyva', 'Adriana Murillo', 'Emilio Encarnacion', 'Claudia Martinez', 'Ruben Valdes',
    'Steffi Garcia', 'Alejandra Granados', 'Patricia Juarez', 'Patricia Collins', 'Reyna Aurora',
    'Jose Luis', 'Gustavo Ochoa', 'Carlos Occelli', 'Jazmin Hernandez', 'Nahilet Chavez',
    'Paty De', 'Jesus Alejandro', 'Morgana Continue', 'Giovanni Rojas', 'Salvador Gutierrez',
    'Javier Montalvo', 'Melanie Ruiz', 'Aurora Melendez', 'Dafne Gutierrez', 'Julia Kohana',
    'Luis Nuci', 'Hugo Castro', 'Adolfo Castro', 'Paulina Aburto', 'Jose Maria', 'Axel Aguilar',
    'Martha Emma', 'Adriana Vazquez', 'Kin Manuel', 'Liliana Herrera', 'Paulo Cunha',
    'Maria Isabel', 'Thalia Rodriguez', 'Jesica Garcia', 'Deyanira Silva', 'Yoonsik Yu',
    'Olga Ruiz', 'Patricia Gaytan', 'Jesus Landa', 'Mariana Cervantes', 'Daniel Salazar',
    'Alejandra Gonzalez', 'David Gonzalez', 'Alfredo Zouiga', 'Osiel Diaz', 'Joyce Paz'
]

# Create DataFrames for comparison
bol_person_df = pd.DataFrame(persons, columns=["Persons"])

# Check for common entries between bol_person_df and unique_people_df
common_people = pd.merge(bol_person_df, unique_people_df, left_on="Persons", right_on="Unique Persons", how="inner")
if common_people.empty:
    st.write("No common persons found.")
else:
    st.write("Common Persons")
    st.write(common_people)

# Check and display column names in bol_consignee_df
st.write("Column names in bol_consignee_df:")
st.write(bol_consignee_df.columns)

# Check for common entries between bol_consignee_df and unique_orgs_df
if 'Consignee' not in bol_consignee_df.columns:
    st.write("Unique consignee CSV does not contain the expected 'Consignee' column.")
else:
    common_organizations = pd.merge(bol_consignee_df, unique_orgs_df, left_on="Consignee", right_on="Unique Organizations", how="inner")
    if common_organizations.empty:
        st.write("No common organizations found.")
    else:
        st.write("Common Organizations")
        st.write(common_organizations)
