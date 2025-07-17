# 📦 dMetrics Trade Intelligence Dashboard

An interactive data science dashboard designed to extract, process, and visualize structured insights from international shipping records, using NLP techniques and machine learning models. This project focuses on trade activities involving Mexico and global stakeholders, aiming to support due diligence, supply chain intelligence, and compliance monitoring.

## 🧠 Project Goals

This app transforms unstructured trade data within Bills of Lading into actionable intelligence. It extracts key entities such as commodities, consignee companies, contact names, phone numbers, and HS codes using a custom-trained NLP model and visualizes trends and relationships in a Streamlit-powered interface.

## 🚀 Key Features

- **Entity Extraction (NLP):**
  - Uses spaCy-based models (custom and `en_core_web_sm`) to extract:
    - Commodity names
    - Contact persons
    - Phone numbers
    - HS codes
    - Weights and quantities
    - Port Logistics

- **Data Cleaning & Parsing:**
  - Regex and linguistic filtering for noisy shipping terms (e.g., “SAID TO CONTAIN”, “PALLETS”, “PACKAGES”)
  - Handles ambiguous or concatenated fields (e.g., contacts, weights)

- **Geolocation:**
  - Resolves country names to coordinates using OpenCage API
  - Supports import/export visualization on world map

- **Interactive Visualization:**
  - Bar & pie charts for:
    - Commodity distribution by consignee
    - Person-level distribution across companies
  - Global map showing trade routes (imports/exports)
  - Metric panels summarizing top people/commodities

## 🛠 Tech Stack

| Component      | Tool/Library              |
|----------------|---------------------------|
| Web Framework  | Streamlit                 |
| NLP            | spaCy (custom + en_core_web_sm) |
| Visualization  | Altair, Plotly, PyDeck     |
| Geocoding      | OpenCage Geocoder          |
| Data Handling  | Pandas, Multiprocessing    |
| File Input     | CSV parsing via Streamlit uploader |

## 🖼 Screenshots

### 🔍 Consignee Commodity Distribution
![Consignee](./Consignee.png)

### 🌍 Global Trade Routes for Mexico
![Location](./Location.png)

### 👤 Person-to-Company Mapping
![People](./People.png)

### 🗃 Overview of Named Entities Extracted
![Overview](./Overview.png)

### 📂 Upload and Parsing Interface
![Upload](./Upload.png)

## 🧪 How It Works

1. **Upload Data:**
   - Upload a `.csv` file with bill of lading data (`BolData5.csv` used for demo)

2. **Automatic Parsing:**
   - Entities are extracted and cleaned using a custom spaCy NLP pipeline
   - Relevant fields (e.g., weights, quantities, contacts) are computed dynamically

3. **Interactive Filtering:**
   - Filter results by Consignee, Person, Country, Time, Port, and Commodity Type
   - Export tables as CSV for offline use

4. **Geospatial Mapping:**
   - Shows trade relationships between Mexico and other nations using arcs
   - Visualizes commodity/person metrics by country

## 💡 Skills Demonstrated

- Named Entity Recognition (NER) customization and deployment
- Regex-based preprocessing
- End-to-end data pipeline design using Streamlit
- Multithreaded CSV processing
- Interactive dashboard UI design
- Geospatial and time-series visualization

## 📁 File Structure
