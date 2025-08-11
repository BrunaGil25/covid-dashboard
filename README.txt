# 🦠 COVID-19 Impact Dashboard

An interactive dashboard built with Streamlit to explore COVID-19 metrics across countries and continents.  
Created by [Bruna Gil Garcia](https://www.linkedin.com/in/bruna-gil-garcia-80656069/).

## 🚀 Features

- 📊 Visualize new cases and deaths over time
- 🌍 Explore geographic snapshots with choropleth maps
- 📈 Track vaccination rates and case fatality ratios
- 🎯 Filter by continent, country, and date range
- 📥 Download filtered data as CSV

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- Plotly
- Altair

## 📦 Installation

To run locally:

```bash
git clone https://github.com/BrunaGil25/covid-dashboard.git
cd covid-dashboard
pip install -r requirements.txt
streamlit run app.py

## 🌐 Live App
Check out the deployed version:
👉 Streamlit App (replace with your actual link)

## 📁 Required Datasets

The dashboard uses two datasets:

| Dataset            | File Name               | Source URL                                                                                      | Repository Status                  |
|--------------------|-------------------------|--------------------------------------------------------------------------------------------------|------------------------------------|
| Vaccination Data   | `vaccinations.csv`      | [GitHub Link](https://github.com/owid/covid-19-data/blob/master/public/data/vaccinations/vaccinations.csv) | ✅ Included as `data/vaccination-data.csv` |
| COVID Metrics      | `owid-covid-data.csv`   | [GitHub Link](https://github.com/owid/covid-19-data/blob/master/public/data/latest/owid-covid-latest.csv) | ⚠️ Not included due to file size      |

---

## ⚙️ Setup Instructions

1. Download the **COVID Metrics** dataset from the link above.
2. Rename the file to: `covid_metrics.csv`
3. Place both datasets in the `data/` directory: 
/data
├── vaccination-data.csv
└── covid_metrics.csv


## How to Use the Dashboard

### 1. 📂 Load Your Data
Specify the paths to your CSV files in the sidebar:
- vaccinations.csv
- covid_metrics.csv

### 2. 🎛️ Apply Filters
Customize your analysis:
- Continent: Select a region or "All"
- Countries: Choose one or more countries
- Date Range: Adjust the timeline with the slider
![Filter Panel](images/filter_panel.png)


### 3. 📊 Choose a Metric for Mapping
Select a metric from the dropdown:
- vaccination_rate_latest
- total_cases
- total_deaths
- cases_per_100k
- deaths_per_100k
![Map Metric Dropdown](images/map_metric_dropdown.png)


### 4. 🌍 View Geographic Snapshot
Visualize selected metrics across countries:
![Vaccination Map](images/vaccination_map.png)


### 5. 📈 Analyze Trends Over Time
Explore patterns in new cases and deaths:
![Cases and Deaths Chart](images/cases_deaths_chart.png)


### 6. 📥 Export Filtered Data
Click the Download CSV button to export your filtered dataset

##📸 Dashboard Overview
![Dashboard Overview](images/dashboard_overview.png)

This view shows:
- Data source paths
- Filter controls
- Key metrics summary
- Download option


This dashboard is designed to be intuitive, flexible, and insightful — perfect for researchers, data enthusiasts, and public health analysts.

## 👩‍💻 About Me
I'm a scientist turned data explorer, passionate about turning raw data into meaningful insights.
Connect with me on GitHub or LinkedIn.

