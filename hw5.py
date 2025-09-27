import cloudscraper # pip install cloudscraper
import json
from statistics import mean
from collections import defaultdict
from datetime import datetime

scraper = cloudscraper.create_scraper()
url1 = "https://api.covidtracking.com/v1/states/"
url2 = "/daily.json"

# Read states in from txt file
with open("/home/ubuntu/data5500_venv/states.txt", "r") as f:
    state_names = [line.strip() for line in f if line.strip()]


for state in state_names:   
    full_url = url1 + state + url2
    response = scraper.get(full_url)

    json_str = response.json()

     # Save JSON file for this state
    with open(f"{state}.json", "w") as f:
        json.dump(json_str, f, indent=2)

    # Extract (date, daily new cases) tuples
    daily_cases = [(entry["date"], entry.get("positiveIncrease", 0)) for entry in json_str]

    # Average number of new daily confirmed cases
    avg_cases = round(mean(c for _, c in daily_cases), 2)

    # Date with highest number of cases
    max_date, max_cases = max(daily_cases, key=lambda x: x[1])

    # Most recent date with zero new cases
    zero_dates = [d for d, c in daily_cases if c == 0]
    most_recent_zero = max(zero_dates) if zero_dates else None

    # Month-year with highest/lowest new cases
    monthly_sums = defaultdict(int)

    for d, c in daily_cases:
        date_obj = datetime.strptime(str(d), "%Y%m%d")  # convert YYYYMMDD → datetime
        month_year = date_obj.strftime("%B %Y")         # e.g., "March 2021"
        monthly_sums[month_year] += c

    high_month, high_cases = max(monthly_sums.items(), key=lambda x: x[1])
    low_month, low_cases = min(monthly_sums.items(), key=lambda x: x[1])

    # Print statements
    print("\n Covid confirmed cases statistics")
    print("State name: ", state)
    print("Average number of new daily confimred cases: ", avg_cases)
    print("Date with highest number of cases: ", max_date, max_cases)
    print("Most recent date with no new confirmed cases: ", most_recent_zero)
    print("Month and year with highest new number of covid cases: ", high_month,",", high_cases)
    print("Month and year with lowest new number of covid cases: ", low_month, ",", low_cases)
