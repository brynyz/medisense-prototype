import pandas as pd
import re

df = pd.read_csv('medisense/data/cleaned/cleaned.csv')

def clean_date_string(s):
    s = s.strip().lower()

    # Replace separators with "-"
    s = re.sub(r"[/.]", "-", s)

    # Replace month names with numbers
    months = {
        "jan": "01", "feb": "02", "mar": "03", "apr": "04",
        "may": "05", "jun": "06", "jul": "07", "aug": "08",
        "sep": "09", "sept": "09", "oct": "10", "nov": "11", "dec": "12"
    }
    for k, v in months.items():
        s = re.sub(rf"{k}[a-z]*", v, s)

    # Remove extra text/commas
    s = re.sub(r"[^\d\-]", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")

    return s


df['course_cleaned'] = (
    df['course']
    .str.lower()
    .str.replace(r'[^\w\s,]', '', regex=True)
    .str.replace(r'\s+', ' ', regex=True)
    .str.replace(r'\b\d+\b', '', regex=True)
    .str.replace(r',+', ',', regex=True)
    .str.strip(' ,')
)

df['gender_cleaned'] = (
    df['gender']
    .str.lower()
    .str.replace(r'[^\w\s,]', '', regex=True)
    .str.replace(r'\s+', ' ', regex=True)
    .str.replace(r'\b\d+\b', '', regex=True)
    .str.replace(r',+', ',', regex=True)
    .str.strip(' ,')
)

unique_course = df['course_cleaned'].unique()
unique_gender = df['gender_cleaned'].unique()

df["normalized"] = df["date"].apply(clean_date_string)
df["clean_date"] = pd.to_datetime(df["normalized"], errors="coerce")

print("Unique Courses:", unique_course)
print("Unique Genders:", unique_gender)

count = 0

for index, row in df.iterrows():
    if row['gender'] == [None] or row['gender'] == '' or pd.isna(row['gender']):
        count += 1

print("Empty Courses Count:", count)

canonical_course = {
    "staff", "bsba", "bscrim", "bse", "beed", "baels", "bscs", "bstm", "bat", "bsit"
    "bsentrep", "bshm", "b"
}

manual_mapping = {
    "bsba": "bsba",
    "bscrim": "bscrim",
    "bse": "bse",
    "beed": "beed",
    "baels": "baels",
    "bscs": "bscs",
    "bstm": "bstm",
    "bat": "bat",
    "bsit": "bsit",
    "bsentrep": "bsentrep",
    "bsent": "bsentrep",
    "bshm": "bshm",

    #staff
    "hr management": "staff",
    "osas": "staff",
    "security": "staff",
    "cashier": "staff",
    "scholarship": "staff",
    "clinic": "staff",
    "library": "staff",
    "extension": "staff",
    "registrar": "staff",
    "qa": "staff",
    "afs": "staff",
}

print(df)