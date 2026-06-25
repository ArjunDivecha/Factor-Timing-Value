---
trigger: always_on
description: 
globs: 
---

# Your rule content
# Excel Date Formatting Guidelines

When writing Excel files in Python code:

1. ALWAYS add proper date formatting for date columns using this pattern:

```python
# After creating the pandas DataFrame with date columns
with pd.ExcelWriter('output.xlsx', engine='xlsxwriter') as writer:
    # Write the DataFrame
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    
    # Access workbook and worksheet objects
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    
    # Create date format - choose preferred format:
    # Common formats: 'yyyy-mm-dd', 'dd-mmm-yyyy', 'mm/dd/yyyy', 'mmmm d, yyyy'
    date_format = workbook.add_format({'num_format': 'dd-mmm-yyyy'})
    
    # Apply format to date column(s)
    date_col_idx = df.columns.get_loc('DateColumnName')
    for row in range(len(df)):
        worksheet.write(row+1, date_col_idx, df['DateColumnName'].iloc[row], date_format)
```

2. For Excel serial dates (numerical dates), convert to datetime first:

```python
def excel_serial_to_datetime(serial_number):
    """Convert Excel serial date to datetime."""
    if not isinstance(serial_number, (int, float)):
        return serial_number
    # Excel epoch starts at 1899-12-30
    # Adjust for Excel's 1900 leap year bug
    if serial_number > 59:
        serial_number -= 1
    return datetime(1899, 12, 30) + timedelta(days=serial_number)

# Apply to DataFrame column
df['DateColumn'] = df['DateColumn'].apply(excel_serial_to_datetime)
```

3. Always set appropriate column widths for date columns:

```python
# Adjust column width for better visibility
worksheet.set_column(date_col_idx, date_col_idx, 15)  # Width of 15
```

4. Use the right date format string based on requirements:
   - 'yyyy-mm-dd' → 2024-01-31
   - 'dd-mmm-yyyy' → 31-Jan-2024
   - 'mm/dd/yyyy' → 01/31/2024
   - 'mmmm d, yyyy' → January 31, 2024

5. For multiple date columns, apply formatting to each:

```python
date_columns = ['Date1', 'Date2']
for date_col in date_columns:
    col_idx = df.columns.get_loc(date_col)
    for row in range(len(df)):
        worksheet.write(row+1, col_idx, df[date_col].iloc[row], date_format)
```
- You can @ files here
- You can use markdown but dont have to
