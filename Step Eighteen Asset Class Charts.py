"""
Step Eighteen Asset Class Charts

INPUT FILES (local):
- T2_Final_Country_Weights.xlsx (Sheet All Periods): country weights from Step Eight/Nine
- Step Factor Categories.xlsx (Sheet Asset Class): Country → Asset Class mapping

OUTPUT FILES:
- T2_Asset_Class.xlsx: asset class weights by period (Final)
- T2_Asset_Class_Weights.pdf: portfolio weights over time
- T2_Asset_Class_Split.pdf: one chart per asset class
- T2_Asset_Class_MissingDataLog.txt: mapping issues and completeness
- T2_Country_Weights.xlsx: country weights by period (Final)
- T2_Country_Weights_Split.pdf: per-country charts

NOTE: Step Fourteen (Target Optimization) is moot; this script now uses only the
Step Eight/Nine output (T2_Final_Country_Weights.xlsx) and no longer compares
against an Optimized portfolio.
"""
# Step Eighteen: Aggregate country weights into asset class weights for each period
# Documented for clarity and reproducibility

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
import os

print("[DEBUG] Working directory:", os.getcwd())

# ========== SECTION 1: Load Data ==========

# File paths (all local)
FINAL_PATH = "T2_Final_Country_Weights.xlsx"
MAPPING_PATH = "Step Factor Categories.xlsx"
OUT_XLSX = "T2_Asset_Class.xlsx"
OUT_PDF = "T2_Asset_Class_Weights.pdf"
OUT_SPLIT_PDF = "T2_Asset_Class_Split.pdf"
OUT_LOG = "T2_Asset_Class_MissingDataLog.txt"
# New country-level outputs
OUT_COUNTRY_XLSX = "T2_Country_Weights.xlsx"
OUT_COUNTRY_SPLIT_PDF = "T2_Country_Weights_Split.pdf"

# Read country weights
def read_weights(filepath, sheet):
    df = pd.read_excel(filepath, sheet_name=sheet)
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    return df

final_df = read_weights(FINAL_PATH, "All Periods")

# Read country to asset class mapping
mapping_df = pd.read_excel(MAPPING_PATH, sheet_name="Asset Class")
country_to_asset = dict(zip(mapping_df.iloc[:,0], mapping_df.iloc[:,1]))

# ========== SECTION 2: Aggregate to Asset Class ==========

missing_countries = set()
missing_log = []

# Helper to aggregate weights by asset class
def aggregate_to_asset(df, country_to_asset, portfolio_name):
    asset_classes = sorted(set(country_to_asset.values()))
    out = []
    completeness = defaultdict(int)
    for idx, row in df.iterrows():
        date = row["Date"]
        asset_weights = dict.fromkeys(asset_classes, 0.0)
        missing = []
        for country in df.columns[1:]:
            weight = row[country]
            asset = country_to_asset.get(country, None)
            if asset is None:
                missing.append(country)
                continue
            asset_weights[asset] += weight
            completeness[country] += int(weight != 0)
        # Fill missing country weights with mean (if any), log
        if missing:
            for m in missing:
                mean_val = row[df.columns[1:]].mean()
                for asset in asset_weights:
                    asset_weights[asset] += mean_val / len(asset_classes)
                missing_log.append(f"{portfolio_name} | {date.date()} | {m}: filled with mean {mean_val:.6f}")
                missing_countries.add(m)
        out.append([date] + [asset_weights[a] for a in asset_classes])
    result_df = pd.DataFrame(out, columns=["Date"] + asset_classes)
    return result_df, completeness

final_asset_df, final_completeness = aggregate_to_asset(final_df, country_to_asset, "Final")

print("[DEBUG] Final asset DataFrame shape:", final_asset_df.shape)
print(final_asset_df.head())

# ========== SECTION 3: Write Output Excel ==========

try:
    with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as writer:
        print(f"[DEBUG] Writing Excel file: {OUT_XLSX}")
        final_asset_df.to_excel(writer, sheet_name="Final", index=False)
        workbook = writer.book
        worksheet = writer.sheets["Final"]
        date_format = workbook.add_format({'num_format': 'dd-mmm-yyyy'})
        worksheet.set_column(0, 0, 15, date_format)
        for i in range(1, len(final_asset_df.columns)):
            worksheet.set_column(i, i, 15)
    print(f"[DEBUG] Excel file written: {OUT_XLSX}")
except Exception as e:
    print(f"[ERROR] Writing Excel file failed: {e}")

# ========== SECTION 3B: Write Country Weights Excel ==========
try:
    with pd.ExcelWriter(OUT_COUNTRY_XLSX, engine="xlsxwriter") as writer:
        print(f"[DEBUG] Writing Country Excel file: {OUT_COUNTRY_XLSX}")
        # Final country weights
        final_countries_df = final_df.copy()
        final_countries_df = final_countries_df.rename(columns={final_countries_df.columns[0]: "Date"})
        final_countries_df.to_excel(writer, sheet_name="Final", index=False)
        workbook = writer.book
        worksheet = writer.sheets["Final"]
        date_format = workbook.add_format({'num_format': 'dd-mmm-yyyy'})
        worksheet.set_column(0, 0, 15, date_format)
        for i in range(1, len(final_countries_df.columns)):
            worksheet.set_column(i, i, 15)
    print(f"[DEBUG] Country Excel file written: {OUT_COUNTRY_XLSX}")
except Exception as e:
    print(f"[ERROR] Writing Country Excel file failed: {e}")

# ========== SECTION 4: Plot Asset Class Weights ==========

try:
    # Original summary chart (Final only)
    plt.figure(figsize=(12, 7))
    for asset in final_asset_df.columns[1:]:
        plt.plot(final_asset_df['Date'], final_asset_df[asset], label=f"Final - {asset}")
    plt.xlabel("Date", fontsize=10)
    plt.ylabel("Asset Class Weight", fontsize=10)
    plt.title("Asset Class Weights Through Time (Final)", fontsize=12)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, ncol=2)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.savefig(OUT_PDF)
    plt.close()
    print(f"[DEBUG] PDF plot written: {OUT_PDF}")

    # New: Separate chart per asset class in a multipage PDF
    from matplotlib.backends.backend_pdf import PdfPages
    asset_classes = final_asset_df.columns[1:]
    with PdfPages(OUT_SPLIT_PDF) as pdf:
        for asset in asset_classes:
            plt.figure(figsize=(10, 5))
            plt.plot(final_asset_df['Date'], final_asset_df[asset], label='Final', color='tab:orange')
            plt.xlabel("Date", fontsize=8)
            plt.ylabel("Weight", fontsize=8)
            plt.title(f"{asset} Asset Class Weight Through Time", fontsize=10)
            plt.legend(fontsize=7)
            plt.xticks(fontsize=7)
            plt.yticks(fontsize=7)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    print(f"[DEBUG] Split PDF (one chart per asset class) written: {OUT_SPLIT_PDF}")

    # ========== SECTION 4B: Plot Country Weights ==========
    countries = final_df.columns[1:]
    with PdfPages(OUT_COUNTRY_SPLIT_PDF) as pdf:
        for country in countries:
            plt.figure(figsize=(10, 5))
            plt.plot(final_df['Date'], final_df[country], label='Final', color='tab:orange')
            plt.xlabel("Date", fontsize=8)
            plt.ylabel("Weight", fontsize=8)
            plt.title(f"{country} Country Weight Through Time", fontsize=10)
            plt.legend(fontsize=7)
            plt.xticks(fontsize=7)
            plt.yticks(fontsize=7)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    print(f"[DEBUG] Split PDF (one chart per country) written: {OUT_COUNTRY_SPLIT_PDF}")
except Exception as e:
    print(f"[ERROR] PDF plot failed: {e}")

# ========== SECTION 5: Log Missing Data and Completeness ==========

try:
    print(f"[DEBUG] Writing log file: {OUT_LOG}")
    with open(OUT_LOG, 'w') as f:
        f.write("Missing country mappings and replacements (mean imputation):\n")
        for line in missing_log:
            f.write(line + "\n")
        f.write("\nData completeness (nonzero count by country):\n")
        f.write(f"\nFinal Portfolio:\n")
        for country, count in final_completeness.items():
            f.write(f"{country}: {count}\n")
    print(f"[DEBUG] Log file written: {OUT_LOG}")
except Exception as e:
    print(f"[ERROR] Writing log file failed: {e}")

print("Step Eighteen complete. Outputs: T2_Asset_Class.xlsx, T2_Asset_Class_Weights.pdf, T2_Asset_Class_MissingDataLog.txt")
