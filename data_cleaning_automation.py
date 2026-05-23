"""
Data Cleaning & Reporting Automation
Thiranex Assignment #4
Author: [Your Name]

Description:
    Automates data cleaning and reporting workflows using Python + pandas.
    Handles missing values, duplicates, inconsistent data, and generates
    automated PDF reports with visual summaries.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import warnings
from io import StringIO
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. SAMPLE DATASET GENERATOR (for demo)
# ─────────────────────────────────────────────

def generate_sample_data(n=200):
    """Generate a realistic messy dataset for demonstration."""
    np.random.seed(42)
    names = ["Alice Johnson", "bob smith", "  Carol White ", "david lee",
             "EVE MARTIN", "Frank Brown", "grace wilson", "Henry Davis",
             "ivy moore", "Jack Taylor"]

    data = {
        "ID":         list(range(1, n+1)),
        "Name":       [names[i % len(names)] for i in range(n)],
        "Age":        np.where(np.random.rand(n) < 0.08, np.nan,
                               np.random.randint(18, 65, n).astype(float)),
        "Department": np.random.choice(["HR", "IT", "Finance", "hr",
                                        "it", "FINANCE", "Marketing", None], n),
        "Salary":     np.where(np.random.rand(n) < 0.06, np.nan,
                               np.random.randint(30000, 120000, n).astype(float)),
        "Joining_Date": pd.date_range("2018-01-01", periods=n, freq="W").strftime("%Y-%m-%d"),
        "Email":      [f"user{i}@company.com" if np.random.rand() > 0.05 else None for i in range(n)],
        "Performance_Score": np.where(np.random.rand(n) < 0.07, np.nan,
                                       np.random.choice([1, 2, 3, 4, 5], n).astype(float)),
    }

    df = pd.DataFrame(data)

    # Inject duplicates (~5%)
    dup_idx = np.random.choice(df.index, size=int(n * 0.05), replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    return df


# ─────────────────────────────────────────────
# 2. DATA CLEANING CLASS
# ─────────────────────────────────────────────

class DataCleaner:
    """Automated data cleaning pipeline with a detailed audit log."""

    def __init__(self, df: pd.DataFrame):
        self.original_df = df.copy()
        self.df = df.copy()
        self.log = []
        self.stats_before = {}
        self.stats_after = {}
        self._capture_stats("before")

    # ── helpers ──────────────────────────────

    def _log(self, msg):
        print(f"  [✓] {msg}")
        self.log.append(msg)

    def _capture_stats(self, label):
        stats = {
            "rows":       len(self.df),
            "cols":       len(self.df.columns),
            "duplicates": self.df.duplicated().sum(),
            "missing":    self.df.isnull().sum().sum(),
        }
        if label == "before":
            self.stats_before = stats
        else:
            self.stats_after = stats

    # ── cleaning steps ────────────────────────

    def remove_duplicates(self):
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        removed = before - len(self.df)
        self._log(f"Removed {removed} duplicate row(s).")
        return self

    def handle_missing_values(self):
        # Numeric: fill with median
        num_cols = self.df.select_dtypes(include="number").columns
        for col in num_cols:
            n_missing = self.df[col].isnull().sum()
            if n_missing:
                median = self.df[col].median()
                self.df[col].fillna(median, inplace=True)
                self._log(f"Filled {n_missing} missing '{col}' values with median ({median:.1f}).")

        # Categorical / object: fill with mode or 'Unknown'
        cat_cols = self.df.select_dtypes(include="object").columns
        for col in cat_cols:
            n_missing = self.df[col].isnull().sum()
            if n_missing:
                mode_val = self.df[col].mode()
                fill_val = mode_val[0] if not mode_val.empty else "Unknown"
                self.df[col].fillna(fill_val, inplace=True)
                self._log(f"Filled {n_missing} missing '{col}' values with '{fill_val}'.")
        return self

    def standardize_text(self):
        str_cols = self.df.select_dtypes(include="object").columns
        for col in str_cols:
            self.df[col] = self.df[col].astype(str).str.strip().str.title()
        self._log(f"Standardized text in columns: {list(str_cols)}")
        return self

    def fix_dtypes(self):
        if "Joining_Date" in self.df.columns:
            self.df["Joining_Date"] = pd.to_datetime(self.df["Joining_Date"], errors="coerce")
            self._log("Converted 'Joining_Date' to datetime.")
        num_candidates = ["Age", "Salary", "Performance_Score"]
        for col in num_candidates:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        self._log("Ensured numeric dtypes on Age, Salary, Performance_Score.")
        return self

    def remove_outliers(self, cols=None, z_thresh=3.0):
        if cols is None:
            cols = self.df.select_dtypes(include="number").columns
        before = len(self.df)
        for col in cols:
            mean, std = self.df[col].mean(), self.df[col].std()
            if std == 0:
                continue
            mask = ((self.df[col] - mean).abs() / std) < z_thresh
            self.df = self.df[mask]
        removed = before - len(self.df)
        self.df.reset_index(drop=True, inplace=True)
        self._log(f"Removed {removed} outlier row(s) (Z-score > {z_thresh}).")
        return self

    def clean(self):
        """Run the full cleaning pipeline."""
        print("\n📋 Running Data Cleaning Pipeline...")
        self.remove_duplicates()
        self.handle_missing_values()
        self.standardize_text()
        self.fix_dtypes()
        self.remove_outliers()
        self._capture_stats("after")
        print(f"\n  Rows: {self.stats_before['rows']} → {self.stats_after['rows']}")
        print(f"  Missing values: {self.stats_before['missing']} → {self.stats_after['missing']}")
        print(f"  Duplicates: {self.stats_before['duplicates']} → {self.stats_after['duplicates']}")
        return self.df


# ─────────────────────────────────────────────
# 3. VISUAL REPORT GENERATOR
# ─────────────────────────────────────────────

class ReportGenerator:
    """Generates a multi-page visual summary report saved as PNG pages."""

    BLUE   = "#1A73E8"
    GREEN  = "#34A853"
    ORANGE = "#FB8C00"
    RED    = "#EA4335"
    GRAY   = "#F1F3F4"
    DARK   = "#202124"
    LIGHT  = "#FFFFFF"

    def __init__(self, cleaner: DataCleaner, output_dir="report_output"):
        self.cleaner = cleaner
        self.df_raw   = cleaner.original_df
        self.df_clean = cleaner.df
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.pages = []

    def _style(self):
        plt.rcParams.update({
            "figure.facecolor": self.LIGHT,
            "axes.facecolor":   self.GRAY,
            "axes.edgecolor":   "#DADCE0",
            "axes.grid":        True,
            "grid.color":       "#E8EAED",
            "grid.linestyle":   "--",
            "font.family":      "DejaVu Sans",
            "text.color":       self.DARK,
        })

    def _header(self, fig, title, subtitle=""):
        fig.text(0.05, 0.96, title, fontsize=18, fontweight="bold",
                 color=self.DARK, va="top")
        fig.text(0.05, 0.925, subtitle, fontsize=10, color="#5F6368", va="top")
        fig.text(0.95, 0.96, f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')}",
                 fontsize=8, color="#9AA0A6", ha="right", va="top")
        fig.add_artist(plt.Line2D([0.05, 0.95], [0.915, 0.915],
                                   transform=fig.transFigure,
                                   color=self.BLUE, linewidth=2))

    # ── Page 1: Summary Scorecard ─────────────

    def _page_summary(self):
        self._style()
        fig = plt.figure(figsize=(12, 8))
        self._header(fig, "Data Cleaning & Reporting Automation",
                     "Thiranex Assignment #4  ·  Executive Summary")

        sb = self.cleaner.stats_before
        sa = self.cleaner.stats_after

        metrics = [
            ("Total Rows",      sb["rows"],   sa["rows"],   "rows"),
            ("Missing Values",  sb["missing"],sa["missing"],"cells"),
            ("Duplicates",      sb["duplicates"],sa["duplicates"],"rows"),
            ("Columns",         sb["cols"],   sa["cols"],   "cols"),
        ]

        colors = [self.BLUE, self.ORANGE, self.RED, self.GREEN]

        for i, (label, before, after, unit) in enumerate(metrics):
            ax = fig.add_axes([0.05 + i*0.24, 0.60, 0.20, 0.28])
            ax.set_facecolor(colors[i])
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.text(0.5, 0.70, str(after), ha="center", va="center",
                    fontsize=28, fontweight="bold", color="white",
                    transform=ax.transAxes)
            ax.text(0.5, 0.40, label, ha="center", fontsize=10,
                    color="white", transform=ax.transAxes)
            ax.text(0.5, 0.18, f"was {before} {unit}", ha="center",
                    fontsize=8, color="rgba(255,255,255,0.75)" if False else "#ffffffbb",
                    transform=ax.transAxes)

        # Cleaning log
        ax2 = fig.add_axes([0.05, 0.10, 0.55, 0.44])
        ax2.set_facecolor(self.GRAY)
        ax2.set_xticks([]); ax2.set_yticks([])
        ax2.set_title("Cleaning Log", fontsize=11, fontweight="bold",
                      color=self.DARK, pad=8)
        for spine in ax2.spines.values():
            spine.set_color("#DADCE0")
        for j, entry in enumerate(self.cleaner.log[:10]):
            ax2.text(0.02, 0.90 - j*0.10, f"• {entry[:80]}",
                     transform=ax2.transAxes, fontsize=8, color=self.DARK,
                     va="top")

        # Before/After bar
        ax3 = fig.add_axes([0.65, 0.10, 0.30, 0.44])
        cats = ["Rows", "Missing", "Duplicates"]
        before_vals = [sb["rows"], sb["missing"], sb["duplicates"]]
        after_vals  = [sa["rows"], sa["missing"],  sa["duplicates"]]
        x = np.arange(len(cats))
        ax3.bar(x - 0.2, before_vals, 0.35, label="Before", color=self.RED,   alpha=0.85)
        ax3.bar(x + 0.2, after_vals,  0.35, label="After",  color=self.GREEN, alpha=0.85)
        ax3.set_xticks(x); ax3.set_xticklabels(cats, fontsize=9)
        ax3.set_title("Before vs After", fontsize=11, fontweight="bold",
                      color=self.DARK)
        ax3.legend(fontsize=8)
        ax3.set_facecolor(self.GRAY)

        path = f"{self.output_dir}/page1_summary.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.pages.append(path)
        print(f"  [✓] Page 1 saved: {path}")

    # ── Page 2: Distribution Analysis ─────────

    def _page_distributions(self):
        self._style()
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle("Distribution Analysis — Cleaned Dataset",
                     fontsize=15, fontweight="bold", color=self.DARK, y=0.98)

        num_cols = self.df_clean.select_dtypes(include="number").columns.tolist()[:6]

        for idx, col in enumerate(num_cols):
            ax = axes[idx // 3][idx % 3]
            data = self.df_clean[col].dropna()
            ax.hist(data, bins=20, color=self.BLUE, alpha=0.80, edgecolor="white")
            ax.axvline(data.mean(),   color=self.RED,    linestyle="--", lw=1.5, label=f"Mean: {data.mean():.1f}")
            ax.axvline(data.median(), color=self.ORANGE, linestyle=":",  lw=1.5, label=f"Median: {data.median():.1f}")
            ax.set_title(col, fontsize=10, fontweight="bold", color=self.DARK)
            ax.legend(fontsize=7)
            ax.set_facecolor(self.GRAY)

        # Hide unused subplots
        for idx in range(len(num_cols), 6):
            axes[idx // 3][idx % 3].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = f"{self.output_dir}/page2_distributions.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.pages.append(path)
        print(f"  [✓] Page 2 saved: {path}")

    # ── Page 3: Department & Performance ──────

    def _page_dept_analysis(self):
        self._style()
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        fig.suptitle("Department & Performance Analysis",
                     fontsize=15, fontweight="bold", color=self.DARK, y=1.02)

        df = self.df_clean.copy()

        # Dept headcount
        if "Department" in df.columns:
            dept_counts = df["Department"].value_counts()
            axes[0].bar(dept_counts.index, dept_counts.values,
                        color=[self.BLUE, self.GREEN, self.ORANGE, self.RED, "#9C27B0"][:len(dept_counts)],
                        edgecolor="white", linewidth=0.5)
            axes[0].set_title("Headcount by Department", fontsize=11, fontweight="bold")
            axes[0].set_xticklabels(dept_counts.index, rotation=30, ha="right", fontsize=8)
            axes[0].set_facecolor(self.GRAY)

        # Avg salary by dept
        if "Salary" in df.columns and "Department" in df.columns:
            avg_sal = df.groupby("Department")["Salary"].mean().sort_values()
            axes[1].barh(avg_sal.index, avg_sal.values, color=self.GREEN, alpha=0.85, edgecolor="white")
            axes[1].set_title("Avg Salary by Department", fontsize=11, fontweight="bold")
            axes[1].set_xlabel("Salary (₹)")
            axes[1].set_facecolor(self.GRAY)
            for i, v in enumerate(avg_sal.values):
                axes[1].text(v + 500, i, f"₹{v:,.0f}", va="center", fontsize=7)

        # Performance score distribution
        if "Performance_Score" in df.columns:
            perf = df["Performance_Score"].value_counts().sort_index()
            axes[2].pie(perf.values, labels=[f"Score {int(k)}" for k in perf.index],
                        autopct="%1.1f%%",
                        colors=[self.RED, self.ORANGE, "#FDD835", self.GREEN, self.BLUE],
                        startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1.5})
            axes[2].set_title("Performance Score Split", fontsize=11, fontweight="bold")

        plt.tight_layout()
        path = f"{self.output_dir}/page3_dept_analysis.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.pages.append(path)
        print(f"  [✓] Page 3 saved: {path}")

    # ── Page 4: Missing Value Heatmap ─────────

    def _page_missing_heatmap(self):
        self._style()
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Missing Values — Before vs After",
                     fontsize=15, fontweight="bold", color=self.DARK)

        for ax, df, label, cmap in [
            (axes[0], self.df_raw,   "BEFORE Cleaning", "Reds"),
            (axes[1], self.df_clean, "AFTER Cleaning",  "Greens"),
        ]:
            miss = df.isnull().astype(int)
            if miss.shape[0] > 50:
                miss = miss.iloc[:50]
            if miss.values.sum() == 0:
                ax.text(0.5, 0.5, "No Missing\nValues ✓",
                        ha="center", va="center", fontsize=16,
                        color=self.GREEN, fontweight="bold",
                        transform=ax.transAxes)
                ax.set_facecolor(self.GRAY)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                sns.heatmap(miss, ax=ax, cbar=False, cmap=cmap,
                            yticklabels=False, linewidths=0.0)
            ax.set_title(label, fontsize=11, fontweight="bold", color=self.DARK)
            ax.set_xlabel("Columns", fontsize=9)

        plt.tight_layout()
        path = f"{self.output_dir}/page4_missing.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.pages.append(path)
        print(f"  [✓] Page 4 saved: {path}")

    # ── Page 5: Correlation Matrix ────────────

    def _page_correlation(self):
        self._style()
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.suptitle("Correlation Matrix — Numeric Features",
                     fontsize=15, fontweight="bold", color=self.DARK)

        num_df = self.df_clean.select_dtypes(include="number")
        corr   = num_df.corr()

        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, ax=ax, mask=mask, annot=True, fmt=".2f",
                    cmap="coolwarm", center=0, linewidths=0.5,
                    linecolor="white", square=True, cbar_kws={"shrink": 0.8})
        ax.set_facecolor(self.GRAY)

        plt.tight_layout()
        path = f"{self.output_dir}/page5_correlation.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.pages.append(path)
        print(f"  [✓] Page 5 saved: {path}")

    # ── Combine all pages ─────────────────────

    def _combine_pages(self):
        """Stitch all page PNGs into one tall combined report image."""
        from PIL import Image as PILImage
        imgs = [PILImage.open(p) for p in self.pages]
        max_w = max(im.width for im in imgs)
        total_h = sum(im.height for im in imgs) + 10 * len(imgs)
        combined = PILImage.new("RGB", (max_w, total_h), (255, 255, 255))
        y = 0
        for im in imgs:
            combined.paste(im, (0, y))
            y += im.height + 10
        out = f"{self.output_dir}/full_report.png"
        combined.save(out)
        print(f"\n  [✓] Full report: {out}")
        return out

    def generate(self):
        print("\n📊 Generating Visual Report...")
        self._page_summary()
        self._page_distributions()
        self._page_dept_analysis()
        self._page_missing_heatmap()
        self._page_correlation()
        try:
            full = self._combine_pages()
        except ImportError:
            full = self.pages[0]
            print("  [!] Pillow not installed; pages saved individually.")
        return self.pages, full

    def export_clean_csv(self):
        path = f"{self.output_dir}/cleaned_data.csv"
        self.df_clean.to_csv(path, index=False)
        print(f"  [✓] Clean CSV saved: {path}")
        return path

    def print_summary_stats(self):
        print("\n📈 Summary Statistics (Cleaned Data):")
        print(self.df_clean.describe(include="all").to_string())


# ─────────────────────────────────────────────
# 4. MAIN ENTRY POINT
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  DATA CLEANING & REPORTING AUTOMATION")
    print("  Thiranex Assignment #4")
    print("=" * 60)

    # ── Step 1: Load data (demo dataset) ──────
    print("\n📂 Step 1: Loading dataset...")
    df_raw = generate_sample_data(n=200)
    print(f"  Loaded {len(df_raw)} rows × {len(df_raw.columns)} columns")
    print(f"  Columns: {list(df_raw.columns)}")

    # ── Step 2: Clean ─────────────────────────
    print("\n🧹 Step 2: Cleaning data...")
    cleaner = DataCleaner(df_raw)
    df_clean = cleaner.clean()

    # ── Step 3: Report ────────────────────────
    print("\n📊 Step 3: Generating report...")
    reporter = ReportGenerator(cleaner, output_dir="report_output")
    pages, full_report = reporter.generate()
    reporter.export_clean_csv()
    reporter.print_summary_stats()

    print("\n" + "=" * 60)
    print("  ✅ DONE!")
    print(f"  Report pages: {len(pages)}")
    print(f"  Output folder: report_output/")
    print("=" * 60)

    return df_clean, pages, full_report


if __name__ == "__main__":
    df_clean, pages, full_report = main()
