import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display_functions import display
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import chi2_contingency, f_oneway
from itertools import combinations
from typing import List, Optional, Dict, Any

sns.set(style="whitegrid")


class DescriptiveAnalyzerPro:
    """
    کلاس تحلیل توصیفی پیشرفته — طراحی شده برای:
      - فیچرها و جدول target جدا هستند (وصل شدن با ID+Year در صورت نیاز)
      - تولید نمودارها (pairplot, violin, stacked bar, heatmaps)
      - تست‌های آماری (chi2 برای دسته‌ای، ANOVA برای عددی)
      - اندازه‌گیری ارتباط (Mutual Info، Cramer's V)
    """

    def __init__(
            self,
            features_df: pd.DataFrame,
            target_df: pd.DataFrame,
            id_col: str = "ID",
            year_col: str = "Year",
            target_col: str = "Tenure",
    ):
        self.features = features_df.copy()
        self.target = target_df.copy()
        self.id_col = id_col
        self.year_col = year_col
        self.target_col = target_col

        # چک اولیه
        for c in [self.id_col, self.year_col]:
            if c not in self.features.columns:
                raise ValueError(f"ستون '{c}' در features_df موجود نیست.")
            if c not in self.target.columns:
                raise ValueError(f"ستون '{c}' در target_df موجود نیست.")
        if self.target_col not in self.target.columns:
            raise ValueError(f"ستون target '{self.target_col}' در target_df نیست.")

    def _join_target(self) -> pd.DataFrame:
        """موقتا features و target را با هم بر اساس ID و Year جوین می‌کند و نتیجه را برمی‌گرداند."""
        df = self.features.merge(
            self.target[[self.id_col, self.year_col, self.target_col]],
            on=[self.id_col, self.year_col],
            how="left",
            validate="m:1",
        )
        return df

    def summary_stats_by_year(
            self,
            numeric_features: Optional[List[str]] = None,
            weight_col: str = "Weight",
            bins: int = 40,
            figsize: tuple = (10, 6),
            show_table: bool = True,
            show_plot: bool = True,
            save: Optional[str] = None,
    ):
        """
        Compute weighted summary stats and optionally plot boxplots/histograms by year.
        Returns: dict {year: DataFrame of stats}
        """
        df = self.features.copy()
        years = sorted(df[self.year_col].unique())
        results = {}

        if numeric_features is None:
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_features = [
                c for c in numeric_features if c not in [self.id_col, self.year_col, weight_col]
            ]

        for year in years:
            df_year = df[df[self.year_col] == year]

            if weight_col and weight_col in df_year.columns:
                w = df_year[weight_col]

                def wmean(x):
                    return np.average(x, weights=w.loc[x.index])

                def wstd(x):
                    avg = np.average(x, weights=w.loc[x.index])
                    return np.sqrt(np.average((x - avg) ** 2, weights=w.loc[x.index]))

                stats = pd.DataFrame(index=numeric_features)
                stats["count"] = w.count()
                stats["mean"] = df_year[numeric_features].apply(wmean)
                stats["std"] = df_year[numeric_features].apply(wstd)
                stats["min"] = df_year[numeric_features].min()
                stats["25%"] = df_year[numeric_features].quantile(0.25)
                stats["median"] = df_year[numeric_features].median()
                stats["50%"] = df_year[numeric_features].quantile(0.50)
                stats["75%"] = df_year[numeric_features].quantile(0.75)
                stats["max"] = df_year[numeric_features].max()

            else:
                stats = df_year[numeric_features].describe().T
                stats["median"] = df_year[numeric_features].median()
                stats = stats[
                    ["count", "mean", "std", "min", "25%", "median", "50%", "75%", "max"]
                ]

            if show_table:
                display(f"📅 Year: {year}")
                display(stats)
                if save:
                    stats.to_excel(f"{save}_{year}.xlsx")

            if show_plot:
                for feat in numeric_features:
                    fig, ax = plt.subplots(figsize=figsize)
                    sns.boxplot(
                        x=self.year_col,
                        y=feat,
                        data=df[df[self.year_col] == year],
                        ax=ax,
                        color="skyblue"
                    )
                    ax.set_title(f"Distribution of {feat} in {year}")
                    if save:
                        fig.savefig(f"{save}_{feat}_{year}.png", bbox_inches="tight")
                    plt.show()

            results[year] = stats

        return results

    def distribution_by_target(
            self,
            numeric_features: Optional[List[str]] = None,
            bins: int = 40,
            figsize: tuple = (12, 8),
            save: Optional[str] = None,
            weight_col: str = "Weight",
            by_year: bool = False
    ) -> Dict[str, plt.Figure]:
        """
        هیستوگرام‌های تفکیکیِ هر فیچر عددی بر اساس target.
        برمی‌گرداند: دیکشنری {feature: figure}
        """

        df = self._join_target()
        if numeric_features is None:
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
            # حذف ID, Year, Weight و ستون‌های غیرتحلیلی
            numeric_features = [
                c
                for c in numeric_features
                if c not in [self.id_col, self.year_col, weight_col]
            ]

        figs = {}
        if pd.api.types.is_categorical_dtype(df[self.target_col]):
            df[self.target_col] = df[self.target_col].cat.add_categories("MISSING")

        years = df[self.year_col].unique() if by_year else [None]

        categories = df[self.target_col].fillna("MISSING").unique()
        for feat in numeric_features:
            for year in years:
                sub_df = df.copy()
                if year is not None:
                    sub_df = sub_df[sub_df[self.year_col] == year]

                if sub_df.empty:
                    continue

                fig, ax = plt.subplots(figsize=figsize)
                for cat in categories:
                    cat_data = sub_df[sub_df[self.target_col] == cat]
                    sns.histplot(
                        data=cat_data,
                        x=feat,
                        weights=cat_data[weight_col] if weight_col in cat_data else None,  # 👈 وزن اینجا اعمال میشه
                        label=str(cat),
                        kde=False,
                        bins=bins,
                        stat="density",
                        alpha=0.5,
                        ax=ax,
                    )

                title = f"Distribution of {feat} by {self.target_col}"
                if year is not None:
                    title += f" (Year {year})"
                ax.set_title(title)
                ax.legend()

                key = f"{feat}_{year}" if year is not None else feat
                figs[key] = fig

                if save:
                    fig.savefig(f"{save}_{key}.png", bbox_inches="tight")
                plt.close(fig)

        return figs

    def target_distribution(self, plot_type='bar', show_table=True):
        """
        نمایش توزیع تارگت به صورت جدول و نمودار.
        plot_type: 'bar', 'pie', 'hbar'
        """
        counts = self.df[self.target_col].value_counts()
        percentages = counts / counts.sum() * 100
        summary = pd.DataFrame({
            "Count": counts,
            "Percentage": percentages.round(2)
        })

        # نمایش جدول
        if show_table:
            display(summary)

        # انتخاب رنگ‌ها
        palette = sns.color_palette("Set2", len(counts))

        if plot_type == 'bar':
            plt.figure(figsize=(8, 5))
            ax = sns.countplot(
                x=self.target_col,
                data=self.df,
                order=counts.index,
                palette=palette
            )
            plt.title("Distribution of Target", fontsize=16, weight='bold')
            plt.ylabel("Count", fontsize=12)
            plt.xlabel(self.target_col, fontsize=12)

            for p, perc in zip(ax.patches, percentages):
                height = p.get_height()
                ax.text(
                    p.get_x() + p.get_width() / 2.,
                    height + max(counts) * 0.01,
                    f'{perc:.1f}%',
                    ha="center",
                    fontsize=11,
                    weight='bold'
                )

            sns.despine()
            plt.tight_layout()
            plt.show()

        elif plot_type == 'pie':
            plt.figure(figsize=(6, 6))
            wedges, texts, autotexts = plt.pie(
                counts,
                labels=counts.index,
                autopct='%1.1f%%',
                startangle=140,
                colors=palette,
                textprops={'fontsize': 12}
            )
            plt.setp(autotexts, size=11, weight="bold", color="white")
            plt.title("Distribution of Target", fontsize=16, weight='bold')
            plt.tight_layout()
            plt.show()

        elif plot_type == 'hbar':
            plt.figure(figsize=(8, 5))
            ax = sns.barplot(
                y=counts.index,
                x=counts.values,
                palette=palette
            )
            plt.title("Distribution of Target", fontsize=16, weight='bold')
            plt.xlabel("Count", fontsize=12)
            plt.ylabel(self.target_col, fontsize=12)

            for p, perc in zip(ax.patches, percentages):
                width = p.get_width()
                ax.text(
                    width + max(counts) * 0.01,
                    p.get_y() + p.get_height() / 2,
                    f'{perc:.1f}%',
                    va="center",
                    fontsize=11,
                    weight='bold'
                )

            sns.despine()
            plt.tight_layout()
            plt.show()

        else:
            print("plot_type must be one of ['bar', 'pie', 'hbar']")

        return summary

    def violin_plots(
            self,
            numeric_features: Optional[List[str]] = None,
            max_cols: int = 3,
            figsize_per: tuple = (5, 4),
            save_prefix: Optional[str] = None,
            weight_col: str = "Weight"
    ) -> Dict[str, plt.Figure]:
        df = self._join_target()

        if numeric_features is None:
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_features = [
                c for c in numeric_features if c not in [self.id_col, self.year_col, weight_col]
            ]

        figs = {}

        for feat in numeric_features:

            sub_df = df.copy()

            # توسعه داده‌ها بر اساس وزن
            sub_df = sub_df.loc[sub_df[weight_col] > 0, [self.target_col, feat, weight_col]]
            sub_df = sub_df.loc[sub_df[weight_col].notna()]

            rows = []
            for _, row in sub_df.iterrows():
                rows.extend([row.drop(weight_col)] * int(round(row[weight_col])))
            sub_df = pd.DataFrame(rows)

            if sub_df.empty:
                continue

            fig, ax = plt.subplots(figsize=figsize_per)
            sns.violinplot(x=self.target_col, y=feat, data=sub_df, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            title = f"Violin plot of {feat} by {self.target_col}"

            ax.set_title(title)

            key = feat
            figs[key] = fig

            if save_prefix:
                fig.savefig(f"{save_prefix}_violin_{key}.png", bbox_inches="tight")
            plt.close(fig)

        return figs

    def pairplot_sample(
            self,
            numeric_features: Optional[List[str]] = None,
            sample_frac: float = 0.02,
            max_vars: int = 6,
            random_state: int = 42,
            save: Optional[str] = None,
    ) -> plt.Figure:
        """
        pairplot/sample scatter matrix برای تعدادی فیچر عددی.
        از نمونه برداری استفاده می‌کنیم تا خیلی سنگین نشه.
        """
        df = self._join_target()
        if numeric_features is None:
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_features = [
                c for c in numeric_features if c not in [self.id_col, self.year_col]
            ]
        numeric_features = numeric_features[:max_vars]

        # اضافه کردن ستون weight به نمونه
        samp = df[numeric_features + [self.target_col, "Weight"]].sample(
            frac=sample_frac, random_state=random_state
        )

        g = sns.pairplot(samp, hue=self.target_col, corner=True, diag_kind="kde")
        if save:
            g.savefig(save, bbox_inches="tight")
            return g.fig

    def stacked_bar_categorical(
            self,
            cat_features: Optional[List[str]] = None,
            top_n: int = 10,
            figsize: tuple = (10, 6),
            save: Optional[str] = None,
    ) -> Dict[str, plt.Figure]:
        """
        برای هر فیچر دسته‌ای یک نمودار stacked bar می‌سازیم:
        سهم هر دسته target در هر دسته فیچر.
        """
        df = self._join_target()
        if cat_features is None:
            cat_features = df.select_dtypes(include=["object", "category"]).columns.tolist()
            cat_features = [c for c in cat_features if c not in [self.target_col]]

        figs = {}
        for feat in cat_features:
            # اگر تعداد دسته‌ها زیاد بود، فقط top_n
            top_cats = df[feat].value_counts().nlargest(top_n).index.tolist()
            sub = df[df[feat].isin(top_cats)].dropna(subset=[self.target_col])

            if sub.empty:
                continue

                # در نظر گرفتن وزن‌ها
            ctab = pd.crosstab(
                sub[feat], sub[self.target_col],
                normalize="index",
                values=sub["weight"],
                aggfunc="sum"
            )

            fig, ax = plt.subplots(figsize=figsize)
            ctab.plot(kind="bar", stacked=True, ax=ax)
            ax.set_title(f"Stacked share of {self.target_col} by {feat}")
            ax.legend(title=self.target_col, bbox_to_anchor=(1.05, 1), loc="upper left")
            figs[feat] = fig
            if save:
                fig.savefig(f"{save}_stacked_{feat}.png", bbox_inches="tight")
            plt.close(fig)
        return figs


    def stacked_bar_categorical(
            self,
            cat_features: Optional[List[str]] = None,
            top_n: int = 10,
            figsize: tuple = (10, 6),
            save: Optional[str] = None,
    ) -> Dict[str, plt.Figure]:
        """
        برای هر فیچر دسته‌ای یک نمودار stacked bar می‌سازیم:
        سهم هر دسته target در هر دسته فیچر.
        """
        df = self._join_target()
        if cat_features is None:
            cat_features = df.select_dtypes(include=["object", "category"]).columns.tolist()
            cat_features = [c for c in cat_features if c not in [self.target_col]]

        figs = {}
        for feat in cat_features:
            # اگر تعداد دسته‌ها زیاد بود، فقط top_n
            top_cats = df[feat].value_counts().nlargest(top_n).index.tolist()
            sub = df[df[feat].isin(top_cats)].dropna(subset=[self.target_col])

            if sub.empty:
                continue

                # در نظر گرفتن وزن‌ها
            ctab = pd.crosstab(
                sub[feat], sub[self.target_col],
                normalize="index",
                values=sub["Weight"],
                aggfunc="sum"
            )

            fig, ax = plt.subplots(figsize=figsize)
            ctab.plot(kind="bar", stacked=True, ax=ax)
            ax.set_title(f"Stacked share of {self.target_col} by {feat}")
            ax.legend(title=self.target_col, bbox_to_anchor=(1.05, 1), loc="upper left")
            figs[feat] = fig
            if save:
                fig.savefig(f"{save}_stacked_{feat}.png", bbox_inches="tight")
            plt.close(fig)
        return figs

    def correlation_heatmaps(
            self,
            numeric_features: Optional[List[str]] = None,
            per_target: bool = True,
            figsize: tuple = (10, 8),
            save_prefix: Optional[str] = None,
    ) -> Dict[str, plt.Figure]:
        """
        نقشه همبستگی کلی و در صورت نیاز نقشه‌های همبستگی به تفکیک target.
        برمی‌گرداند دیکشنری {'overall': fig, 'cat_value': fig, ...}
        """
        df = self._join_target()
        if numeric_features is None:
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_features = [
                c for c in numeric_features if c not in [self.id_col, self.year_col]
            ]

            # اضافه کردن ستون weight به لیست فیچرها (اگه نباشه)
        if "weight" not in numeric_features:
            numeric_features.append("Weight")

        figs = {}
        # Overall
        corr = df[numeric_features].corr()
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr, annot=False, ax=ax, cmap="coolwarm")
        ax.set_title("Overall correlation")
        figs["overall"] = fig
        if save_prefix:
            fig.savefig(f"{save_prefix}_corr_overall.png", bbox_inches="tight")
        plt.close(fig)

        if per_target:
            for cat in df[self.target_col].dropna().unique():
                sub = df[df[self.target_col] == cat]
                if sub.shape[0] < 10:
                    continue
                corr = sub[numeric_features].corr()
                fig, ax = plt.subplots(figsize=figsize)
                sns.heatmap(corr, annot=False, ax=ax, cmap="coolwarm")
                ax.set_title(f"Correlation for target={cat}")
                figs[str(cat)] = fig
                if save_prefix:
                    fig.savefig(f"{save_prefix}_corr_{cat}.png", bbox_inches="tight")
                plt.close(fig)

        return figs


# ----------------- آماری / اندازه‌گیری ارتباط -----------------
    def cramers_v(self, x: pd.Series, y: pd.Series) -> float:
        """محاسبه Cramér's V برای دو متغیر دسته‌ای."""
        ct = pd.crosstab(x, y)
        if ct.size == 0:
            return np.nan
        chi2, p, dof, ex = chi2_contingency(ct, correction=False)
        n = ct.sum().sum()
        if n == 0:
            return np.nan
        phi2 = chi2 / n
        r, k = ct.shape
        # اصلاح برای bias
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        denom = min((kcorr - 1), (rcorr - 1))
        if denom == 0:
            return np.nan
        return np.sqrt(phi2corr / denom)

    def chi2_tests_categorical(
            self, cat_features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        برای هر فیچر دسته‌ای، آزمون chi2 (crosstab بین فیچر و target)
        خروجی: جدول با columns = [feature, chi2, p_value, dof, cramers_v]
        """
        df = self._join_target().dropna(subset=[self.target_col])
        if cat_features is None:
            cat_features = df.select_dtypes(include=["object", "category"]).columns.tolist()
            cat_features = [c for c in cat_features if c not in [self.target_col]]
        rows = []
        for feat in cat_features:
            try:
                ct = pd.crosstab(df[feat].fillna("MISSING"), df[self.target_col])
                if ct.size == 0:
                    continue
                chi2, p, dof, ex = chi2_contingency(ct)
                cv = self.cramers_v(df[feat].fillna("MISSING"), df[self.target_col])
                rows.append({"feature": feat, "chi2": chi2, "p_value": p, "dof": dof, "cramers_v": cv})
            except Exception as e:
                rows.append({"feature": feat, "chi2": np.nan, "p_value": np.nan, "dof": np.nan, "cramers_v": np.nan})
        return pd.DataFrame(rows).sort_values("p_value")

    def anova_numeric_by_target(
            self, numeric_features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        ANOVA (f_oneway) برای هر فیچر عددی: آیا میانگین در گروه‌های مختلف target تفاوت معناداری دارد؟
        خروجی: feature, f_stat, p_value
        """
        df = self._join_target().dropna(subset=[self.target_col])
        if numeric_features is None:
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_features = [c for c in numeric_features if c not in [self.id_col, self.year_col]]
        rows = []
        groups = df[self.target_col].unique()
        for feat in numeric_features:
            samples = [df.loc[df[self.target_col] == g, feat].dropna().values for g in groups]
            # نیاز به حداقل دو گروه با داده
            valid = [s for s in samples if len(s) > 1]
            if len(valid) < 2:
                rows.append({"feature": feat, "f_stat": np.nan, "p_value": np.nan})
                continue
            try:
                f_stat, p = f_oneway(*valid)
                rows.append({"feature": feat, "f_stat": f_stat, "p_value": p})
            except Exception:
                rows.append({"feature": feat, "f_stat": np.nan, "p_value": np.nan})
        return pd.DataFrame(rows).sort_values("p_value")

    def mutual_info_ranking(
            self,
            numeric_features: Optional[List[str]] = None,
            cat_features: Optional[List[str]] = None,
            discrete_features: Optional[List[str]] = None,
            random_state: int = 0,
    ) -> pd.DataFrame:
        """
        Mutual Information (برای برآورد اهمیت فیچرها نسبت به target).
        این متد فقط برای target دسته‌ای مناسب است (classification-like).
        """
        df = self._join_target().dropna(subset=[self.target_col]).copy()
        # ساخت X و y
        if numeric_features is None:
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_features = [c for c in numeric_features if c not in [self.id_col, self.year_col]]
        if cat_features is None:
            cat_features = df.select_dtypes(include=["object", "category"]).columns.tolist()
            cat_features = [c for c in cat_features if c not in [self.target_col]]

        # برای فیچرهای دسته‌ای: label encoding ساده
        X = pd.DataFrame(index=df.index)
        for c in numeric_features:
            X[c] = df[c].fillna(0)
        for c in cat_features:
            X[c] = pd.factorize(df[c].fillna("MISSING"))[0]

        if X.shape[0] < 5:
            raise ValueError("دیتا برای محاسبه Mutual Info خیلی کم است.")
        y = pd.factorize(df[self.target_col])[0]
        # همه‌ی فیچر‌ها discrete یا continuous مشخص شود
        if discrete_features is None:
            discrete = [1 if c in cat_features else 0 for c in X.columns]
        else:
            discrete = [1 if c in discrete_features else 0 for c in X.columns]

        mi = mutual_info_classif(X.fillna(0), y, discrete_features=np.array(discrete), random_state=random_state)
        res = pd.DataFrame({"feature": X.columns, "mutual_info": mi}).sort_values("mutual_info", ascending=False)
        return res

    def quick_report(self, top_cat: int = 10, top_num: int = 10) -> Dict[str, Any]:
        """
        یک گزارش سریع ترکیبی (summary, top categorical chi2, top numeric anova, mutual info)
        برمی‌گرداند dict از نتایج.
        """
        summary = self.summary_stats()
        chi2 = self.chi2_tests_categorical().head(top_cat)
        anova = self.anova_numeric_by_target().head(top_num)
        try:
            mi = self.mutual_info_ranking().head(20)
        except Exception as e:
            mi = pd.DataFrame({"feature": [], "mutual_info": []})
        return {"summary": summary, "chi2": chi2, "anova": anova, "mutual_info": mi}
