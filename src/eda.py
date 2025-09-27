import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display_functions import display


class DescriptiveAnalyzer:
    def __init__(self, features_df, target_df, id_col='ID', year_col='Year', target_col='Tenure'):
        self.features_df = features_df.copy()
        self.target_df = target_df.copy()
        self.id_col = id_col
        self.year_col = year_col
        self.target_col = target_col

        self.df = pd.merge(features_df, target_df, on=[id_col, year_col], how='left')

    def target_distribution(self, plot_type='bar', show_table=True, weight_col='Weight', by_year=True):
        results = {}

        if by_year and "Year" in self.df.columns:
            years = sorted(self.df["Year"].unique())
        else:
            years = [None]

        for year in years:
            if year is not None:
                df_year = self.df[self.df["Year"] == year]
            else:
                df_year = self.df

            # شمارش وزنی یا ساده
            if weight_col and weight_col in df_year.columns:
                counts = df_year.groupby(self.target_col)[weight_col].sum()
            else:
                counts = df_year[self.target_col].value_counts()

            percentages = counts / counts.sum() * 100
            summary = pd.DataFrame({
                "Weighted_Count" if weight_col else "Count": counts,
                "Percentage": percentages.round(2)
            })

            if show_table:
                print(f"\nYear: {year if year else 'All'}")
                display(summary)

            # فقط یک مثال: رسم bar plot
            if plot_type == 'bar':
                plt.figure(figsize=(8, 5))
                palette = sns.color_palette("Set2", len(counts))
                ax = sns.barplot(
                    x=counts.index,
                    y=counts.values,
                    palette=palette
                )
                plt.title(f"Distribution of Target ({year if year else 'All'})", fontsize=16, weight='bold')
                plt.ylabel("Weighted Count" if weight_col else "Count", fontsize=12)
                plt.xlabel(self.target_col, fontsize=12)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

                for p, perc in zip(ax.patches, percentages):
                    height = p.get_height()
                    ax.text(
                        p.get_x() + p.get_width() / 2.,
                        height + max(counts) * 0.01,
                        f'{perc:.1f}%',
                        ha="center",
                        fontsize=11,
                        weight='bold',
                        color='black'
                    )

                sns.despine()
                plt.tight_layout()
                plt.show()

            results[year if year else "All"] = summary

        return results

    def target_by_category(self, cat_feature):
        ctab = pd.crosstab(self.df[cat_feature], self.df[self.target_col], normalize='index') * 100
        print(ctab)
        ctab.plot(kind='bar', stacked=True, figsize=(8, 5))
        plt.title(f"{self.target_col} by {cat_feature}")
        plt.ylabel("Percentage")
        plt.show()

    def numeric_summary_by_target(self, num_features):
        summary = self.df.groupby(self.target_col)[num_features].agg(['mean', 'median', 'std'])
        print(summary)
        return summary

    def numeric_boxplot_by_target(self, num_features):
        for col in num_features:
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=self.target_col, y=col, data=self.df)
            plt.title(f"{col} by {self.target_col}")
            plt.show()

    def correlation_heatmap(self, num_features):
        temp_df = self.df[num_features + [self.target_col]].copy()
        temp_df[self.target_col] = temp_df[self.target_col].astype('category').cat.codes
        corr = temp_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.show()
        return corr

    def missing_values(self):
        miss = self.df.isnull().sum()
        print("Missing values:\n", miss[miss > 0])
        return miss[miss > 0]

    def numeric_distribution(self, num_features):
        for col in num_features:
            plt.figure(figsize=(6, 4))
            sns.histplot(self.df[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.show()
