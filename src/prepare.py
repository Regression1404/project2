import hbsir

from src import TableInspector
import pandas as pd
from functools import reduce
from sklearn.preprocessing import StandardScaler


def encode_column(df, col, method="onehot"):
    if method == "onehot":
        encoded = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df.drop(columns=[col]), encoded], axis=1)
    elif method == "label":
        df[col] = df[col].astype("category").cat.codes
    else:
        raise ValueError("method must be 'onehot' or 'label'")
    return df


def prepare_household_information(form="normalized"):
    household_info_inspector = TableInspector(table_name="household_information", form=form)
    household_info_inspector.load_table()
    cols_to_drop = ["Main_Household", "Alternative_Household", "Season", "Month"]
    existing_cols = [c for c in cols_to_drop if c in household_info_inspector.df.columns]
    household_info_inspector.drop_columns(existing_cols)
    household_info_inspector.inspect_columns()
    household_info_inspector.encode_categorical()
    return household_info_inspector.df


def prepare_house_specifications():
    house_spec_inspector = TableInspector("house_specifications")
    house_spec_inspector.load_table()
    house_spec_inspector.drop_columns(["Other_Dwelling_Types", "No_Appliances", "Tenure"])
    house_spec_inspector.inspect_columns()
    house_spec_inspector.encode_categorical()
    return house_spec_inspector.df


def prepare_members_properties():
    members_prop_inspector = TableInspector("members_properties")
    members_prop_inspector.load_table()
    members_prop_inspector.inspect_columns()

    members_df = members_prop_inspector.df

    group_cols = ["ID", "Year"]

    literate_count = members_df.groupby(group_cols)["Is_Literate"].sum(min_count=1).rename("Literate_Count")

    student_count = members_df.groupby(group_cols)["Is_Student"].sum(min_count=1).rename("Student_Count")

    age_stats = members_df.groupby(group_cols)["Age"].agg(
        Mean_Age="mean",
        Max_Age="max",
        Min_Age="min"
    )

    head_edu = (
        members_df[members_df["Relationship"] == "Head"]
        .set_index(group_cols)["Education_Level"]
        .rename("Head_Education")
    )

    activity_status = pd.get_dummies(members_df["Activity_Status"])
    activity_status[group_cols] = members_df[group_cols]
    activity_agg = activity_status.groupby(group_cols).sum()

    marital_status = pd.get_dummies(members_df["Marital_Status"])
    marital_status[group_cols] = members_df[group_cols]
    marital_agg = marital_status.groupby(group_cols).sum()

    sex_status = pd.get_dummies(members_df["Sex"])
    sex_status[group_cols] = members_df[group_cols]
    sex_agg = sex_status.groupby(group_cols).sum()

    result = pd.concat(
        [literate_count, student_count, age_stats,
         head_edu, activity_agg, marital_agg, sex_agg],
        axis=1
    ).reset_index(drop=False).sort_values(by=["Year", "ID"]).reset_index(drop=True)

    if "index" in result.columns:
        result = result.drop(columns="index")

    result = encode_column(result, "Head_Education")

    return result


def prepare_outlays():
    outlays_inspector = TableInspector("Original_Outlays")
    outlays_inspector.load_table()
    outlays_inspector.inspect_columns()

    outlays = outlays_inspector.df

    outlays = outlays.groupby(["ID", "Year"]).agg(
        Total_Gross_Expenditure=("Gross_Expenditure", "sum"),
        Total_Net_Expenditure=("Net_Expenditure", "sum"),
    ).reset_index().sort_values(by=["Year", "ID"]).reset_index(drop=True)

    return outlays


def prepare_home():
    home_inspector = TableInspector("home")
    home_inspector.load_table()
    home_inspector.inspect_columns()

    home = home_inspector.df

    home = (
        home
        .groupby(["ID", "Year"])["Expenditure"]
        .sum()
        .rename("Home_Expenditure")
        .reset_index()
        .sort_values(by=["Year", "ID"]).reset_index(drop=True)
    )

    return home


def prepare_furniture():
    furniture_inspector = TableInspector("furniture")
    furniture_inspector.load_table()
    furniture_inspector.inspect_columns()

    furniture = furniture_inspector.df

    furniture = (
        furniture
        .groupby(["ID", "Year"])["Expenditure"]
        .sum()
        .rename("Furniture_Expenditure")
        .reset_index()
        .sort_values(by=["Year", "ID"]).reset_index(drop=True)
    )

    return furniture


def prepare_imputed_rent():
    imputed_rent_inspector = TableInspector("Imputed_Rent")
    imputed_rent_inspector.load_table()
    imputed_rent_inspector.inspect_columns()

    df = imputed_rent_inspector.df.rename(
        columns={"Income": "Imputed_Rent_Income"}
    )

    df_pivot = df.pivot_table(
        index=["ID", "Year"],
        columns="Income_Type",
        values="Imputed_Rent_Income",
        aggfunc="sum"
    ).reset_index()

    df_pivot = df_pivot.rename(columns={
        "NonCash_ImputedRent_Ownership": "Imputed_Rent_Ownership",
        "NonCash_ImputedRent_Mortgage": "Imputed_Rent_Mortgage"
    })

    df_pivot.columns.name = None

    return df_pivot


def merge_dataframes(dfs, on=None, how="outer"):
    if on is None:
        on = ["ID", "Year"]
    if not dfs:
        raise ValueError("No dataframes provided")

    merged_df = reduce(lambda left, right: pd.merge(left, right, on=on, how=how), dfs)
    return merged_df


def add_composite_features(df):
    df = df.copy()

    df["Density"] = df["Members"] / (df["Number_of_Rooms"].replace(0, 1))

    df["Income_Expenditure_Ratio"] = df["Income"] / (df["Gross_Expenditure"].replace(0, 1))

    df["Area_per_Capita"] = (df["House_Area"].clip(lower=0) / df["Number_of_Rooms"]).astype(float)

    return df


def add_geographical_features(df):
    df = df.copy()

    df = hbsir.add_attribute(table=df, name="Urban_Rural")

    df = hbsir.add_attribute(table=df, name="Province")

    return df


def standardize_numeric(df, numeric_cols=None):
    df = df.copy()

    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ["ID", "Year"]]

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    print(f"Standardized columns: {numeric_cols}")
    return df


def add_quintile_decile(df, income_col="Income", weight_col="Weight"):
    df = df.copy()

    df = hbsir.calculate.add_quantile(
        table=df,
        on_column=income_col,
        bins=5,
        quantile_column_name="Income_Quintile",
        weight_column=weight_col,
        annual=True
    )

    df = hbsir.calculate.add_decile(
        table=df,
        on_column=income_col,
        quantile_column_name="Income_Decile",
        weight_column=weight_col,
        annual=True
    )

    return df
