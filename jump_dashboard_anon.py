import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import numpy as np
from scipy import stats

# ---------------------------------------------------------
# Utah Color Scheme
# ---------------------------------------------------------
U_RED = "#CC0000"
U_BLACK = "#000000"
U_GRAY = "#808080"

st.set_page_config(page_title="Utah Gymnastics Dashboard", layout="wide")

# ---------------------------------------------------------
# Load James's data (your correct file paths)
# ---------------------------------------------------------
finaldf = pd.read_pickle("finaldf.pkl")
fulljumpdf = pd.read_csv("fulljumpdf.csv")

fulljumpdf = pd.read_csv("fulljumpdf.csv")


# ---------------------------------------------------------
# Load ActiveGym jump data (all 3 seasons, DJ only)
# ---------------------------------------------------------
@st.cache_data
def load_jump_data():
    df = pd.read_excel("ActiveGym Jumps.xlsx", header=6)
    df = df[df['Test Type'].str.upper() == 'DROP JUMP'].copy()
    df['Test Date'] = pd.to_datetime(df['Test Date'], errors='coerce')

    def assign_season(date):
        if pd.isna(date):
            return None
        year = date.year
        return f"{year}-{year+1}" if date.month >= 8 else f"{year-1}-{year}"

    df['Season'] = df['Test Date'].apply(assign_season)
    df = df[df['Season'].isin(['2023-2024', '2024-2025', '2025-2026'])]
    return df

jump_df = load_jump_data()

RSI_COL = 'RSI (Flight Time/Contact Time)'
JH_COL = 'Jump Height (Flight Time) [cm]'
CT_COL = 'Contact Time [s]'
# ---------------------------------------------------------
# Calculate per-athlete, per-season fatigue flags
# ---------------------------------------------------------
@st.cache_data
def calculate_flags(df):
    df['test_day'] = df['Test Date'].dt.date
    session = (
        df.groupby(['Athlete', 'Season', 'test_day'])[[RSI_COL, JH_COL, CT_COL]]
        .mean()
        .reset_index()
    )
    session['test_day'] = pd.to_datetime(session['test_day'])

    baseline = (
        session.groupby(['Athlete', 'Season'])[RSI_COL]
        .agg(['mean', 'std'])
        .reset_index()
        .rename(columns={'mean': 'rsi_mean', 'std': 'rsi_sd'})
    )

    session = session.merge(baseline, on=['Athlete', 'Season'])
    session['rsi_zscore'] = (session[RSI_COL] - session['rsi_mean']) / session['rsi_sd']

    def flag(z):
        if pd.isna(z):
            return 'Unknown'
        if z <= -2:
            return 'High Concern'
        elif z <= -1:
            return 'Watch'
        return 'Normal'

    session['fatigue_flag'] = session['rsi_zscore'].apply(flag)
    return session

session_df = calculate_flags(jump_df)

# ---------------------------------------------------------
# RSI trend significance per athlete per season
# ---------------------------------------------------------
@st.cache_data
def calculate_trends():
    ag = pd.read_excel("ActiveGym Jumps.xlsx", header=6)
    ag = ag[ag['Test Type'].str.upper() == 'DROP JUMP'].copy()
    ag['Test Date'] = pd.to_datetime(ag['Test Date'], errors='coerce')
    ag['month'] = ag['Test Date'].dt.month
    ag = ag[ag['month'].isin([1, 2, 3, 4])]

    def assign_season(date):
        if pd.isna(date):
            return None
        year = date.year
        return f"{year}-{year+1}" if date.month >= 8 else f"{year-1}-{year}"

    ag['Season'] = ag['Test Date'].apply(assign_season)
    ag = ag[ag['Season'].isin(['2023-2024', '2024-2025', '2025-2026'])]

    results = []
    ag_rsi = RSI_COL

    for (athlete, season), group in ag.groupby(['Athlete', 'Season']):
        group = group.dropna(subset=[ag_rsi]).sort_values('Test Date')
        if len(group) < 3:
            continue

        days = (group['Test Date'] - group['Test Date'].min()).dt.days
        slope, intercept, r, p, se = stats.linregress(days, group[ag_rsi])

        results.append({
            'Athlete': athlete,
            'Season': season,
            'slope': slope,
            'p_value': p,
            'significant': p < 0.05,
            'direction': 'Declining' if slope < 0 else 'Improving'
        })

    return pd.DataFrame(results)

trend_df = calculate_trends()

# ---------------------------------------------------------
# Anonymize athlete names (consistent mapping across all dataframes)
# ---------------------------------------------------------
all_names = pd.concat([
    finaldf["Athlete_x"].dropna(),
    session_df["Athlete"].dropna(),
    trend_df["Athlete"].dropna()
]).unique()

# Sort alphabetically so numbering is stable across app restarts
all_names_sorted = sorted(all_names)
athlete_map = {name: f"Athlete {i+1}" for i, name in enumerate(all_names_sorted)}

finaldf["Athlete_x"] = finaldf["Athlete_x"].map(athlete_map).fillna(finaldf["Athlete_x"])
session_df["Athlete"] = session_df["Athlete"].map(athlete_map).fillna(session_df["Athlete"])
trend_df["Athlete"] = trend_df["Athlete"].map(athlete_map).fillna(trend_df["Athlete"])

# ---------------------------------------------------------
# Clean finaldf
# ---------------------------------------------------------
finaldf["Date"] = pd.to_datetime(finaldf["Date"], errors="coerce")
finaldf["Test Date"] = pd.to_datetime(finaldf["Test Date"], errors="coerce")
finaldf["Days Between Jump and Meet"] = (finaldf["Date"] - finaldf["Test Date"]).dt.days
finaldf["All Events"] = finaldf[["Vault Score", "Bar Score", "Beam Score", "Floor Score"]].mean(axis=1)

def assign_season(date):
    if pd.isna(date):
        return None
    year = date.year
    return f"{year}-{year+1}" if date.month >= 8 else f"{year-1}-{year}"

finaldf["Season"] = finaldf["Date"].apply(assign_season)
fulljumpdf["Test Date"] = pd.to_datetime(fulljumpdf["Test Date"], errors="coerce")
fulljumpdf["Season"] = fulljumpdf["Test Date"].apply(assign_season)

# ---------------------------------------------------------
# Normalize dates (year → 2000)
# ---------------------------------------------------------
fulljumpdf["MonthDay"] = fulljumpdf["Test Date"].dt.strftime("%m-%d")
fulljumpdf["MonthDay_dt"] = pd.to_datetime(fulljumpdf["MonthDay"], format="%m-%d").apply(
    lambda d: d.replace(year=2000)
)

finaldf["Date_norm"] = finaldf["Date"].apply(
    lambda d: d.replace(year=2000) if pd.notna(d) else d
)
seasons = ["2023-2024", "2024-2025", "2025-2026"]

dj_metrics = {
    "RSI": RSI_COL,
    "RSI-mod": "RSI (JH (Flight Time)/Contact Time) [m/s]",
    "Jump Height": "Rebound Jump Height (Flight Time) [cm]",
    "Contact Time": "Rebound Contact Time [ms]",
    "Peak Force": "Peak Drop Landing Force [N]",
    "Peak RFD": "Drop Landing RFD [N/s]",
}

cmj_metrics = {
    "RSI": RSI_COL,
    "Jump Height": "Rebound Jump Height (Flight Time) [cm]",
    "Peak Force": "Peak Drop Landing Force [N]",
    "Peak RFD": "Drop Landing RFD [N/s]",
}

score_cols = ["Vault Score", "Bar Score", "Beam Score", "Floor Score", "All Events"]

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.header("View Mode")
view_mode = st.sidebar.selectbox("Select View", ["Athlete", "Team", "Fatigue Overview"])
season_choice = st.sidebar.selectbox("Select Season", ["Compare all"] + seasons)

# ---------------------------------------------------------
# ATHLETE VIEW
# ---------------------------------------------------------
if view_mode == "Athlete":
    st.title("Utah Gymnastics Dashboard — Athlete View")

    if season_choice == "Compare all":
        df_season = finaldf[finaldf["Season"].notna()]
        jump_season = session_df.copy()
    else:
        df_season = finaldf[finaldf["Season"] == season_choice]
        jump_season = session_df[session_df['Season'] == season_choice]

    athletes = sorted(df_season["Athlete_x"].dropna().unique())

    if len(athletes) == 0:
        st.warning("No athletes found for this season.")
    else:
        athlete = st.sidebar.selectbox("Select Athlete", athletes)

        event_default = "All Events"
        event = st.sidebar.selectbox(
            "Event for Regression",
            score_cols,
            index=score_cols.index(event_default)
        )

        df_a = df_season[df_season["Athlete_x"] == athlete].sort_values("Date")
        jump_a = jump_season[jump_season['Athlete'] == athlete].sort_values('test_day')

        st.write(f"### Athlete: {athlete} — Season: {season_choice}")

        # ---- Fatigue status banner ----
        if not jump_a.empty:
            latest = jump_a.sort_values('test_day').iloc[-1]
            flag = latest['fatigue_flag']
            flag_color = {
                'Normal': '🟢', 'Watch': '🟠', 'High Concern': '🔴', 'Unknown': '⚪'
            }.get(flag, '⚪')
            flag_rate = (jump_a['fatigue_flag'] != 'Normal').mean() * 100

            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Current Fatigue Status",
                f"{flag_color} {flag}",
                help="Based on most recent session RSI vs athlete's own season average"
            )
            col2.metric(
                "Season Flag Rate",
                f"{flag_rate:.1f}%",
                help="% of sessions flagged Watch or High Concern this season"
            )
            col3.metric(
                "Sessions Tested",
                len(jump_a)
            )

            # RSI trend significance
            selected_season = season_choice if season_choice != 'Compare all' else '2025-2026'
            athlete_trend = trend_df[
                (trend_df['Athlete'] == athlete) &
                (trend_df['Season'] == selected_season)
            ]
            if not athlete_trend.empty:
                t = athlete_trend.iloc[0]
                if t['significant']:
                    direction = t['direction']
                    color = "red" if direction == "Declining" else "green"
                    st.markdown(
                        f"**Season RSI Trend:** :{color}[{direction} — statistically significant (p={t['p_value']:.3f})]"
                    )
                else:
                    st.markdown(
                        f"**Season RSI Trend:** No significant trend detected (p={t['p_value']:.3f})"
                    )

        if df_a["Days Between Jump and Meet"].notna().sum() > 0:
            avg_days = df_a["Days Between Jump and Meet"].mean()
            st.write(f"**Average time between jump and meet:** {avg_days:.1f} days")

        # ---- RSI trend with flagged sessions overlaid ----
        st.subheader("RSI Across the Season — Flagged Sessions Highlighted")

        if not jump_a.empty and RSI_COL in jump_a.columns:
            fig = go.Figure()

            # Main RSI line
            fig.add_trace(go.Scatter(
                x=jump_a['test_day'],
                y=jump_a[RSI_COL],
                mode='lines+markers',
                name='Session RSI',
                line=dict(color=U_BLACK, width=2),
                marker=dict(size=6, color=U_BLACK)
            ))

            # Watch sessions
            watch = jump_a[jump_a['fatigue_flag'] == 'Watch']
            if not watch.empty:
                fig.add_trace(go.Scatter(
                    x=watch['test_day'],
                    y=watch[RSI_COL],
                    mode='markers',
                    name='Watch',
                    marker=dict(size=14, color='orange',
                                line=dict(color='black', width=1.5))
                ))

            # High Concern sessions
            concern = jump_a[jump_a['fatigue_flag'] == 'High Concern']
            if not concern.empty:
                fig.add_trace(go.Scatter(
                    x=concern['test_day'],
                    y=concern[RSI_COL],
                    mode='markers',
                    name='High Concern',
                    marker=dict(size=14, color=U_RED,
                                line=dict(color='black', width=1.5))
                ))

            # Athlete average line
            fig.add_hline(
                y=jump_a['rsi_mean'].iloc[0],
                line_dash='dash',
                line_color=U_GRAY,
                annotation_text="Season average"
            )

            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="RSI",
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)

        # ---- Other DJ metric trends ----
        st.subheader("Other Drop Jump Metric Trends")
        for label, col in dj_metrics.items():
            if col not in df_a.columns:
                continue
            df_m = df_a.dropna(subset=[col])
            if df_m.empty:
                continue
            fig = px.line(
                df_m, x="Date", y=col,
                markers=True,
                color_discrete_sequence=[U_BLACK],
                title=f"{col} over time"
            )
            st.plotly_chart(fig, use_container_width=True)

        # ---- Regression section ----
        st.subheader("Top Regression Relationships: Drop Jump → Meet Performance")
        regression_results = []
        for label, col in dj_metrics.items():
            if col not in df_a.columns or event not in df_a.columns:
                continue
            df_reg = df_a.dropna(subset=[col, event])
            if len(df_reg) < 4:
                continue
            X = sm.add_constant(df_reg[col])
            y = df_reg[event]
            model = sm.OLS(y, X).fit()
            regression_results.append({
                "label": label, "col": col, "event": event,
                "coef": model.params[col], "intercept": model.params["const"],
                "r2": model.rsquared, "df_reg": df_reg
            })

        if not regression_results:
            st.write("**Insufficient data for regression (need ≥4 jump–score pairs).**")
        else:
            regression_results.sort(key=lambda x: x["r2"], reverse=True)
            for res in regression_results[:2]:
                st.write(f"### {res['col']} → {res['event']}")
                st.write(f"Coefficient: {res['coef']:.6f}")
                st.write(f"Intercept: {res['intercept']:.4f}")
                st.write(f"R²: {res['r2']:.4f}")
                fig_reg = px.scatter(
                    res["df_reg"], x=res["col"], y=res["event"],
                    trendline="ols",
                    color_discrete_sequence=[U_RED],
                    title=f"{res['col']} vs {res['event']}"
                )
                st.plotly_chart(fig_reg, use_container_width=True)
# ---------------------------------------------------------
# TEAM VIEW
# ---------------------------------------------------------
elif view_mode == "Team":
    st.title("Utah Gymnastics Dashboard — Team View")

    st.subheader("CMJ Strength/Power/RFD Trends (Normalized by Calendar Date)")

    if season_choice == "Compare all":
        df_cmj = fulljumpdf[fulljumpdf["Season"].isin(seasons)]
    else:
        df_cmj = fulljumpdf[fulljumpdf["Season"] == season_choice]

    for label, col in cmj_metrics.items():
        if col not in df_cmj.columns:
            continue

        df_tr = (
            df_cmj.dropna(subset=[col, "Season", "MonthDay_dt"])
            .groupby(["Season", "MonthDay_dt"])[col]
            .mean()
            .reset_index()
        )

        if df_tr.empty:
            continue

        if season_choice == "Compare all":
            fig_tr = px.line(
                df_tr,
                x="MonthDay_dt",
                y=col,
                color="Season",
                markers=True,
                color_discrete_sequence=[U_RED, U_BLACK, U_GRAY],
                title=f"{col} — Season Comparison (CMJ)"
            )
        else:
            fig_tr = px.line(
                df_tr,
                x="MonthDay_dt",
                y=col,
                markers=True,
                color_discrete_sequence=[U_RED],
                title=f"{col} — {season_choice} (CMJ)"
            )

        fig_tr.update_layout(xaxis_title="Calendar Date", yaxis_title=col)
        fig_tr.update_xaxes(tickformat="%b %d")
        st.plotly_chart(fig_tr, use_container_width=True)

    st.subheader("Team Scoring Trends")

    if season_choice == "Compare all":
        df_score = finaldf[finaldf["Season"].isin(seasons)]
    else:
        df_score = finaldf[finaldf["Season"] == season_choice]

    team_score_cols = ["Vault Score", "Bar Score", "Beam Score", "Floor Score"]

    for sc in team_score_cols:
        df_ss = (
            df_score.dropna(subset=[sc, "Season", "Date_norm"])
            .groupby(["Season", "Date_norm"])[sc]
            .mean()
            .reset_index()
        )


        fig_score = px.line(
            df_ss,
            x="Date_norm",
            y=sc,
            color="Season",
            markers=True,
            color_discrete_sequence=[U_RED, U_BLACK, U_GRAY],
            title=f"{sc} — Team Trend"
        )
        fig_score.update_xaxes(tickformat="%b %d")
        st.plotly_chart(fig_score, use_container_width=True)

# ---------------------------------------------------------
# FATIGUE OVERVIEW
# ---------------------------------------------------------
elif view_mode == "Fatigue Overview":
    st.title("Team Fatigue Overview")

    st.subheader("Fatigue Flags by Athlete")

    flag_counts = (
        session_df.groupby(['Athlete', 'fatigue_flag'])
        .size()
        .reset_index(name='count')
    )

    fig_flags = px.bar(
        flag_counts,
        x='Athlete',
        y='count',
        color='fatigue_flag',
        color_discrete_map={
            'Normal': 'green',
            'Watch': 'orange',
            'High Concern': 'red',
            'Unknown': 'gray'
        },
        title="Fatigue Flags per Athlete"
    )
    st.plotly_chart(fig_flags, use_container_width=True)

    st.subheader("Monthly Fatigue Rates")

    session_df['month'] = session_df['test_day'].dt.month
    monthly = (
        session_df.groupby(['month', 'fatigue_flag'])
        .size()
        .reset_index(name='count')
    )

    fig_month = px.bar(
        monthly,
        x='month',
        y='count',
        color='fatigue_flag',
        color_discrete_map={
            'Normal': 'green',
            'Watch': 'orange',
            'High Concern': 'red',
            'Unknown': 'gray'
        },
        title="Monthly Fatigue Flag Distribution"
    )
    st.plotly_chart(fig_month, use_container_width=True)

    st.subheader("RSI Trend Summary")

    fig_trend = px.bar(
        trend_df,
        x='Athlete',
        y='slope',
        color='direction',
        color_discrete_map={'Improving': 'green', 'Declining': 'red'},
        title="RSI Trend Direction (Slope)"
    )
    st.plotly_chart(fig_trend, use_container_width=True)
