import streamlit as st
from garminconnect import Garmin
from datetime import date, timedelta
import pandas as pd
from datetime import datetime

import altair as alt

import requests
import os

from dotenv import load_dotenv
load_dotenv()

import os
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
print(HF_API_TOKEN)


def call_huggingface(prompt):
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.4
        }
    }

    response = requests.post(
        "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.3",
        headers=headers,
        json=payload
    )

    data = response.json()

    # HF returns a list with generated_text
    if isinstance(data, list) and "generated_text" in data[0]:
        return data[0]["generated_text"]

    # Some models return a dict
    if "generated_text" in data:
        return data["generated_text"]

    return str(data)


def load_training_plan():
    df = pd.read_csv("trainingplan.csv")
    # Normalize date column to datetime
    df["date"] = pd.to_datetime(df["date"])
    return df

def get_todays_workout(plan_df):
    today = datetime.today().date()
    row = plan_df[plan_df["date"] == pd.to_datetime(today)]

    if row.empty:
        return None

    return row.iloc[0].to_dict()

def get_garmin_client():
    email = st.secrets.get("garmin_email")
    password = st.secrets.get("garmin_password")

    if not email or not password:
        st.error("Garmin credentials not found in secrets.toml.")
        st.stop()

    client = Garmin(email, password)
    client.login()
    return client

def get_latest_run_summary(client):
    activities = client.get_activities(0, 1)  # most recent
    if not activities:
        return None

    a = activities[0]
    # st.write(a) # Debug for format

    return {
        "name": a.get("activityName"),
        "type": a.get("activityType", {}).get("typeKey"),
        "distance_miles": a.get("distance") / 1609.34 if a.get("distance") else None,
        "duration_min": a.get("duration") / 60 if a.get("duration") else None,
        "avg_hr": a.get("averageHR"),
        "max_hr": a.get("maxHR"),
        "avg_cadence": a.get("averageRunningCadenceInStepsPerMinute"),
        "max_cadence": a.get("maxRunningCadenceInStepsPerMinute"),
        # Convert Garmin's meters/second to minutes/mile
        "avg_pace_formatted": format_pace_mmss((26.8224 / a.get("averageSpeed")) if a.get("averageSpeed") else None),
        "start_time": a.get("startTimeLocal"),
    }
def get_run_history(client, days=90):
    """
    Fetches past Garmin running activities and returns a dataframe
    with the columns needed for weekly trend analysis.
    """
    end = date.today()
    start = end - timedelta(days=days)

    activities = client.get_activities_by_date(start.isoformat(), end.isoformat())
    if not activities:
        return pd.DataFrame()

    rows = []
    for a in activities:
        if a.get("activityType", {}).get("typeKey") != "running":
            continue

        distance_mi = (a.get("distance") or 0) / 1609.34
        duration_min = (a.get("duration") or 0) / 60
        pace = (duration_min / distance_mi) if distance_mi > 0 else None

        rows.append({
            "date": pd.to_datetime(a.get("startTimeLocal")).date(),
            "distance_mi": distance_mi,
            "duration_min": duration_min,
            "pace": pace,
            "avg_hr": a.get("averageHR"),
            "avg_cadence": a.get("averageRunningCadenceInStepsPerMinute"),
            "workout_type": "Easy Run"  # fallback if plan doesn't specify
        })

    return pd.DataFrame(rows)

def get_all_runs_with_splits(client, days=90):
    """
    Fetches all running activities for the last X days,
    including splits/laps for each activity.
    """
    end = date.today()
    start = end - timedelta(days=days)

    activities = client.get_activities_by_date(start.isoformat(), end.isoformat())
    if not activities:
        return []

    all_runs = []

    for a in activities:
        if a.get("activityType", {}).get("typeKey") != "running":
            continue

        activity_id = a.get("activityId")
        splits_raw = get_laps(client, activity_id)

        # Normalize Garmin lap formats
        lap_list = []
        if isinstance(splits_raw, dict):
            lap_list = splits_raw.get("lapDTOs", [])
        elif isinstance(splits_raw, list):
            lap_list = splits_raw

        # Convert laps to structured rows
        laps = []
        for lap in lap_list:
            distance_mi = (lap.get("distance", 0) / 1609.34)
            duration_min = (lap.get("duration", 0) / 60)
            pace = (duration_min / distance_mi) if distance_mi > 0 else None

            laps.append({
                "lap": lap.get("lapNumber") or lap.get("lapIndex"),
                "distance_mi": round(distance_mi, 2),
                "duration_min": round(duration_min, 2),
                "pace_min_per_mi": pace,
                "avg_cadence": lap.get("averageRunCadence"),
                "elev_gain": lap.get("elevationGain"),
                "elev_loss": lap.get("elevationLoss"),
            })

        # Build run summary
        distance_mi = (a.get("distance") or 0) / 1609.34
        duration_min = (a.get("duration") or 0) / 60
        pace = (duration_min / distance_mi) if distance_mi > 0 else None

        run = {
            "activity_id": activity_id,
            "name": a.get("activityName"),
            "date": str(a.get("startTimeLocal")),
            "distance_mi": distance_mi,
            "duration_min": duration_min,
            "pace_min_per_mi": pace,
            "avg_hr": a.get("averageHR"),
            "max_hr": a.get("maxHR"),
            "avg_cadence": a.get("averageRunningCadenceInStepsPerMinute"),
            "splits": laps
        }

        all_runs.append(run)

    return all_runs
def compute_global_training_metrics(all_runs):
    """
    Computes season-wide metrics the conversational coach can use.
    """
    if not all_runs:
        return {}

    total_miles = sum(r["distance_mi"] for r in all_runs)
    avg_pace = sum(r["pace_min_per_mi"] for r in all_runs if r["pace_min_per_mi"]) / len(all_runs)
    avg_hr = sum(r["avg_hr"] for r in all_runs if r["avg_hr"]) / len(all_runs)

    # Aerobic trend: compare first 20% vs last 20%
    n = len(all_runs)
    early = all_runs[:max(1, n // 5)]
    late = all_runs[-max(1, n // 5):]

    def avg_pace_hr(runs):
        paces = [r["pace_min_per_mi"] for r in runs if r["pace_min_per_mi"]]
        hrs = [r["avg_hr"] for r in runs if r["avg_hr"]]
        if not paces or not hrs:
            return None
        return (sum(paces) / len(paces), sum(hrs) / len(hrs))

    early_stats = avg_pace_hr(early)
    late_stats = avg_pace_hr(late)

    return {
        "total_miles": round(total_miles, 1),
        "avg_pace": round(avg_pace, 2),
        "avg_hr": round(avg_hr, 1),
        "early_period_stats": early_stats,
        "late_period_stats": late_stats,
        "num_runs": n
    }

def compute_weekly_trends(run_history_df, plan_df):
    # run_history_df: your logged Garmin runs
    # plan_df: your trainingplan.csv

    # Normalize dates
    run_history_df["date"] = pd.to_datetime(run_history_df["date"])
    plan_df["date"] = pd.to_datetime(plan_df["date"])

    # Add week number
    run_history_df["week"] = run_history_df["date"].dt.isocalendar().week
    plan_df["week"] = plan_df["date"].dt.isocalendar().week

    weekly = {}

    for week in sorted(run_history_df["week"].unique()):
        actual_week = run_history_df[run_history_df["week"] == week]
        planned_week = plan_df[plan_df["week"] == week]

        weekly[week] = {
            "planned_mileage": planned_week["distance_mi"].sum(),
            "actual_mileage": actual_week["distance_mi"].sum(),
            "mileage_diff": actual_week["distance_mi"].sum() - planned_week["distance_mi"].sum(),
            "avg_easy_pace": actual_week[actual_week["workout_type"] == "Easy Run"]["pace"].mean() if not actual_week.empty else None,
            "avg_tempo_pace": actual_week[actual_week["workout_type"] == "Tempo Run"]["pace"].mean() if not actual_week.empty else None,
            "avg_long_pace": actual_week[actual_week["workout_type"] == "Long Run"]["pace"].mean() if not actual_week.empty else None,
            "avg_hr": actual_week["avg_hr"].mean() if "avg_hr" in actual_week else None,
            "avg_cadence": actual_week["avg_cadence"].mean() if "avg_cadence" in actual_week else None,
        }

    return weekly

def get_current_week_summary(weekly_trends):
    import datetime
    current_week = datetime.date.today().isocalendar().week
    return weekly_trends.get(current_week, None)


def get_season_summary(run_history_df, plan_df):
    # Calculate total mileage for the whole 16-week plan vs actual
    total_planned = plan_df["distance_mi"].sum()
    total_actual = run_history_df["distance_mi"].sum()

    # Calculate consistency (days ran / days planned)
    days_planned = len(plan_df[plan_df["distance_mi"] > 0])
    days_actual = len(run_history_df)
    consistency_pct = (days_actual / days_planned) * 100

    # Find the "Trend": Is pace getting faster at the same HR?
    # This is a key indicator of aerobic fitness growth.
    # (Logic: Compare average pace/HR ratio of first 2 weeks vs last 2 weeks)

    return {
        "total_completion_pct": round((total_actual / total_planned) * 100, 1),
        "consistency_score": round(consistency_pct, 1),
        "total_miles_ran": round(total_actual, 1),
        "remaining_miles_in_plan": round(total_planned - total_actual, 1)
    }

def get_training_week_number(plan_df, target_date):
    """Returns the training week index for any given date."""
    plan_df["date"] = pd.to_datetime(plan_df["date"])
    raw_start = plan_df["date"].min()
    plan_start = raw_start - pd.Timedelta(days=raw_start.weekday())
    target_date = pd.to_datetime(target_date)
    days_into_plan = (target_date - plan_start).days
    return (days_into_plan // 7) + 1


def get_week_plan(plan_df, week_number):
    """Returns all workouts for a specific training week."""
    plan_df["date"] = pd.to_datetime(plan_df["date"])
    raw_start = plan_df["date"].min()
    plan_start = raw_start - pd.Timedelta(days=raw_start.weekday())
    plan_df["training_week"] = ((plan_df["date"] - plan_start).dt.days // 7) + 1
    return plan_df[plan_df["training_week"] == week_number].sort_values("date")


def format_pace(pace_min_per_mile):
    if pace_min_per_mile is None:
        return "N/A"
    minutes = int(pace_min_per_mile)
    seconds = int((pace_min_per_mile - minutes) * 60)
    return f"{minutes}:{seconds:02d} min/mi"
def format_pace_mmss(pace_min_per_mile):
    if pace_min_per_mile is None or pace_min_per_mile == "N/A":
        return "N/A"
    minutes = int(pace_min_per_mile)
    seconds = int(round((pace_min_per_mile - minutes) * 60))
    if seconds == 60:
        minutes += 1
        seconds = 0
    return f"{minutes}:{seconds:02d} min/mi"

def get_split_paces(activity):
    splits = activity.get("splitSummaries", [])
    if not splits:
        return None

    split_data = []
    for s in splits:
        distance_miles = s.get("distance") / 1609.34 if s.get("distance") else None
        duration_min = s.get("duration") / 60 if s.get("duration") else None
        pace_min_per_mile = (duration_min / distance_miles) if distance_miles and duration_min else None
        split_data.append({
            "split_number": s.get("splitNumber"),
            "distance_miles": round(distance_miles, 2),
            "duration_min": round(duration_min, 2),
            "pace_min_per_mile": round(pace_min_per_mile, 2) if pace_min_per_mile else "N/A"
        })
    return split_data
def get_laps(client, activity_id):
    try:
        laps = client.get_activity_splits(activity_id)
        return laps
    except Exception as e:
        return None
def build_season_coach_prompt(ai_input, season_summary):
    return f"""
You are a high-level Running Coach. You are looking at Joel's training for a Half Marathon on 2026-09-06.

HISTORICAL CONTEXT:
- Total Plan Progress: {season_summary['total_completion_pct']}% of mileage completed.
- Consistency: Joel has hit {season_summary['consistency_score']}% of his planned sessions.
- Weekly Mileage Trend: {ai_input['weekly_trends']}

TODAY'S DATA:
- Activity: {ai_input['run_summary']}
- Plan: {ai_input['todays_plan']}

COACHING OBJECTIVE:
1. Briefly comment on today's run.
2. Look at the last 3-4 weeks. Is he over-training (running too fast/too much) or under-training?
3. Mention the Half Marathon goal (Sept 6). Based on his current pace trends, is he on track for his 9:15-9:40 target?
4. Give one "Macro" tip (e.g., 'Focus on sleep this week' or 'You've earned a step-back week').

Write this as a supportive, data-driven coach.
"""

def get_comparison_dataframe(run_history_df, plan_df):
    # Ensure dates are datetime objects
    run_history_df['date'] = pd.to_datetime(run_history_df['date'])
    plan_df['date'] = pd.to_datetime(plan_df['date'])

    # Aggregate Garmin runs by date (in case you ran twice in one day)
    actual_daily = run_history_df.groupby('date')['distance_mi'].sum().reset_index()
    actual_daily.columns = ['date', 'actual_dist']

    # Merge with the plan
    comparison_df = pd.merge(plan_df, actual_daily, on='date', how='left')
    comparison_df['actual_dist'] = comparison_df['actual_dist'].fillna(0)

    # Calculate Cumulative Sums
    comparison_df['cumulative_planned'] = comparison_df['distance_mi'].cumsum()
    comparison_df['cumulative_actual'] = comparison_df['actual_dist'].cumsum()

    return comparison_df

def compare_run_to_plan(latest_run, todays_plan, laps_df=None):
    if latest_run is None or todays_plan is None:
        return {"status": "incomplete", "message": "Missing run data or training plan for today."}

    result = {}

    # Planned
    planned_dist = float(todays_plan.get("distance_mi", 0) or 0)
    planned_pace_min = todays_plan.get("pace_min")
    planned_pace_max = todays_plan.get("pace_max")
    workout_type = todays_plan.get("workout_type")

    # Actual
    actual_dist = latest_run.get("distance_miles") or 0
    actual_pace_str = latest_run.get("avg_pace_formatted")  # "mm:ss min/mi"

    # Convert pace string "mm:ss min/mi" -> minutes as float
    def pace_str_to_float(p):
        if not p or p == "N/A":
            return None
        mm_ss = p.split()[0]
        m, s = mm_ss.split(":")
        return int(m) + int(s) / 60.0

    actual_pace = pace_str_to_float(actual_pace_str)

    def mmss_to_float(mmss):
        if not isinstance(mmss, str) or ":" not in mmss:
            return None
        m, s = mmss.split(":")
        return int(m) + int(s) / 60.0

    planned_pace_min_f = mmss_to_float(planned_pace_min)
    planned_pace_max_f = mmss_to_float(planned_pace_max)

    result["workout_type"] = workout_type
    result["planned_distance"] = planned_dist
    result["actual_distance"] = actual_dist
    result["planned_pace_min"] = planned_pace_min_f
    result["planned_pace_max"] = planned_pace_max_f
    result["actual_pace"] = actual_pace

    # Distance deviation
    result["distance_diff"] = actual_dist - planned_dist

    # Pace classification
    if actual_pace and planned_pace_min_f and planned_pace_max_f:
        if actual_pace < planned_pace_min_f - 0.05:
            result["pace_vs_plan"] = "faster_than_planned"
        elif actual_pace > planned_pace_max_f + 0.05:
            result["pace_vs_plan"] = "slower_than_planned"
        else:
            result["pace_vs_plan"] = "within_plan"
    else:
        result["pace_vs_plan"] = "unknown"

    # Optional: lap consistency
    if laps_df is not None and "Pace (min/mi)" in laps_df.columns:
        result["lap_pace_std"] = laps_df["Pace (min/mi)"].std()
    else:
        result["lap_pace_std"] = None

    return result


def build_ai_input(latest_run, todays_plan, comparison, laps_df, weekly_trends, current_week_summary, all_runs, global_metrics, current_week_plan, next_week_plan):
    latest_run_date = None
    if latest_run and latest_run.get("start_time"):
        latest_run_date = str(latest_run["start_time"])

    today_date = str(datetime.today().date())

    latest_run_is_today = (
        latest_run_date is not None and today_date in latest_run_date
    )

    return {
        "today_date": today_date,
        "latest_run_date": latest_run_date,
        "latest_run_is_today": latest_run_is_today,

        "run_summary": latest_run,
        "todays_plan": todays_plan,
        "comparison": comparison,
        "laps": laps_df.to_dict("records") if laps_df is not None else None,

        # FULL TRAINING HISTORY + SPLITS
        "all_runs": all_runs,

        # GLOBAL METRICS
        "global_metrics": global_metrics,

        # WEEKLY TRENDS
        "weekly_trends": weekly_trends,
        "current_week_summary": current_week_summary,

        # EMPHASIZED PLANS
        "current_week_plan": current_week_plan,
        "next_week_plan": next_week_plan,

        "meta": {
            "race_goal": {
                "race_name": "Half Marathon",
                "race_date": "2026-09-06",
                "target_pace_range": "9:15–9:40"
            }
        }
    }


def build_coach_prompt(ai_input):
    return f"""
You are an experienced running coach helping a 44-year-old male runner train for a half marathon on 2026-09-06.

===========================
DATE CHECK (CRITICAL)
===========================
- Today is: {ai_input["today_date"]}
- Latest run occurred on: {ai_input["latest_run_date"]}
- Did the latest run occur today? {ai_input["latest_run_is_today"]}

Before giving feedback:
1. Determine whether the latest run happened today or earlier.
2. If the latest run was NOT today, do NOT treat it as today's workout.
3. Base all analysis on the correct temporal relationship.

===========================
FULL TRAINING HISTORY (last 90 days, including splits)
===========================
{ai_input["all_runs"]}

===========================
GLOBAL TRAINING METRICS
===========================
{ai_input["global_metrics"]}

===========================
HIGH-PRIORITY PLANS (emphasize these)
===========================
CURRENT WEEK PLAN:
{ai_input["current_week_plan"]}

NEXT WEEK PLAN:
{ai_input["next_week_plan"]}

===========================
SECONDARY CONTEXT
===========================
TODAY'S PLAN:
{ai_input["todays_plan"]}

LATEST RUN SUMMARY:
{ai_input["run_summary"]}

PLAN VS ACTUAL COMPARISON:
{ai_input["comparison"]}

LAPS (if available):
{ai_input["laps"]}

WEEKLY TRENDS:
{ai_input["weekly_trends"]}

CURRENT WEEK SUMMARY:
{ai_input["current_week_summary"]}

RACE GOAL:
{ai_input["meta"]["race_goal"]}

===========================
YOUR TASK
===========================
- Evaluate how well the latest run matched the plan *only if it occurred today*.
- If the latest run was on a previous day, evaluate it as a past workout.
- Consider splits, trends, global metrics, and plan alignment.
- Emphasize the CURRENT WEEK PLAN and NEXT WEEK PLAN.
- Provide 2–3 specific, practical suggestions for upcoming runs.
- Keep the tone encouraging but honest.

Respond with a single, coherent paragraph or two.
"""


# --- Hugging Face-based AI Coach Feedback ---

def get_llm_coach_feedback(ai_input):
    prompt = build_coach_prompt(ai_input)

    HF_API_TOKEN = st.secrets.get("HF_API_TOKEN") or os.getenv("HF_API_TOKEN")
    if not HF_API_TOKEN:
        return "Error: HF_API_TOKEN not found."

    # REMOVED /hf-inference/ from the URL.
    # This allows the router to pick any available provider (Together, Sambanova, etc.)
    url = "https://router.huggingface.co/v1/chat/completions"

    # We will use Llama 3.1 8B as it is the most widely supported on the router
    model_id = "meta-llama/Llama-3.1-8B-Instruct"

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are a professional running coach."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)

        # Handle the common 403 error for gated models
        if response.status_code == 403:
            # Fallback to an open model that doesn't require a license
            payload["model"] = "Qwen/Qwen2.5-72B-Instruct"
            response = requests.post(url, headers=headers, json=payload, timeout=30)

        if response.status_code != 200:
            return f"Router Error {response.status_code}: {response.text}"

        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return f"Request failed: {str(e)}"

############### Start of the Streamlit code ################

st.set_page_config(page_title="Garmin AI Coach")

st.title("Garmin AI Coach")
st.write("If you can see this, your Streamlit app is working.")

###############################################
# 📊 TRAINING PROGRESS DASHBOARD (Cumulative + Weekly)
###############################################

st.header("📊 Training Progress Dashboard")

if st.button("Show Training Progress"):
    client = st.session_state["garmin_client"]

    # Load data
    plan_df = load_training_plan()
    run_history_df = get_run_history(client, days=120)

    if run_history_df.empty:
        st.warning("No running history available.")
        st.stop()

    # --- Cumulative Comparison ---
    comparison_df = get_comparison_dataframe(run_history_df, plan_df)

    st.subheader("📈 Cumulative Mileage: Planned vs Actual")

    chart_df = comparison_df[["date", "cumulative_planned", "cumulative_actual"]]
    chart_df = chart_df.set_index("date")

    st.line_chart(chart_df)

    # --- Weekly Mileage ---
    st.subheader("📅 Weekly Mileage Comparison")

    weekly_trends = compute_weekly_trends(run_history_df, plan_df)

    weekly_rows = []
    for week, data in weekly_trends.items():
        weekly_rows.append({
            "Week": week,
            "Planned Miles": data["planned_mileage"],
            "Actual Miles": data["actual_mileage"]
        })

    weekly_df = pd.DataFrame(weekly_rows)

    # Melt into long format for Altair
    weekly_long = weekly_df.melt(
        id_vars="Week",
        value_vars=["Planned Miles", "Actual Miles"],
        var_name="Type",
        value_name="Miles"
    )

    chart = (
        alt.Chart(weekly_long)
        .mark_bar()
        .encode(
            x=alt.X("Week:O", title="Training Week"),
            y=alt.Y("Miles:Q", title="Mileage"),
            color=alt.Color("Type:N", title=""),
            column=alt.Column("Type:N", title=None)  # side-by-side bars
        )
        .properties(height=300)
    )

    st.altair_chart(chart, use_container_width=True)
    # --- Summary Stats ---
    st.subheader("📌 Season Summary")

    season_summary = get_season_summary(run_history_df, plan_df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Miles Ran", season_summary["total_miles_ran"])
    col2.metric("Plan Completion", f"{season_summary['total_completion_pct']}%")
    col3.metric("Consistency", f"{season_summary['consistency_score']}%")
    col4.metric("Miles Remaining", season_summary["remaining_miles_in_plan"])


st.header("Recent Garmin activities")

if "garmin_client" not in st.session_state:
    st.session_state["garmin_client"] = Garmin(st.secrets["garmin_email"], st.secrets["garmin_password"])
    st.session_state["garmin_client"].login()


if st.button("Load my recent activities"):
    client = st.session_state["garmin_client"]

    today = date.today()
    week_ago = today - timedelta(days=7)

    activities = client.get_activities_by_date(week_ago.isoformat(), today.isoformat())
    df = pd.DataFrame(activities)

    st.dataframe(df)
st.header("Latest run")

if st.button("Show latest run"):
    client = st.session_state["garmin_client"]

    # Get latest activity summary
    activities = client.get_activities(0, 1)
    if not activities:
        st.warning("No activities found.")
        st.stop()

    activity = activities[0]
    activity_id = activity.get("activityId")

    latest = get_latest_run_summary(client)
    st.subheader("Summary")
    # st.json(latest) # Debug print to confirm structure

    # Fetch laps (full activity details)
    laps = get_laps(client, activity_id)
    # st.write(laps)  # Debug print to confirm structure

    # Garmin returns a dict with "lapDTOs" inside
    lap_list = []
    if isinstance(laps, dict):
        lap_list = laps.get("lapDTOs", [])
    elif isinstance(laps, list):
        # Some versions return the list directly
        lap_list = laps

    if lap_list:
        lap_rows = []
        for lap in lap_list:
            distance_mi = lap.get("distance", 0) / 1609.34
            duration_min = lap.get("duration", 0) / 60
            pace = (duration_min / distance_mi) if distance_mi > 0 else None

            lap_rows.append({
                "Lap": lap.get("lapNumber") or lap.get("lapIndex"),
                "Distance (mi)": round(distance_mi, 2),
                "Duration (min)": round(duration_min, 2),
                "Pace (min/mi)": format_pace_mmss(pace),
                "Avg Cadence": lap.get("averageRunCadence"),
                "Elevation Gain": lap.get("elevationGain"),
                "Elevation Loss": lap.get("elevationLoss"),
            })

        st.subheader("Laps")
        st.dataframe(pd.DataFrame(lap_rows))
    else:
        st.info("No lap data available.")

    plan_df = load_training_plan()
    todays_plan = get_todays_workout(plan_df)

    st.subheader("Today's Planned Workout")
    if todays_plan:
        st.json(todays_plan)
    else:
        st.info("No planned workout for today.")

###############################################
# 📅 Weekly Training Plan Viewer (Plan-relative weeks, fixed)
###############################################

st.header("📅 This Week's Training Plan")

if st.button("Show This Week's Plan"):
    plan_df = load_training_plan()

    # Ensure date column is datetime64
    plan_df["date"] = pd.to_datetime(plan_df["date"])

    # Determine the earliest date in the plan
    raw_start = plan_df["date"].min()

    # Align the plan start to the Monday of that week
    plan_start = raw_start - pd.Timedelta(days=raw_start.weekday())

    # Today's date
    today = pd.to_datetime(datetime.today().date())

    # Compute training week number (Week 1 = first aligned week)
    days_into_plan = (today - plan_start).days
    training_week = (days_into_plan // 7) + 1

    # Add training week index to the dataframe
    plan_df["training_week"] = ((plan_df["date"] - plan_start).dt.days // 7) + 1

    # Filter to the current training week
    week_plan = plan_df[plan_df["training_week"] == training_week]

    if week_plan.empty:
        st.info(f"No training plan found for Training Week {training_week}.")
    else:
        week_plan = week_plan.sort_values("date")

        display_rows = []
        for _, row in week_plan.iterrows():
            display_rows.append({
                "Day": row["day"],
                "Date": row["date"].strftime("%a %b %d"),
                "Workout": row["workout_type"],
                "Miles": row["distance_mi"],
                "Pace Range": f"{row['pace_min']}–{row['pace_max']}",
                "Notes": row.get("notes", "")
            })

        st.subheader(f"Training Plan — Week {training_week}")
        st.table(pd.DataFrame(display_rows))

st.header("AI Coach Feedback (LLM)")

if st.button("Get AI LLM feedback"):
    client = st.session_state["garmin_client"]

    # Latest run
    latest = get_latest_run_summary(client)

    # Get latest activity + laps
    activities = client.get_activities(0, 1)
    activity = activities[0] if activities else None
    laps_df = None
    if activity:
        activity_id = activity.get("activityId")
        laps = get_laps(client, activity_id)

        if laps and isinstance(laps, dict):
            lap_list = laps.get("lapDTOs", [])
        elif laps and isinstance(laps, list):
            lap_list = laps
        else:
            lap_list = []

        if lap_list:
            lap_rows = []
            for lap in lap_list:
                distance_mi = lap.get("distance", 0) / 1609.34
                duration_min = lap.get("duration", 0) / 60
                pace = (duration_min / distance_mi) if distance_mi > 0 else None
                lap_rows.append({
                    "Lap": lap.get("lapIndex"),
                    "Distance (mi)": distance_mi,
                    "Duration (min)": duration_min,
                    "Pace (min/mi)": pace,
                })
            laps_df = pd.DataFrame(lap_rows)

    # Load training plan
    plan_df = load_training_plan()
    todays_plan = get_todays_workout(plan_df)

    # --- NEW: compute current + next week plans ---
    today = datetime.today().date()
    current_training_week = get_training_week_number(plan_df, today)
    next_training_week = current_training_week + 1

    current_week_plan = get_week_plan(plan_df, current_training_week).to_dict("records")

    # Comparison
    next_week_plan = get_week_plan(plan_df, next_training_week).to_dict("records")
    comparison = compare_run_to_plan(latest, todays_plan, laps_df)

    # Weekly trends
    run_history_df = get_run_history(client)
    weekly_trends = compute_weekly_trends(run_history_df, plan_df)
    current_week_summary = get_current_week_summary(weekly_trends)

    # Full run history + splits
    all_runs = get_all_runs_with_splits(client, days=90)

    # Global metrics
    global_metrics = compute_global_training_metrics(all_runs)

    # Build AI input
    ai_input = build_ai_input(
        latest,
        todays_plan,
        comparison,
        laps_df,
        weekly_trends,
        current_week_summary,
        all_runs,
        global_metrics,
        current_week_plan,
        next_week_plan
    )

    # Get feedback
    feedback = get_llm_coach_feedback(ai_input)

    st.subheader("AI Coach Feedback")
    st.write(feedback)


###############################################
# 🔥 INTERACTIVE AI COACH CHAT SECTION
###############################################

st.header("💬 AI Coach Chat")

# Initialize chat history
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

# --- Build Context for the AI ---
def build_chat_context():
    client = st.session_state["garmin_client"]

    # Full run history + splits (last 90 days)
    all_runs = get_all_runs_with_splits(client, days=90)

    # Global metrics across the 90-day block
    global_metrics = compute_global_training_metrics(all_runs)

    # Full training plan
    plan_df = load_training_plan()
    todays_plan = get_todays_workout(plan_df)

    # Weekly trends from Garmin history
    run_history_df = get_run_history(client, days=90)
    weekly_trends = compute_weekly_trends(run_history_df, plan_df)
    current_week_summary = get_current_week_summary(weekly_trends)

    # Latest run summary
    latest = get_latest_run_summary(client)

    # Date logic
    today_date = str(datetime.today().date())
    latest_run_date = None
    if latest and latest.get("start_time"):
        latest_run_date = str(latest["start_time"])

    latest_run_is_today = (
        latest_run_date is not None and today_date in latest_run_date
    )

    # Determine current and next training week
    today = datetime.today().date()
    current_training_week = get_training_week_number(plan_df, today)
    next_training_week = current_training_week + 1

    # Extract week plans
    current_week_plan = get_week_plan(plan_df, current_training_week).to_dict("records")
    next_week_plan = get_week_plan(plan_df, next_training_week).to_dict("records")

    return {
        "today_date": today_date,
        "latest_run_date": latest_run_date,
        "latest_run_is_today": latest_run_is_today,

        # FULL TRAINING PLAN (for context)
        "training_plan": plan_df.to_dict("records"),

        # FULL 90-DAY RUN HISTORY + SPLITS
        "all_runs": all_runs,

        # GLOBAL METRICS (season-wide)
        "global_metrics": global_metrics,

        # WEEKLY TRENDS (Garmin)
        "weekly_trends": weekly_trends,
        "current_week_summary": current_week_summary,

        # TODAY + LATEST RUN
        "todays_plan": todays_plan,
        "latest_run": latest,

        # EMPHASIZED PLANS
        "current_week_plan": current_week_plan,
        "next_week_plan": next_week_plan,

        # RACE GOAL
        "race_goal": {
            "race_name": "Half Marathon",
            "race_date": "2026-09-06",
            "target_pace_range": "9:15–9:40"
        }
    }



# --- Build Prompt for Chat ---
def build_chat_prompt(messages, context):
    return f"""
You are Joel's personal running coach. Use ALL training data below to answer his question accurately.

===========================
DATE CHECK (CRITICAL)
===========================
- Today is: {context["today_date"]}
- Latest run occurred on: {context["latest_run_date"]}
- Did the latest run occur today? {context["latest_run_is_today"]}

Before answering:
1. Determine whether the latest run happened today or earlier.
2. If the latest run was NOT today, do NOT treat it as today's workout.
3. Base all analysis on the correct temporal relationship.

===========================
FULL TRAINING PLAN (for context)
===========================
{context["training_plan"]}

===========================
FULL TRAINING HISTORY (last 90 days, including splits)
===========================
{context["all_runs"]}

===========================
GLOBAL TRAINING METRICS (last 90 days)
===========================
{context["global_metrics"]}

===========================
HIGH-PRIORITY PLANS (emphasize these)
===========================
CURRENT WEEK PLAN:
{context["current_week_plan"]}

NEXT WEEK PLAN:
{context["next_week_plan"]}

===========================
SECONDARY CONTEXT
===========================
TODAY'S PLAN:
{context["todays_plan"]}

LATEST RUN:
{context["latest_run"]}

WEEKLY TRENDS:
{context["weekly_trends"]}

CURRENT WEEK SUMMARY:
{context["current_week_summary"]}

RACE GOAL:
{context["race_goal"]}

===========================
CHAT HISTORY
===========================
{messages}

===========================
YOUR TASK
===========================
- Answer Joel's latest question using ALL available training data.
- Consider splits, trends, global metrics, and plan alignment.
- Emphasize the CURRENT WEEK PLAN and NEXT WEEK PLAN when giving advice.
- If the latest run was not today, clearly state that before giving advice.
- Provide specific, practical, forward-looking coaching guidance.
- Keep the tone encouraging but honest.

Respond with a single, coherent paragraph or two.
"""


# --- LLM Call for Chat ---
def chat_llm_call(prompt):
    HF_API_TOKEN = st.secrets.get("HF_API_TOKEN") or os.getenv("HF_API_TOKEN")
    if not HF_API_TOKEN:
        return "Error: HF_API_TOKEN not found."

    url = "https://router.huggingface.co/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a professional running coach."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=payload, timeout=30)

    # If model is gated, fallback to Qwen
    if response.status_code == 403:
        payload["model"] = "Qwen/Qwen2.5-72B-Instruct"
        response = requests.post(url, headers=headers, json=payload, timeout=30)

    if response.status_code != 200:
        return f"Router Error {response.status_code}: {response.text}"

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()

# --- Chat UI ---
for msg in st.session_state["chat_messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask your AI coach anything...")

if user_input:
    # Add user message
    st.session_state["chat_messages"].append({"role": "user", "content": user_input})

    # Build context + prompt
    context = build_chat_context()
    prompt = build_chat_prompt(st.session_state["chat_messages"], context)

    # Get LLM response
    coach_reply = chat_llm_call(prompt)

    # Add assistant message
    st.session_state["chat_messages"].append({"role": "assistant", "content": coach_reply})

    # Display assistant message
    with st.chat_message("assistant"):
        st.write(coach_reply)
