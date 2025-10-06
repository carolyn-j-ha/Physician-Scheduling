# streamlit run utility_editor_improved.py
"""
Physician Utility Editor - Improved Version
- Separate Calendar and Table filters
- Improved calendar layout with date numbers visible
- Bundle handling optimized (view-only expansion)
- Undo functionality
- 6-month heatmap overview
- Personal score statistics
- Unavailability support
"""

from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ============================================
# CONFIG
# ============================================
st.set_page_config(page_title="Utility Editor", layout="wide")

# ROOT = "c:\\Users\\1004c\\OneDrive\\바탕 화면\\RA\\Code\\"
# DATA_DIR = Path(os.path.join(ROOT, "data"))
DATA_DIR = Path("data")

PERIOD_ORDER = ["Day", "Night", "24h"]
SERVICE_ORDER = ["PICU", "PCICU", "SED", "SMI"]
TOKENS = {"abc123": 101, "def456": 205, "test": 101}

BUNDLE_DESC = {
    "PICU_day_weekday": "Mon-Fri",
    "PCICU_day_weekday": "Fri-Thu",
    "PCICU_weekend_pair": "Sat-Sun"
}

SERVICE_COLORS = {
    "PICU": (15, 76, 129),
    "PCICU": (123, 31, 162),
    "SED": (27, 94, 32),
    "SMI": (230, 81, 0)
}


# ============================================
# UTILITY COLORS
# ============================================

def get_color_by_score(service: str, score: float, available: bool = True) -> str:
    """Service color + Score-based opacity"""
    if not available:
        return "rgba(189, 189, 189, 0.5)"
    
    base = SERVICE_COLORS.get(service, (100, 100, 100))
    
    # Score 0.5-2.0 → normalized 0-1
    normalized = (score - 0.5) / 1.5
    normalized = max(0, min(1, normalized))
    
    # Power function for contrast
    opacity = 0.2 + (normalized ** 1.5) * 0.8  # 0.2-1.0 range
    
    return f'rgba({base[0]}, {base[1]}, {base[2]}, {opacity})'


# ============================================
# TOY DATA - REALISTIC 1 MONTH
# ============================================

@st.cache_data
def generate_toy_data():
    """
    Realistic 1-month toy data with proper bundle structure:
    - PICU Day: Mon-Fri bundle (same week)
    - PCICU Day: Fri-Thu bundle (cross-week)
    - Individual slots only for Night/24h
    """
    physicians = pd.DataFrame([
        {"i_id": 101, "name": "Smith", "qualification": "PCICU_pred", "unavailable_dates": "2026-01-10,2026-01-20"},
        {"i_id": 205, "name": "Jones", "qualification": "PICU_only", "unavailable_dates": "2026-01-15"}
    ])
    
    utilities = []
    start = pd.Timestamp("2026-01-01")  # Thursday
    pid = 101
    
    # ============================================
    # BUNDLE CONFIGURATIONS
    # ============================================
    
    # PICU Day: Mon-Fri bundles (same week)
    picu_weeks = [
        ("2026-01-05", "PICU_day_weekday_W01", 1.25),  # Week 1
        ("2026-01-12", "PICU_day_weekday_W02", 1.15),  # Week 2
        ("2026-01-19", "PICU_day_weekday_W03", 1.35),  # Week 3
        ("2026-01-26", "PICU_day_weekday_W04", 1.05),  # Week 4
    ]
    
    for monday_str, bundle_id, score in picu_weeks:
        monday = pd.Timestamp(monday_str)
        dates = [(monday + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(5)]
        utilities.append({
            "i_id": pid,
            "slot_type": "bundle",
            "date": ",".join(dates),
            "svc": "PICU",
            "period": "Day",
            "bundle_id": bundle_id,
            "score": score
        })
    
    # PCICU Day: Fri-Thu bundles (cross-week)
    pcicu_bundles = [
        (["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"], "PCICU_day_weekday_W01", 1.18),  # Thu + Mon-Thu
        (["2026-01-09", "2026-01-12", "2026-01-13", "2026-01-14", "2026-01-15"], "PCICU_day_weekday_W02", 1.08),  # Fri + Mon-Thu
        (["2026-01-16", "2026-01-19", "2026-01-20", "2026-01-21", "2026-01-22"], "PCICU_day_weekday_W03", 1.28),  # Fri + Mon-Thu
        (["2026-01-23", "2026-01-26", "2026-01-27", "2026-01-28", "2026-01-29"], "PCICU_day_weekday_W04", 0.98),  # Fri + Mon-Thu
    ]
    
    for dates, bundle_id, score in pcicu_bundles:
        utilities.append({
            "i_id": pid,
            "slot_type": "bundle",
            "date": ",".join(dates),
            "svc": "PCICU",
            "period": "Day",
            "bundle_id": bundle_id,
            "score": score
        })
    
    # ============================================
    # INDIVIDUAL SLOTS
    # ============================================
    
    # Collect bundle dates to skip
    bundle_slots = set()
    for util in utilities:
        if util["slot_type"] == "bundle":
            for d in util["date"].split(","):
                bundle_slots.add((d, util["svc"], util["period"]))
    
    # Generate individual slots for January (31 days)
    for d in range(31):
        date = (start + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        date_obj = pd.Timestamp(date)
        weekday = date_obj.weekday()  # 0=Mon, 6=Sun
        
        # Weekday (Mon-Fri)
        if weekday < 5:
            # PICU Night (Day는 bundle만 존재)
            if (date, "PICU", "Night") not in bundle_slots:
                utilities.append({
                    "i_id": pid,
                    "slot_type": "individual",
                    "date": date,
                    "svc": "PICU",
                    "period": "Night",
                    "bundle_id": None,
                    "score": round(np.random.uniform(0.7, 1.5), 3)
                })
            
            # PCICU Night (Day는 bundle만 존재)
            if (date, "PCICU", "Night") not in bundle_slots:
                utilities.append({
                    "i_id": pid,
                    "slot_type": "individual",
                    "date": date,
                    "svc": "PCICU",
                    "period": "Night",
                    "bundle_id": None,
                    "score": round(np.random.uniform(0.6, 1.4), 3)
                })
            
            # SED: Mon-Fri Day only
            utilities.append({
                "i_id": pid,
                "slot_type": "individual",
                "date": date,
                "svc": "SED",
                "period": "Day",
                "bundle_id": None,
                "score": round(np.random.uniform(0.8, 1.6), 3)
            })
            
            # SMI: Tue Day only
            if weekday == 1:  # Tuesday
                utilities.append({
                    "i_id": pid,
                    "slot_type": "individual",
                    "date": date,
                    "svc": "SMI",
                    "period": "Day",
                    "bundle_id": None,
                    "score": round(np.random.uniform(0.9, 1.7), 3)
                })
        
        # Weekend (Sat-Sun): 24h only
        else:
            # PICU 24h
            utilities.append({
                "i_id": pid,
                "slot_type": "individual",
                "date": date,
                "svc": "PICU",
                "period": "24h",
                "bundle_id": None,
                "score": round(np.random.uniform(0.7, 1.5), 3)
            })
            
            # PCICU 24h
            utilities.append({
                "i_id": pid,
                "slot_type": "individual",
                "date": date,
                "svc": "PCICU",
                "period": "24h",
                "bundle_id": None,
                "score": round(np.random.uniform(0.6, 1.4), 3)
            })
    
    return physicians, pd.DataFrame(utilities)


# ============================================
# DATA LOADING
# ============================================

@st.cache_data
def load_data():
    phys_path = DATA_DIR / "physicians.csv"
    util_path = DATA_DIR / "utilities.csv"
    
    if phys_path.exists() and util_path.exists():
        return pd.read_csv(phys_path), pd.read_csv(util_path)
    
    st.warning("Real data not found. Using toy data.")
    return generate_toy_data()

def get_qual_services(qual: str):
    return {
        "PICU_only": ["PICU", "SED", "SMI"],
        "PCICU_only": ["PCICU", "SED", "SMI"],
        "PCICU_pred": ["PICU", "PCICU", "SED", "SMI"]
    }.get(qual, [])

# ============================================
# DATA PREPARATION (IMPROVED)
# ============================================

def create_calendar_view(util_data: pd.DataFrame, unavailable_dates: list) -> pd.DataFrame:
    """
    View-only expansion for calendar rendering
    Bundle은 원본 유지, calendar 표시용으로만 확장
    """
    rows = []
    for idx, row in util_data.iterrows():
        if row["slot_type"] == "bundle":
            dates = row["date"].split(",")
            for date in dates:
                rows.append({
                    "original_idx": idx,
                    "date": date,
                    "svc": row["svc"],
                    "period": row["period"],
                    "score": row["score"],
                    "is_bundle": True,
                    "available": date not in unavailable_dates
                })
        else:
            rows.append({
                "original_idx": idx,
                "date": row["date"],
                "svc": row["svc"],
                "period": row["period"],
                "score": row["score"],
                "is_bundle": False,
                "available": row["date"] not in unavailable_dates
            })
    
    df = pd.DataFrame(rows)
    df["date_parsed"] = pd.to_datetime(df["date"])
    return df

def prepare_edit_data(util_data: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for editing table"""
    df = util_data.copy()
    df["Type"] = df["slot_type"].apply(lambda x: "📦 Bundle" if x == "bundle" else "📄 Individual")
    df["Date"] = df.apply(lambda r: 
        f"{r['date'].split(',')[0]} → {r['date'].split(',')[-1]}" if r["slot_type"] == "bundle" 
        else r["date"], axis=1)
    df["Bundle"] = df.apply(lambda r: 
        BUNDLE_DESC.get(r["bundle_id"].rsplit("_", 1)[0], "") if r["bundle_id"] else "", axis=1)
    return df


# ============================================
# VISUALIZATION - HEATMAP OVERVIEW
# ============================================

def create_heatmap_overview(cal_data: pd.DataFrame, services: list):
    """6-month overview heatmap"""
    # Y-axis categories: Service-Period combinations
    y_categories = []
    for svc in services:
        for period in PERIOD_ORDER:
            y_categories.append(f"{svc}-{period}")
    
    dates = sorted(cal_data["date_parsed"].unique())
    
    fig = go.Figure()
    
    for y_idx, cat in enumerate(y_categories):
        svc, period = cat.split("-")
        
        for date in dates:
            mask = (cal_data["date_parsed"] == date) & \
                   (cal_data["svc"] == svc) & \
                   (cal_data["period"] == period)
            subset = cal_data[mask]
            
            if len(subset) > 0:
                row = subset.iloc[0]
                score = row["score"]
                available = row["available"]
                color = get_color_by_score(svc, score, available)
                
                fig.add_trace(go.Scatter(
                    x=[date], y=[y_idx],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=color,
                        symbol='square',
                        line=dict(width=1, color='white')
                    ),
                    hovertext=f"{date.strftime('%Y-%m-%d')}<br>{cat}<br>Score: {score:.2f}<br>{'Bundle' if row['is_bundle'] else 'Individual'}",
                    hoverinfo='text',
                    showlegend=False
                ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=100, r=20, t=40, b=40),
        yaxis=dict(
            tickvals=list(range(len(y_categories))), 
            ticktext=y_categories,
            tickfont=dict(size=10)
        ),
        xaxis=dict(title="Date", tickangle=-45),
        title="6-Month Overview Heatmap",
        plot_bgcolor='white',
        hovermode='closest'
    )
    
    return fig


# ============================================
# VISUALIZATION - CALENDAR (IMPROVED)
# ============================================

def create_calendar(cal_data, date_range, svc_filter, period_filter):
    """
    Improved calendar with:
    - Day on top, Night on bottom (D-N order)
    - Unavailable dates show empty cells
    - Full viewport fit with proper scaling
    - Variable cell count per day
    """
    mask = (
        (cal_data["date_parsed"] >= date_range[0]) &
        (cal_data["date_parsed"] <= date_range[1]) &
        (cal_data["svc"].isin(svc_filter)) &
        (cal_data["period"].isin(period_filter))
    )
    sub = cal_data[mask].copy()
    
    if sub.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data for selected filters", x=0.5, y=0.5, 
                          showarrow=False, font=dict(size=16))
        fig.update_layout(height=400)
        return fig
    
    dates = sorted(sub["date_parsed"].unique())
    first_monday = dates[0] - pd.Timedelta(days=dates[0].weekday())
    
    fig = go.Figure()
    week_num = 0
    current_monday = first_monday
    
    # Calculate number of weeks for auto-sizing
    last_date = dates[-1]
    total_weeks = ((last_date - first_monday).days // 7) + 1
    
    while current_monday <= dates[-1]:
        for day_offset in range(7):
            current_date = current_monday + pd.Timedelta(days=day_offset)
            if current_date < dates[0] or current_date > dates[-1]:
                continue
            
            date_data = sub[sub["date_parsed"] == current_date]
            
            x_base, y_base = day_offset, -week_num
            
            # Date header (20% of cell)
            fig.add_shape(
                type="rect", 
                x0=x_base, y0=y_base, 
                x1=x_base+1, y1=y_base+0.20,
                fillcolor='rgba(245,245,245,1)', 
                line=dict(color='#999', width=1)
            )
            
            fig.add_annotation(
                x=x_base+0.5, y=y_base+0.10, 
                text=f"<b>{current_date.day}</b>",
                showarrow=False, 
                font=dict(size=13, color='#333'),
                xanchor='center', yanchor='middle'
            )
            
            # Check if unavailable
            if len(date_data) == 0:
                continue
                
            is_unavailable = not date_data['available'].all()
            
            if is_unavailable:
                # Empty cell for unavailable dates
                fig.add_shape(
                    type="rect",
                    x0=x_base, y0=y_base+0.20,
                    x1=x_base+1, y1=y_base+1,
                    fillcolor='rgba(250,250,250,1)',
                    line=dict(color='#ddd', width=1)
                )
                continue
            
            # Separate Day, Night, and 24h shifts
            day_data = date_data[date_data["period"] == "Day"]
            night_data = date_data[date_data["period"] == "Night"]
            h24_data = date_data[date_data["period"] == "24h"]
            
            n_cols = len(svc_filter)
            cell_w = 0.95 / n_cols
            
            # If 24h shift exists (weekend), show only 24h with "24h" label
            if len(h24_data) > 0:
                # Add "24h" label on left
                fig.add_annotation(
                    x=x_base-0.05, y=y_base+0.57,
                    text="<b>24h</b>",
                    showarrow=False,
                    font=dict(size=8, color='#666'),
                    xanchor='right', yanchor='middle'
                )
                
                for _, row in h24_data.iterrows():
                    if row["svc"] not in svc_filter:
                        continue
                    svc_idx = svc_filter.index(row["svc"])
                    
                    x = x_base + 0.025 + svc_idx * cell_w
                    y = y_base + 0.20
                    h = 0.75
                    
                    color = get_color_by_score(row["svc"], row["score"], row["available"])
                    border = dict(
                        color='gold' if row["is_bundle"] else '#ddd', 
                        width=3 if row["is_bundle"] else 1
                    )
                    
                    fig.add_shape(
                        type="rect", 
                        x0=x, y0=y, 
                        x1=x+cell_w, y1=y+h,
                        fillcolor=color, 
                        line=border
                    )
                    
                    text_color = 'white' if row['score'] > 1.2 else 'black'
                    
                    fig.add_annotation(
                        x=x+cell_w/2, y=y+h/2, 
                        text=f"{row['score']:.2f}",
                        showarrow=False, 
                        font=dict(size=9, color=text_color, family="Arial Black")
                    )
            else:
                # Weekday: Day on top (D), Night on bottom (N)
                
                # Day shifts on TOP (40% of available space)
                if len(day_data) > 0:
                    for _, row in day_data.iterrows():
                        if row["svc"] not in svc_filter:
                            continue
                        svc_idx = svc_filter.index(row["svc"])
                        
                        x = x_base + 0.025 + svc_idx * cell_w
                        y = y_base + 0.60  # TOP position for Day
                        h = 0.35
                        
                        color = get_color_by_score(row["svc"], row["score"], row["available"])
                        border = dict(
                            color='gold' if row["is_bundle"] else '#ddd', 
                            width=3 if row["is_bundle"] else 1
                        )
                        
                        fig.add_shape(
                            type="rect", 
                            x0=x, y0=y, 
                            x1=x+cell_w, y1=y+h,
                            fillcolor=color, 
                            line=border
                        )
                        
                        text_color = 'white' if row['score'] > 1.2 else 'black'
                        
                        fig.add_annotation(
                            x=x+cell_w/2, y=y+h/2, 
                            text=f"{row['score']:.2f}",
                            showarrow=False, 
                            font=dict(size=9, color=text_color, family="Arial Black")
                        )
                
                # Night shifts on BOTTOM (40% of available space)
                if len(night_data) > 0:
                    for _, row in night_data.iterrows():
                        if row["svc"] not in svc_filter:
                            continue
                        svc_idx = svc_filter.index(row["svc"])
                        
                        x = x_base + 0.025 + svc_idx * cell_w
                        y = y_base + 0.20  # BOTTOM position for Night
                        h = 0.35
                        
                        color = get_color_by_score(row["svc"], row["score"], row["available"])
                        border = dict(
                            color='gold' if row["is_bundle"] else '#ddd', 
                            width=3 if row["is_bundle"] else 1
                        )
                        
                        fig.add_shape(
                            type="rect", 
                            x0=x, y0=y, 
                            x1=x+cell_w, y1=y+h,
                            fillcolor=color, 
                            line=border
                        )
                        
                        text_color = 'white' if row['score'] > 1.2 else 'black'
                        
                        fig.add_annotation(
                            x=x+cell_w/2, y=y+h/2, 
                            text=f"{row['score']:.2f}",
                            showarrow=False, 
                            font=dict(size=9, color=text_color, family="Arial Black")
                        )
        
        week_num += 1
        current_monday += pd.Timedelta(days=7)
    
    # Add D/N labels only once on the left of Monday column
    for k in range(week_num):
        yb = -k  # 그 주의 y_base

        if "Day" in period_filter:
            fig.add_annotation(
                x=-0.15, y=yb + 0.775,
                text="<b>D</b>", showarrow=False,
                font=dict(size=10, color='#666'),
                xanchor='right', yanchor='middle'
            )

        if "Night" in period_filter:
            fig.add_annotation(
                x=-0.15, y=yb + 0.375,
                text="<b>N</b>", showarrow=False,
                font=dict(size=10, color='#666'),
                xanchor='right', yanchor='middle'
            )
    
    # Add 24h labels on the right of Sunday (once per week)
    if "24h" in period_filter:
        for k in range(week_num):
            yb = -k
            fig.add_annotation(
                x=7.05,          # Sun(6) 오른쪽 바깥
                y=yb + 0.57,     # 24h 박스 중앙 높이(플러스여야 위아래가 맞아요)
                text="<b>24h</b>",
                showarrow=False,
                font=dict(size= 9, color='#666'),
                xanchor='left', yanchor='middle'
            )
    
    # Proper scaling: 0 to 7 (exactly 7 days), extended for 24h label
    fig.update_xaxes(
        range=[-0.15, 7.3], 
        tickvals=list(range(7)),
        ticktext=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], 
        side="top", 
        showgrid=False
    )
    fig.update_yaxes(
        range=[-week_num+0.5, 0.5], 
        showticklabels=False, 
        showgrid=False
    )
    
    # Proper scaling: 0 to 7 (exactly 7 days)
    fig.update_xaxes(
        range=[-0.15, 7], 
        tickvals=list(range(7)),
        ticktext=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], 
        side="top", 
        showgrid=False
    )
    fig.update_yaxes(
        range=[-week_num+0.5, 0.5], 
        showticklabels=False, 
        showgrid=False
    )
    
    # Auto-fit height
    cal_height = max(450, min(900, total_weeks * 130))
    
    fig.update_layout(
        height=cal_height, 
        margin=dict(l=40, r=20, t=40, b=10),
        plot_bgcolor='white', 
        showlegend=False
    )
    
    return fig


# ============================================
# SESSION STATE
# ============================================

def init_session_state():
    if "initialized" not in st.session_state:
        params = dict(st.query_params) if hasattr(st, 'query_params') else {}
        pid = TOKENS.get(params.get("token", "test"), 101)
        
        physicians, utilities = load_data()
        phys = physicians[physicians["i_id"] == pid].iloc[0]
        util = utilities[utilities["i_id"] == pid].copy()
        
        # Parse unavailable dates
        unavail = phys.get("unavailable_dates", "")
        unavail_list = [d.strip() for d in str(unavail).split(",") if d.strip()] if unavail else []
        
        st.session_state.physician_id = pid
        st.session_state.physician_name = phys["name"]
        st.session_state.eligible_services = get_qual_services(phys["qualification"])
        st.session_state.unavailable_dates = unavail_list
        st.session_state.data = util
        st.session_state.original_data = util.copy()
        st.session_state.calendar_data = create_calendar_view(util, unavail_list)
        st.session_state.edit_data = prepare_edit_data(util)
        st.session_state.changes_pending = False
        st.session_state.change_log = []
        st.session_state.history_stack = []
        
        # Separate filters
        st.session_state.cal_svc_filter = st.session_state.eligible_services.copy()
        st.session_state.cal_period_filter = PERIOD_ORDER.copy()
        st.session_state.tbl_svc_filter = st.session_state.eligible_services.copy()
        st.session_state.tbl_period_filter = PERIOD_ORDER.copy()
        st.session_state.tbl_type_filter = ["individual", "bundle"]
        
        st.session_state.total_applied_changes = 0
        st.session_state.initialized = True


# ============================================
# MAIN
# ============================================

def main():
    init_session_state()
    
    st.title(f"Dr. {st.session_state.physician_name}'s Utility Editor")
    
    # ============================================
    # SIDEBAR
    # ============================================
    with st.sidebar:
        st.header("📊 Statistics")
        
        current_scores = st.session_state.data["score"]
        
        col1, col2 = st.columns(2)
        col1.metric("Min Score", f"{current_scores.min():.3f}")
        col2.metric("Max Score", f"{current_scores.max():.3f}")
        col1.metric("Mean", f"{current_scores.mean():.3f}")
        col2.metric("Median", f"{current_scores.median():.3f}")
        
        # Distribution histogram
        fig_dist = go.Figure(data=[go.Histogram(
            x=current_scores,
            nbinsx=20,
            marker_color='steelblue'
        )])
        fig_dist.update_layout(
            height=150,
            margin=dict(l=20,r=20,t=20,b=20),
            xaxis_title="Score",
            yaxis_title="Count",
            showlegend=False
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        st.divider()
        
        # Status
        st.subheader("Status")
        if st.session_state.changes_pending:
            st.warning("⚠️ Unsaved changes")
        else:
            st.success("✓ All saved")
        
        st.metric("Total Changes Applied", st.session_state.total_applied_changes)
        # st.metric("Undo Steps Available", len(st.session_state.history_stack))
        
        st.divider()
        
        # Actions
        st.subheader("Export")
        
        csv = st.session_state.data[["i_id","slot_type","date","svc","period","score","bundle_id"]].to_csv(index=False)
        st.download_button(
            "📥 Export CSV", 
            csv, 
            f"utilities_{st.session_state.physician_id}.csv",
            "text/csv", 
            use_container_width=True
        )
    
    # ============================================
    # MAIN CONTENT
    # ============================================
    
    # Tabs
    tab1, tab2 = st.tabs(["📅 Calendar View", "📈 Heatmap Overview"])
    
    with tab1:
        st.subheader("Calendar View")
        
        # Calendar filters
        with st.expander("📅 Calendar Filters", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                cal_svc = st.multiselect(
                    "Services", 
                    st.session_state.eligible_services,
                    default=st.session_state.cal_svc_filter,
                    key="cal_svc"
                )
                st.session_state.cal_svc_filter = cal_svc
            
            with col2:
                cal_period = st.multiselect(
                    "Periods", 
                    PERIOD_ORDER, 
                    default=st.session_state.cal_period_filter,
                    key="cal_period"
                )
                st.session_state.cal_period_filter = cal_period
            
            with col3:
                min_date = st.session_state.calendar_data["date_parsed"].min().date()
                max_date = st.session_state.calendar_data["date_parsed"].max().date()
                date_range = st.date_input(
                    "Date Range", 
                    value=(min_date, max_date),
                    min_value=min_date, 
                    max_value=max_date
                )
        
        if isinstance(date_range, tuple) and len(date_range) == 2:
            date_range = (pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))
        else:
            date_range = (pd.Timestamp(min_date), pd.Timestamp(max_date))
        
        # Render calendar
        if cal_svc and cal_period:
            # Add service labels ABOVE the calendar
            service_label_html = "<div style='margin-bottom: 10px; padding: 8px; background-color: #f8f9fa; border-radius: 4px;'>"
            service_label_html += "<b>Services:</b> "
            for idx, svc in enumerate(cal_svc):
                color_rgb = SERVICE_COLORS.get(svc, (0,0,0))
                color_hex = f"#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}"
                service_label_html += f"<span style='color: {color_hex}; font-weight: bold; margin-right: 15px;'>{svc}</span>"
            service_label_html += "</div>"
            st.markdown(service_label_html, unsafe_allow_html=True)

            fig = create_calendar(
                st.session_state.calendar_data, 
                date_range, 
                cal_svc, 
                cal_period
            )
            st.plotly_chart(fig, use_container_width=True, key="calendar")
        else:
            st.warning("Please select at least one service and period")
    
    with tab2:
        st.subheader("6-Month Overview Heatmap")
        st.info("Heatmap view disabled in demo mode (only 2 weeks of data)")
        # fig_heatmap = create_heatmap_overview(
        #     st.session_state.calendar_data,
        #     st.session_state.eligible_services
        # )
        # st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # ============================================
    # TABLE EDITOR
    # ============================================
    
    st.divider()
    st.subheader("✏️ Edit Scores")
    
    st.info("""
    💡 **Editing Tips**:
    - **Bundle scores**: PICU and PCICU day shifts are grouped as a bundle — all dates in that bundle share the same score.
    - **Making multiple edits:** You can adjust several cells at once. Remember to click Apply Changes afterward.
    - **Note**: It might take a short moment before your updates appear on the calendar.
    """)
    
    # Table filters
    with st.expander("📊 Table Filters", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            tbl_type = st.multiselect(
                "Type",
                ["individual", "bundle"],
                default=st.session_state.tbl_type_filter,
                key="tbl_type"
            )
            st.session_state.tbl_type_filter = tbl_type
        
        with col2:
            tbl_svc = st.multiselect(
                "Services",
                st.session_state.eligible_services,
                default=st.session_state.tbl_svc_filter,
                key="tbl_svc"
            )
            st.session_state.tbl_svc_filter = tbl_svc
        
        with col3:
            tbl_period = st.multiselect(
                "Periods",
                PERIOD_ORDER,
                default=st.session_state.tbl_period_filter,
                key="tbl_period"
            )
            st.session_state.tbl_period_filter = tbl_period
        
        with col4:
            search_date = st.text_input("Search Date (YYYY-MM-DD)", "")
    
    # Apply filters
    mask = (
        (st.session_state.edit_data["slot_type"].isin(tbl_type)) &
        (st.session_state.edit_data["svc"].isin(tbl_svc)) &
        (st.session_state.edit_data["period"].isin(tbl_period))
    )
    
    if search_date:
        mask = mask & st.session_state.edit_data["date"].str.contains(search_date)
    
    filtered = st.session_state.edit_data[mask][["Type","Date","svc","period","Bundle","score"]].copy()
    
    if len(filtered) == 0:
        st.warning("No data matching filters")
        return
    
    # Form 밖으로 Undo/Reset 버튼 이동
    col1, col2, col3 = st.columns([3,1,1])
    with col2:
        if st.button("↩️ Undo", use_container_width=True, disabled=len(st.session_state.history_stack)==0):
            if len(st.session_state.history_stack) > 0:
                last_state = st.session_state.history_stack.pop()
                st.session_state.data = last_state["data"]
                st.session_state.change_log = last_state["change_log"]
                st.session_state.calendar_data = create_calendar_view(
                    st.session_state.data, 
                    st.session_state.unavailable_dates
                )
                st.session_state.edit_data = prepare_edit_data(st.session_state.data)
                st.success("↩️ Undone!")
                st.rerun()
    
    with col3:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.data = st.session_state.original_data.copy()
            st.session_state.calendar_data = create_calendar_view(
                st.session_state.data, 
                st.session_state.unavailable_dates
            )
            st.session_state.edit_data = prepare_edit_data(st.session_state.data)
            st.session_state.change_log = []
            st.session_state.history_stack = []
            st.session_state.total_applied_changes = 0
            st.session_state.changes_pending = False
            st.success("🔄 All changes reset!")
            st.rerun()
    
    # Data editor in form (Apply 버튼만 form 안에)
    with st.form("edit_form"):
        edited = st.data_editor(
            filtered, 
            hide_index=True, 
            use_container_width=True, 
            num_rows="fixed",
            disabled=["Type","Date","svc","period","Bundle"],
            column_config={
                "score": st.column_config.NumberColumn(
                    "Score", 
                    min_value=0.5, 
                    max_value=2.0,
                    step=0.01, 
                    format="%.3f"
                )
            },
            height=400
        )
        
        apply = st.form_submit_button("✅ Apply Changes", type="primary", use_container_width=True)
    
    if apply:
        # Save current state to history before making changes
        st.session_state.history_stack.append({
            "data": st.session_state.data.copy(),
            "timestamp": datetime.now(),
            "change_log": st.session_state.change_log.copy()
        })
        
        # Keep only last 10 states
        if len(st.session_state.history_stack) > 10:
            st.session_state.history_stack.pop(0)
        
        applied = 0

        # Reset index to ensure alignment
        filtered_reset = filtered.reset_index(drop=False)  # Keep original index as column
        edited_reset = edited.reset_index(drop=True)

        for idx in range(len(edited_reset)):
            old = filtered_reset.iloc[idx]["score"]
            new = edited_reset.iloc[idx]["score"]
            
            if abs(old - new) > 0.001:
                orig_idx = filtered_reset.iloc[idx]["index"]
                
                # Update edit_data
                st.session_state.edit_data.at[orig_idx, "score"] = new
                
                # Update main data
                data_row = st.session_state.data.iloc[orig_idx]
                st.session_state.data.at[orig_idx, "score"] = new
                
                # Update calendar_data (all matching rows)
                # Bundle 처리: bundle_id가 같은 모든 row 업데이트
                if data_row["slot_type"] == "bundle" and pd.notna(data_row["bundle_id"]):
                    bundle_id = data_row["bundle_id"]
                    bundle_mask = (st.session_state.data["bundle_id"] == bundle_id)
                    st.session_state.data.loc[bundle_mask, "score"] = new
                    st.session_state.edit_data.loc[bundle_mask, "score"] = new
                    
                    # Calendar data도 bundle_id로 업데이트
                    cal_bundle_mask = (
                        (st.session_state.calendar_data["svc"] == data_row["svc"]) &
                        (st.session_state.calendar_data["period"] == data_row["period"]) &
                        (st.session_state.calendar_data["is_bundle"] == True)
                    )
                    # Bundle dates 확인
                    dates = data_row["date"].split(",")
                    for d in dates:
                        date_mask = cal_bundle_mask & (st.session_state.calendar_data["date"] == d)
                        st.session_state.calendar_data.loc[date_mask, "score"] = new
                else:
                    # Individual slot: 단일 업데이트
                    cal_mask = (
                        (st.session_state.calendar_data["date"] == data_row["date"]) &
                        (st.session_state.calendar_data["svc"] == data_row["svc"]) &
                        (st.session_state.calendar_data["period"] == data_row["period"]) &
                        (st.session_state.calendar_data["is_bundle"] == False)
                    )
                    st.session_state.calendar_data.loc[cal_mask, "score"] = new
                
                # Log change
                row_data = st.session_state.edit_data.loc[orig_idx]
                st.session_state.change_log.append({
                    "time": datetime.now().isoformat(),
                    "date": row_data["Date"],
                    "svc": row_data["svc"],
                    "period": row_data["period"],
                    "old": old,
                    "new": new
                })
                applied += 1
        
        if applied > 0:
            st.session_state.total_applied_changes += applied
        
        st.session_state.changes_pending = False
        
        if applied > 0:
            st.success(f"✅ Applied {applied} changes!")
            st.rerun()
        else:
            st.info("No changes detected")


if __name__ == "__main__":
    main()