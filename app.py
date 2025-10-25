"""
Physician Utility Editor - FINAL VERSION

Based on v10 structure with corrections:
- Token authentication from v11
- Color contrast improvement
- Hours, Score, Per-Hour all editable
- Auto-calculation between fields
- Min/max constraints
"""

from __future__ import annotations
import os
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ============================================
# CONFIG
# ============================================
st.set_page_config(
    page_title="Utility Editor",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = "./"
DATA_DIR = Path(os.path.join(ROOT, "data", "processed"))
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)

PERIOD_ORDER = ["Day", "Night", "24h"]
SERVICE_ORDER = ["PICU", "PCICU", "Sedation", "Smilow"]

# (6) Period display mapping
PERIOD_DISPLAY_MAP = {
    "Day": "AM",
    "Night": "PM",
    "24h": "24h"
}
PERIOD_DISPLAY_TO_INTERNAL = {
    "AM": "Day",
    "PM": "Night",
    "24h": "24h"
}
PERIOD_DISPLAY_ORDER = ["AM", "PM", "24h"]

SERVICE_NAME_MAP = {
    "SED": "Sedation",
    "SMI": "Smilow",
    "Sedation": "Sedation",
    "Smilow": "Smilow",
    "PICU": "PICU",
    "PCICU": "PCICU"
}

SERVICE_COLORS = {
    "PICU": (15, 76, 129),
    "PCICU": (123, 31, 162),
    "Sedation": (27, 94, 32),
    "Smilow": (230, 81, 0)
}

# ============================================
# TOKEN MANAGEMENT
# ============================================

def load_tokens() -> dict:
   """Load physician tokens from Streamlit secrets"""
   if "tokens" in st.secrets:
       return dict(st.secrets["tokens"])
   return {"test": None}

TOKENS = load_tokens()

if not TOKENS:
    TOKENS = {"test": None}

# ============================================
# UTILITY COLORS
# ============================================

def get_color_by_score(service: str, score: float, available: bool = True) -> str:
    """
    Service color + Score-based opacity
    Improved contrast for 8-9 range
    """
    if not available:
        return "rgba(189, 189, 189, 0.5)"
    
    base = SERVICE_COLORS.get(service, (100, 100, 100))
    
    # score 범위: 0-10
    normalized = score / 10.0
    normalized = max(0, min(1, normalized))
    
    # 색상 대비 개선: opacity 0.15 ~ 0.95
    opacity = 0.15 + (normalized ** 1.2) * 0.80
    
    return f'rgba({base[0]}, {base[1]}, {base[2]}, {opacity})'

# ============================================
# UTILITY FUNCTIONS
# ============================================

def get_qual_services(qual: str):
    """Get eligible services for qualification"""
    qual_lower = qual.lower()
    
    if qual_lower in ["both", "pcicu_pred"]:
        return ["PICU", "PCICU", "Sedation", "Smilow"]
    elif qual_lower == "pcicu_only":
        return ["PCICU", "Sedation", "Smilow"]
    return SERVICE_ORDER

def check_availability(availability_df: pd.DataFrame, physician_id: int, date: str, period: str) -> bool:
    """Check period-specific availability"""
    avail_row = availability_df[
        (availability_df['i_id'] == physician_id) &
        (availability_df['date'] == date)
    ]
    
    if len(avail_row) == 0:
        return True
    
    am = avail_row.iloc[0]['AM']
    pm = avail_row.iloc[0]['PM']
    h24 = avail_row.iloc[0]['H24']
    
    if period == 'Day':
        return (am == 1) or (h24 == 1)
    elif period == 'Night':
        return (pm == 1) or (h24 == 1)
    elif period == '24h':
        return h24 == 1
    return True

# ============================================
# DATA LOADING
# ============================================

@st.cache_data
def load_data():
    """Load physicians and availability"""
    phys_path = DATA_DIR / "physicians.csv"
    avail_path = DATA_DIR / "availability.csv"
    
    if not phys_path.exists():
        st.error("❌ physicians.csv not found!")
        st.stop()
    
    physicians = pd.read_csv(phys_path)
    physicians['i_id'] = physicians['i_id'].astype(int)
    
    if avail_path.exists():
        availability = pd.read_csv(avail_path)
    else:
        st.error("❌ availability.csv not found!")
        st.stop()
    
    return physicians, availability

def load_physician_utility(physician_id: int) -> pd.DataFrame:
    """Load specific physician's utility file"""
    util_path = DATA_DIR / "utility" / f"utilities_{physician_id}.csv"
    
    if not util_path.exists():
        st.error(f"❌ utilities_{physician_id}.csv not found!")
        return pd.DataFrame()
    
    utilities = pd.read_csv(util_path)
    
    if 'svc' in utilities.columns:
        utilities['svc'] = utilities['svc'].map(lambda x: SERVICE_NAME_MAP.get(x, x))
    
    # Ensure per_hour_utility exists
    if 'per_hour_utility' not in utilities.columns:
        if 'hours' in utilities.columns:
            utilities['per_hour_utility'] = utilities['score'] / utilities['hours']
        else:
            utilities['per_hour_utility'] = 0.0
    
    return utilities

# ============================================
# DATA PREPARATION
# ============================================

def create_calendar_view(util_data: pd.DataFrame, availability_df: pd.DataFrame, 
                         physician_id: int) -> pd.DataFrame:
    """Create calendar view with availability check"""
    rows = []
    
    for idx, row in util_data.iterrows():
        avail = check_availability(availability_df, physician_id, row["date"], row["period"])
        
        per_hour_utility = row.get('per_hour_utility', 0)
        if per_hour_utility == 0 and 'hours' in row and row['hours'] > 0:
            per_hour_utility = row['score'] / row['hours']
        
        if avail:
            rows.append({
                "original_idx": idx,
                "date": row["date"],
                "svc": row["svc"],
                "period": row["period"],
                "score": row["score"],
                "per_hour_utility": per_hour_utility,
                "hours": row.get("hours", 1),
                "available": True
            })
    
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df["date_parsed"] = pd.to_datetime(df["date"])
    return df

def prepare_edit_data(util_data: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for editing table"""
    df = util_data.copy()
    df["original_data_idx"] = df.index
    df["Date"] = pd.to_datetime(df["date"]).dt.strftime('%m-%d-%Y')
    return df

# ============================================
# VISUALIZATION - CALENDAR
# ============================================

def create_6month_calendar(cal_data, start_date, num_weeks, svc_filter, period_filter):
    """Create weekly calendar view"""
    fig = go.Figure()
    
    current_monday = pd.Timestamp(start_date) - pd.Timedelta(days=start_date.weekday())
    week_num = 0
    
    for _ in range(num_weeks):
        y_base = -week_num
        
        for dow in range(7):
            date = current_monday + pd.Timedelta(days=dow)
            x_base = dow
            
            date_data = cal_data[cal_data["date_parsed"] == date]
            
            # Draw day border
            fig.add_shape(type="rect", x0=x_base, y0=y_base+0.20, 
                         x1=x_base+1, y1=y_base+0.95,
                         line=dict(color='#bbb', width=1), fillcolor='rgba(255,255,255,0)')
            
            # Date label
            fig.add_annotation(x=x_base+0.5, y=y_base+0.08, 
                             text=f"<b>{date.day}</b>", showarrow=False,
                             font=dict(size=10, color='#333'))
            
            if len(date_data) == 0:
                continue
            
            # Filter by period
            day_data = date_data[date_data["period"] == "Day"]
            night_data = date_data[date_data["period"] == "Night"]
            h24_data = date_data[date_data["period"] == "24h"]
            
            n_cols = len(svc_filter)
            cell_w = 0.95 / n_cols
            
            if len(h24_data) > 0:
                for _, row in h24_data.iterrows():
                    if row["svc"] not in svc_filter:
                        continue
                    svc_idx = svc_filter.index(row["svc"])
                    
                    x = x_base + 0.025 + svc_idx * cell_w
                    y = y_base + 0.20
                    h = 0.75
                    
                    color = get_color_by_score(row["svc"], row["score"], row["available"])
                    border = dict(color='#ddd', width=1)
                    
                    fig.add_shape(type="rect", x0=x, y0=y, x1=x+cell_w, y1=y+h,
                                fillcolor=color, line=border)
                    
                    text_color = 'white' if row['score'] > 5 else 'black'
                    per_hour = row.get('per_hour_utility', 0)
                    display_text = f"{row['score']:.2f}<br>({per_hour:.3f})" if per_hour > 0 else f"{row['score']:.2f}"
                    
                    fig.add_annotation(
                        x=x+cell_w/2, y=y+h/2, text=display_text,
                        showarrow=False, font=dict(size=8, color=text_color, family="Arial Black")
                    )
            else:
                if len(day_data) > 0:
                    for _, row in day_data.iterrows():
                        if row["svc"] not in svc_filter:
                            continue
                        svc_idx = svc_filter.index(row["svc"])
                        
                        x = x_base + 0.025 + svc_idx * cell_w
                        y = y_base + 0.60
                        h = 0.35
                        
                        color = get_color_by_score(row["svc"], row["score"], row["available"])
                        border = dict(color='#ddd', width=1)
                        
                        fig.add_shape(type="rect", x0=x, y0=y, x1=x+cell_w, y1=y+h,
                                    fillcolor=color, line=border)
                        
                        text_color = 'white' if row['score'] > 5 else 'black'
                        per_hour = row.get('per_hour_utility', 0)
                        display_text = f"{row['score']:.2f}<br>({per_hour:.3f})" if per_hour > 0 else f"{row['score']:.2f}"
                        
                        fig.add_annotation(
                            x=x+cell_w/2, y=y+h/2, text=display_text,
                            showarrow=False, font=dict(size=8, color=text_color, family="Arial Black")
                        )
                
                if len(night_data) > 0:
                    for _, row in night_data.iterrows():
                        if row["svc"] not in svc_filter:
                            continue
                        svc_idx = svc_filter.index(row["svc"])
                        
                        x = x_base + 0.025 + svc_idx * cell_w
                        y = y_base + 0.20
                        h = 0.35
                        
                        color = get_color_by_score(row["svc"], row["score"], row["available"])
                        border = dict(color='#ddd', width=1)
                        
                        fig.add_shape(type="rect", x0=x, y0=y, x1=x+cell_w, y1=y+h,
                                    fillcolor=color, line=border)
                        
                        text_color = 'white' if row['score'] > 5 else 'black'
                        per_hour = row.get('per_hour_utility', 0)
                        display_text = f"{row['score']:.2f}<br>({per_hour:.3f})" if per_hour > 0 else f"{row['score']:.2f}"
                        
                        fig.add_annotation(
                            x=x+cell_w/2, y=y+h/2, text=display_text,
                            showarrow=False, font=dict(size=8, color=text_color, family="Arial Black")
                        )
        
        week_num += 1
        current_monday += pd.Timedelta(days=7)
    
    # Period labels
    for k in range(week_num):
        yb = -k
        if "Day" in period_filter:
            fig.add_annotation(x=-0.15, y=yb + 0.775, text="<b>AM</b>", showarrow=False,
                             font=dict(size=10, color='#666'), xanchor='right', yanchor='middle')
        if "Night" in period_filter:
            fig.add_annotation(x=-0.15, y=yb + 0.375, text="<b>PM</b>", showarrow=False,
                             font=dict(size=10, color='#666'), xanchor='right', yanchor='middle')
    
    if "24h" in period_filter:
        for k in range(week_num):
            fig.add_annotation(x=7.05, y=-k + 0.57, text="<b>24h</b>", showarrow=False,
                             font=dict(size=9, color='#666'), xanchor='left', yanchor='middle')
    
    fig.update_xaxes(range=[-0.15, 7.3], tickvals=list(range(7)),
                     ticktext=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], 
                     side='top', tickfont=dict(size=11, color='#333'))
    
    fig.update_yaxes(range=[-week_num+0.15, 1.05], showticklabels=False)
    
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        plot_bgcolor='white',
        height=max(300, week_num * 110),
        margin=dict(l=50, r=50, t=30, b=10),
        showlegend=False
    )
    
    return fig

# ============================================
# VISUALIZATION - HEATMAP
# ============================================

def create_6month_heatmap(cal_data: pd.DataFrame, eligible_services: list, show_scores: bool = False):
    """6-month calendar heatmap (달력 형태)"""
    if cal_data.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, 
                          showarrow=False, font=dict(size=16))
        fig.update_layout(height=600)
        return fig
    
    dates = sorted(cal_data["date_parsed"].unique())
    if len(dates) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        return fig
    
    first_date = dates[0]
    last_date = dates[-1]
    first_monday = first_date - pd.Timedelta(days=first_date.weekday())
    
    fig = go.Figure()
    week_num = 0
    current_monday = first_monday
    total_weeks = ((last_date - first_monday).days // 7) + 1
    
    current_month = None
    month_positions = []
    
    while current_monday <= last_date:
        for day_offset in range(7):
            current_date = current_monday + pd.Timedelta(days=day_offset)
            if current_date < first_date or current_date > last_date:
                continue
            
            if current_date.month != current_month:
                current_month = current_date.month
                month_positions.append({
                    'week': week_num,
                    'month': current_date.strftime('%B'),
                    'date': current_date
                })
            
            date_data = cal_data[cal_data["date_parsed"] == current_date]
            
            x_base, y_base = day_offset, -week_num
            
            # Date header
            fig.add_shape(
                type="rect", x0=x_base, y0=y_base, x1=x_base+1, y1=y_base+0.15,
                fillcolor='rgba(245,245,245,1)', line=dict(color='#999', width=0.5)
            )
            
            fig.add_annotation(
                x=x_base+0.5, y=y_base+0.075, text=f"<b>{current_date.day}</b>",
                showarrow=False, font=dict(size=8, color='#333'),
                xanchor='center', yanchor='middle'
            )
            
            # (1) No shift - 흰색
            if len(date_data) == 0:
                fig.add_shape(
                    type="rect", x0=x_base, y0=y_base+0.15, x1=x_base+1, y1=y_base+1,
                    fillcolor='rgba(255,255,255,1)',  # 흰색
                    line=dict(color='#ddd', width=0.5)
                )
                continue
            
            # (1) Unavailable - 흰색
            is_unavailable = not date_data['available'].all()
            
            if is_unavailable:
                fig.add_shape(
                    type="rect", x0=x_base, y0=y_base+0.15, x1=x_base+1, y1=y_base+1,
                    fillcolor='rgba(255,255,255,1)',  # 흰색
                    line=dict(color='#ddd', width=0.5)
                )
                continue
            
            day_data = date_data[date_data["period"] == "Day"]
            night_data = date_data[date_data["period"] == "Night"]
            h24_data = date_data[date_data["period"] == "24h"]
            
            n_cols = len(eligible_services)
            cell_w = 0.95 / n_cols
            
            if len(h24_data) > 0:
                for _, row in h24_data.iterrows():
                    if row["svc"] not in eligible_services:
                        continue
                    svc_idx = eligible_services.index(row["svc"])
                    
                    x = x_base + 0.025 + svc_idx * cell_w
                    y = y_base + 0.15
                    h = 0.80
                    
                    color = get_color_by_score(row["svc"], row["score"], row["available"])
                    border = dict(color='#ddd', width=0.5)
                    
                    fig.add_shape(type="rect", x0=x, y0=y, x1=x+cell_w, y1=y+h,
                                fillcolor=color, line=border)
                    
                    if show_scores:
                        text_color = 'white' if row['score'] > 5 else 'black'
                        per_hour = row.get('per_hour_utility', 0)
                        display_text = f"{row['score']:.1f}<br>({per_hour:.2f})" if per_hour > 0 else f"{row['score']:.1f}"
                        
                        fig.add_annotation(
                            x=x+cell_w/2, y=y+h/2, text=display_text,
                            showarrow=False, font=dict(size=6, color=text_color, family="Arial Black")
                        )
            else:
                if len(day_data) > 0:
                    for _, row in day_data.iterrows():
                        if row["svc"] not in eligible_services:
                            continue
                        svc_idx = eligible_services.index(row["svc"])
                        
                        x = x_base + 0.025 + svc_idx * cell_w
                        y = y_base + 0.55
                        h = 0.40
                        
                        color = get_color_by_score(row["svc"], row["score"], row["available"])
                        border = dict(color='#ddd', width=0.5)
                        
                        fig.add_shape(type="rect", x0=x, y0=y, x1=x+cell_w, y1=y+h,
                                    fillcolor=color, line=border)
                        
                        if show_scores:
                            text_color = 'white' if row['score'] > 5 else 'black'
                            per_hour = row.get('per_hour_utility', 0)
                            display_text = f"{row['score']:.1f}<br>({per_hour:.2f})" if per_hour > 0 else f"{row['score']:.1f}"
                            
                            fig.add_annotation(
                                x=x+cell_w/2, y=y+h/2, text=display_text,
                                showarrow=False, font=dict(size=6, color=text_color, family="Arial Black")
                            )
                
                if len(night_data) > 0:
                    for _, row in night_data.iterrows():
                        if row["svc"] not in eligible_services:
                            continue
                        svc_idx = eligible_services.index(row["svc"])
                        
                        x = x_base + 0.025 + svc_idx * cell_w
                        y = y_base + 0.15
                        h = 0.40
                        
                        color = get_color_by_score(row["svc"], row["score"], row["available"])
                        border = dict(color='#ddd', width=0.5)
                        
                        fig.add_shape(type="rect", x0=x, y0=y, x1=x+cell_w, y1=y+h,
                                    fillcolor=color, line=border)
                        
                        if show_scores:
                            text_color = 'white' if row['score'] > 5 else 'black'
                            per_hour = row.get('per_hour_utility', 0)
                            display_text = f"{row['score']:.1f}<br>({per_hour:.2f})" if per_hour > 0 else f"{row['score']:.1f}"
                            
                            fig.add_annotation(
                                x=x+cell_w/2, y=y+h/2, text=display_text,
                                showarrow=False, font=dict(size=6, color=text_color, family="Arial Black")
                            )
        
        week_num += 1
        current_monday += pd.Timedelta(days=7)
    
    # Month separators and labels
    for i, month_info in enumerate(month_positions):
        y_pos = -month_info['week']
        
        fig.add_shape(
            type="line",
            x0=-0.2, x1=7.2,
            y0=y_pos + 1, y1=y_pos + 1,
            line=dict(color='rgba(0,0,0,0.3)', width=2.5)
        )
        
        fig.add_annotation(
            x=-0.5, y=y_pos + 0.5,
            text=f"<b>{month_info['month']}</b>",
            showarrow=False,
            font=dict(size=11, color='#333'),
            xanchor='right',
            yanchor='middle',
            textangle=0
        )
    
    fig.update_xaxes(
        range=[-0.6, 7.3], 
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
    
    cal_height = max(600, min(1200, total_weeks * 50))
    
    fig.update_layout(
        height=cal_height, 
        margin=dict(l=100, r=20, t=40, b=10),
        plot_bgcolor='white', 
        showlegend=False,
        title="6-Month Calendar Overview"
    )
    
    return fig

# ============================================
# AUTHENTICATION
# ============================================

def authenticate():
    """Token-based authentication"""
    st.title("🔐 Physician Login")
    
    query_params = st.query_params
    token = query_params.get("token", None)
    
    if token:
        if token in TOKENS:
            physician_id = TOKENS[token]
            
            physicians, availability = load_data()
            phys_info = physicians[physicians['i_id'] == physician_id]
            
            if len(phys_info) > 0:
                st.session_state.authenticated = True
                st.session_state.physician_id = physician_id
                st.session_state.physician_name = phys_info.iloc[0]['name']
                st.session_state.qualification = phys_info.iloc[0]['qualification']
                st.success(f"✅ Authenticated as Dr. {st.session_state.physician_name}")
                st.rerun()
            else:
                st.error("❌ Invalid physician ID")
        else:
            st.error("❌ Invalid token")
    
    st.info("💡 Please use the URL provided to you, or enter your token below:")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        manual_token = st.text_input("Enter your token", type="password")
    with col2:
        st.write("")
        st.write("")
        if st.button("Login", type="primary"):
            if manual_token in TOKENS:
                physician_id = TOKENS[manual_token]
                physicians, availability = load_data()
                phys_info = physicians[physicians['i_id'] == physician_id]
                
                if len(phys_info) > 0:
                    st.session_state.authenticated = True
                    st.session_state.physician_id = physician_id
                    st.session_state.physician_name = phys_info.iloc[0]['name']
                    st.session_state.qualification = phys_info.iloc[0]['qualification']
                    st.rerun()
                else:
                    st.error("❌ Invalid physician ID")
            else:
                st.error("❌ Invalid token")

# ============================================
# SESSION STATE
# ============================================

def init_session_state():
    """Initialize session state"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.physician_id = None
        st.session_state.physician_name = None
        st.session_state.qualification = None
    
    if not st.session_state.authenticated:
        return
    
    if 'initialized' in st.session_state and st.session_state.initialized:
        return
    
    pid = st.session_state.physician_id
    
    physicians, availability = load_data()
    
    phys_match = physicians[physicians["i_id"] == pid]
    if len(phys_match) == 0:
        st.error(f"❌ Physician ID {pid} not found!")
        st.stop()
    
    phys = phys_match.iloc[0].to_dict()
    util = load_physician_utility(pid)
    
    if util.empty:
        st.error("No utility data found!")
        st.stop()
    
    avail = availability[availability["i_id"] == pid].copy()

    calendar_data = create_calendar_view(util, avail, pid)
    edit_data = prepare_edit_data(util)
    
    st.session_state.eligible_services = get_qual_services(phys["qualification"])
    st.session_state.availability_df = avail
    
    st.session_state.data = util.copy(deep=True)
    st.session_state.original_data = util.copy(deep=True)
    st.session_state.calendar_data = calendar_data
    st.session_state.edit_data = edit_data
    
    st.session_state.changes_pending = False
    st.session_state.change_log = []
    st.session_state.history_stack = []
    
    # Filters
    st.session_state.cal_svc_filter = st.session_state.eligible_services.copy()
    st.session_state.cal_period_filter = PERIOD_ORDER.copy()
    st.session_state.tbl_svc_filter = st.session_state.eligible_services.copy()
    st.session_state.tbl_period_filter = PERIOD_ORDER.copy()
    
    # Date range
    min_date = calendar_data["date_parsed"].min().date()
    default_end = min((pd.Timestamp(min_date) + pd.Timedelta(days=30)).date(), 
                     calendar_data["date_parsed"].max().date())
    st.session_state.cal_date_range = (min_date, default_end)
    
    st.session_state.total_applied_changes = 0
    st.session_state.heatmap_fig = None
    st.session_state.heatmap_last_refresh = None
    st.session_state.form_key = 0
    
    st.session_state.initialized = True

# ============================================
# MAIN
# ============================================

def main():
    init_session_state()
    
    if not st.session_state.authenticated:
        authenticate()
        return
    
    st.title(f"Dr. {st.session_state.physician_name}'s Utility Editor")
    
    # Sidebar
    with st.sidebar:
        st.header(f"👤 Dr. {st.session_state.physician_name}")
        st.caption(f"ID: {st.session_state.physician_id}")
        st.caption(f"Qualification: {st.session_state.qualification}")
        
        st.divider()
        st.header("📊 Statistics")
        
        scores = st.session_state.data["score"]
        col1, col2 = st.columns(2)
        col1.metric("Min", f"{scores.min():.2f}")
        col2.metric("Max", f"{scores.max():.2f}")
        col1.metric("Mean", f"{scores.mean():.2f}")
        col2.metric("Median", f"{scores.median():.2f}")

        st.info("📌 **Score Ranges**:\n\n"
                "• **Score**: 0 - 10\n\n"
                "• **Hours**: 4 - 25")
        
        st.divider()
        st.subheader("Status")
        st.success("✓ All saved")
        st.metric("Total Changes", st.session_state.total_applied_changes)
        
        if st.session_state.heatmap_last_refresh:
            st.caption(f"Heatmap: {st.session_state.heatmap_last_refresh}")
        else:
            st.caption("Heatmap: Not generated yet")
        
        st.divider()
        
        st.markdown("**💾 Save & Export**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.caption("Save changes to server (required for persistence)")
            if st.button("💾 Save to File", use_container_width=True, type="primary"):
                output_path = DATA_DIR / "utility" / f"utilities_{st.session_state.physician_id}.csv"
                st.session_state.data.to_csv(output_path, index=False)
                
                log_path = DATA_DIR / "logs" / f"changes_{st.session_state.physician_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                log_path.parent.mkdir(exist_ok=True, parents=True)
                
                if len(st.session_state.change_log) > 0:
                    log_df = pd.DataFrame(st.session_state.change_log)
                    log_df.to_csv(log_path, index=False)
                
                st.success(f"✅ Saved!")
                st.session_state.original_data = st.session_state.data.copy(deep=True)
        
        with col2:
            # (2) Export CSV button
            st.caption("Download current version to your computer")
            csv_data = st.session_state.data.to_csv(index=False)
            st.download_button(
                label="📤 Export CSV",
                data=csv_data,
                file_name=f"utilities_{st.session_state.physician_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        st.caption("⚠️ **Save to File before logout** to keep changes. Export CSV for local backup.")
        
        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Calendar section
    st.subheader("📅 Calendar View")
    
    # Create tabs for Calendar and Heatmap
    tab1, tab2 = st.tabs(["📅 Calendar View", "🔥 Heatmap Overview"])
    
    with tab1:
        # (3) 색상 범례 - 글자 자체를 색상으로
        st.markdown("**Service Colors:**")
        service_labels = []
        for service, color in SERVICE_COLORS.items():
            service_labels.append(
                f'<span style="color: rgb{color}; font-weight: bold; font-size: 16px; margin-right: 20px;">{service}</span>'
            )
        st.markdown(" ".join(service_labels), unsafe_allow_html=True)
        
        st.divider()
        
        col1, col2, col3 = st.columns([2, 2, 2])
        
        with col1:
            date_range = st.date_input(
                "Date Range",
                value=st.session_state.cal_date_range,
                key="cal_dates"
            )
            
            if isinstance(date_range, tuple) and len(date_range) == 2:
                st.session_state.cal_date_range = date_range
        
        with col2:
            cal_svc = st.multiselect(
                "Services",
                st.session_state.eligible_services,
                default=st.session_state.cal_svc_filter,
                key="cal_svc"
            )
            st.session_state.cal_svc_filter = cal_svc
        
        with col3:
            # (6) Period를 AM/PM/24h로 표시
            cal_period_display = st.multiselect(
                "Periods",
                PERIOD_DISPLAY_ORDER,
                default=[PERIOD_DISPLAY_MAP[p] for p in st.session_state.cal_period_filter],
                key="cal_period"
            )
            # 내부적으로는 Day/Night/24h로 변환
            st.session_state.cal_period_filter = [PERIOD_DISPLAY_TO_INTERNAL[p] for p in cal_period_display]
        
        # (5) Weeks 계산 - 자동으로 date range에 맞춤
        if len(st.session_state.cal_svc_filter) > 0 and len(st.session_state.cal_period_filter) > 0:
            start_date, end_date = st.session_state.cal_date_range
            num_weeks = ((end_date - start_date).days // 7) + 1
            
            # Filter calendar data
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            
            mask = (
                (st.session_state.calendar_data["date_parsed"] >= start_ts) &
                (st.session_state.calendar_data["date_parsed"] <= end_ts) &
                (st.session_state.calendar_data["svc"].isin(st.session_state.cal_svc_filter)) &
                (st.session_state.calendar_data["period"].isin(st.session_state.cal_period_filter))
            )
            filtered_cal_data = st.session_state.calendar_data[mask]
            
            fig_cal = create_6month_calendar(
                filtered_cal_data,
                start_date,
                num_weeks,
                st.session_state.cal_svc_filter,
                st.session_state.cal_period_filter
            )
            st.plotly_chart(fig_cal, use_container_width=True)
        else:
            st.warning("⚠️ Please select at least one service and one period")
    
    with tab2:
        # (3) 색상 범례 - 글자 자체를 색상으로
        st.markdown("**Service Colors:**")
        service_labels = []
        for service, color in SERVICE_COLORS.items():
            service_labels.append(
                f'<span style="color: rgb{color}; font-weight: bold; font-size: 16px; margin-right: 20px;">{service}</span>'
            )
        st.markdown(" ".join(service_labels), unsafe_allow_html=True)
        
        st.divider()
        
        st.info("💡 **Tip**: Click 'Generate Heatmap' to create the overview, then 'Refresh' to update with changes.")
        
        # Generate button
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔥 Generate Heatmap", use_container_width=True, type="primary"):
                with st.spinner("Generating 6-month heatmap..."):
                    fig_heatmap = create_6month_heatmap(
                        st.session_state.calendar_data,
                        st.session_state.eligible_services
                    )
                    st.session_state.heatmap_fig = fig_heatmap
                    st.session_state.heatmap_last_refresh = datetime.now().strftime('%H:%M:%S')
                    st.success("✅ Heatmap generated!")
        
        # Refresh button (only if heatmap exists)
        with col2:
            if st.session_state.heatmap_fig is not None:
                if st.button("🔄 Refresh Heatmap", use_container_width=True):
                    with st.spinner("Updating heatmap..."):
                        # Rebuild calendar_data from current data state
                        fresh_cal_data = create_calendar_view(
                            st.session_state.data,
                            st.session_state.availability_df,
                            st.session_state.physician_id
                        )
                        st.session_state.calendar_data = fresh_cal_data
                        
                        fig_heatmap = create_6month_heatmap(
                            fresh_cal_data,
                            st.session_state.eligible_services
                        )
                        st.session_state.heatmap_fig = fig_heatmap
                        st.session_state.heatmap_last_refresh = datetime.now().strftime('%H:%M:%S')
                        st.success("✅ Heatmap refreshed!")
                        st.rerun()
        
        # Show heatmap if it exists
        if st.session_state.heatmap_fig is not None:
            st.plotly_chart(st.session_state.heatmap_fig, use_container_width=True)
        else:
            st.info("👆 Click 'Generate Heatmap' to create the 6-month overview")
    
    # Table editor
    st.divider()
    st.subheader("✏️ Edit Scores")
    
    st.info("💡 **Editable Fields**: Score (0-10) and Per-Hour \n\n"
            "- Edit **Score** → Per-Hour automatically updates (Score ÷ Hours)\n"
            "- Edit **Per-Hour** → Score automatically updates (Per-Hour × Hours)\n"
            "- If adjusting the Per-Hour value would make the total Score exceed 10, the system will block that change.")
    
    with st.expander("📊 Table Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tbl_svc = st.multiselect("Services", st.session_state.eligible_services,
                                    default=st.session_state.tbl_svc_filter, key="tbl_svc")
            st.session_state.tbl_svc_filter = tbl_svc
        
        with col2:
            # (6) Period를 AM/PM/24h로 표시
            tbl_period_display = st.multiselect("Periods", PERIOD_DISPLAY_ORDER,
                                       default=[PERIOD_DISPLAY_MAP[p] for p in st.session_state.tbl_period_filter], 
                                       key="tbl_period")
            # 내부적으로는 Day/Night/24h로 변환
            tbl_period = [PERIOD_DISPLAY_TO_INTERNAL[p] for p in tbl_period_display]
            st.session_state.tbl_period_filter = tbl_period
        
        with col3:
            search_date = st.text_input("Search Date (MM-DD-YYYY)", "", 
                                       help="Search within date range above")
    
    # Filter data
    start_date, end_date = st.session_state.cal_date_range
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    
    edit_data_with_parsed = st.session_state.edit_data.copy()
    edit_data_with_parsed["date_parsed"] = pd.to_datetime(edit_data_with_parsed["date"])
    
    # Ensure per_hour_utility exists
    if 'per_hour_utility' not in edit_data_with_parsed.columns:
        edit_data_with_parsed['per_hour_utility'] = edit_data_with_parsed.apply(
            lambda r: r['score'] / r.get('hours', 1) if r.get('hours', 1) > 0 else 0, axis=1
        )
    
    date_mask = (
        (edit_data_with_parsed["date_parsed"] >= start_ts) &
        (edit_data_with_parsed["date_parsed"] <= end_ts)
    )
    
    mask = (
        date_mask &
        (edit_data_with_parsed["svc"].isin(tbl_svc)) &
        (edit_data_with_parsed["period"].isin(tbl_period))
    )
    
    if search_date:
        try:
            search_parts = search_date.split('-')
            if len(search_parts) == 3:
                search_formatted = f"{search_parts[2]}-{search_parts[0]}-{search_parts[1]}"
                mask = mask & edit_data_with_parsed["date"].str.contains(search_formatted, na=False)
            else:
                mask = mask & (
                    edit_data_with_parsed["Date"].str.contains(search_date, na=False) |
                    edit_data_with_parsed["date"].str.contains(search_date, na=False)
                )
        except:
            mask = mask & (
                edit_data_with_parsed["Date"].str.contains(search_date, na=False) |
                edit_data_with_parsed["date"].str.contains(search_date, na=False)
            )
    
    filtered_base = edit_data_with_parsed[mask].copy()
    filtered_base.reset_index(drop=True, inplace=True)
    
    # (6) Period를 AM/PM/24h로 표시
    # filtered_base에 original_data_idx 유지
    filtered = filtered_base[["Date","svc","period","hours","score","per_hour_utility","original_data_idx"]].copy()
    filtered["Period Display"] = filtered["period"].map(PERIOD_DISPLAY_MAP)
    
    # Display용 DataFrame - per_hour_utility는 score/hours로 계산해서 추가
    # (원본 per_hour_utility 값을 사용하지 않음으로써 편집 가능하게 만듦)
    filtered_display = filtered[["Date","svc","Period Display","hours","score"]].copy()
    # 현재 score/hours로 계산된 값을 표시 (이 값과 다르게 편집하면 감지됨)
    filtered_display["per_hour_utility"] = filtered_display["score"] / filtered_display["hours"]
    
    if len(filtered_display) == 0:
        st.warning("No data matching filters")
        return
    
    # Undo/Reset buttons
    col1, col2, col3 = st.columns([3,1,1])
    
    with col2:
        undo_clicked = st.button("↩️ Undo", use_container_width=True, 
                                disabled=len(st.session_state.history_stack)==0,
                                key="undo_btn")
    
    with col3:
        reset_clicked = st.button("🔄 Reset All", use_container_width=True,
                                  key="reset_btn")
    
    if undo_clicked and len(st.session_state.history_stack) > 0:
        last_state = st.session_state.history_stack.pop()
        
        st.session_state.data = last_state["data"].copy(deep=True)
        st.session_state.change_log = last_state["change_log"].copy()
        
        st.session_state.calendar_data = create_calendar_view(
            st.session_state.data,
            st.session_state.availability_df,
            st.session_state.physician_id
        )
        st.session_state.edit_data = prepare_edit_data(st.session_state.data)
        st.session_state.total_applied_changes = len(st.session_state.change_log)
        
        st.success(f"↩️ Undone! Restored to {last_state['timestamp'].strftime('%H:%M:%S')}")
        st.rerun()
    
    if reset_clicked:
        st.session_state.data = st.session_state.original_data.copy(deep=True)
        
        st.session_state.calendar_data = create_calendar_view(
            st.session_state.data,
            st.session_state.availability_df,
            st.session_state.physician_id
        )
        st.session_state.edit_data = prepare_edit_data(st.session_state.data)
        st.session_state.change_log = []
        st.session_state.history_stack = []
        st.session_state.total_applied_changes = 0
        st.session_state.changes_pending = False
        
        st.success("🔄 Reset complete!")
        st.rerun()
    
    # Data editor
    with st.form(key=f"edit_form_{st.session_state.form_key}"):
        edited = st.data_editor(
            filtered_display,
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            disabled=["Date","svc","Period Display","hours"],
            column_config={
                "hours": st.column_config.NumberColumn(
                    "Hours (4-25)",
                    min_value=4.0,
                    max_value=25.0,
                    step=0.5,
                    format="%.1f",
                    help="Edit hours (Per-Hour will auto-update)"
                ),
                "score": st.column_config.NumberColumn(
                    "Score (0-10)",
                    min_value=0.0,
                    max_value=10.0,
                    step=0.1,
                    format="%.2f",
                    help="Edit score (Per-Hour will auto-update)"
                ),
                "per_hour_utility": st.column_config.NumberColumn(
                    "Per Hour Score",
                    min_value=0.0,
                    max_value=3.0,
                    step=0.01,
                    format="%.3f",
                    help="Edit per-hour utility (Score will auto-update if within 0-10 range)"
                )
            },
            height=400,
            key=f"data_editor_{st.session_state.form_key}"
        )
        
        apply = st.form_submit_button("✅ Apply Changes", type="primary", use_container_width=True)
    
    if apply:
        st.session_state.history_stack.append({
            "data": st.session_state.data.copy(deep=True),
            "timestamp": datetime.now(),
            "change_log": st.session_state.change_log.copy()
        })
        
        if len(st.session_state.history_stack) > 10:
            st.session_state.history_stack.pop(0)
        
        applied = 0
        edited_reset = edited.reset_index(drop=True)

        for display_idx in range(len(edited_reset)):
            # filtered에서 original_data_idx 가져오기 (filtered_display가 아님!)
            orig_data_idx = filtered.iloc[display_idx]["original_data_idx"]

            # Get old values from stored data
            old_hours = st.session_state.data.loc[orig_data_idx, "hours"]
            old_score = st.session_state.data.loc[orig_data_idx, "score"]
            
            # Get old per_hour - ensure it exists
            if "per_hour_utility" in st.session_state.data.columns:
                old_per_hour_stored = st.session_state.data.loc[orig_data_idx, "per_hour_utility"]
            else:
                old_per_hour_stored = old_score / old_hours if old_hours > 0 else 0.0
            
            # Calculate what per_hour is displayed (score/hours)
            old_per_hour_calculated = old_score / old_hours if old_hours > 0 else 0.0
            
            # Get new values from edited table
            new_hours = edited_reset.iloc[display_idx]["hours"]
            new_score = edited_reset.iloc[display_idx]["score"]
            new_per_hour = edited_reset.iloc[display_idx]["per_hour_utility"]
            
            # Check what changed (hours should never change since it's disabled)
            hours_changed = abs(old_hours - new_hours) > 0.001
            score_changed = abs(old_score - new_score) > 0.001
            # Compare new_per_hour with the CALCULATED value (not stored value)
            per_hour_changed = abs(old_per_hour_calculated - new_per_hour) > 0.001
            
            if hours_changed or score_changed or per_hour_changed:
                data_row = st.session_state.data.loc[orig_data_idx]
                
                # Auto-calculation logic (hours는 disabled이므로 항상 old_hours 사용)
                if score_changed and not per_hour_changed:
                    # Only score changed → recalculate per_hour
                    final_score = new_score
                    final_hours = old_hours  # hours는 변경 불가
                    final_per_hour = new_score / final_hours if final_hours > 0 else 0.0
                    should_count = True  # score 변경은 항상 counting
                    
                elif per_hour_changed and not score_changed:
                    # Only per_hour changed → recalculate score
                    final_per_hour = new_per_hour
                    final_hours = old_hours  # hours는 변경 불가
                    final_score_unconstrained = new_per_hour * final_hours
                    
                    # Check if score will be constrained
                    if final_score_unconstrained > 10.0:
                        st.warning(f"⚠️ Row {display_idx+1}: Calculated score ({final_score_unconstrained:.2f}) exceeds maximum (10.0). Change not counted.")
                        final_score = old_score  # ← 기존 값 유지
                        final_per_hour = old_per_hour_calculated
                        should_count = False  # ← counting 안됨
                    elif final_score_unconstrained < 0.0:
                        st.warning(f"⚠️ Row {display_idx+1}: Calculated score ({final_score_unconstrained:.2f}) is below minimum (0.0). Change not counted.")
                        final_score = old_score  # ← 기존 값 유지 (여기도!)
                        final_per_hour = old_per_hour_calculated  # ← 추가!
                        should_count = False
                    else:
                        final_score = final_score_unconstrained
                        should_count = True  # 범위 내면 counting
                    
                else:
                    # Multiple changed (or hours somehow changed) → prioritize score
                    final_score = new_score
                    final_hours = old_hours  # hours는 변경 불가
                    final_per_hour = new_score / final_hours if final_hours > 0 else 0.0
                    should_count = True  # score 변경은 항상 counting
                
                # Apply constraints
                final_score = max(0.0, min(10.0, final_score))
                final_per_hour = max(0.0, min(3.0, final_per_hour))
                
                # Update data
                st.session_state.data.at[orig_data_idx, "hours"] = final_hours
                st.session_state.data.at[orig_data_idx, "score"] = final_score
                st.session_state.data.at[orig_data_idx, "per_hour_utility"] = final_per_hour
                
                # Update calendar data
                cal_mask = (
                    (st.session_state.calendar_data["date"] == data_row["date"]) &
                    (st.session_state.calendar_data["svc"] == data_row["svc"]) &
                    (st.session_state.calendar_data["period"] == data_row["period"])
                )
                st.session_state.calendar_data.loc[cal_mask, "hours"] = final_hours
                st.session_state.calendar_data.loc[cal_mask, "score"] = final_score
                st.session_state.calendar_data.loc[cal_mask, "per_hour_utility"] = final_per_hour
                
                # Only add to change log if should_count is True
                if should_count:
                    st.session_state.change_log.append({
                        "time": datetime.now().isoformat(),
                        "date": data_row["date"],
                        "svc": data_row["svc"],
                        "period": data_row["period"],
                        "old_score": old_score,
                        "new_score": final_score,
                        "old_hours": old_hours,
                        "new_hours": final_hours
                    })
                    applied += 1
        
        if applied > 0:
            st.session_state.edit_data = prepare_edit_data(st.session_state.data)
            st.session_state.total_applied_changes += applied
            st.session_state.form_key += 1
            
            st.success(f"✅ Applied {applied} changes! Remember to save before logging out.")
            st.rerun()
        else:
            st.info("No changes detected")

if __name__ == "__main__":
    main()
