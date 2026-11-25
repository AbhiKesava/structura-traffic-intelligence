"""
STRUCTURA — TRAFFIC INTELLIGENCE (Streamlit Companion App)
Matches the Structura design system from Image 1 & Image 2.
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Structura — Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Obsidian & Sage Green CSS matching Image 1
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
        background-color: #090c0a !important;
        color: #f2f6f3 !important;
    }
    
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #0e120f !important;
        border-right: 1px solid #1a221c !important;
        padding-top: 1rem;
    }
    
    section[data-testid="stSidebar"] * {
        color: #7d8f82 !important;
    }
    
    .structura-card {
        background: #121613;
        border: 1px solid #1e2620;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 16px;
    }
    
    .sage-card {
        background: #b8cca6 !important;
        border: 1px solid #b8cca6 !important;
        border-radius: 14px;
        padding: 18px 20px;
        color: #0e140f !important;
    }
    
    .metric-caption {
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.1em;
        color: #526156;
        text-transform: uppercase;
    }
    
    .big-num {
        font-size: 34px;
        font-weight: 800;
        color: #f2f6f3;
        line-height: 1.1;
    }
    
    .badge-high {
        background: rgba(249, 115, 22, 0.18);
        color: #fb923c;
        font-size: 10px;
        font-weight: 800;
        padding: 3px 9px;
        border-radius: 6px;
    }
    
    .badge-mod {
        background: rgba(217, 119, 6, 0.16);
        color: #fbbf24;
        font-size: 10px;
        font-weight: 800;
        padding: 3px 9px;
        border-radius: 6px;
    }
    
    .badge-low {
        background: rgba(126, 168, 114, 0.16);
        color: #a3c997;
        font-size: 10px;
        font-weight: 800;
        padding: 3px 9px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar (Image 1)
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
        <div style="width: 32px; height: 32px; background: #141b16; border: 1px solid #28372b; border-radius: 8px; display: flex; align-items: center; justify-content: center;">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <rect width="24" height="24" rx="6" fill="#141b16" />
                <path d="M 6 18 L 12 5 L 18 18 M 8.5 13.5 L 15.5 13.5" stroke="#b8cca6" stroke-width="2.5" stroke-linecap="round"/>
            </svg>
        </div>
        <div>
            <div style="font-size: 13px; font-weight: 800; color: #ffffff; letter-spacing: 0.12em;">STRUCTURA</div>
            <div style="font-size: 8.5px; font-weight: 700; color: #526156; letter-spacing: 0.14em;">TRAFFIC INTELLIGENCE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("WORKSPACE")
    st.markdown("**● Overview**")
    st.markdown("• Live traffic map")
    st.markdown("• Forecast explorer")
    st.markdown("• Route comparison")
    st.markdown("• Saved corridors")

    st.caption("INSIGHTS")
    st.markdown("• Reports")
    st.markdown("• Data sources")

    st.markdown("---")
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; padding: 10px; background: #121714; border: 1px solid #1a221c; border-radius: 12px;">
        <div style="width: 28px; height: 28px; border-radius: 50%; background: #1f2a22; color: #b8cca6; font-size: 11px; font-weight: 800; display: flex; align-items: center; justify-content: center; border: 1px solid #314235;">AI</div>
        <div>
            <div style="font-size: 12px; font-weight: 700; color: #ffffff;">Alex Taylor</div>
            <div style="font-size: 10px; color: #7d8f82;">Traffic analyst</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main Header (Image 1)
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("<div style='font-size: 10px; font-weight: 800; letter-spacing: 0.14em; color: #526156;'>MONDAY, NOVEMBER 9, 2026</div>", unsafe_allow_html=True)
    st.markdown("<h1 style='font-size: 28px; font-weight: 800; color: #ffffff; margin-top: 2px; margin-bottom: 4px;'>Good morning, Alex.</h1>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 13px; color: #7d8f82;'>Here's the traffic outlook across your saved corridors.</div>", unsafe_allow_html=True)

with col_h2:
    st.markdown("""
    <div style='display: flex; justify-content: flex-end; align-items: center; gap: 10px; margin-top: 14px;'>
        <div style='background: #121613; border: 1px solid #1e2620; padding: 6px 14px; border-radius: 20px; font-size: 11px; color: #7d8f82;'>
            <span style='color: #10b981; font-weight: bold;'>●</span> Data updated 2 min ago
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# 4 KPI Summary Cards (Image 1)
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="structura-card">
        <div class="metric-caption">NETWORK SPEED</div>
        <div style="display: flex; align-items: baseline; justify-content: space-between; margin: 8px 0;">
            <span class="big-num">47 <span style="font-size: 14px; color: #7d8f82;">mph</span></span>
            <svg width="50" height="24" viewBox="0 0 50 24">
                <path d="M 2 20 L 14 18 L 24 19 L 36 10 L 48 4" fill="none" stroke="#7ea872" stroke-width="2"/>
            </svg>
        </div>
        <div style="font-size: 11px; color: #7d8f82;"><span style="color: #7ea872; font-weight: 700;">↑ 6%</span> vs. yesterday</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="structura-card">
        <div class="metric-caption">ACTIVE SLOWDOWNS</div>
        <div class="big-num" style="margin: 8px 0;">12</div>
        <div style="font-size: 11px; color: #7d8f82;">
            <span style="color: #ef4444;">●</span> 3 severe · <span style="color: #f97316;">●</span> 5 high · <span style="color: #fbbf24;">●</span> 4 moderate
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="structura-card">
        <div class="metric-caption">AVG. DELAY</div>
        <div style="display: flex; align-items: baseline; margin: 8px 0;">
            <span class="big-num">+8 <span style="font-size: 14px; color: #7d8f82;">min</span></span>
        </div>
        <div style="font-size: 11px; color: #7d8f82;"><span style="color: #fbbf24; font-weight: 700;">↑ 2 min</span> in the last hour</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="sage-card">
        <div style="font-size: 10px; font-weight: 800; letter-spacing: 0.1em; color: #2b392d;">BEST DEPARTURE WINDOW</div>
        <div style="font-size: 26px; font-weight: 800; color: #0e140f; margin: 10px 0 6px 0;">10:30 – 11:00</div>
        <div style="font-size: 11.5px; color: #2c3c2e; font-weight: 600;">Typically 18% faster than now</div>
    </div>
    """, unsafe_allow_html=True)

# Middle Section: Corridor Conditions & Speed Forecast (Image 1)
col_mid1, col_mid2 = st.columns([1.5, 1])

with col_mid1:
    st.markdown("""
    <div class="structura-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <div>
                <div style="font-size: 14.5px; font-weight: 700; color: #ffffff;">Corridor conditions</div>
                <div style="font-size: 11.5px; color: #7d8f82;">Los Angeles · Live and forecasted traffic</div>
            </div>
            <div style="background: #0d120f; border: 1px solid #1e2620; padding: 4px 10px; border-radius: 8px; font-size: 11px; color: #f2f6f3;">
                Next 30 minutes ⌄
            </div>
        </div>
        <svg viewBox="0 0 680 290" style="width: 100%; height: 260px; background: #101512; border-radius: 8px;">
            <!-- Grid Lines -->
            <g stroke="#17221b" stroke-width="1">
                <line x1="0" y1="60" x2="680" y2="60" />
                <line x1="0" y1="120" x2="680" y2="120" />
                <line x1="0" y1="180" x2="680" y2="180" />
                <line x1="0" y1="240" x2="680" y2="240" />
                <line x1="120" y1="0" x2="120" y2="290" />
                <line x1="240" y1="0" x2="240" y2="290" />
                <line x1="360" y1="0" x2="360" y2="290" />
                <line x1="480" y1="0" x2="480" y2="290" />
                <line x1="600" y1="0" x2="600" y2="290" />
            </g>
            <!-- Orange Cross Highway -->
            <path d="M 40 145 C 160 155, 280 180, 425 160 C 490 148, 560 190, 640 225" stroke="#d97746" stroke-width="8" stroke-linecap="round" fill="none" opacity="0.88"/>
            <!-- Sage Green Highway -->
            <path d="M 40 225 C 150 220, 260 184, 425 160 C 495 142, 560 124, 650 122" stroke="#b2c6a0" stroke-width="8" stroke-linecap="round" fill="none"/>
            <!-- Nodes -->
            <circle cx="265" cy="184" r="7" fill="#101512" stroke="#b2c6a0" stroke-width="3" />
            <circle cx="425" cy="160" r="8" fill="#101512" stroke="#f97316" stroke-width="3.5" />
            <circle cx="425" cy="160" r="3" fill="#f97316" />
            <!-- Floating Tooltip -->
            <g transform="translate(360, 108)">
                <rect width="130" height="42" rx="7" fill="#0d120f" stroke="#253228" stroke-width="1.2"/>
                <text x="65" y="16" text-anchor="middle" font-size="9.5" font-weight="700" fill="#85968a">I-405 NORTH</text>
                <text x="65" y="32" text-anchor="middle" font-size="12" font-weight="800" fill="#ffffff">32 mph &ndash; <tspan fill="#fb923c">HIGH</tspan></text>
            </g>
        </svg>
    </div>
    """, unsafe_allow_html=True)

with col_mid2:
    st.markdown("""
    <div class="structura-card">
        <div style="font-size: 14.5px; font-weight: 700; color: #ffffff;">Speed forecast</div>
        <div style="font-size: 11.5px; color: #7d8f82; margin-bottom: 8px;">Average network speed</div>
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
            <span style="font-size: 28px; font-weight: 800; color: #ffffff;">47 mph</span>
            <span style="background: rgba(126, 168, 114, 0.16); color: #a3c997; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 6px;">+6% today</span>
        </div>
        <svg viewBox="0 0 340 170" style="width: 100%; height: 160px;">
            <g stroke="#1a231d" stroke-width="1" stroke-dasharray="2 3">
                <line x1="26" y1="20" x2="330" y2="20" />
                <line x1="26" y1="60" x2="330" y2="60" />
                <line x1="26" y1="100" x2="330" y2="100" />
                <line x1="26" y1="140" x2="330" y2="140" />
            </g>
            <g font-size="9" fill="#58695d" text-anchor="end">
                <text x="22" y="24">65</text>
                <text x="22" y="64">50</text>
                <text x="22" y="104">35</text>
                <text x="22" y="144">20</text>
            </g>
            <!-- Baseline -->
            <path d="M 30 105 C 90 98, 180 90, 330 75" fill="none" stroke="#4a5a4f" stroke-width="1.8" stroke-dasharray="4 3" />
            <!-- Predicted Wavy Curve -->
            <path d="M 30 102 C 60 102, 85 82, 110 80 C 135 78, 160 105, 185 100 C 215 92, 240 62, 270 58 C 295 54, 315 38, 330 22" fill="none" stroke="#b2c6a0" stroke-width="2.6" stroke-linecap="round"/>
            <g font-size="9" fill="#66776b" text-anchor="middle">
                <text x="35" y="164">NOW</text>
                <text x="135" y="164">+15M</text>
                <text x="235" y="164">+30M</text>
                <text x="320" y="164">+60M</text>
            </g>
        </svg>
    </div>
    """, unsafe_allow_html=True)

# Corridors to watch (Image 1)
st.markdown("""
<div class="structura-card">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
        <span style="font-size: 14px; font-weight: 700; color: #ffffff;">Corridors to watch</span>
        <span style="font-size: 11.5px; color: #7d8f82;">View all corridors &rarr;</span>
    </div>
    <div style="border-top: 1px solid #1a221c; padding: 12px 0; display: flex; justify-content: space-between; align-items: center;">
        <span style="font-weight: 600; color: #ffffff; width: 30%;">I-405 North - Sepulveda Pass</span>
        <span style="color: #7d8f82; width: 15%;">32 mph</span>
        <span class="badge-high">HIGH</span>
        <span style="color: #7d8f82; width: 30%;">Expected to clear in 24 min</span>
        <span style="color: #f97316; font-size: 16px;">&rarr;</span>
    </div>
    <div style="border-top: 1px solid #1a221c; padding: 12px 0; display: flex; justify-content: space-between; align-items: center;">
        <span style="font-weight: 600; color: #ffffff; width: 30%;">US-101 South - Hollywood</span>
        <span style="color: #7d8f82; width: 15%;">41 mph</span>
        <span class="badge-mod">MODERATE</span>
        <span style="color: #7d8f82; width: 30%;">Steady for the next hour</span>
        <span style="color: #7d8f82; font-size: 16px;">&rarr;</span>
    </div>
    <div style="border-top: 1px solid #1a221c; padding: 12px 0; display: flex; justify-content: space-between; align-items: center;">
        <span style="font-weight: 600; color: #ffffff; width: 30%;">I-10 East - Downtown LA</span>
        <span style="color: #7d8f82; width: 15%;">52 mph</span>
        <span class="badge-low">LOW</span>
        <span style="color: #7d8f82; width: 30%;">Flowing normally</span>
        <span style="color: #7d8f82; font-size: 16px;">&rarr;</span>
    </div>
</div>
""", unsafe_allow_html=True)
