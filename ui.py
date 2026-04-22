# ui.py — Movie Recommendation System with Black & Gold Theme + Animations
# Run with: streamlit run ui.py

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
import ast
import base64
import time
from PIL import Image
import io
import random

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG — Must be first Streamlit command
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="🎬 CINEMASCOPE | Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════
# CUSTOM CSS — BLACK & GOLD THEME WITH ANIMATIONS
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Main container */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0a0a0a 100%);
        color: #e0e0e0;
    }
    
    /* Animated gradient overlay */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle at 20% 50%, rgba(255, 210, 0, 0.08) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
        animation: pulseGlow 8s ease-in-out infinite;
    }
    
    @keyframes pulseGlow {
        0%, 100% { opacity: 0.4; }
        50% { opacity: 0.8; }
    }
    
    /* Floating film strip animation */
    .film-strip {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 60px;
        background: repeating-linear-gradient(90deg, 
            rgba(255,210,0,0.15) 0px, 
            rgba(255,210,0,0.15) 20px,
            transparent 20px,
            transparent 40px);
        z-index: 0;
        pointer-events: none;
        animation: slideStrip 20s linear infinite;
    }
    
    @keyframes slideStrip {
        0% { background-position: 0 0; }
        100% { background-position: 80px 0; }
    }
    
    /* Title styling */
    .main-title {
        text-align: center;
        font-family: 'Orbitron', monospace;
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 50%, #FFD700 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 30px rgba(255, 215, 0, 0.3);
        letter-spacing: 3px;
        animation: titleGlow 3s ease-in-out infinite;
        margin-bottom: 10px;
    }
    
    @keyframes titleGlow {
        0%, 100% { text-shadow: 0 0 20px rgba(255, 215, 0, 0.3); }
        50% { text-shadow: 0 0 50px rgba(255, 215, 0, 0.6); }
    }
    
    .sub-title {
        text-align: center;
        color: #c0c0c0;
        font-size: 1rem;
        letter-spacing: 2px;
        margin-bottom: 30px;
        font-weight: 300;
    }
    
    /* Movie Cards with hover animation */
    .movie-card {
        background: linear-gradient(135deg, rgba(20,20,30,0.95) 0%, rgba(10,10,20,0.98) 100%);
        border: 1px solid rgba(255, 215, 0, 0.3);
        border-radius: 20px;
        padding: 20px;
        margin: 12px 0;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        position: relative;
        overflow: hidden;
    }
    
    .movie-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,215,0,0.1), transparent);
        transition: left 0.6s ease;
    }
    
    .movie-card:hover::before {
        left: 100%;
    }
    
    .movie-card:hover {
        border-color: #FFD700;
        transform: translateY(-6px) scale(1.01);
        box-shadow: 0 12px 30px rgba(255, 215, 0, 0.2);
    }
    
    .movie-rank {
        font-size: 2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
    }
    
    .movie-title-card {
        font-size: 1.2rem;
        font-weight: 700;
        color: #fff;
        margin-left: 10px;
    }
    
    .score-badge {
        background: linear-gradient(90deg, #FFD700, #FFA500);
        color: #0a0a0a;
        font-weight: 800;
        padding: 5px 16px;
        border-radius: 30px;
        font-size: 0.85rem;
        display: inline-block;
        margin-top: 8px;
    }
    
    .cluster-badge {
        background: rgba(255,215,0,0.15);
        color: #FFD700;
        padding: 5px 14px;
        border-radius: 30px;
        font-size: 0.8rem;
        border: 1px solid rgba(255,215,0,0.3);
        margin-left: 8px;
    }
    
    .same-cluster {
        background: rgba(46,204,113,0.2);
        color: #2ecc71;
        border-color: #2ecc71;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        border-left: 5px solid #FFD700;
        padding-left: 16px;
        margin: 30px 0 20px 0;
        letter-spacing: -0.5px;
    }
    
    /* Stat boxes */
    .stat-box {
        background: rgba(0,0,0,0.6);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 18px;
        text-align: center;
        border: 1px solid rgba(255,215,0,0.25);
        transition: all 0.3s ease;
    }
    
    .stat-box:hover {
        border-color: #FFD700;
        transform: scale(1.02);
    }
    
    .stat-number {
        font-size: 2.2rem;
        font-weight: 900;
        color: #FFD700;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #aaa;
        margin-top: 6px;
    }
    
    /* Compare boxes */
    .compare-box {
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 700;
        backdrop-filter: blur(8px);
        animation: fadeInUp 0.6s ease;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .high-sim { background: rgba(46,204,113,0.2); border: 2px solid #2ecc71; color: #2ecc71; }
    .mid-sim { background: rgba(241,196,15,0.2); border: 2px solid #f1c40f; color: #f1c40f; }
    .low-sim { background: rgba(231,76,60,0.2); border: 2px solid #e74c3c; color: #e74c3c; }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: rgba(5,5,15,0.95);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255,215,0,0.2);
    }
    
    section[data-testid="stSidebar"] .sidebar-content {
        background: transparent;
    }
    
    /* Buttons with animation */
    .stButton > button {
        background: linear-gradient(90deg, #FFD700, #FFA500);
        color: #0a0a0a;
        font-weight: 800;
        border: none;
        border-radius: 40px;
        padding: 12px 32px;
        font-size: 1rem;
        width: 100%;
        transition: all 0.3s ease;
        cursor: pointer;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(255, 215, 0, 0.4);
        background: linear-gradient(90deg, #FFE44D, #FFB347);
    }
    
    .stButton > button:active {
        transform: translateY(1px);
    }
    
    /* Input fields */
    .stSelectbox > div > div, .stTextInput > div > div {
        background: rgba(20,20,40,0.8) !important;
        border: 1px solid rgba(255,215,0,0.4) !important;
        color: white !important;
        border-radius: 30px !important;
        transition: all 0.3s;
    }
    
    .stSelectbox > div > div:hover, .stTextInput > div > div:hover {
        border-color: #FFD700 !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #aaa;
        font-weight: 600;
        border-radius: 30px;
        padding: 8px 24px;
        transition: all 0.3s;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(255,215,0,0.2), rgba(255,165,0,0.1));
        color: #FFD700 !important;
        border-bottom: none;
        border-radius: 30px;
    }
    
    /* Running text animation */
    .running-text {
        position: fixed;
        bottom: 20px;
        left: 0;
        width: 100%;
        background: rgba(0,0,0,0.7);
        backdrop-filter: blur(5px);
        padding: 8px 0;
        z-index: 100;
        overflow: hidden;
        white-space: nowrap;
        border-top: 1px solid rgba(255,215,0,0.3);
        border-bottom: 1px solid rgba(255,215,0,0.3);
        pointer-events: none;
    }
    
    .running-text span {
        display: inline-block;
        animation: marquee 25s linear infinite;
        font-size: 0.85rem;
        color: #FFD700;
        letter-spacing: 2px;
    }
    
    @keyframes marquee {
        0% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }
    
    /* Poster floating effect */
    @keyframes floatPoster {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-15px) rotate(1deg); }
    }
    
    /* Hide default elements */
    #MainMenu, footer { visibility: hidden; }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #1a1a1a;
    }
    ::-webkit-scrollbar-thumb {
        background: #FFD700;
        border-radius: 10px;
    }
</style>

<div class="film-strip"></div>
<div class="running-text">
    <span>🎬 CINEMASCOPE • Movie recommandation system • Krisha - 20240802247 • Vikram - 20240802261 • Aditya - 20240802267 • 🎬</span>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# LOAD MODEL FILES
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    """Load all pickle files once and cache them."""
    try:
        final_df   = pickle.load(open('output/movies.pkl',     'rb'))
        similarity = pickle.load(open('output/similarity.pkl', 'rb'))
        kmeans     = pickle.load(open('output/kmeans.pkl',     'rb'))
        vectors    = pickle.load(open('output/vectors.pkl',    'rb'))
        return final_df, similarity, kmeans, vectors, True
    except FileNotFoundError as e:
        return None, None, None, None, False

final_df, similarity, kmeans, vectors, loaded = load_models()

if not loaded:
    st.error("❌ Model files not found! Please run `recommender.py` first.")
    st.code("python recommender.py", language="bash")
    st.stop()

# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════

st.markdown('<div class="main-title">🎬 CINEMASCOPE</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-POWERED MOVIE RECOMMENDATION ENGINE</div>', unsafe_allow_html=True)
st.markdown("---")

# ═══════════════════════════════════════════════════════════════
# SIDEBAR — STATS & SETTINGS
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ CINEMA SETTINGS")
    
    top_n = st.slider("🎯 Number of Recommendations", 3, 12, 6)
    
    same_cluster = st.checkbox("🎭 Same Cluster Only", value=False,
                                help="Show recommendations only from the same K-Means cluster")
    
    st.markdown("---")
    st.markdown("### 📊 CINEMATIC DATABASE")
    
    total_movies   = len(final_df)
    total_clusters = final_df['cluster'].nunique()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{total_movies}</div>
            <div class="stat-label">🎬 Movies</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{total_clusters}</div>
            <div class="stat-label">✨ Clusters</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🔍 SEARCH VAULT")
    search_query = st.text_input("Find a movie", placeholder="e.g., Inception")
    if search_query:
        results = final_df[
            final_df['title'].str.contains(search_query, case=False, na=False)
        ]['title'].values[:8]
        if len(results) > 0:
            st.markdown("**🎞️ Matches:**")
            for r in results:
                st.markdown(f"• {r}")
        else:
            st.warning("No matches in the archive.")
    
    st.markdown("---")
    st.markdown("### 🧠 AI TECHNIQUES")
    st.markdown("""
    <div style="font-size:0.85rem">
    ✅ CountVectorizer<br>
    ✅ Cosine Similarity<br>
    ✅ K-Means Clustering<br>
    ✅ L2 Normalization<br>
    ✅ PCA Visualization<br>
    ✅ Star Schema
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 RECOMMEND ENGINE",
    "⚖️ CINEMA COMPARE",
    "🔭 CLUSTER EXPLORER",
    "📊 VISUAL INSIGHTS"
])

# ───────────────────────────────────────────────────────────────
# TAB 1 — RECOMMEND
# ───────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">🎯 GET PERSONALIZED RECOMMENDATIONS</div>',
                unsafe_allow_html=True)
    
    selected_movie = st.selectbox(
        "🌟 Select your favorite movie:",
        sorted(final_df['title'].values),
        index=0
    )
    
    if st.button("🎬 GENERATE RECOMMENDATIONS", use_container_width=True):
        with st.spinner("🎞️ Analyzing cinematic DNA..."):
            time.sleep(0.5)  # Smooth animation effect
            
            idx = final_df[final_df['title'] == selected_movie].index[0]
            cluster = int(final_df.iloc[idx]['cluster'])
            
            distances = list(enumerate(similarity[idx]))
            sorted_movies = sorted(distances, key=lambda x: x[1], reverse=True)
            
            if same_cluster:
                sorted_movies = [(i, s) for i, s in sorted_movies if final_df.iloc[i]['cluster'] == cluster]
            
            # Movie info display
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-number">🎬</div>
                    <div class="stat-label">{selected_movie[:25]}</div>
                </div>""", unsafe_allow_html=True)
            with col_b:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-number">{cluster}</div>
                    <div class="stat-label">🎭 Cluster ID</div>
                </div>""", unsafe_allow_html=True)
            with col_c:
                cluster_size = len(final_df[final_df['cluster'] == cluster])
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-number">{cluster_size}</div>
                    <div class="stat-label">📀 Movies in Cluster</div>
                </div>""", unsafe_allow_html=True)
            
            st.markdown('<div class="section-header">⭐ TOP RECOMMENDATIONS</div>',
                        unsafe_allow_html=True)
            
            shown = 0
            for index, score in sorted_movies[1:]:
                if shown >= top_n:
                    break
                
                rec_title = final_df.iloc[index]['title']
                rec_cluster = int(final_df.iloc[index]['cluster'])
                is_same = rec_cluster == cluster
                cluster_cls = "same-cluster" if is_same else ""
                cluster_lbl = "✅ SAME CLUSTER" if is_same else f"🔄 CLUSTER {rec_cluster}"
                
                st.markdown(f"""
                <div class="movie-card">
                    <span class="movie-rank">#{shown+1}</span>
                    <span class="movie-title-card">{rec_title}</span><br>
                    <span class="score-badge">🎯 SCORE: {round(score, 4)}</span>
                    <span class="cluster-badge {cluster_cls}">{cluster_lbl}</span>
                </div>
                """, unsafe_allow_html=True)
                
                shown += 1
            
            # Similarity visualization
            st.markdown('<div class="section-header">📊 SIMILARITY METRICS</div>',
                        unsafe_allow_html=True)
            
            chart_titles = []
            chart_scores = []
            chart_colors = []
            
            for index, score in sorted_movies[1:top_n+1]:
                chart_titles.append(final_df.iloc[index]['title'][:30])
                chart_scores.append(round(score, 4))
                chart_colors.append('#FFD700' if final_df.iloc[index]['cluster'] == cluster else '#FF8C00')
            
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_alpha(0)
            ax.set_facecolor('#0a0a1a')
            bars = ax.barh(chart_titles[::-1], chart_scores[::-1],
                           color=chart_colors[::-1], edgecolor='#FFD700', linewidth=0.8)
            for bar, score in zip(bars, chart_scores[::-1]):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{score:.4f}', va='center', color='#FFD700', fontsize=10, fontweight='bold')
            ax.set_xlabel('COSINE SIMILARITY SCORE', color='white', fontweight='bold')
            ax.tick_params(colors='white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for spine in ['bottom','left']:
                ax.spines[spine].set_color('#FFD700')
            st.pyplot(fig)
            plt.close()

# ───────────────────────────────────────────────────────────────
# TAB 2 — COMPARE MOVIES
# ───────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">⚖️ CINEMATIC DUEL</div>',
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        movie1 = st.selectbox("🎬 FIRST MOVIE", sorted(final_df['title'].values), key='m1')
    with col2:
        movie2 = st.selectbox("🎭 SECOND MOVIE", sorted(final_df['title'].values),
                               index=min(1, len(final_df)-1), key='m2')
    
    if st.button("⚖️ COMPARE NOW", use_container_width=True):
        idx1 = final_df[final_df['title'] == movie1].index[0]
        idx2 = final_df[final_df['title'] == movie2].index[0]
        
        score = round(float(similarity[idx1][idx2]), 4)
        cluster1 = int(final_df.iloc[idx1]['cluster'])
        cluster2 = int(final_df.iloc[idx2]['cluster'])
        
        if score >= 0.4:
            verdict = "🟢 HIGHLY SIMILAR"
            css_cls = "high-sim"
        elif score >= 0.15:
            verdict = "🟡 MODERATELY SIMILAR"
            css_cls = "mid-sim"
        else:
            verdict = "🔴 DISTINCTLY DIFFERENT"
            css_cls = "low-sim"
        
        st.markdown(f"""
        <div class="compare-box {css_cls}">
            <i class="fas fa-chart-line"></i> SIMILARITY SCORE: {score} &nbsp;|&nbsp; {verdict}
        </div>""", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{cluster1}</div>
                <div class="stat-label">{movie1[:20]} Cluster</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            match_icon = "✅ MATCH" if cluster1 == cluster2 else "❌ NO MATCH"
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{match_icon}</div>
                <div class="stat-label">Cluster Match</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{cluster2}</div>
                <div class="stat-label">{movie2[:20]} Cluster</div>
            </div>""", unsafe_allow_html=True)
        
        # Common keywords
        tags1 = set(final_df.iloc[idx1]['tags'].split())
        tags2 = set(final_df.iloc[idx2]['tags'].split())
        stop = {'the','a','an','in','of','and','to','is','it','on','with','his','her','for','as'}
        common = sorted([t for t in (tags1 & tags2) if t not in stop])[:20]
        
        st.markdown('<div class="section-header">🔗 SHARED DNA</div>', unsafe_allow_html=True)
        if common:
            tag_html = " ".join([
                f'<span style="background:rgba(255,210,0,0.15);border:1px solid #FFD700;border-radius:30px;padding:6px 14px;margin:5px;display:inline-block;color:#FFD700;font-size:0.85rem;">{t}</span>'
                for t in common[:15]
            ])
            st.markdown(tag_html, unsafe_allow_html=True)
        else:
            st.info("🎭 No common cinematic DNA found between these movies.")

# ───────────────────────────────────────────────────────────────
# TAB 3 — EXPLORE CLUSTERS
# ───────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">🔭 CLUSTER EXPLORATION</div>',
                unsafe_allow_html=True)
    
    selected_cluster = st.selectbox(
        "🎪 SELECT CLUSTER TO EXPLORE:",
        sorted(final_df['cluster'].unique()),
        format_func=lambda x: f"✨ CLUSTER {x}"
    )
    
    cluster_movies = final_df[final_df['cluster'] == selected_cluster]['title'].values
    
    st.markdown(f"""
    <div class="stat-box" style="margin-bottom:25px;">
        <div class="stat-number">{len(cluster_movies)}</div>
        <div class="stat-label">🎬 Movies in Cluster {selected_cluster}</div>
    </div>""", unsafe_allow_html=True)
    
    cluster_search = st.text_input("🔎 Filter within cluster", placeholder="Type to search...")
    if cluster_search:
        cluster_movies = [m for m in cluster_movies if cluster_search.lower() in m.lower()]
    
    cols = st.columns(3)
    for i, title in enumerate(cluster_movies[:30]):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="movie-card" style="padding:12px; margin:5px 0;">
                <span style="color:#FFD700; font-weight:800;">#{i+1}</span>
                <span style="color:white; margin-left:8px;">{title[:35]}</span>
            </div>""", unsafe_allow_html=True)
    
    if len(cluster_movies) > 30:
        st.info(f"🎞️ Showing 30 of {len(cluster_movies)} movies in this cinematic universe.")

# ───────────────────────────────────────────────────────────────
# TAB 4 — VISUALIZATIONS
# ───────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">📊 CINEMATIC DATA VISUALIZATION</div>',
                unsafe_allow_html=True)
    
    viz_choice = st.selectbox("📈 SELECT VISUALIZATION:", [
        "🏔️ Cluster Size Distribution",
        "🎭 Top 15 Most Common Genres",
        "🍕 Movies per Cluster (Pie Chart)",
    ])
    
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor('#0a0a1a')
    ax.set_facecolor('#0a0a1a')
    
    GOLD_GRADIENT = ['#FFD700', '#FFC107', '#FFB300', '#FFA000', '#FF8F00']
    
    if viz_choice == "🏔️ Cluster Size Distribution":
        counts = final_df['cluster'].value_counts().sort_index()
        bars = ax.bar(counts.index, counts.values,
                      color=[GOLD_GRADIENT[i % len(GOLD_GRADIENT)] for i in range(len(counts))],
                      edgecolor='#FFD700', linewidth=1)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    str(int(bar.get_height())), ha='center', color='#FFD700', fontsize=10, fontweight='bold')
        ax.set_xlabel('CLUSTER ID', color='white', fontweight='bold')
        ax.set_ylabel('NUMBER OF MOVIES', color='white', fontweight='bold')
        ax.set_title('MOVIES PER K-MEANS CLUSTER', color='#FFD700', fontsize=14, fontweight='bold')
        
    elif viz_choice == "🎭 Top 15 Most Common Genres":
        try:
            movies_raw = pd.read_csv('dataset/tmdb_5000_movies.csv')
            all_g = []
            for g_str in movies_raw['genres']:
                try:
                    for g in ast.literal_eval(g_str):
                        all_g.append(g['name'])
                except:
                    pass
            genre_counts = Counter(all_g).most_common(15)
        except:
            all_genres = []
            for tags in final_df['tags']:
                words = tags.split()
                all_genres.extend(words[:5])
            genre_counts = Counter(all_genres).most_common(15)
        
        labels = [g[0] for g in genre_counts]
        values = [g[1] for g in genre_counts]
        bars = ax.barh(labels[::-1], values[::-1],
                       color=[GOLD_GRADIENT[i % len(GOLD_GRADIENT)] for i in range(len(labels))],
                       edgecolor='#FFD700', linewidth=0.8)
        for bar, val in zip(bars, values[::-1]):
            ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2,
                    str(val), va='center', color='#FFD700', fontsize=10, fontweight='bold')
        ax.set_xlabel('MOVIE COUNT', color='white', fontweight='bold')
        ax.set_title('TOP 15 GENRES', color='#FFD700', fontsize=14, fontweight='bold')
        
    elif viz_choice == "🍕 Movies per Cluster (Pie Chart)":
        counts = final_df['cluster'].value_counts().sort_index()
        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=[f'C{i}' for i in counts.index],
            autopct='%1.1f%%',
            colors=GOLD_GRADIENT * 3,
            textprops={'color': 'white', 'fontsize': 10, 'fontweight': 'bold'},
            wedgeprops={'linewidth': 2, 'edgecolor': '#0a0a1a'}
        )
        for at in autotexts:
            at.set_color('#0a0a1a')
            at.set_fontsize(9)
            at.set_fontweight('bold')
        ax.set_title('CLUSTER DISTRIBUTION', color='#FFD700', fontsize=14, fontweight='bold')
    
    for spine in ax.spines.values():
        spine.set_color('#FFD700')
    ax.tick_params(colors='white')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#666; font-size:0.8rem; padding:20px;">
    🎬 CINEMASCOPE · AI MOVIE RECOMMENDATION SYSTEM · POWERED BY PYTHON & STREAMLIT
</div>
""", unsafe_allow_html=True)