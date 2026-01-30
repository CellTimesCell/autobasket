"""
AutoBasket - NBA Dashboard
==========================
Полноценный веб-интерфейс для мониторинга NBA ставок

Показывает:
- Текущий банкролл и историю
- Сегодняшние игры и ставки  
- Историю выигрышей/проигрышей
- Статистику модели
- Аномалии в live-играх
- Прогнозы экспертов

Запуск: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import sqlite3
import json
import sys
import os
from pathlib import Path

# Путь к проекту
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

# === PAGE CONFIG ===

st.set_page_config(
    page_title="AutoBasket NBA - Betting Intelligence",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === STYLES ===

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 5px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 14px;
        opacity: 0.8;
    }
    .win-card { 
        background: linear-gradient(135deg, #155724 0%, #28a745 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 5px 0;
    }
    .loss-card { 
        background: linear-gradient(135deg, #721c24 0%, #dc3545 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 5px 0;
    }
    .pending-card { 
        background: linear-gradient(135deg, #856404 0%, #ffc107 100%);
        padding: 15px;
        border-radius: 10px;
        color: black;
        margin: 5px 0;
    }
    .anomaly-alert {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff8e8e 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .game-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .bet-placed {
        border-left: 5px solid #28a745;
    }
    .no-bet {
        border-left: 5px solid #6c757d;
    }
</style>
""", unsafe_allow_html=True)


# === DATABASE FUNCTIONS ===

def get_db_path(db_name: str) -> Path:
    """Возвращает путь к базе данных"""
    return PROJECT_DIR / db_name


def load_json_file(filename: str) -> list:
    """Загружает JSON файл"""
    filepath = PROJECT_DIR / filename
    if filepath.exists():
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except:
            pass
    return []


def save_json_file(filename: str, data: list):
    """Сохраняет JSON файл"""
    filepath = PROJECT_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def get_knowledge_base_stats() -> dict:
    """Статистика Knowledge Base"""
    db_path = get_db_path("team_knowledge.db")
    if not db_path.exists():
        return {'teams': 0, 'games': 0, 'seasons': 'N/A'}
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM team_profiles")
        teams = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM historical_games")
        games = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(data_from_year), MAX(data_to_year) FROM team_profiles")
        row = cursor.fetchone()
        seasons = f"{row[0]}-{row[1]}" if row and row[0] else "N/A"
        
        conn.close()
        return {'teams': teams, 'games': games, 'seasons': seasons}
    except Exception as e:
        return {'teams': 0, 'games': 0, 'seasons': f'Error: {e}'}


def get_expert_stats() -> dict:
    """Статистика Expert Picks"""
    db_path = get_db_path("expert_picks.db")
    if not db_path.exists():
        return {'experts': 0, 'picks_today': 0, 'sharp': 0}
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM experts")
        experts = cursor.fetchone()[0]
        
        today = date.today().isoformat()
        cursor.execute("SELECT COUNT(*) FROM picks WHERE DATE(timestamp) = ?", (today,))
        picks_today = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM experts WHERE is_sharp = 1")
        sharp = cursor.fetchone()[0]
        
        conn.close()
        return {'experts': experts, 'picks_today': picks_today, 'sharp': sharp}
    except:
        return {'experts': 0, 'picks_today': 0, 'sharp': 0}


def load_bankroll_data() -> dict:
    """Загружает данные банкролла"""
    # Пытаемся прочитать из state файла
    state_file = PROJECT_DIR / "system_state.json"
    if state_file.exists():
        try:
            with open(state_file, 'r') as f:
                data = json.load(f)
                return {
                    'current': data.get('bankroll', 200.0),
                    'initial': data.get('initial_bankroll', 200.0),
                    'peak': data.get('peak_bankroll', 200.0),
                    'history': data.get('bankroll_history', [])
                }
        except:
            pass
    
    return {
        'current': 200.0,
        'initial': 200.0,
        'peak': 200.0,
        'history': []
    }


def load_bets_history() -> list:
    """Загружает историю ставок"""
    return load_json_file("bets_history.json")


def load_predictions() -> list:
    """Загружает предсказания"""
    return load_json_file("predictions.json")


def get_todays_games_from_state() -> list:
    """Получает сегодняшние игры из state"""
    state_file = PROJECT_DIR / "system_state.json"
    if state_file.exists():
        try:
            with open(state_file, 'r') as f:
                data = json.load(f)
                return data.get('todays_games', [])
        except:
            pass
    return []


# === SESSION STATE ===

# Автоматическое обновление каждые 30 секунд
if 'last_auto_refresh' not in st.session_state:
    st.session_state.last_auto_refresh = datetime.now()

# Проверяем нужно ли обновить
time_since_refresh = (datetime.now() - st.session_state.last_auto_refresh).total_seconds()
if time_since_refresh > 30:
    st.session_state.last_auto_refresh = datetime.now()
    st.rerun()

# Загружаем свежие данные при КАЖДОМ рендере (не кешируем!)
bankroll_data = load_bankroll_data()
bets_history = load_bets_history()
todays_games = get_todays_games_from_state()

if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()


# === SIDEBAR ===

with st.sidebar:
    st.title("🏀 AutoBasket")
    st.caption("NBA Betting Intelligence")
    
    st.divider()
    
    # Банкролл - используем свежие данные
    br = bankroll_data
    change_pct = ((br['current'] - br['initial']) / br['initial'] * 100) if br['initial'] > 0 else 0
    
    st.metric(
        "💰 Bankroll",
        f"${br['current']:.2f}",
        f"{change_pct:+.1f}%"
    )
    
    # Knowledge Base
    kb = get_knowledge_base_stats()
    st.metric("📚 Teams in KB", kb['teams'])
    st.metric("🎮 Historical Games", f"{kb['games']:,}")
    
    # Expert picks
    ep = get_expert_stats()
    st.metric("🎤 Experts", ep['experts'])
    st.metric("📊 Picks Today", ep['picks_today'])
    
    st.divider()
    
    # Навигация
    page = st.radio(
        "📍 Navigation",
        ["📊 Overview", "🎯 Today's Games", "📈 Bet History", 
         "🔍 Analytics", "⚠️ Live Monitor", "⚙️ Settings"]
    )
    
    st.divider()
    
    st.caption(f"Updated: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
    st.caption("Auto-refresh: 30s")
    
    if st.button("🔄 Refresh Now"):
        st.session_state.last_refresh = datetime.now()
        st.rerun()


# === PAGES ===

if page == "📊 Overview":
    st.title("📊 NBA Dashboard Overview")
    
    # Главные метрики
    col1, col2, col3, col4 = st.columns(4)
    
    br = bankroll_data  # Используем свежие данные
    bets = bets_history
    
    wins = len([b for b in bets if b.get('status') == 'won'])
    losses = len([b for b in bets if b.get('status') == 'lost'])
    pending = len([b for b in bets if b.get('status') == 'pending'])
    total_profit = sum(b.get('profit', 0) for b in bets if b.get('profit'))
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${br['current']:.2f}</div>
            <div class="metric-label">Current Bankroll</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{win_rate:.1f}%</div>
            <div class="metric-label">Win Rate ({wins}W-{losses}L)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${total_profit:+.2f}</div>
            <div class="metric-label">Total Profit</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        roi = (total_profit / br['initial'] * 100) if br['initial'] > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{roi:+.1f}%</div>
            <div class="metric-label">ROI</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Два столбца
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📈 Bankroll History")
        
        history = br.get('history', [])
        if history and len(history) > 1:
            df = pd.DataFrame(history)
            fig = px.line(df, x='date', y='value',
                         title="Bankroll Over Time")
            fig.update_traces(line_color='#2d5a87', line_width=3)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Demo data
            dates = pd.date_range(end=date.today(), periods=30)
            values = [200 + i * 1.5 + (i % 7) * 2 for i in range(30)]
            df = pd.DataFrame({'date': dates, 'value': values})
            fig = px.line(df, x='date', y='value',
                         title="Bankroll Over Time (Demo)")
            fig.update_traces(line_color='#2d5a87', line_width=3)
            st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.subheader("🎯 Recent Bets")
        
        recent_bets = bets[-10:] if bets else []
        
        if recent_bets:
            for bet in reversed(recent_bets):
                status = bet.get('status', 'pending')
                if status == 'won':
                    card_class = 'win-card'
                    emoji = '✅'
                elif status == 'lost':
                    card_class = 'loss-card'
                    emoji = '❌'
                else:
                    card_class = 'pending-card'
                    emoji = '⏳'
                
                profit = bet.get('profit', 0)
                st.markdown(f"""
                <div class="{card_class}">
                    <b>{emoji} {bet.get('bet_team', 'Unknown')}</b><br>
                    ${bet.get('amount', 0):.2f} @ {bet.get('odds', 1.0):.2f}<br>
                    <small>Edge: {bet.get('edge', 0):.1%} | Profit: ${profit:+.2f}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No bets recorded yet. Start the autonomous system!")
    
    st.divider()
    
    # System Status
    st.subheader("🖥️ System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📚 Knowledge Base", f"{kb['teams']} teams", f"{kb['games']:,} games")
    
    with col2:
        st.metric("🎤 Expert Tracker", f"{ep['experts']} experts", f"{ep['sharp']} sharp")
    
    with col3:
        st.metric("📊 Predictions", len(load_predictions()), "total")
    
    with col4:
        st.metric("⏳ Pending Bets", pending)


elif page == "🎯 Today's Games":
    st.title("🎯 Today's NBA Games")
    
    st.info(f"📅 {date.today().strftime('%A, %B %d, %Y')}")
    
    games = todays_games  # Используем уже загруженные данные
    bets = bets_history
    todays_bets_dict = {b.get('game_id', ''): b for b in bets if b.get('placed_at', '').startswith(date.today().isoformat())}
    
    if games:
        for game in games:
            game_id = game.get('game_id', '')
            bet = todays_bets_dict.get(game_id)
            
            card_class = "game-card bet-placed" if bet else "game-card no-bet"
            
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.markdown(f"### {game.get('away_team', 'Away')} @ {game.get('home_team', 'Home')}")
            
            with col2:
                prob = game.get('predicted_home_prob', 0.5)
                st.metric("Home Win", f"{prob:.1%}")
            
            with col3:
                spread = game.get('spread', 0)
                st.metric("Spread", f"{spread:+.1f}")
            
            with col4:
                if bet:
                    st.success(f"🎯 ${bet['amount']:.0f} on {bet['bet_team']}")
                else:
                    st.info("No bet")
            
            # Details expander
            with st.expander("📊 Details"):
                dcol1, dcol2 = st.columns(2)
                with dcol1:
                    st.write(f"**Home:** {game.get('home_team')}")
                    home_trends = game.get('home_trends', {})
                    st.write(f"Record: {home_trends.get('record', 'N/A')}")
                    st.write(f"Streak: {home_trends.get('current_streak', 'N/A')}")
                
                with dcol2:
                    st.write(f"**Away:** {game.get('away_team')}")
                    away_trends = game.get('away_trends', {})
                    st.write(f"Record: {away_trends.get('record', 'N/A')}")
                    st.write(f"Streak: {away_trends.get('current_streak', 'N/A')}")
            
            st.divider()
    else:
        st.warning("No games loaded. Start the autonomous system to fetch today's games.")
        
        # Demo
        st.subheader("📋 Example View")
        demo = [
            {"away": "Warriors", "home": "Timberwolves", "prob": 0.405, "spread": 2.5, "bet": "away", "amount": 10},
            {"away": "Lakers", "home": "Bulls", "prob": 0.48, "spread": 2.5, "bet": None, "amount": 0},
            {"away": "Trail Blazers", "home": "Celtics", "prob": 0.53, "spread": 2.5, "bet": None, "amount": 0},
        ]
        
        for g in demo:
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                st.markdown(f"### {g['away']} @ {g['home']}")
            with col2:
                st.metric("Home Win", f"{g['prob']:.1%}")
            with col3:
                st.metric("Spread", f"{g['spread']:+.1f}")
            with col4:
                if g['bet']:
                    st.success(f"🎯 ${g['amount']} on {g['bet']}")
                else:
                    st.info("No value")
            st.divider()


elif page == "📈 Bet History":
    st.title("📈 Bet History")
    
    bets = load_bets_history()
    
    if bets:
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            status_filter = st.selectbox("Status", ["All", "Won", "Lost", "Pending"])
        with col2:
            days_back = st.slider("Days back", 1, 90, 30)
        with col3:
            sort_by = st.selectbox("Sort by", ["Date (newest)", "Date (oldest)", "Amount", "Profit"])
        
        # Filter
        cutoff = (datetime.now() - timedelta(days=days_back)).isoformat()
        filtered = [b for b in bets if b.get('placed_at', '') >= cutoff]
        
        if status_filter != "All":
            filtered = [b for b in filtered if b.get('status', '').lower() == status_filter.lower()]
        
        # Sort
        if sort_by == "Date (newest)":
            filtered = sorted(filtered, key=lambda x: x.get('placed_at', ''), reverse=True)
        elif sort_by == "Date (oldest)":
            filtered = sorted(filtered, key=lambda x: x.get('placed_at', ''))
        elif sort_by == "Amount":
            filtered = sorted(filtered, key=lambda x: x.get('amount', 0), reverse=True)
        elif sort_by == "Profit":
            filtered = sorted(filtered, key=lambda x: x.get('profit', 0), reverse=True)
        
        # Stats
        st.divider()
        col1, col2, col3, col4, col5 = st.columns(5)
        
        wins = len([b for b in filtered if b.get('status') == 'won'])
        losses = len([b for b in filtered if b.get('status') == 'lost'])
        total_wagered = sum(b.get('amount', 0) for b in filtered)
        total_profit = sum(b.get('profit', 0) for b in filtered if b.get('profit'))
        
        with col1:
            st.metric("Total Bets", len(filtered))
        with col2:
            st.metric("Wins", wins)
        with col3:
            st.metric("Losses", losses)
        with col4:
            st.metric("Wagered", f"${total_wagered:.2f}")
        with col5:
            st.metric("Profit", f"${total_profit:+.2f}")
        
        st.divider()
        
        # Table
        if filtered:
            df = pd.DataFrame(filtered)
            cols = ['placed_at', 'bet_team', 'amount', 'odds', 'edge', 'status', 'profit']
            cols = [c for c in cols if c in df.columns]
            
            # Format
            if 'placed_at' in df.columns:
                df['placed_at'] = pd.to_datetime(df['placed_at']).dt.strftime('%Y-%m-%d %H:%M')
            if 'edge' in df.columns:
                df['edge'] = df['edge'].apply(lambda x: f"{x:.1%}" if x else "")
            if 'profit' in df.columns:
                df['profit'] = df['profit'].apply(lambda x: f"${x:+.2f}" if x else "")
            
            st.dataframe(df[cols], use_container_width=True, hide_index=True)
        
        # Profit chart
        st.subheader("💰 Cumulative Profit")
        
        if filtered:
            profits = [b.get('profit', 0) for b in sorted(filtered, key=lambda x: x.get('placed_at', ''))]
            cumulative = []
            total = 0
            for p in profits:
                total += (p or 0)
                cumulative.append(total)
            
            fig = px.line(y=cumulative, title="Cumulative Profit Over Bets")
            fig.update_traces(line_color='#28a745' if cumulative[-1] > 0 else '#dc3545', line_width=3)
            fig.update_layout(xaxis_title="Bet #", yaxis_title="Cumulative Profit ($)")
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("No bet history yet.")


elif page == "🔍 Analytics":
    st.title("🔍 Analytics")
    
    tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "📈 Edge Analysis", "🎯 Calibration"])
    
    with tab1:
        st.subheader("Model Performance")
        
        predictions = load_predictions()
        settled = [p for p in predictions if p.get('settled')]
        
        if settled:
            correct = len([p for p in settled if p.get('was_correct')])
            total = len(settled)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predictions", total)
            with col2:
                st.metric("Correct", correct)
            with col3:
                st.metric("Accuracy", f"{correct/total:.1%}" if total > 0 else "N/A")
            
            # Accuracy over time
            if len(settled) > 10:
                window = 20
                accuracies = []
                for i in range(window, len(settled)):
                    recent = settled[i-window:i]
                    acc = len([p for p in recent if p.get('was_correct')]) / window
                    accuracies.append(acc)
                
                fig = px.line(y=accuracies, title=f"Rolling {window}-game Accuracy")
                fig.add_hline(y=0.5, line_dash="dash", line_color="red")
                fig.add_hline(y=0.55, line_dash="dash", line_color="green")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No settled predictions yet.")
    
    with tab2:
        st.subheader("Edge Analysis")
        
        bets = load_bets_history()
        settled_bets = [b for b in bets if b.get('status') in ['won', 'lost']]
        
        if settled_bets and len(settled_bets) > 5:
            df = pd.DataFrame(settled_bets)
            
            if 'edge' in df.columns and 'profit' in df.columns:
                fig = px.scatter(df, x='edge', y='profit',
                               color='status',
                               color_discrete_map={'won': 'green', 'lost': 'red'},
                               title="Edge vs Profit")
                st.plotly_chart(fig, use_container_width=True)
                
                # Edge distribution
                fig2 = px.histogram(df, x='edge', nbins=15,
                                   title="Edge Distribution in Bets")
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Need more bets for edge analysis.")
    
    with tab3:
        st.subheader("Calibration")
        
        predictions = load_predictions()
        settled = [p for p in predictions if p.get('settled')]
        
        if len(settled) > 20:
            # Bucket predictions
            buckets = {i: {'count': 0, 'wins': 0} for i in range(10)}
            
            for p in settled:
                prob = p.get('predicted_home_prob', 0.5)
                bucket = min(int(prob * 10), 9)
                buckets[bucket]['count'] += 1
                if p.get('was_correct'):
                    buckets[bucket]['wins'] += 1
            
            data = []
            for bucket, stats in buckets.items():
                if stats['count'] >= 3:
                    expected = (bucket + 0.5) / 10
                    actual = stats['wins'] / stats['count']
                    data.append({'expected': expected, 'actual': actual, 'n': stats['count']})
            
            if data:
                df = pd.DataFrame(data)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                        name='Perfect', line=dict(dash='dash', color='gray')))
                fig.add_trace(go.Scatter(x=df['expected'], y=df['actual'],
                                        mode='markers+lines', name='Actual',
                                        marker=dict(size=df['n']*2)))
                fig.update_layout(title="Calibration Plot",
                                 xaxis_title="Predicted Probability",
                                 yaxis_title="Actual Win Rate")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need more predictions for calibration analysis.")


elif page == "⚠️ Live Monitor":
    st.title("⚠️ Live Game Monitor")
    
    st.info("Monitor active games for anomalies and live betting opportunities.")
    
    # В реальности это из live_monitor
    st.subheader("🔥 Active Anomalies")
    
    # Demo anomalies
    demo_anomalies = [
        {
            "type": "UNDERDOG_LEADING",
            "game": "Mavericks @ Lakers",
            "description": "Dallas (аутсайдер 40%) лидирует +8 очков во 2-й четверти!",
            "quarter": 2, "time": "5:30",
            "opportunity": True, "side": "AWAY", "edge": 0.12
        }
    ]
    
    for a in demo_anomalies:
        st.markdown(f"""
        <div class="anomaly-alert">
            <h4>🚨 {a['type']}</h4>
            <p><b>{a['game']}</b> | Q{a['quarter']} {a['time']}</p>
            <p>{a['description']}</p>
            {"<p>🎯 <b>BET OPPORTUNITY:</b> " + a['side'] + f" (Edge: {a['edge']:.0%})</p>" if a['opportunity'] else ""}
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("📋 Anomaly Types")
    
    with st.expander("🔥 UNDERDOG_LEADING"):
        st.write("Аутсайдер неожиданно лидирует с хорошим отрывом (6+ очков)")
    
    with st.expander("🏃 MOMENTUM_SHIFT"):
        st.write("Команда набрала 10+ очков подряд без ответа")
    
    with st.expander("⚠️ FAVORITE_STRUGGLING"):
        st.write("Фаворит проигрывает 5+ очков")


elif page == "⚙️ Settings":
    st.title("⚙️ Settings")
    
    tab1, tab2, tab3 = st.tabs(["💰 Bankroll", "🎯 Betting", "🔔 Notifications"])
    
    with tab1:
        st.subheader("Bankroll Management")
        
        br = bankroll_data  # Используем свежие данные
        
        new_br = st.number_input("Set Bankroll ($)", 
                                min_value=10.0, max_value=100000.0,
                                value=float(br['current']), step=10.0)
        
        if st.button("💾 Update Bankroll"):
            # Save to state file
            state_file = PROJECT_DIR / "system_state.json"
            state = {}
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state = json.load(f)
            
            state['bankroll'] = new_br
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            st.success(f"Bankroll updated to ${new_br:.2f}")
            st.rerun()  # Перезагружаем чтобы обновить данные
        
        st.divider()
        
        st.subheader("Progressive Betting Tiers")
        
        tiers_data = [
            ["$0-$200", "5%", "$5-$10"],
            ["$200-$400", "4.5%", "$10-$18"],
            ["$400-$700", "4%", "$15-$28"],
            ["$700-$1000", "3.5%", "$25-$35"],
            ["$1000-$1500", "3%", "$30-$45"],
            ["$2000-$3000", "2.5%", "$50-$75"],
            ["$3000+", "2.5%", "$75-$125"],
        ]
        
        df_tiers = pd.DataFrame(tiers_data, columns=["Bankroll", "Risk %", "Bet Size"])
        st.dataframe(df_tiers, hide_index=True, use_container_width=True)
    
    with tab2:
        st.subheader("Betting Parameters")
        
        min_edge = st.slider("Minimum Edge", 0.01, 0.10, 0.03, 0.01)
        st.caption(f"Only bet when edge > {min_edge:.0%}")
        
        max_bets = st.slider("Max Daily Bets", 1, 10, 6)
        st.caption(f"Maximum {max_bets} bets per day")
        
        daily_risk = st.slider("Daily Risk Limit", 0.05, 0.30, 0.15, 0.01)
        st.caption(f"Risk up to {daily_risk:.0%} of bankroll daily")
    
    with tab3:
        st.subheader("Telegram Notifications")
        
        telegram_on = st.checkbox("Enable Telegram", value=True)
        
        if telegram_on:
            st.text_input("Bot Token", type="password", key="tg_token")
            st.text_input("Chat ID", key="tg_chat")
        
        st.divider()
        
        st.write("Notify on:")
        st.checkbox("New bet placed", value=True, key="n1")
        st.checkbox("Bet result", value=True, key="n2")
        st.checkbox("Live anomalies", value=True, key="n3")
        st.checkbox("Daily summary", value=True, key="n4")


# === FOOTER ===

st.divider()
st.caption("AutoBasket NBA v2.0 | 🏀 Betting Intelligence System")
