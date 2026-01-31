"""
AutoBasket - Autonomous Betting System
======================================
Полностью автономная система:
- Собирает данные об играх NBA
- Анализирует команды, травмы, тренеров
- Делает ставки через Claude AI
- Мониторит игры в реальном времени
- Учится на результатах
- Работает 24/7
"""

import os
import sys
import time
import logging
import schedule
import subprocess
import threading
import webbrowser
import json
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
from pathlib import Path

# Timezone support
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    from backports.zoneinfo import ZoneInfo  # Fallback

# MST/MDT timezone (Mountain Time)
MST = ZoneInfo("America/Denver")

# Загружаем .env
from dotenv import load_dotenv
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('autobasket.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AutoBasket')

# === ИМПОРТЫ МОДУЛЕЙ ===
try:
    from config.settings import config, BetCategory, BetStatus
    from core.bankroll_manager import BankrollManager
    from core.prediction_engine import BasketballPredictor, GameFeatures, Prediction
    from core.elo_system import EloRatingSystem
    from core.value_finder import ValueBetFinder, BettingPortfolioManager
    from core.live_monitor import LiveGameMonitor, LiveScoreProvider, GameStatus, AnomalyDetector, LiveAnomaly, AnomalyType
    from data.database import Database
    from data.api_clients import DataService, NBADataClient, OddsAPIClient
    from data.historical_db import HistoricalDatabase, PatternAnalyzer
    from data.team_knowledge import TeamKnowledgeBase
    from analysis.team_analytics import DeepAnalyzer, TeamAnalytics, InjuryTracker
    from analysis.coach_analytics import CoachAnalyzer, CoachDatabase
    from analysis.historical import HistoricalAnalyzer
    from analysis.expert_picks import ExpertPicksTracker, AutoScraper, ExpertPicksScheduler
    from ml.self_learning import SelfLearner, PredictionTracker
    from ml.claude_analytics import ClaudeAnalyzer, SmartBettingAdvisor, GameContext
    from utils.notifications import NotificationManager
    from utils.discipline import DisciplineManager
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Run: pip install -r requirements.txt")
    sys.exit(1)


class AutoBasketSystem:
    """
    Главная автономная система
    """
    
    def __init__(self):
        logger.info("=" * 60)
        logger.info("🏀 AUTOBASKET AUTONOMOUS SYSTEM")
        logger.info("=" * 60)
        
        # === ИНИЦИАЛИЗАЦИЯ КОМПОНЕНТОВ ===
        
        # База данных
        self.db = Database()
        self.historical_db = HistoricalDatabase()
        
        # Управление банкроллом
        self.bankroll = BankrollManager(
            initial_bankroll=config.bankroll.initial_bankroll
        )
        logger.info(f"💰 Bankroll: ${self.bankroll.bankroll:.2f}")
        
        # Elo рейтинги
        self.elo = EloRatingSystem()
        
        # ML Prediction
        self.predictor = BasketballPredictor()
        
        # Value Finder
        self.value_finder = ValueBetFinder(self.bankroll)
        self.portfolio = BettingPortfolioManager(self.bankroll)
        
        # Анализаторы
        self.team_analyzer = DeepAnalyzer()
        self.coach_analyzer = CoachAnalyzer()
        self.injury_tracker = InjuryTracker()
        self.historical_analyzer = HistoricalAnalyzer()
        self.pattern_analyzer = PatternAnalyzer(self.historical_db)
        
        # Self-Learning
        self.prediction_tracker = PredictionTracker(storage_path="predictions.json")
        self.learner = SelfLearner(tracker=self.prediction_tracker)
        
        # Claude AI (если есть ключ)
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if api_key:
            self.claude = ClaudeAnalyzer(api_key=api_key)
            self.advisor = SmartBettingAdvisor(self.claude)
            logger.info("🧠 Claude AI: ENABLED")
        else:
            self.claude = None
            self.advisor = None
            logger.warning("🧠 Claude AI: DISABLED (no API key)")
        
        # Data Services
        self.data_service = DataService()
        self.odds_client = OddsAPIClient(api_key=os.getenv('ODDS_API_KEY'))
        
        # Team Knowledge Base (исторические данные)
        self.knowledge_base = TeamKnowledgeBase()
        kb_stats = self.knowledge_base.get_stats()
        logger.info(f"📚 Knowledge Base: {kb_stats['teams_with_data']} teams, {kb_stats['total_historical_games']} games")
        
        # Expert Picks Tracker (мнения экспертов)
        self.expert_tracker = ExpertPicksTracker()
        self.expert_scraper = AutoScraper(self.expert_tracker)
        self.expert_scheduler = ExpertPicksScheduler(
            self.expert_tracker, 
            odds_api_key=os.getenv('ODDS_API_KEY')
        )
        logger.info("🎤 Expert Picks Tracker: initialized")
        
        # Live Monitor с детекцией аномалий
        self.live_monitor = LiveGameMonitor(
            update_interval=60,
            alert_callback=self._on_live_alert,
            anomaly_callback=self._on_anomaly_detected
        )
        
        # Notifications - загружаем токены из .env
        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        discord_webhook = os.getenv('DISCORD_WEBHOOK_URL')

        self.notifications = NotificationManager(
            telegram_token=telegram_token,
            telegram_chat_id=telegram_chat_id,
            discord_webhook=discord_webhook
        )

        if telegram_token and telegram_chat_id:
            logger.info("📱 Telegram notifications: ENABLED")
        else:
            logger.warning("📱 Telegram notifications: DISABLED (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env)")

        if discord_webhook:
            logger.info("💬 Discord notifications: ENABLED")
        else:
            logger.info("💬 Discord notifications: DISABLED")
        
        # Discipline
        self.discipline = DisciplineManager()
        
        # Состояние
        self.todays_games: List[Dict] = []
        self.todays_bets: List[Dict] = []
        self.active_bets: Dict[str, Dict] = {}
        self.is_running = False
        self.dashboard_process = None
        
        # Пути для сохранения данных
        self.state_file = Path(__file__).parent / "system_state.json"
        self.bets_file = Path(__file__).parent / "bets_history.json"
        self.predictions_file = Path(__file__).parent / "predictions.json"
        
        # История банкролла
        self.bankroll_history = []
        self._record_bankroll()
        
        # Загружаем историю ставок
        self._load_bets_history()
        
        logger.info("✅ System initialized")
    
    def _load_bets_history(self):
        """Загружает историю ставок из файла и восстанавливает active_bets"""
        if self.bets_file.exists():
            try:
                with open(self.bets_file, 'r') as f:
                    self.todays_bets = json.load(f)
                logger.info(f"Loaded {len(self.todays_bets)} bets from history")

                # CRITICAL FIX: Восстанавливаем active_bets из pending ставок!
                pending_bets = [b for b in self.todays_bets if b.get('status') == 'pending']
                for bet in pending_bets:
                    # Используем комбинацию команд как ключ для поиска
                    match_key = self._make_match_key(bet.get('home_team', ''), bet.get('away_team', ''))
                    if match_key:
                        self.active_bets[match_key] = bet
                        # Также сохраняем по game_id для совместимости
                        if bet.get('game_id'):
                            self.active_bets[str(bet['game_id'])] = bet

                if pending_bets:
                    logger.info(f"✅ Restored {len(pending_bets)} pending bets to active_bets")

            except Exception as e:
                logger.warning(f"Could not load bets history: {e}")
                self.todays_bets = []

        # Загружаем состояние системы
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)

                # Восстанавливаем bankroll history
                self.bankroll_history = state.get('bankroll_history', [])

                # Восстанавливаем bankroll если был сохранён
                saved_bankroll = state.get('bankroll')
                if saved_bankroll and saved_bankroll > 0:
                    self.bankroll.bankroll = saved_bankroll
                    logger.info(f"💰 Restored bankroll: ${saved_bankroll:.2f}")

                logger.info(f"Restored state: bankroll history has {len(self.bankroll_history)} entries")
            except Exception as e:
                logger.warning(f"Could not restore state: {e}")

    def _make_match_key(self, home_team: str, away_team: str) -> str:
        """Создаёт уникальный ключ матча для сопоставления ставок"""
        if not home_team or not away_team:
            return ""
        # Нормализуем названия команд для сравнения
        home_norm = self._normalize_team_name(home_team)
        away_norm = self._normalize_team_name(away_team)
        return f"{away_norm}@{home_norm}"

    def _normalize_team_name(self, name: str) -> str:
        """Нормализует название команды для сравнения"""
        # Убираем лишние пробелы и приводим к нижнему регистру
        name = name.lower().strip()

        # Маппинг разных вариантов названий
        name_mapping = {
            'la lakers': 'lakers',
            'los angeles lakers': 'lakers',
            'la clippers': 'clippers',
            'los angeles clippers': 'clippers',
            'golden state warriors': 'warriors',
            'gs warriors': 'warriors',
            'boston celtics': 'celtics',
            'miami heat': 'heat',
            'denver nuggets': 'nuggets',
            'phoenix suns': 'suns',
            'milwaukee bucks': 'bucks',
            'philadelphia 76ers': '76ers',
            'philly 76ers': '76ers',
            'minnesota timberwolves': 'timberwolves',
            'sacramento kings': 'kings',
            'detroit pistons': 'pistons',
            'new york knicks': 'knicks',
            'ny knicks': 'knicks',
            'cleveland cavaliers': 'cavaliers',
            'oklahoma city thunder': 'thunder',
            'okc thunder': 'thunder',
            'dallas mavericks': 'mavericks',
            'memphis grizzlies': 'grizzlies',
            'atlanta hawks': 'hawks',
            'brooklyn nets': 'nets',
            'new orleans pelicans': 'pelicans',
            'chicago bulls': 'bulls',
            'houston rockets': 'rockets',
            'indiana pacers': 'pacers',
            'orlando magic': 'magic',
            'portland trail blazers': 'blazers',
            'trail blazers': 'blazers',
            'san antonio spurs': 'spurs',
            'toronto raptors': 'raptors',
            'utah jazz': 'jazz',
            'washington wizards': 'wizards',
            'charlotte hornets': 'hornets',
        }

        # Проверяем маппинг
        for full_name, short_name in name_mapping.items():
            if full_name in name or name in full_name:
                return short_name

        # Если не нашли в маппинге, берём последнее слово (обычно это nickname)
        parts = name.split()
        return parts[-1] if parts else name
    
    def _record_bankroll(self):
        """Записывает текущий банкролл в историю"""
        self.bankroll_history.append({
            'date': datetime.now().isoformat(),
            'value': self.bankroll.bankroll
        })
        # Оставляем только последние 365 записей
        if len(self.bankroll_history) > 365:
            self.bankroll_history = self.bankroll_history[-365:]
    
    def _save_state(self):
        """Сохраняет текущее состояние для dashboard"""
        try:
            # Состояние системы
            state = {
                'bankroll': self.bankroll.bankroll,
                'initial_bankroll': self.bankroll.initial_bankroll,
                'peak_bankroll': self.bankroll.peak_bankroll,
                'todays_games': self.todays_games,
                'active_bets_count': len(self.active_bets),
                'last_update': datetime.now().isoformat(),
                'bankroll_history': getattr(self, 'bankroll_history', [])
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            # История ставок
            with open(self.bets_file, 'w') as f:
                json.dump(self.todays_bets, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    # =========================================================================
    # DASHBOARD
    # =========================================================================
    
    def start_dashboard(self):
        """Запускает Streamlit dashboard в отдельном процессе"""
        try:
            dashboard_path = Path(__file__).parent / "dashboard.py"
            
            if not dashboard_path.exists():
                logger.warning("Dashboard file not found")
                return
            
            logger.info("🖥️ Starting dashboard...")
            
            # Запускаем streamlit в фоне
            self.dashboard_process = subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", str(dashboard_path), 
                 "--server.port", "8501", "--server.headless", "true"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            time.sleep(3)  # Даем время на запуск
            
            # Открываем в браузере
            webbrowser.open("http://localhost:8501")
            
            logger.info("🖥️ Dashboard running at http://localhost:8501")
            
        except Exception as e:
            logger.error(f"Failed to start dashboard: {e}")
    
    def stop_dashboard(self):
        """Останавливает dashboard"""
        if self.dashboard_process:
            self.dashboard_process.terminate()
            logger.info("Dashboard stopped")
    
    # =========================================================================
    # СБОР ДАННЫХ
    # =========================================================================
    
    def fetch_todays_games(self) -> List[Dict]:
        """Получает все игры на сегодня и собирает недостающие данные"""
        logger.info("📅 Fetching today's games...")
        
        games = []
        
        try:
            # Получаем игры
            nba_games = self.data_service.get_todays_games()
            
            if not nba_games:
                logger.warning("No games found for today")
                return []
            
            logger.info(f"Found {len(nba_games)} games")
            
            # Собираем ID всех команд
            all_team_ids = set()
            for game in nba_games:
                all_team_ids.add(game.home_team_id)
                all_team_ids.add(game.away_team_id)
            
            # Проверяем какие команды без данных
            missing_teams = self.knowledge_base.get_missing_teams(list(all_team_ids))
            
            if missing_teams:
                logger.info(f"\n📊 Need to collect data for {len(missing_teams)} teams...")
                
                for team_id in missing_teams:
                    team_name = self._get_team_name_by_id(team_id)
                    logger.info(f"   🔄 Collecting history for {team_name}...")
                    
                    try:
                        games_collected = self.knowledge_base.collect_team_history(
                            team_id=team_id,
                            years_back=10,
                            progress_callback=lambda msg: logger.info(f"      {msg}")
                        )
                        logger.info(f"   ✅ Collected {games_collected} games for {team_name}")
                    except Exception as e:
                        logger.error(f"   ❌ Error collecting {team_name}: {e}")
                    
                    time.sleep(1)  # Пауза между командами
                
                logger.info("📊 Data collection complete!\n")
            
            # Получаем коэффициенты
            odds_data = {}
            if self.odds_client.api_key:
                try:
                    odds_list = self.odds_client.get_nba_odds()
                    for odd in odds_list:
                        key = f"{odd.away_team}_{odd.home_team}"
                        odds_data[key] = odd
                    logger.info(f"Fetched odds for {len(odds_data)} games")
                except Exception as e:
                    logger.error(f"Error fetching odds: {e}")
            
            # Собираем полную информацию по каждой игре
            for game in nba_games:
                game_info = self._build_game_info_with_knowledge(game, odds_data)
                if game_info:
                    games.append(game_info)
                    
                    # Сохраняем в recent_games
                    self.knowledge_base.save_todays_game(
                        game_id=game_info['game_id'],
                        home_team_id=game.home_team_id,
                        home_team=game_info['home_team'],
                        away_team_id=game.away_team_id,
                        away_team=game_info['away_team'],
                        our_prediction=game_info['predicted_home_prob'],
                        analysis=game_info.get('analysis_summary')
                    )
                
                time.sleep(0.5)  # Rate limiting
            
            self.todays_games = games
            logger.info(f"✅ Processed {len(games)} games")
            
            # Сохраняем состояние для dashboard
            self._save_state()
            
        except Exception as e:
            logger.error(f"Error fetching games: {e}")
        
        return games
    
    def _get_team_name_by_id(self, team_id: int) -> str:
        """Получает название команды по ID"""
        team_names = {
            1610612737: "Atlanta Hawks",
            1610612738: "Boston Celtics", 
            1610612739: "Cleveland Cavaliers",
            1610612740: "New Orleans Pelicans",
            1610612741: "Chicago Bulls",
            1610612742: "Dallas Mavericks",
            1610612743: "Denver Nuggets",
            1610612744: "Golden State Warriors",
            1610612745: "Houston Rockets",
            1610612746: "LA Clippers",
            1610612747: "Los Angeles Lakers",
            1610612748: "Miami Heat",
            1610612749: "Milwaukee Bucks",
            1610612750: "Minnesota Timberwolves",
            1610612751: "Brooklyn Nets",
            1610612752: "New York Knicks",
            1610612753: "Orlando Magic",
            1610612754: "Indiana Pacers",
            1610612755: "Philadelphia 76ers",
            1610612756: "Phoenix Suns",
            1610612757: "Portland Trail Blazers",
            1610612758: "Sacramento Kings",
            1610612759: "San Antonio Spurs",
            1610612760: "Oklahoma City Thunder",
            1610612761: "Toronto Raptors",
            1610612762: "Utah Jazz",
            1610612763: "Memphis Grizzlies",
            1610612764: "Washington Wizards",
            1610612765: "Detroit Pistons",
            1610612766: "Charlotte Hornets",
        }
        return team_names.get(team_id, f"Team {team_id}")
    
    def _build_game_info_with_knowledge(self, game, odds_data: Dict) -> Optional[Dict]:
        """Собирает полную информацию об игре используя базу знаний"""
        try:
            home_team = game.home_team
            away_team = game.away_team
            home_id = game.home_team_id
            away_id = game.away_team_id
            
            logger.info(f"  Analyzing: {away_team} @ {home_team}")
            
            # Получаем данные из базы знаний
            home_profile = self.knowledge_base.get_team_profile(home_id)
            away_profile = self.knowledge_base.get_team_profile(away_id)
            
            home_trends = self.knowledge_base.get_team_trends(home_id, 10)
            away_trends = self.knowledge_base.get_team_trends(away_id, 10)
            
            h2h = self.knowledge_base.get_h2h_history(home_id, away_id, 10)
            
            # Elo рейтинги
            home_elo = self.elo.get_rating(home_team)
            away_elo = self.elo.get_rating(away_team)
            
            # Коэффициенты
            odds_key = f"{away_team}_{home_team}"
            game_odds = odds_data.get(odds_key)
            
            if game_odds:
                home_odds = game_odds.home_odds
                away_odds = game_odds.away_odds
                spread = game_odds.spread
                total = game_odds.total
            else:
                home_odds = 1.91
                away_odds = 1.91
                spread = -2.5 if home_elo > away_elo else 2.5
                total = 220.0
            
            # Формируем features для ML используя реальные данные
            features = GameFeatures(
                game_id=game.game_id if hasattr(game, 'game_id') else 0,
                home_team=home_team,
                away_team=away_team,
                game_date=datetime.now(),
                home_elo=home_elo,
                away_elo=away_elo,
                home_win_pct_last10=home_trends.get('win_pct', 0.5) if home_trends else 0.5,
                away_win_pct_last10=away_trends.get('win_pct', 0.5) if away_trends else 0.5,
                home_home_record=home_profile.home_win_pct if home_profile else 0.5,
                away_road_record=away_profile.away_win_pct if away_profile else 0.5,
                home_streak=self._parse_streak(home_trends.get('current_streak', 'W0')) if home_trends else 0,
                away_streak=self._parse_streak(away_trends.get('current_streak', 'W0')) if away_trends else 0,
                home_rest_days=1,
                away_rest_days=1,
                avg_points_scored=home_profile.avg_points_scored if home_profile else 110,
                avg_points_allowed=home_profile.avg_points_allowed if home_profile else 110,
            )
            
            # ML предсказание
            prediction = self.predictor.predict(features)
            
            # Формируем саммари анализа
            analysis_summary = {
                'home_profile': {
                    'win_pct': home_profile.all_time_win_pct if home_profile else 0,
                    'home_win_pct': home_profile.home_win_pct if home_profile else 0,
                    'avg_scored': home_profile.avg_points_scored if home_profile else 0,
                    'avg_allowed': home_profile.avg_points_allowed if home_profile else 0,
                },
                'away_profile': {
                    'win_pct': away_profile.all_time_win_pct if away_profile else 0,
                    'away_win_pct': away_profile.away_win_pct if away_profile else 0,
                    'avg_scored': away_profile.avg_points_scored if away_profile else 0,
                    'avg_allowed': away_profile.avg_points_allowed if away_profile else 0,
                },
                'home_recent': home_trends,
                'away_recent': away_trends,
                'h2h': h2h,
            }
            
            return {
                'game_id': game.game_id,
                'home_team': home_team,
                'away_team': away_team,
                'home_team_id': home_id,
                'away_team_id': away_id,
                'game_time': getattr(game, 'game_time', 'TBD'),
                'status': game.status,
                
                # Рейтинги
                'home_elo': home_elo,
                'away_elo': away_elo,
                
                # Предсказание
                'predicted_home_prob': prediction.home_win_prob,
                'predicted_margin': prediction.predicted_margin or 0,
                'predicted_total': prediction.predicted_total or 220,
                'model_confidence': prediction.confidence,
                
                # Коэффициенты
                'home_odds': home_odds,
                'away_odds': away_odds,
                'spread': spread,
                'total_line': total,
                
                # Из базы знаний
                'home_trends': home_trends,
                'away_trends': away_trends,
                'h2h': h2h,
                'home_injuries': [],
                'away_injuries': [],
                
                # Полный анализ
                'analysis_summary': analysis_summary,
                'features': features.__dict__ if hasattr(features, '__dict__') else {}
            }
            
        except Exception as e:
            logger.error(f"Error building game info: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # =========================================================================
    # ПРИНЯТИЕ РЕШЕНИЙ О СТАВКАХ
    # =========================================================================
    
    def analyze_and_bet(self):
        """Анализирует игры и делает ставки"""
        logger.info("\n" + "=" * 60)
        logger.info("🎯 ANALYZING GAMES AND PLACING BETS")
        logger.info("=" * 60)
        
        if not self.todays_games:
            logger.warning("No games to analyze")
            return
        
        # Считаем игры с потенциальным value
        potential_value_games = 0
        for game in self.todays_games:
            home_implied = 1 / game['home_odds']
            away_implied = 1 / game['away_odds']
            home_edge = game['predicted_home_prob'] - home_implied
            away_edge = (1 - game['predicted_home_prob']) - away_implied
            if max(home_edge, away_edge) > 0.03:
                potential_value_games += 1
        
        # Планируем бюджет на день
        plan = self.bankroll.plan_daily_bets(potential_value_games)
        
        logger.info(f"\n💰 DAILY BUDGET PLAN:")
        logger.info(f"   Bankroll: ${plan['bankroll']:.2f} ({plan['tier']})")
        logger.info(f"   Daily budget: ${plan['daily_budget']:.2f}")
        logger.info(f"   Potential value games: {plan['potential_games']}")
        logger.info(f"   Recommended bets: {plan['recommended_bets']}")
        logger.info(f"   Per-bet budget: ${plan['per_bet_budget']:.2f}")
        logger.info("")
        
        for game in self.todays_games:
            self._process_game_for_betting(game)
        
        # Итоги
        logger.info(f"\n📊 Today's bets: {len(self.todays_bets)}")
        total_wagered = sum(b['amount'] for b in self.todays_bets)
        logger.info(f"💵 Total wagered: ${total_wagered:.2f}")
    
    def _process_game_for_betting(self, game: Dict):
        """Обрабатывает одну игру для ставки"""
        home = game['home_team']
        away = game['away_team']
        game_id = game.get('game_id', f"{date.today().isoformat()}_{away}_{home}")
        
        logger.info(f"\n🏀 {away} @ {home}")
        logger.info(f"   📊 Predicted: {game['predicted_home_prob']:.1%} home win")
        logger.info(f"   🎯 Confidence: {game['model_confidence']:.1%}")
        logger.info(f"   💰 Odds: Home {game['home_odds']:.2f} / Away {game['away_odds']:.2f}")
        
        # Проверяем консенсус экспертов
        try:
            expert_consensus = self.expert_tracker.get_expert_consensus(game_id, home, away)
            if expert_consensus.get('total_picks', 0) > 0:
                logger.info(f"   🎤 Experts: {expert_consensus['home_picks']} for {home}, {expert_consensus['away_picks']} for {away}")
                if expert_consensus.get('sharp_picks', {}).get('home', 0) > 0:
                    logger.info(f"   🔥 Sharp money on {home}")
                elif expert_consensus.get('sharp_picks', {}).get('away', 0) > 0:
                    logger.info(f"   🔥 Sharp money on {away}")
        except Exception as e:
            logger.debug(f"Could not get expert consensus: {e}")
            expert_consensus = {}
        
        # Проверяем дисциплину
        can_bet, warnings = self.discipline.check_can_bet()
        if not can_bet:
            logger.warning(f"   ⛔ Betting locked: {[w.value for w in warnings]}")
            return
        
        # Ищем value
        home_implied = 1 / game['home_odds']
        away_implied = 1 / game['away_odds']
        
        home_edge = game['predicted_home_prob'] - home_implied
        away_edge = (1 - game['predicted_home_prob']) - away_implied
        
        # Корректируем edge на основе экспертов (если есть strong consensus)
        if expert_consensus.get('total_picks', 0) >= 3:
            expert_home_pct = expert_consensus.get('home_pct', 0.5)
            
            # Если эксперты сильно за home (>70%), добавляем бонус
            if expert_home_pct > 0.7:
                home_edge += 0.01  # +1% edge
                logger.info(f"   📈 Expert bonus: +1% edge for {home}")
            elif expert_home_pct < 0.3:
                away_edge += 0.01
                logger.info(f"   📈 Expert bonus: +1% edge for {away}")
        
        # Выбираем лучшую сторону
        if home_edge > away_edge and home_edge > 0.03:
            bet_side = 'home'
            bet_team = home
            bet_odds = game['home_odds']
            edge = home_edge
            our_prob = game['predicted_home_prob']
        elif away_edge > 0.03:
            bet_side = 'away'
            bet_team = away
            bet_odds = game['away_odds']
            edge = away_edge
            our_prob = 1 - game['predicted_home_prob']
        else:
            logger.info("   ❌ No value found")
            return
        
        logger.info(f"   ✅ Value found: {bet_side.upper()} ({bet_team})")
        logger.info(f"   📈 Edge: {edge:.1%}")
        
        # Прогрессивный размер ставки на основе банкролла и edge
        bet_amount = self.bankroll.get_bet_size_for_edge(edge, game['model_confidence'])
        
        # Показываем текущий тир
        pct, min_bet, max_bet, tier = self.bankroll.get_current_tier()
        logger.info(f"   💰 Tier: {tier}, Base bet: ${self.bankroll.get_base_bet_size()}")
        logger.info(f"   💵 Calculated bet: ${bet_amount:.2f}")
        
        # Claude AI валидация (если доступен и ставка значительная)
        if self.advisor and bet_amount >= 10:
            claude_result = self._get_claude_validation(game, bet_side, bet_amount, edge)
            
            if claude_result:
                logger.info(f"   🧠 Claude: {claude_result['final_recommendation'].upper()}")
                
                if claude_result['final_recommendation'] == 'skip':
                    logger.info(f"   ⚠️ Claude says skip: {claude_result.get('reasoning', [])}")
                    return
                
                if claude_result['final_recommendation'] == 'reduce':
                    bet_amount *= 0.5
                    logger.info(f"   📉 Reduced bet to ${bet_amount:.2f}")
        
        # Размещаем ставку
        self._place_bet(game, bet_side, bet_team, bet_odds, bet_amount, edge)
    
    def _get_claude_validation(self, game: Dict, bet_side: str, bet_amount: float, edge: float) -> Optional[Dict]:
        """Получает валидацию от Claude"""
        try:
            context = GameContext(
                game_id=str(game['game_id']),
                home_team=game['home_team'],
                away_team=game['away_team'],
                game_date=date.today(),
                predicted_home_prob=game['predicted_home_prob'],
                predicted_margin=game['predicted_margin'],
                predicted_total=game['predicted_total'],
                model_confidence=game['model_confidence'],
                home_odds=game['home_odds'],
                away_odds=game['away_odds'],
                spread_line=game['spread'],
                total_line=game['total_line'],
                home_record=game['home_trends'].get('record', '') if game.get('home_trends') else '',
                away_record=game['away_trends'].get('record', '') if game.get('away_trends') else '',
                home_streak=game['home_trends'].get('current_streak', '') if game.get('home_trends') else '',
                away_streak=game['away_trends'].get('current_streak', '') if game.get('away_trends') else '',
                h2h_summary=f"{game['h2h'].get('team1_wins', 0)}-{game['h2h'].get('team2_wins', 0)} in last {game['h2h'].get('total_games', 0)}" if game.get('h2h') else "No H2H data",
                injuries=game.get('home_injuries', []) + game.get('away_injuries', [])
            )
            
            return self.advisor.get_recommendation(
                context=context,
                ml_recommendation='bet',
                ml_edge=edge,
                bet_amount=bet_amount
            )
        except Exception as e:
            logger.error(f"Claude validation error: {e}")
            return None
    
    def _place_bet(self, game: Dict, side: str, team: str, odds: float, amount: float, edge: float):
        """Размещает ставку"""
        # Проверяем банкролл
        if amount > self.bankroll.bankroll:
            amount = self.bankroll.get_base_bet_size()
        
        # Проверяем оставшийся дневной бюджет
        if amount > self.bankroll.today_remaining:
            logger.warning(f"   ⚠️ Reducing bet to remaining daily budget: ${self.bankroll.today_remaining:.2f}")
            amount = self.bankroll.today_remaining
        
        if amount < 5:
            logger.warning("   ❌ Insufficient budget for bet")
            return
        
        # Записываем ставку
        bet = {
            'bet_id': f"BET_{game['game_id']}_{int(time.time())}",
            'game_id': game['game_id'],
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'bet_side': side,
            'bet_team': team,
            'odds': odds,
            'amount': amount,
            'edge': edge,
            'predicted_prob': game['predicted_home_prob'] if side == 'home' else 1 - game['predicted_home_prob'],
            'placed_at': datetime.now().isoformat(),
            'status': 'pending'
        }
        
        # Обновляем банкролл и дневной учёт
        self.bankroll.bankroll -= amount
        self.bankroll.today_risked += amount
        self.bankroll.today_remaining -= amount
        self.bankroll.today_bets_count += 1
        
        # Сохраняем
        self.todays_bets.append(bet)
        # Сохраняем по game_id
        self.active_bets[str(game['game_id'])] = bet
        # CRITICAL: Также сохраняем по match_key для сопоставления с ESPN
        match_key = self._make_match_key(game['home_team'], game['away_team'])
        if match_key:
            self.active_bets[match_key] = bet
        
        # Записываем предсказание для обучения
        self.prediction_tracker.record_prediction(
            game_id=game['game_id'],
            home_team=game['home_team'],
            away_team=game['away_team'],
            predicted_home_prob=game['predicted_home_prob'],
            confidence=game['model_confidence'],
            features=game['features']
        )
        
        # Добавляем в live monitor
        self.live_monitor.add_prediction(
            game_id=str(game['game_id']),
            home_team=game['home_team'],
            away_team=game['away_team'],
            predicted_home_prob=game['predicted_home_prob'],
            predicted_margin=game['predicted_margin'],
            predicted_total=game['predicted_total']
        )
        self.live_monitor.add_bet(str(game['game_id']), side, amount, game['spread'])
        
        logger.info(f"   💰 BET PLACED: ${amount:.2f} on {team} @ {odds:.2f}")
        logger.info(f"   📊 Potential win: ${amount * (odds - 1):.2f}")
        logger.info(f"   💵 Remaining daily budget: ${self.bankroll.today_remaining:.2f}")
        
        # Уведомление
        self.notifications.send_message(
            f"🎯 NEW BET\n{game['away_team']} @ {game['home_team']}\n"
            f"${amount:.2f} on {team} @ {odds:.2f}\n"
            f"Edge: {edge:.1%}"
        )
        
        # Сохраняем состояние для dashboard
        self._save_state()
    
    # =========================================================================
    # МОНИТОРИНГ В РЕАЛЬНОМ ВРЕМЕНИ
    # =========================================================================
    
    def start_live_monitoring(self):
        """Запускает мониторинг игр"""
        if not self.active_bets:
            logger.info("No active bets to monitor")
            return
        
        logger.info("\n📺 Starting live game monitoring...")
        self.live_monitor.start_monitoring()
    
    def stop_live_monitoring(self):
        """Останавливает мониторинг"""
        self.live_monitor.stop_monitoring()
    
    def _on_live_alert(self, alert):
        """Обработчик алертов от live monitor"""
        logger.info(f"🚨 ALERT: {alert.message}")
        self.notifications.send_message(f"🚨 {alert.message}")
    
    def _on_anomaly_detected(self, anomaly: LiveAnomaly):
        """
        Обработчик аномалий во время игры
        
        Это как твой пример с Lakers vs Dallas:
        - Аутсайдер лидирует во 2-м тайме
        - Система детектирует и предлагает live-ставку
        """
        logger.info(f"\n{'🔥' * 20}")
        logger.info(f"🚨 LIVE ANOMALY: {anomaly.anomaly_type.value}")
        logger.info(f"   {anomaly.description}")
        logger.info(f"   Q{anomaly.quarter} {anomaly.time_remaining} | Score: {anomaly.current_score}")
        
        # Уведомление в Telegram
        msg = f"""
🚨 LIVE ANOMALY DETECTED
━━━━━━━━━━━━━━━━━━━━━━━

{anomaly.away_team} @ {anomaly.home_team}
Q{anomaly.quarter} {anomaly.time_remaining}
Score: {anomaly.current_score}

{anomaly.description}
Confidence: {anomaly.confidence:.0%}
"""
        
        # Если есть возможность для ставки
        if anomaly.bet_opportunity:
            logger.info(f"   🎯 BET OPPORTUNITY!")
            logger.info(f"   Recommended: {anomaly.recommended_side.upper()}")
            logger.info(f"   Edge estimate: {anomaly.edge_estimate:.1%}")
            
            msg += f"""
🎯 LIVE BET OPPORTUNITY:
   Side: {anomaly.recommended_side.upper()}
   Edge: {anomaly.edge_estimate:.1%}
"""
            
            # Можно автоматически ставить если edge > 10%
            if anomaly.edge_estimate > 0.10 and anomaly.confidence > 0.7:
                logger.info(f"   ⚡ HIGH CONFIDENCE - Consider live bet!")
                
                # TODO: Автоматическая live-ставка (опционально)
                # Пока только уведомление, так как live-ставки требуют
                # интеграции с букмекером для получения live odds
        
        logger.info(f"{'🔥' * 20}\n")
        
        self.notifications.send_message(msg)
    
    def check_and_settle_bets(self):
        """Проверяет завершенные игры и рассчитывает ставки"""
        logger.info("\n🔍 Checking for finished games...")

        if not self.active_bets:
            logger.info("   No active bets to settle")
            return

        # Получаем текущие результаты
        live_games = self.live_monitor.update()

        settled_count = 0
        for game in live_games:
            if game.status != GameStatus.FINAL:
                continue

            # CRITICAL FIX: Ищем ставку по названиям команд, а не только по ID
            bet = self._find_bet_for_game(game)
            if not bet:
                continue

            logger.info(f"🎯 Found finished game: {game.away_team} @ {game.home_team}")
            logger.info(f"   Final score: {game.score.away_score} - {game.score.home_score}")

            # Определяем результат
            home_won = game.score.home_score > game.score.away_score
            bet_won = (bet['bet_side'] == 'home' and home_won) or \
                      (bet['bet_side'] == 'away' and not home_won)

            # Рассчитываем
            if bet_won:
                profit = bet['amount'] * (bet['odds'] - 1)
                self.bankroll.bankroll += bet['amount'] + profit
                bet['status'] = 'won'
                bet['profit'] = profit
                bet['final_score'] = f"{game.score.away_score}-{game.score.home_score}"
                logger.info(f"✅ WON: {bet['bet_team']} - Profit: ${profit:.2f}")
            else:
                bet['status'] = 'lost'
                bet['profit'] = -bet['amount']
                bet['final_score'] = f"{game.score.away_score}-{game.score.home_score}"
                logger.info(f"❌ LOST: {bet['bet_team']} - Loss: ${bet['amount']:.2f}")

            # Записываем результат для обучения
            try:
                # Fix: handle both string and int game_id types
                gid = game.game_id
                if isinstance(gid, int):
                    game_id_int = gid
                elif isinstance(gid, str) and gid.isdigit():
                    game_id_int = int(gid)
                else:
                    game_id_int = hash(str(gid)) % 10000000

                self.prediction_tracker.record_result(
                    game_id=game_id_int,
                    home_won=home_won,
                    margin=game.score.margin
                )
            except Exception as e:
                logger.warning(f"Could not record result for learning: {e}")

            # Записываем в discipline
            self.discipline.record_result(bet_won)

            # Убираем из активных - по всем возможным ключам
            self._remove_bet_from_active(bet, game)

            settled_count += 1

            # Обновляем ставку в todays_bets
            for i, b in enumerate(self.todays_bets):
                if b.get('bet_id') == bet.get('bet_id'):
                    self.todays_bets[i] = bet
                    break

            # Уведомление
            emoji = "✅" if bet_won else "❌"
            self.notifications.send_message(
                f"{emoji} BET SETTLED\n"
                f"{bet['away_team']} @ {bet['home_team']}\n"
                f"Score: {game.score.away_score}-{game.score.home_score}\n"
                f"Result: {'WON' if bet_won else 'LOST'}\n"
                f"P&L: ${bet['profit']:+.2f}\n"
                f"Bankroll: ${self.bankroll.bankroll:.2f}"
            )

        if settled_count > 0:
            logger.info(f"\n📊 Settled {settled_count} bet(s)")
            self._record_bankroll()

        # Сохраняем состояние после settle
        self._save_state()

    def _find_bet_for_game(self, game) -> Optional[Dict]:
        """Находит нашу ставку для данной игры по названиям команд"""
        # Создаём ключ матча
        match_key = self._make_match_key(game.home_team, game.away_team)

        # Сначала пробуем найти по match_key
        if match_key in self.active_bets:
            return self.active_bets[match_key]

        # Пробуем по game_id (на случай если совпадают)
        if game.game_id in self.active_bets:
            return self.active_bets[game.game_id]

        # Если не нашли напрямую, ищем по нормализованным названиям
        game_home_norm = self._normalize_team_name(game.home_team)
        game_away_norm = self._normalize_team_name(game.away_team)

        for key, bet in self.active_bets.items():
            bet_home_norm = self._normalize_team_name(bet.get('home_team', ''))
            bet_away_norm = self._normalize_team_name(bet.get('away_team', ''))

            # Проверяем совпадение команд
            if bet_home_norm == game_home_norm and bet_away_norm == game_away_norm:
                return bet

        return None

    def _remove_bet_from_active(self, bet: Dict, game):
        """Удаляет ставку из active_bets по всем возможным ключам"""
        keys_to_remove = []

        # Собираем все ключи которые нужно удалить
        match_key = self._make_match_key(game.home_team, game.away_team)
        if match_key in self.active_bets:
            keys_to_remove.append(match_key)

        if game.game_id in self.active_bets:
            keys_to_remove.append(game.game_id)

        bet_game_id = str(bet.get('game_id', ''))
        if bet_game_id and bet_game_id in self.active_bets:
            keys_to_remove.append(bet_game_id)

        # Удаляем
        for key in keys_to_remove:
            if key in self.active_bets:
                del self.active_bets[key]
    
    # =========================================================================
    # ОБУЧЕНИЕ
    # =========================================================================
    
    def run_learning_cycle(self):
        """Запускает цикл обучения"""
        logger.info("\n🎓 Running learning cycle...")
        
        result = self.learner.learn_from_results()
        
        if result['status'] == 'success':
            logger.info(f"   Accuracy: {result['performance']['accuracy']:.1%}")
            logger.info(f"   Brier Score: {result['performance']['brier_score']:.4f}")
            logger.info(f"   Trend: {result['performance']['trend']}")
            logger.info(f"   Adjustments made: {result['adjustments']}")
        else:
            logger.info(f"   Status: {result['status']}")
        
        return result
    
    # =========================================================================
    # ОТЧЕТЫ
    # =========================================================================
    
    def print_daily_report(self):
        """Выводит дневной отчет"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 DAILY REPORT")
        logger.info("=" * 60)
        
        stats = self.bankroll.get_statistics()
        
        logger.info(f"\n💰 BANKROLL")
        logger.info(f"   Current: ${self.bankroll.bankroll:.2f}")
        logger.info(f"   Daily P&L: ${self.bankroll.bankroll - self.bankroll.day_start_balance:+.2f}")
        logger.info(f"   Peak: ${self.bankroll.peak_bankroll:.2f}")
        
        logger.info(f"\n📈 TODAY'S BETS: {len(self.todays_bets)}")
        
        for bet in self.todays_bets:
            status_emoji = "⏳" if bet['status'] == 'pending' else "✅" if bet['status'] == 'won' else "❌"
            logger.info(f"   {status_emoji} {bet['bet_team']} @ {bet['odds']:.2f} - ${bet['amount']:.2f}")
        
        won = sum(1 for b in self.todays_bets if b['status'] == 'won')
        lost = sum(1 for b in self.todays_bets if b['status'] == 'lost')
        pending = sum(1 for b in self.todays_bets if b['status'] == 'pending')
        
        logger.info(f"\n   Won: {won} | Lost: {lost} | Pending: {pending}")
        
        total_profit = sum(b.get('profit', 0) for b in self.todays_bets if b['status'] != 'pending')
        logger.info(f"   Total P&L: ${total_profit:+.2f}")
        
        logger.info("=" * 60)
    
    def print_games_summary(self):
        """Выводит сводку по играм с данными из базы знаний"""
        logger.info("\n" + "=" * 60)
        logger.info("🏀 TODAY'S GAMES SUMMARY")
        logger.info(f"⏰ Current time: {self.get_mst_time().strftime('%I:%M %p MST')}")
        logger.info("=" * 60)

        for game in self.todays_games:
            home = game['home_team']
            away = game['away_team']
            prob = game['predicted_home_prob']
            game_time = game.get('game_time', 'TBD')
            mst_time = self.format_game_time_mst(game_time) if game_time != 'TBD' else 'TBD'

            logger.info(f"\n{away} @ {home}")
            logger.info(f"   ⏰ Game time: {mst_time}")
            logger.info(f"   📊 Prediction: {prob:.1%} home win")
            logger.info(f"   📈 Spread: {home} {game['spread']}")
            logger.info(f"   🎯 Total: {game['total_line']}")
            
            # Данные из базы знаний
            home_trends = game.get('home_trends', {})
            away_trends = game.get('away_trends', {})
            h2h = game.get('h2h', {})
            
            if home_trends:
                logger.info(f"   🏠 {home}: Last 10: {home_trends.get('record', 'N/A')}, Streak: {home_trends.get('current_streak', 'N/A')}")
            
            if away_trends:
                logger.info(f"   ✈️  {away}: Last 10: {away_trends.get('record', 'N/A')}, Streak: {away_trends.get('current_streak', 'N/A')}")
            
            if h2h and h2h.get('total_games', 0) > 0:
                logger.info(f"   🤝 H2H: {h2h['team1_wins']}-{h2h['team2_wins']} (last {h2h['total_games']} games)")
            
            if game.get('home_injuries'):
                logger.info(f"   ⚠️ {home} injuries: {len(game['home_injuries'])}")
            if game.get('away_injuries'):
                logger.info(f"   ⚠️ {away} injuries: {len(game['away_injuries'])}")
    
    # =========================================================================
    # ГЛАВНЫЙ ЦИКЛ
    # =========================================================================
    
    def run_daily_cycle(self):
        """Запускает полный дневной цикл"""
        logger.info("\n" + "🌅 " * 20)
        logger.info("🌅 STARTING NEW DAILY CYCLE")
        logger.info("🌅 " * 20)

        # Сначала проверяем и settle'им оставшиеся ставки с предыдущего дня
        if self.active_bets:
            logger.info(f"\n🔄 Checking {len(self.active_bets)} pending bets from previous day...")
            self.check_and_settle_bets()

        # Сбрасываем дневные данные
        self.todays_games = []

        # CRITICAL FIX: НЕ стираем pending ставки! Фильтруем только settled
        pending_bets = [b for b in self.todays_bets if b.get('status') == 'pending']
        settled_bets = [b for b in self.todays_bets if b.get('status') != 'pending']

        if pending_bets:
            logger.info(f"⏳ Keeping {len(pending_bets)} pending bets")
        if settled_bets:
            logger.info(f"📊 Previous day settled: {len([b for b in settled_bets if b.get('status') == 'won'])} won, {len([b for b in settled_bets if b.get('status') == 'lost'])} lost")

        # Начинаем новый день с pending ставками
        self.todays_bets = pending_bets

        self.bankroll.reset_daily()
        
        # 1. Собираем данные об играх
        self.fetch_todays_games()
        
        if not self.todays_games:
            logger.info("No games today. Waiting...")
            return
        
        # 2. Собираем мнения экспертов (автоматически)
        try:
            logger.info("\n🎤 Collecting expert picks...")
            expert_results = self.expert_scheduler.run_collection_cycle()
            if expert_results:
                logger.info(f"   Collected {expert_results.get('total_picks', 0)} expert picks")
        except Exception as e:
            logger.warning(f"Expert picks collection failed: {e}")
        
        # 3. Выводим сводку
        self.print_games_summary()
        
        # 4. Анализируем и делаем ставки
        self.analyze_and_bet()
        
        # Сохраняем состояние для dashboard
        self._save_state()
        self._record_bankroll()  # Записываем банкролл в историю
        
        # 5. Запускаем мониторинг
        self.start_live_monitoring()
    
    def collect_expert_picks(self):
        """Собирает прогнозы экспертов (вызывается по расписанию)"""
        logger.info("\n🎤 Scheduled expert picks collection...")
        
        try:
            results = self.expert_scheduler.run_collection_cycle()
            
            if results and results.get('total_picks', 0) > 0:
                logger.info(f"   ✅ Collected {results['total_picks']} picks from {len(results.get('sources', {}))} sources")
                
                # Уведомляем
                if self.notifications:
                    self.notifications.send_message(
                        f"🎤 Expert picks collected: {results['total_picks']} picks"
                    )
            else:
                logger.info("   No new picks found")
                
        except Exception as e:
            logger.error(f"Expert collection failed: {e}")
    
    def run_forever(self):
        """Запускает систему в бесконечном режиме"""
        self.is_running = True
        
        logger.info("\n" + "🚀 " * 20)
        logger.info("🚀 AUTOBASKET STARTING IN AUTONOMOUS MODE")
        logger.info("🚀 " * 20)
        
        # Запускаем dashboard
        self.start_dashboard()
        
        # Запускаем первый цикл
        self.run_daily_cycle()
        
        # Расписание
        schedule.every().day.at("00:05").do(self.run_daily_cycle)
        schedule.every(5).minutes.do(self.check_and_settle_bets)
        schedule.every().day.at("10:00").do(self.collect_expert_picks)  # Утренний сбор
        schedule.every().day.at("17:00").do(self.collect_expert_picks)  # Вечерний сбор перед играми
        schedule.every().day.at("23:55").do(self.run_learning_cycle)
        schedule.every().day.at("23:58").do(self.print_daily_report)
        
        logger.info("\n⏰ Schedule configured:")
        logger.info("   00:05 - New daily cycle (games + data collection)")
        logger.info("   10:00 - Morning expert picks collection")
        logger.info("   17:00 - Pre-game expert picks collection")
        logger.info("   Every 5 min - Check finished games")
        logger.info("   23:55 - Learning cycle")
        logger.info("   23:58 - Daily report")
        
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(30)
                
        except KeyboardInterrupt:
            logger.info("\n\n⛔ Shutting down...")
            self.stop_live_monitoring()
            self.stop_dashboard()
            self.print_daily_report()
            logger.info("👋 Goodbye!")
    
    # =========================================================================
    # УТИЛИТЫ
    # =========================================================================
    
    def _get_team_id(self, team_name: str) -> int:
        """Возвращает ID команды"""
        # Упрощенная карта
        team_ids = {
            'Los Angeles Lakers': 1610612747,
            'Golden State Warriors': 1610612744,
            'Boston Celtics': 1610612738,
            'Miami Heat': 1610612748,
            'Denver Nuggets': 1610612743,
            'Phoenix Suns': 1610612756,
            'Milwaukee Bucks': 1610612749,
            'Philadelphia 76ers': 1610612755,
            'Minnesota Timberwolves': 1610612750,
            'Sacramento Kings': 1610612758,
            'Detroit Pistons': 1610612765,
            'New York Knicks': 1610612752,
            'Cleveland Cavaliers': 1610612739,
            'Oklahoma City Thunder': 1610612760,
            'Dallas Mavericks': 1610612742,
            'Memphis Grizzlies': 1610612763,
        }
        return team_ids.get(team_name, 0)
    
    def _get_team_abbr(self, team_name: str) -> str:
        """Возвращает аббревиатуру команды"""
        abbrs = {
            'Los Angeles Lakers': 'LAL',
            'Golden State Warriors': 'GSW',
            'Boston Celtics': 'BOS',
            'Miami Heat': 'MIA',
            'Denver Nuggets': 'DEN',
            'Phoenix Suns': 'PHX',
            'Milwaukee Bucks': 'MIL',
            'Philadelphia 76ers': 'PHI',
            'Minnesota Timberwolves': 'MIN',
            'Sacramento Kings': 'SAC',
            'Detroit Pistons': 'DET',
            'New York Knicks': 'NYK',
            'Cleveland Cavaliers': 'CLE',
            'Oklahoma City Thunder': 'OKC',
            'Dallas Mavericks': 'DAL',
            'Memphis Grizzlies': 'MEM',
        }
        return abbrs.get(team_name, 'UNK')
    
    def _parse_record(self, record: str) -> float:
        """Парсит рекорд типа '7-3' в win%"""
        try:
            if not record or '-' not in record:
                return 0.5
            parts = record.split('-')
            if len(parts) < 2:
                return 0.5
            wins = int(parts[0].strip())
            losses = int(parts[1].strip())
            return wins / (wins + losses) if wins + losses > 0 else 0.5
        except (ValueError, IndexError, AttributeError):
            return 0.5
    
    def _parse_streak(self, streak: str) -> int:
        """Парсит streak типа 'W3' или 'L2'"""
        try:
            if streak.startswith('W'):
                return int(streak[1:])
            elif streak.startswith('L'):
                return -int(streak[1:])
            return 0
        except:
            return 0
    
    def _safe_get_rating(self, analysis: Dict, key: str, default: float) -> float:
        """Безопасно получает рейтинг из анализа"""
        if not analysis:
            return default
        
        # Пробуем получить из advanced_stats
        adv = analysis.get('advanced_stats')
        if adv:
            if hasattr(adv, key):
                return getattr(adv, key, default)
            if isinstance(adv, dict):
                return adv.get(key, default)
        
        # Пробуем получить напрямую
        return analysis.get(key, default)
    
    def _injury_to_dict(self, injury) -> Dict:
        """Конвертирует травму в dict"""
        return {
            'player': injury.player_name,
            'team': injury.team_abbr,
            'status': injury.status.value,
            'injury': injury.injury_type
        }

    # =========================================================================
    # TIMEZONE HELPERS
    # =========================================================================

    def get_mst_time(self) -> datetime:
        """Возвращает текущее время в MST"""
        return datetime.now(MST)

    def convert_to_mst(self, dt: datetime) -> datetime:
        """Конвертирует datetime в MST"""
        if dt.tzinfo is None:
            # Assume UTC if no timezone
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        return dt.astimezone(MST)

    def format_game_time_mst(self, game_time_str: str) -> str:
        """Форматирует время игры в MST"""
        try:
            # Пробуем разные форматы
            formats = [
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S",
                "%I:%M %p ET",
                "%I:%M%p ET",
            ]

            for fmt in formats:
                try:
                    if "ET" in game_time_str:
                        # Eastern Time
                        time_part = game_time_str.replace(" ET", "").replace("ET", "")
                        dt = datetime.strptime(time_part, fmt.replace(" ET", "").replace("ET", ""))
                        # Assume today's date
                        dt = dt.replace(year=date.today().year, month=date.today().month, day=date.today().day)
                        dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))
                    else:
                        dt = datetime.strptime(game_time_str, fmt)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=ZoneInfo("UTC"))

                    mst_time = dt.astimezone(MST)
                    return mst_time.strftime("%I:%M %p MST")
                except ValueError:
                    continue

            # Если не удалось распарсить, возвращаем как есть
            return f"{game_time_str} (local)"
        except Exception:
            return game_time_str

    def get_games_schedule_mst(self) -> List[Dict]:
        """Возвращает расписание игр с временем в MST"""
        schedule = []
        for game in self.todays_games:
            game_time = game.get('game_time', 'TBD')
            mst_time = self.format_game_time_mst(game_time) if game_time != 'TBD' else 'TBD'

            schedule.append({
                'matchup': f"{game['away_team']} @ {game['home_team']}",
                'time_mst': mst_time,
                'original_time': game_time,
                'game_id': game['game_id']
            })

        return schedule


# =========================================================================
# ТОЧКА ВХОДА
# =========================================================================

def select_trading_mode() -> str:
    """Меню выбора режима торговли: Paper или Real"""
    print("\n" + "=" * 50)
    print("💰 SELECT TRADING MODE")
    print("=" * 50)
    print("\n  1. 📝 Paper Trading (виртуальные деньги)")
    print("     - Безопасно для тестирования")
    print("     - Никаких реальных ставок")
    print("     - Полная симуляция")
    print()
    print("  2. 💵 Real Money (Kalshi API)")
    print("     - РЕАЛЬНЫЕ ДЕНЬГИ!")
    print("     - Требуется Kalshi аккаунт")
    print("     - Риск потери средств")
    print()

    try:
        mode_choice = input("Select mode [1-2, default=1]: ").strip()
    except (EOFError, KeyboardInterrupt):
        mode_choice = "1"

    if mode_choice == "2":
        print("\n" + "⚠️ " * 20)
        print("⚠️  WARNING: REAL MONEY MODE SELECTED!")
        print("⚠️  You are about to trade with REAL money.")
        print("⚠️  Losses are possible and permanent.")
        print("⚠️ " * 20)

        try:
            confirm = input("\nType 'I UNDERSTAND' to continue: ").strip()
        except (EOFError, KeyboardInterrupt):
            confirm = ""

        if confirm != "I UNDERSTAND":
            print("\n❌ Real money mode cancelled. Using Paper Trading.")
            return "paper"

        # Проверяем наличие API ключей
        kalshi_key = os.getenv('KALSHI_API_KEY')
        kalshi_secret = os.getenv('KALSHI_API_SECRET')

        if not kalshi_key or not kalshi_secret:
            print("\n❌ Kalshi API credentials not found!")
            print("   Set KALSHI_API_KEY and KALSHI_API_SECRET in .env")
            print("   Falling back to Paper Trading.")
            return "paper"

        print("\n✅ Real Money mode confirmed")
        return "real"

    print("\n✅ Paper Trading mode selected")
    return "paper"


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🏀 AUTOBASKET AUTONOMOUS BETTING SYSTEM")
    print("=" * 60)
    print("\nCommands:")
    print("  1. Run autonomous mode (24/7)")
    print("  2. Run single analysis cycle")
    print("  3. Show today's games")
    print("  4. Check pending bets status")
    print("  5. Exit")
    print()

    try:
        choice = input("Select option [1-5]: ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = "1"

    if choice == "5":
        print("Exiting...")
        sys.exit(0)

    # Показываем меню выбора режима для опций 1, 2
    trading_mode = "paper"
    if choice in ["1", "2"]:
        trading_mode = select_trading_mode()

    print(f"\n🎮 Trading Mode: {trading_mode.upper()}")
    print("=" * 60)

    system = AutoBasketSystem()

    # Показываем статус pending ставок
    pending_count = len([b for b in system.todays_bets if b.get('status') == 'pending'])
    if pending_count > 0:
        print(f"\n⏳ Found {pending_count} pending bets from previous session")
        for bet in system.todays_bets:
            if bet.get('status') == 'pending':
                print(f"   • {bet.get('bet_team', 'Unknown')} @ {bet.get('odds', 0):.2f} - ${bet.get('amount', 0):.2f}")

    if choice == "1":
        system.run_forever()

    elif choice == "2":
        system.run_daily_cycle()

        # Ждем завершения игр
        print("\nPress Ctrl+C to stop monitoring and exit")
        try:
            while system.active_bets:
                system.check_and_settle_bets()
                time.sleep(60)
        except KeyboardInterrupt:
            pass

        system.run_learning_cycle()
        system.print_daily_report()

    elif choice == "3":
        system.fetch_todays_games()
        system.print_games_summary()

    elif choice == "4":
        # Показываем статус pending ставок и пытаемся их settle
        print("\n" + "=" * 60)
        print("⏳ PENDING BETS STATUS")
        print("=" * 60)

        pending = [b for b in system.todays_bets if b.get('status') == 'pending']
        if not pending:
            print("\n✅ No pending bets found")
        else:
            print(f"\nFound {len(pending)} pending bet(s):\n")
            for bet in pending:
                print(f"  🎯 {bet.get('away_team', '?')} @ {bet.get('home_team', '?')}")
                print(f"     Bet: {bet.get('bet_team', '?')} @ {bet.get('odds', 0):.2f}")
                print(f"     Amount: ${bet.get('amount', 0):.2f}")
                print(f"     Placed: {bet.get('placed_at', '?')}")
                print()

            print("\n🔍 Checking for finished games...")
            system.check_and_settle_bets()

            # Показываем обновлённый статус
            still_pending = len([b for b in system.todays_bets if b.get('status') == 'pending'])
            settled = len(pending) - still_pending
            if settled > 0:
                print(f"\n✅ Settled {settled} bet(s)")
            if still_pending > 0:
                print(f"⏳ {still_pending} bet(s) still pending")

            system.print_daily_report()

    else:
        print("Invalid option. Exiting...")
