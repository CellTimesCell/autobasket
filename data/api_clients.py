"""
AutoBasket - API Clients
========================
Клиенты для получения данных из внешних источников
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, date, timedelta
from dataclasses import dataclass
import json

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from nba_api.stats.endpoints import (
        leaguegamefinder, teamgamelog,
        playergamelog, commonteamroster, teamestimatedmetrics
    )
    from nba_api.live.nba.endpoints import scoreboard as live_scoreboard
    from nba_api.stats.static import teams as nba_teams
    NBA_API_AVAILABLE = True
    USE_LIVE_SCOREBOARD = True
except ImportError as e:
    print(f"nba_api import error: {e}")
    try:
        # Fallback - старый импорт
        from nba_api.stats.endpoints import (
            leaguegamefinder, teamgamelog,
            playergamelog, commonteamroster, teamestimatedmetrics
        )
        from nba_api.stats.static import teams as nba_teams
        NBA_API_AVAILABLE = True
        USE_LIVE_SCOREBOARD = False
    except:
        NBA_API_AVAILABLE = False
        USE_LIVE_SCOREBOARD = False
except Exception as e:
    print(f"nba_api unexpected error: {e}")
    NBA_API_AVAILABLE = False
    USE_LIVE_SCOREBOARD = False

import sys
sys.path.append('..')
from config.settings import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# === NBA DATA CLIENT ===

@dataclass
class NBAGame:
    """Структура игры NBA"""
    game_id: str
    game_date: date
    home_team_id: int
    home_team: str
    home_team_abbr: str
    away_team_id: int
    away_team: str
    away_team_abbr: str
    
    # Результат (если игра завершена)
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    status: str = "scheduled"  # scheduled, live, final
    
    # Дополнительно
    arena: Optional[str] = None
    game_time: Optional[str] = None


@dataclass  
class TeamStats:
    """Статистика команды"""
    team_id: int
    team_name: str
    team_abbr: str
    
    # Record
    wins: int = 0
    losses: int = 0
    win_pct: float = 0.0
    
    # Home/Away
    home_wins: int = 0
    home_losses: int = 0
    road_wins: int = 0
    road_losses: int = 0
    
    # Advanced
    off_rating: float = 110.0
    def_rating: float = 110.0
    net_rating: float = 0.0
    pace: float = 100.0
    
    # Streak
    streak: int = 0  # Положительный = победы, отрицательный = поражения
    last_10: str = ""  # например "7-3"


class NBADataClient:
    """
    Клиент для получения данных NBA
    
    Использует nba_api для официальной статистики NBA
    """
    
    def __init__(self, request_delay: float = None):
        self.delay = request_delay or config.api.nba_api_delay
        self._last_request = 0
        
        if not NBA_API_AVAILABLE:
            logger.warning("nba_api не установлен. Установите: pip install nba_api")
        
        # Кэш команд
        self._teams_cache = {}
        self._load_teams()
    
    def _rate_limit(self):
        """Соблюдаем rate limit"""
        elapsed = time.time() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request = time.time()
    
    def _load_teams(self):
        """Загружает список команд NBA"""
        if not NBA_API_AVAILABLE:
            return
        
        try:
            all_teams = nba_teams.get_teams()
            for team in all_teams:
                self._teams_cache[team['id']] = {
                    'name': team['full_name'],
                    'abbr': team['abbreviation'],
                    'city': team['city']
                }
        except Exception as e:
            logger.error(f"Ошибка загрузки команд: {e}")
    
    def get_team_name(self, team_id: int) -> str:
        """Получает название команды по ID"""
        team = self._teams_cache.get(team_id, {})
        return team.get('name', f'Team {team_id}')
    
    def get_team_abbr(self, team_id: int) -> str:
        """Получает аббревиатуру команды"""
        team = self._teams_cache.get(team_id, {})
        return team.get('abbr', 'UNK')
    
    def get_todays_games(self) -> List[NBAGame]:
        """Получает игры на сегодня"""
        if not NBA_API_AVAILABLE:
            logger.warning("nba_api недоступен")
            return []
        
        try:
            self._rate_limit()
            
            # Используем live scoreboard API
            if USE_LIVE_SCOREBOARD:
                sb = live_scoreboard.ScoreBoard()
                data = sb.get_dict()
                
                games = []
                for game in data.get('scoreboard', {}).get('games', []):
                    home_team = game.get('homeTeam', {})
                    away_team = game.get('awayTeam', {})
                    
                    # Определяем статус
                    game_status = game.get('gameStatus', 1)
                    if game_status == 1:
                        status = 'scheduled'
                    elif game_status == 2:
                        status = 'live'
                    else:
                        status = 'final'
                    
                    games.append(NBAGame(
                        game_id=game.get('gameId', ''),
                        game_date=date.today(),
                        home_team_id=home_team.get('teamId', 0),
                        home_team=home_team.get('teamName', ''),
                        home_team_abbr=home_team.get('teamTricode', ''),
                        away_team_id=away_team.get('teamId', 0),
                        away_team=away_team.get('teamName', ''),
                        away_team_abbr=away_team.get('teamTricode', ''),
                        home_score=home_team.get('score', 0),
                        away_score=away_team.get('score', 0),
                        status=status
                    ))
                
                logger.info(f"Найдено {len(games)} игр на сегодня (live API)")
                return games
            
            # Fallback: используем leaguegamefinder
            else:
                today_str = date.today().strftime('%m/%d/%Y')
                finder = leaguegamefinder.LeagueGameFinder(
                    date_from_nullable=today_str,
                    date_to_nullable=today_str,
                    league_id_nullable='00'
                )
                
                games_df = finder.get_data_frames()[0]
                
                if games_df.empty:
                    return []
                
                # Группируем по game_id
                game_ids = games_df['GAME_ID'].unique()
                games = []
                
                for game_id in game_ids:
                    game_data = games_df[games_df['GAME_ID'] == game_id]
                    
                    if len(game_data) != 2:
                        continue
                    
                    home_row = game_data[game_data['MATCHUP'].str.contains('vs.')].iloc[0]
                    away_row = game_data[game_data['MATCHUP'].str.contains('@')].iloc[0]
                    
                    games.append(NBAGame(
                        game_id=game_id,
                        game_date=date.today(),
                        home_team_id=int(home_row['TEAM_ID']),
                        home_team=home_row['TEAM_NAME'],
                        home_team_abbr=home_row['TEAM_ABBREVIATION'],
                        away_team_id=int(away_row['TEAM_ID']),
                        away_team=away_row['TEAM_NAME'],
                        away_team_abbr=away_row['TEAM_ABBREVIATION'],
                        home_score=int(home_row.get('PTS', 0) or 0),
                        away_score=int(away_row.get('PTS', 0) or 0),
                        status='final' if home_row.get('PTS') else 'scheduled'
                    ))
                
                logger.info(f"Найдено {len(games)} игр на сегодня (game finder)")
                return games
            
        except Exception as e:
            logger.error(f"Ошибка получения игр: {e}")
            return []
    
    def _parse_game_status(self, status_id: int) -> str:
        """Парсит статус игры"""
        status_map = {
            1: 'scheduled',
            2: 'live',
            3: 'final'
        }
        return status_map.get(status_id, 'unknown')
    
    def get_team_stats(self, team_id: int, season: str = "2024-25") -> Optional[TeamStats]:
        """Получает статистику команды"""
        if not NBA_API_AVAILABLE:
            return None
        
        try:
            self._rate_limit()
            
            # Получаем лог игр команды
            game_log = teamgamelog.TeamGameLog(
                team_id=team_id,
                season=season
            )
            games = game_log.get_normalized_dict()['TeamGameLog']
            
            if not games:
                return None
            
            # Считаем статистику
            wins = sum(1 for g in games if g['WL'] == 'W')
            losses = sum(1 for g in games if g['WL'] == 'L')
            
            home_games = [g for g in games if 'vs.' in g['MATCHUP']]
            road_games = [g for g in games if '@' in g['MATCHUP']]
            
            home_wins = sum(1 for g in home_games if g['WL'] == 'W')
            home_losses = sum(1 for g in home_games if g['WL'] == 'L')
            road_wins = sum(1 for g in road_games if g['WL'] == 'W')
            road_losses = sum(1 for g in road_games if g['WL'] == 'L')
            
            # Последние 10
            last_10_games = games[:10]
            l10_wins = sum(1 for g in last_10_games if g['WL'] == 'W')
            l10_losses = 10 - l10_wins
            
            # Streak
            streak = 0
            if games:
                current = games[0]['WL']
                for g in games:
                    if g['WL'] == current:
                        streak += 1 if current == 'W' else -1
                    else:
                        break
            
            return TeamStats(
                team_id=team_id,
                team_name=self.get_team_name(team_id),
                team_abbr=self.get_team_abbr(team_id),
                wins=wins,
                losses=losses,
                win_pct=wins / (wins + losses) if (wins + losses) > 0 else 0,
                home_wins=home_wins,
                home_losses=home_losses,
                road_wins=road_wins,
                road_losses=road_losses,
                streak=streak,
                last_10=f"{l10_wins}-{l10_losses}"
            )
            
        except Exception as e:
            logger.error(f"Ошибка получения статистики команды: {e}")
            return None
    
    def get_team_advanced_stats(self, team_id: int, season: str = "2024-25") -> Dict:
        """Получает продвинутую статистику"""
        if not NBA_API_AVAILABLE:
            return {}
        
        try:
            self._rate_limit()
            
            metrics = teamestimatedmetrics.TeamEstimatedMetrics(season=season)
            data = metrics.get_normalized_dict()['TeamEstimatedMetrics']
            
            for team in data:
                if team['TEAM_ID'] == team_id:
                    return {
                        'off_rating': team.get('E_OFF_RATING', 110),
                        'def_rating': team.get('E_DEF_RATING', 110),
                        'net_rating': team.get('E_NET_RATING', 0),
                        'pace': team.get('E_PACE', 100)
                    }
            
            return {}
            
        except Exception as e:
            logger.error(f"Ошибка получения advanced stats: {e}")
            return {}


# === ODDS API CLIENT ===

@dataclass
class GameOdds:
    """Коэффициенты на игру"""
    game_id: str
    home_team: str
    away_team: str
    commence_time: datetime
    
    # Moneyline
    home_odds: float = 1.90
    away_odds: float = 1.90
    
    # Best odds
    home_best_odds: float = 1.90
    away_best_odds: float = 1.90
    home_best_bookmaker: Optional[str] = None
    away_best_bookmaker: Optional[str] = None
    
    # Spread
    spread_line: Optional[float] = None
    home_spread_odds: float = 1.90
    away_spread_odds: float = 1.90
    
    # Total
    total_line: Optional[float] = None
    over_odds: float = 1.90
    under_odds: float = 1.90


class OddsAPIClient:
    """
    Клиент для получения коэффициентов
    
    Использует The Odds API (https://the-odds-api.com/)
    Бесплатный план: 500 запросов/месяц
    """
    
    BASE_URL = "https://api.the-odds-api.com/v4"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.api.odds_api_key
        self.regions = config.api.odds_api_regions
        
        if not self.api_key:
            logger.warning("ODDS_API_KEY не установлен. Получите на https://the-odds-api.com/")
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Выполняет запрос к API"""
        if not REQUESTS_AVAILABLE:
            logger.error("requests не установлен")
            return None
        
        if not self.api_key:
            return None
        
        params = params or {}
        params['apiKey'] = self.api_key
        
        try:
            url = f"{self.BASE_URL}/{endpoint}"
            response = requests.get(url, params=params, timeout=10)
            
            # Логируем оставшиеся запросы
            remaining = response.headers.get('x-requests-remaining')
            if remaining:
                logger.info(f"Осталось запросов к Odds API: {remaining}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка запроса к Odds API: {e}")
            return None
    
    def get_nba_odds(
        self,
        markets: List[str] = None,
        bookmakers: List[str] = None
    ) -> List[GameOdds]:
        """
        Получает коэффициенты на игры NBA
        
        Args:
            markets: Типы ставок ['h2h', 'spreads', 'totals']
            bookmakers: Букмекеры для сравнения
        
        Returns:
            Список GameOdds
        """
        markets = markets or ['h2h', 'spreads', 'totals']
        bookmakers = bookmakers or config.api.bookmakers
        
        params = {
            'regions': ','.join(self.regions),
            'markets': ','.join(markets),
            'bookmakers': ','.join(bookmakers),
            'oddsFormat': 'decimal'
        }
        
        data = self._make_request('sports/basketball_nba/odds', params)
        
        if not data:
            return []
        
        odds_list = []
        
        for game in data:
            game_odds = self._parse_game_odds(game)
            if game_odds:
                odds_list.append(game_odds)
        
        return odds_list
    
    def _parse_game_odds(self, game_data: Dict) -> Optional[GameOdds]:
        """Парсит коэффициенты игры"""
        try:
            home_team = game_data.get('home_team', '')
            away_team = game_data.get('away_team', '')
            
            odds = GameOdds(
                game_id=game_data.get('id', ''),
                home_team=home_team,
                away_team=away_team,
                commence_time=datetime.fromisoformat(
                    game_data.get('commence_time', '').replace('Z', '+00:00')
                )
            )
            
            # Обрабатываем коэффициенты от разных букмекеров
            best_home = {'odds': 0, 'bookmaker': None}
            best_away = {'odds': 0, 'bookmaker': None}
            
            for bookmaker in game_data.get('bookmakers', []):
                bookmaker_name = bookmaker.get('title', '')
                
                for market in bookmaker.get('markets', []):
                    market_key = market.get('key', '')
                    
                    if market_key == 'h2h':
                        for outcome in market.get('outcomes', []):
                            team = outcome.get('name', '')
                            price = outcome.get('price', 1.0)
                            
                            if team == home_team:
                                if price > best_home['odds']:
                                    best_home = {'odds': price, 'bookmaker': bookmaker_name}
                            elif team == away_team:
                                if price > best_away['odds']:
                                    best_away = {'odds': price, 'bookmaker': bookmaker_name}
                    
                    elif market_key == 'spreads':
                        for outcome in market.get('outcomes', []):
                            if outcome.get('name') == home_team:
                                odds.spread_line = outcome.get('point')
                                odds.home_spread_odds = outcome.get('price', 1.90)
                            else:
                                odds.away_spread_odds = outcome.get('price', 1.90)
                    
                    elif market_key == 'totals':
                        for outcome in market.get('outcomes', []):
                            if outcome.get('name') == 'Over':
                                odds.total_line = outcome.get('point')
                                odds.over_odds = outcome.get('price', 1.90)
                            else:
                                odds.under_odds = outcome.get('price', 1.90)
            
            odds.home_odds = best_home['odds'] or 1.90
            odds.away_odds = best_away['odds'] or 1.90
            odds.home_best_odds = best_home['odds']
            odds.away_best_odds = best_away['odds']
            odds.home_best_bookmaker = best_home['bookmaker']
            odds.away_best_bookmaker = best_away['bookmaker']
            
            return odds
            
        except Exception as e:
            logger.error(f"Ошибка парсинга коэффициентов: {e}")
            return None
    
    def get_remaining_requests(self) -> Optional[int]:
        """Проверяет оставшиеся запросы"""
        # Делаем легкий запрос к списку видов спорта
        if not REQUESTS_AVAILABLE or not self.api_key:
            return None
        
        try:
            response = requests.get(
                f"{self.BASE_URL}/sports",
                params={'apiKey': self.api_key},
                timeout=5
            )
            return int(response.headers.get('x-requests-remaining', 0))
        except:
            return None


# === MOCK DATA FOR TESTING ===

class MockDataProvider:
    """
    Провайдер тестовых данных
    Используется когда API недоступны
    """
    
    @staticmethod
    def get_mock_games() -> List[NBAGame]:
        """Возвращает тестовые игры"""
        today = date.today()
        
        return [
            NBAGame(
                game_id="0022400001",
                game_date=today,
                home_team_id=1610612747,
                home_team="Los Angeles Lakers",
                home_team_abbr="LAL",
                away_team_id=1610612744,
                away_team="Golden State Warriors",
                away_team_abbr="GSW",
                status="scheduled",
                game_time="19:30"
            ),
            NBAGame(
                game_id="0022400002",
                game_date=today,
                home_team_id=1610612738,
                home_team="Boston Celtics",
                home_team_abbr="BOS",
                away_team_id=1610612748,
                away_team="Miami Heat",
                away_team_abbr="MIA",
                status="scheduled",
                game_time="20:00"
            ),
            NBAGame(
                game_id="0022400003",
                game_date=today,
                home_team_id=1610612743,
                home_team="Denver Nuggets",
                home_team_abbr="DEN",
                away_team_id=1610612756,
                away_team="Phoenix Suns",
                away_team_abbr="PHX",
                status="scheduled",
                game_time="21:00"
            ),
        ]
    
    @staticmethod
    def get_mock_odds() -> List[GameOdds]:
        """Возвращает тестовые коэффициенты"""
        now = datetime.now()
        
        return [
            GameOdds(
                game_id="0022400001",
                home_team="Los Angeles Lakers",
                away_team="Golden State Warriors",
                commence_time=now + timedelta(hours=3),
                home_odds=1.75,
                away_odds=2.10,
                home_best_odds=1.78,
                away_best_odds=2.15,
                home_best_bookmaker="Pinnacle",
                away_best_bookmaker="DraftKings",
                spread_line=-3.5,
                total_line=225.5
            ),
            GameOdds(
                game_id="0022400002",
                home_team="Boston Celtics",
                away_team="Miami Heat",
                commence_time=now + timedelta(hours=4),
                home_odds=1.45,
                away_odds=2.75,
                home_best_odds=1.48,
                away_best_odds=2.80,
                home_best_bookmaker="Bet365",
                away_best_bookmaker="FanDuel",
                spread_line=-7.5,
                total_line=218.5
            ),
            GameOdds(
                game_id="0022400003",
                home_team="Denver Nuggets",
                away_team="Phoenix Suns",
                commence_time=now + timedelta(hours=5),
                home_odds=1.65,
                away_odds=2.25,
                home_best_odds=1.68,
                away_best_odds=2.30,
                home_best_bookmaker="BetMGM",
                away_best_bookmaker="Pinnacle",
                spread_line=-4.5,
                total_line=230.0
            ),
        ]
    
    @staticmethod
    def get_mock_team_stats() -> Dict[str, TeamStats]:
        """Возвращает тестовую статистику команд"""
        return {
            "Los Angeles Lakers": TeamStats(
                team_id=1610612747,
                team_name="Los Angeles Lakers",
                team_abbr="LAL",
                wins=28, losses=18,
                win_pct=0.609,
                home_wins=16, home_losses=7,
                road_wins=12, road_losses=11,
                off_rating=115.5, def_rating=112.3,
                net_rating=3.2, pace=100.5,
                streak=3, last_10="7-3"
            ),
            "Golden State Warriors": TeamStats(
                team_id=1610612744,
                team_name="Golden State Warriors",
                team_abbr="GSW",
                wins=25, losses=21,
                win_pct=0.543,
                home_wins=14, home_losses=9,
                road_wins=11, road_losses=12,
                off_rating=118.0, def_rating=115.5,
                net_rating=2.5, pace=102.3,
                streak=-1, last_10="5-5"
            ),
            "Boston Celtics": TeamStats(
                team_id=1610612738,
                team_name="Boston Celtics",
                team_abbr="BOS",
                wins=35, losses=10,
                win_pct=0.778,
                home_wins=19, home_losses=3,
                road_wins=16, road_losses=7,
                off_rating=122.5, def_rating=110.2,
                net_rating=12.3, pace=99.8,
                streak=5, last_10="8-2"
            ),
        }


# === UNIFIED DATA SERVICE ===

class DataService:
    """
    Единый сервис данных
    Автоматически выбирает источник (API или mock)
    """
    
    def __init__(
        self,
        use_real_api: bool = True,
        odds_api_key: str = None
    ):
        self.use_real_api = use_real_api
        
        # Инициализируем клиенты
        self.nba_client = NBADataClient() if use_real_api else None
        self.odds_client = OddsAPIClient(odds_api_key) if use_real_api else None
        self.mock = MockDataProvider()
    
    def get_todays_games(self) -> List[NBAGame]:
        """Получает игры на сегодня"""
        if self.use_real_api and self.nba_client:
            games = self.nba_client.get_todays_games()
            if games:
                return games
        
        logger.info("Используем тестовые данные игр")
        return self.mock.get_mock_games()
    
    def get_odds(self) -> List[GameOdds]:
        """Получает коэффициенты"""
        if self.use_real_api and self.odds_client:
            odds = self.odds_client.get_nba_odds()
            if odds:
                return odds
        
        logger.info("Используем тестовые коэффициенты")
        return self.mock.get_mock_odds()
    
    def get_team_stats(self, team_name: str) -> Optional[TeamStats]:
        """Получает статистику команды"""
        mock_stats = self.mock.get_mock_team_stats()
        return mock_stats.get(team_name)


# === ТЕСТИРОВАНИЕ ===

if __name__ == "__main__":
    print("=== Тест API Clients ===\n")
    
    # Тест с mock данными
    print("Тест 1: Mock Data Provider")
    print("-" * 40)
    
    mock = MockDataProvider()
    
    games = mock.get_mock_games()
    print(f"Тестовых игр: {len(games)}")
    for game in games:
        print(f"  {game.away_team_abbr} @ {game.home_team_abbr} ({game.game_time})")
    
    odds = mock.get_mock_odds()
    print(f"\nТестовых коэффициентов: {len(odds)}")
    for o in odds:
        print(f"  {o.away_team} @ {o.home_team}: {o.away_odds:.2f} / {o.home_odds:.2f}")
        print(f"    Spread: {o.spread_line}, Total: {o.total_line}")
    
    # Тест DataService
    print("\n\nТест 2: Data Service")
    print("-" * 40)
    
    service = DataService(use_real_api=False)
    
    games = service.get_todays_games()
    odds = service.get_odds()
    
    print(f"Игр получено: {len(games)}")
    print(f"Коэффициентов получено: {len(odds)}")
    
    stats = service.get_team_stats("Boston Celtics")
    if stats:
        print(f"\nСтатистика Boston Celtics:")
        print(f"  Record: {stats.wins}-{stats.losses} ({stats.win_pct:.1%})")
        print(f"  Net Rating: {stats.net_rating:+.1f}")
        print(f"  Streak: {stats.streak}")
    
    # Проверка реального API (если ключ есть)
    print("\n\nТест 3: Real API Check")
    print("-" * 40)
    
    if NBA_API_AVAILABLE:
        print("✅ nba_api доступен")
    else:
        print("❌ nba_api не установлен (pip install nba_api)")
    
    odds_key = os.getenv('ODDS_API_KEY')
    if odds_key:
        client = OddsAPIClient(odds_key)
        remaining = client.get_remaining_requests()
        print(f"✅ Odds API ключ найден. Осталось запросов: {remaining}")
    else:
        print("❌ ODDS_API_KEY не установлен")
