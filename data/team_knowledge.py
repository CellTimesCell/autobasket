"""
AutoBasket - Team Knowledge Base
================================
Сбор, хранение и анализ исторических данных команд.

Логика:
1. Видим игру → проверяем есть ли данные по командам
2. Нет данных → собираем за 10 лет
3. Есть данные → используем + дополняем свежими
4. Храним отдельно: historical (прошлое) и recent (текущий сезон)
"""

import os
import sqlite3
import logging
import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import json

try:
    from nba_api.stats.endpoints import (
        teamgamelog, franchisehistory, commonteamroster,
        leaguegamefinder, teamyearbyyearstats
    )
    from nba_api.stats.static import teams as nba_teams
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TeamHistoricalGame:
    """Историческая игра команды"""
    game_id: str
    game_date: str
    season: str
    
    # Команды
    team_id: int
    team_name: str
    team_abbr: str
    opponent_id: int
    opponent_name: str
    opponent_abbr: str
    
    # Локация
    is_home: bool
    arena: str = ""
    
    # Результат
    won: bool = False
    team_score: int = 0
    opponent_score: int = 0
    margin: int = 0
    
    # Статистика игры
    fg_pct: float = 0.0
    fg3_pct: float = 0.0
    ft_pct: float = 0.0
    rebounds: int = 0
    assists: int = 0
    steals: int = 0
    blocks: int = 0
    turnovers: int = 0
    
    # Контекст (заполняется позже)
    coach: str = ""
    injuries: str = ""  # JSON список
    odds_spread: float = 0.0
    odds_total: float = 0.0
    odds_moneyline: float = 0.0


@dataclass
class TeamProfile:
    """Профиль команды с накопленными знаниями"""
    team_id: int
    team_name: str
    team_abbr: str
    
    # Базовая инфо
    city: str = ""
    arena: str = ""
    conference: str = ""
    division: str = ""
    
    # Исторические данные
    data_from_year: int = 2015
    data_to_year: int = 2025
    total_games_collected: int = 0
    last_updated: str = ""
    
    # Агрегированная статистика
    all_time_wins: int = 0
    all_time_losses: int = 0
    all_time_win_pct: float = 0.0
    
    home_wins: int = 0
    home_losses: int = 0
    home_win_pct: float = 0.0
    
    away_wins: int = 0
    away_losses: int = 0
    away_win_pct: float = 0.0
    
    # Средние показатели
    avg_points_scored: float = 0.0
    avg_points_allowed: float = 0.0
    avg_margin: float = 0.0
    
    # Тренды (JSON)
    yearly_records: str = ""  # {"2024": "45-37", "2023": "42-40", ...}
    coaches_history: str = ""  # [{"coach": "Name", "years": "2020-2024", "record": "150-120"}, ...]


class TeamKnowledgeBase:
    """
    База знаний о командах NBA
    
    Хранит:
    - historical_games: все игры за 10+ лет
    - team_profiles: профили команд с агрегатами
    - h2h_records: история личных встреч
    - recent_games: игры текущего сезона (обновляются ежедневно)
    """
    
    def __init__(self, db_path: str = "team_knowledge.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        
        # Кэш команд NBA
        self._teams_cache = {}
        if NBA_API_AVAILABLE:
            self._load_teams()
        
        # Rate limiting для API
        self._last_request = 0
        self.request_delay = 0.6
    
    def _create_tables(self):
        """Создает таблицы"""
        cursor = self.conn.cursor()
        
        # Профили команд
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS team_profiles (
                team_id INTEGER PRIMARY KEY,
                team_name TEXT,
                team_abbr TEXT,
                city TEXT,
                arena TEXT,
                conference TEXT,
                division TEXT,
                data_from_year INTEGER,
                data_to_year INTEGER,
                total_games_collected INTEGER DEFAULT 0,
                last_updated TEXT,
                all_time_wins INTEGER DEFAULT 0,
                all_time_losses INTEGER DEFAULT 0,
                all_time_win_pct REAL DEFAULT 0,
                home_wins INTEGER DEFAULT 0,
                home_losses INTEGER DEFAULT 0,
                home_win_pct REAL DEFAULT 0,
                away_wins INTEGER DEFAULT 0,
                away_losses INTEGER DEFAULT 0,
                away_win_pct REAL DEFAULT 0,
                avg_points_scored REAL DEFAULT 0,
                avg_points_allowed REAL DEFAULT 0,
                avg_margin REAL DEFAULT 0,
                yearly_records TEXT,
                coaches_history TEXT
            )
        """)
        
        # Исторические игры
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT,
                game_date TEXT,
                season TEXT,
                team_id INTEGER,
                team_name TEXT,
                team_abbr TEXT,
                opponent_id INTEGER,
                opponent_name TEXT,
                opponent_abbr TEXT,
                is_home INTEGER,
                arena TEXT,
                won INTEGER,
                team_score INTEGER,
                opponent_score INTEGER,
                margin INTEGER,
                fg_pct REAL,
                fg3_pct REAL,
                ft_pct REAL,
                rebounds INTEGER,
                assists INTEGER,
                steals INTEGER,
                blocks INTEGER,
                turnovers INTEGER,
                coach TEXT,
                injuries TEXT,
                odds_spread REAL,
                odds_total REAL,
                odds_moneyline REAL,
                UNIQUE(game_id, team_id)
            )
        """)
        
        # Индексы
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_team ON historical_games(team_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_date ON historical_games(game_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_season ON historical_games(season)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_opponent ON historical_games(opponent_id)")
        
        # H2H записи
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS h2h_records (
                team1_id INTEGER,
                team2_id INTEGER,
                total_games INTEGER DEFAULT 0,
                team1_wins INTEGER DEFAULT 0,
                team2_wins INTEGER DEFAULT 0,
                avg_total_score REAL DEFAULT 0,
                avg_margin REAL DEFAULT 0,
                last_updated TEXT,
                games_json TEXT,
                PRIMARY KEY (team1_id, team2_id)
            )
        """)
        
        # Свежие игры (текущий сезон)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recent_games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT UNIQUE,
                game_date TEXT,
                home_team_id INTEGER,
                home_team TEXT,
                away_team_id INTEGER,
                away_team TEXT,
                home_score INTEGER,
                away_score INTEGER,
                status TEXT,
                our_prediction REAL,
                our_bet_side TEXT,
                our_bet_amount REAL,
                our_bet_result TEXT,
                analysis_json TEXT,
                created_at TEXT
            )
        """)
        
        self.conn.commit()
    
    def _load_teams(self):
        """Загружает список команд NBA"""
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
    
    def _rate_limit(self):
        """Rate limiting"""
        elapsed = time.time() - self._last_request
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request = time.time()
    
    # =========================================================================
    # ПРОВЕРКА НАЛИЧИЯ ДАННЫХ
    # =========================================================================
    
    def has_team_data(self, team_id: int, min_games: int = 100) -> bool:
        """Проверяет есть ли достаточно данных по команде"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT total_games_collected FROM team_profiles WHERE team_id = ?",
            (team_id,)
        )
        row = cursor.fetchone()
        
        if not row:
            return False
        
        return row['total_games_collected'] >= min_games
    
    def get_team_profile(self, team_id: int) -> Optional[TeamProfile]:
        """Получает профиль команды"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM team_profiles WHERE team_id = ?", (team_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        return TeamProfile(**dict(row))
    
    def get_missing_teams(self, team_ids: List[int]) -> List[int]:
        """Возвращает список команд без данных"""
        missing = []
        for team_id in team_ids:
            if not self.has_team_data(team_id):
                missing.append(team_id)
        return missing
    
    # =========================================================================
    # СБОР ИСТОРИЧЕСКИХ ДАННЫХ
    # =========================================================================
    
    def collect_team_history(
        self,
        team_id: int,
        years_back: int = 10,
        progress_callback=None
    ) -> int:
        """
        Собирает полную историю команды за N лет
        
        Returns:
            Количество собранных игр
        """
        if not NBA_API_AVAILABLE:
            logger.error("nba_api не доступен")
            return 0
        
        team_info = self._teams_cache.get(team_id, {})
        team_name = team_info.get('name', f'Team {team_id}')
        team_abbr = team_info.get('abbr', 'UNK')
        
        logger.info(f"📊 Collecting history for {team_name} ({years_back} years)...")
        
        current_year = datetime.now().year
        start_year = current_year - years_back
        
        total_games = 0
        yearly_records = {}
        
        for year in range(start_year, current_year + 1):
            season = f"{year}-{str(year+1)[-2:]}"
            
            if progress_callback:
                progress_callback(f"Fetching {team_name} {season}...")
            
            try:
                self._rate_limit()
                
                # Получаем лог игр сезона
                game_log = teamgamelog.TeamGameLog(
                    team_id=team_id,
                    season=season,
                    season_type_all_star='Regular Season'
                )
                
                games_df = game_log.get_data_frames()[0]
                
                if games_df.empty:
                    continue
                
                season_wins = 0
                season_losses = 0
                
                for _, row in games_df.iterrows():
                    game = self._parse_game_log_row(row, team_id, team_name, team_abbr, season)
                    
                    if game:
                        self._save_historical_game(game)
                        total_games += 1
                        
                        if game.won:
                            season_wins += 1
                        else:
                            season_losses += 1
                
                yearly_records[season] = f"{season_wins}-{season_losses}"
                logger.info(f"   {season}: {season_wins}-{season_losses} ({len(games_df)} games)")
                
                time.sleep(0.5)  # Дополнительная пауза между сезонами
                
            except Exception as e:
                logger.error(f"Error fetching {season}: {e}")
                continue
        
        # Обновляем профиль команды
        self._update_team_profile(team_id, team_name, team_abbr, yearly_records, total_games)
        
        logger.info(f"✅ Collected {total_games} games for {team_name}")
        
        return total_games
    
    def _parse_game_log_row(
        self,
        row,
        team_id: int,
        team_name: str,
        team_abbr: str,
        season: str
    ) -> Optional[TeamHistoricalGame]:
        """Парсит строку из TeamGameLog"""
        try:
            matchup = row['MATCHUP']
            is_home = 'vs.' in matchup
            
            # Определяем оппонента
            if is_home:
                opponent_abbr = matchup.split('vs.')[-1].strip()
            else:
                opponent_abbr = matchup.split('@')[-1].strip()
            
            # Находим ID оппонента
            opponent_id = 0
            opponent_name = opponent_abbr
            for tid, tinfo in self._teams_cache.items():
                if tinfo['abbr'] == opponent_abbr:
                    opponent_id = tid
                    opponent_name = tinfo['name']
                    break
            
            # Результат
            wl = row['WL']
            won = wl == 'W'
            
            team_score = int(row['PTS'])
            
            # Очки оппонента вычисляем из +/-
            plus_minus = row.get('PLUS_MINUS', 0) or 0
            if won:
                opponent_score = team_score - abs(plus_minus)
            else:
                opponent_score = team_score + abs(plus_minus)
            
            return TeamHistoricalGame(
                game_id=row['Game_ID'],
                game_date=row['GAME_DATE'],
                season=season,
                team_id=team_id,
                team_name=team_name,
                team_abbr=team_abbr,
                opponent_id=opponent_id,
                opponent_name=opponent_name,
                opponent_abbr=opponent_abbr,
                is_home=is_home,
                won=won,
                team_score=team_score,
                opponent_score=opponent_score,
                margin=team_score - opponent_score,
                fg_pct=float(row.get('FG_PCT', 0) or 0),
                fg3_pct=float(row.get('FG3_PCT', 0) or 0),
                ft_pct=float(row.get('FT_PCT', 0) or 0),
                rebounds=int(row.get('REB', 0) or 0),
                assists=int(row.get('AST', 0) or 0),
                steals=int(row.get('STL', 0) or 0),
                blocks=int(row.get('BLK', 0) or 0),
                turnovers=int(row.get('TOV', 0) or 0)
            )
            
        except Exception as e:
            logger.error(f"Error parsing game row: {e}")
            return None
    
    def _save_historical_game(self, game: TeamHistoricalGame):
        """Сохраняет игру в БД"""
        cursor = self.conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO historical_games (
                    game_id, game_date, season, team_id, team_name, team_abbr,
                    opponent_id, opponent_name, opponent_abbr, is_home, arena,
                    won, team_score, opponent_score, margin,
                    fg_pct, fg3_pct, ft_pct, rebounds, assists, steals, blocks, turnovers,
                    coach, injuries, odds_spread, odds_total, odds_moneyline
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                game.game_id, game.game_date, game.season,
                game.team_id, game.team_name, game.team_abbr,
                game.opponent_id, game.opponent_name, game.opponent_abbr,
                int(game.is_home), game.arena,
                int(game.won), game.team_score, game.opponent_score, game.margin,
                game.fg_pct, game.fg3_pct, game.ft_pct,
                game.rebounds, game.assists, game.steals, game.blocks, game.turnovers,
                game.coach, game.injuries,
                game.odds_spread, game.odds_total, game.odds_moneyline
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error saving game: {e}")
    
    def _update_team_profile(
        self,
        team_id: int,
        team_name: str,
        team_abbr: str,
        yearly_records: Dict,
        total_games: int
    ):
        """Обновляет профиль команды с агрегатами"""
        cursor = self.conn.cursor()
        
        # Считаем агрегаты
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(won) as wins,
                SUM(CASE WHEN is_home = 1 THEN 1 ELSE 0 END) as home_games,
                SUM(CASE WHEN is_home = 1 AND won = 1 THEN 1 ELSE 0 END) as home_wins,
                SUM(CASE WHEN is_home = 0 THEN 1 ELSE 0 END) as away_games,
                SUM(CASE WHEN is_home = 0 AND won = 1 THEN 1 ELSE 0 END) as away_wins,
                AVG(team_score) as avg_scored,
                AVG(opponent_score) as avg_allowed,
                AVG(margin) as avg_margin
            FROM historical_games
            WHERE team_id = ?
        """, (team_id,))
        
        stats = cursor.fetchone()
        
        total = stats['total'] or 0
        wins = stats['wins'] or 0
        losses = total - wins
        
        home_games = stats['home_games'] or 0
        home_wins = stats['home_wins'] or 0
        home_losses = home_games - home_wins
        
        away_games = stats['away_games'] or 0
        away_wins = stats['away_wins'] or 0
        away_losses = away_games - away_wins
        
        # Сохраняем профиль
        cursor.execute("""
            INSERT OR REPLACE INTO team_profiles (
                team_id, team_name, team_abbr,
                total_games_collected, last_updated,
                all_time_wins, all_time_losses, all_time_win_pct,
                home_wins, home_losses, home_win_pct,
                away_wins, away_losses, away_win_pct,
                avg_points_scored, avg_points_allowed, avg_margin,
                yearly_records
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            team_id, team_name, team_abbr,
            total, datetime.now().isoformat(),
            wins, losses, wins / total if total > 0 else 0,
            home_wins, home_losses, home_wins / home_games if home_games > 0 else 0,
            away_wins, away_losses, away_wins / away_games if away_games > 0 else 0,
            stats['avg_scored'] or 0,
            stats['avg_allowed'] or 0,
            stats['avg_margin'] or 0,
            json.dumps(yearly_records)
        ))
        
        self.conn.commit()
    
    # =========================================================================
    # ПОЛУЧЕНИЕ ДАННЫХ ДЛЯ АНАЛИЗА
    # =========================================================================
    
    def get_team_games(
        self,
        team_id: int,
        limit: int = 100,
        season: str = None
    ) -> List[TeamHistoricalGame]:
        """Получает игры команды"""
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM historical_games WHERE team_id = ?"
        params = [team_id]
        
        if season:
            query += " AND season = ?"
            params.append(season)
        
        query += " ORDER BY game_date DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        
        games = []
        for row in cursor.fetchall():
            games.append(TeamHistoricalGame(
                game_id=row['game_id'],
                game_date=row['game_date'],
                season=row['season'],
                team_id=row['team_id'],
                team_name=row['team_name'],
                team_abbr=row['team_abbr'],
                opponent_id=row['opponent_id'],
                opponent_name=row['opponent_name'],
                opponent_abbr=row['opponent_abbr'],
                is_home=bool(row['is_home']),
                won=bool(row['won']),
                team_score=row['team_score'],
                opponent_score=row['opponent_score'],
                margin=row['margin'],
                fg_pct=row['fg_pct'],
                fg3_pct=row['fg3_pct'],
                ft_pct=row['ft_pct'],
                rebounds=row['rebounds'],
                assists=row['assists'],
                steals=row['steals'],
                blocks=row['blocks'],
                turnovers=row['turnovers']
            ))
        
        return games
    
    def get_h2h_history(
        self,
        team1_id: int,
        team2_id: int,
        limit: int = 20
    ) -> Dict:
        """Получает историю личных встреч"""
        cursor = self.conn.cursor()
        
        # Игры где team1 играл против team2
        cursor.execute("""
            SELECT * FROM historical_games
            WHERE team_id = ? AND opponent_id = ?
            ORDER BY game_date DESC
            LIMIT ?
        """, (team1_id, team2_id, limit))
        
        games = cursor.fetchall()
        
        team1_wins = sum(1 for g in games if g['won'])
        team2_wins = len(games) - team1_wins
        
        avg_total = sum(g['team_score'] + g['opponent_score'] for g in games) / len(games) if games else 0
        avg_margin = sum(g['margin'] for g in games) / len(games) if games else 0
        
        return {
            'total_games': len(games),
            'team1_wins': team1_wins,
            'team2_wins': team2_wins,
            'avg_total': avg_total,
            'avg_margin': avg_margin,
            'last_5': [
                {
                    'date': g['game_date'],
                    'score': f"{g['team_score']}-{g['opponent_score']}",
                    'winner': g['team_name'] if g['won'] else g['opponent_name']
                }
                for g in games[:5]
            ]
        }
    
    def get_team_trends(self, team_id: int, games: int = 10) -> Dict:
        """Получает текущие тренды команды"""
        recent = self.get_team_games(team_id, limit=games)
        
        if not recent:
            return {}
        
        wins = sum(1 for g in recent if g.won)
        
        # Текущая серия
        streak = 0
        streak_type = 'W' if recent[0].won else 'L'
        for g in recent:
            if (g.won and streak_type == 'W') or (not g.won and streak_type == 'L'):
                streak += 1
            else:
                break
        
        # Домашние/выездные в последних N
        home_games = [g for g in recent if g.is_home]
        away_games = [g for g in recent if not g.is_home]
        
        return {
            'last_n': games,
            'record': f"{wins}-{games - wins}",
            'win_pct': wins / games,
            'current_streak': f"{streak_type}{streak}",
            'avg_scored': sum(g.team_score for g in recent) / len(recent),
            'avg_allowed': sum(g.opponent_score for g in recent) / len(recent),
            'home_record': f"{sum(1 for g in home_games if g.won)}-{len(home_games) - sum(1 for g in home_games if g.won)}" if home_games else "0-0",
            'away_record': f"{sum(1 for g in away_games if g.won)}-{len(away_games) - sum(1 for g in away_games if g.won)}" if away_games else "0-0"
        }
    
    # =========================================================================
    # СОХРАНЕНИЕ СВЕЖИХ ИГР
    # =========================================================================
    
    def save_todays_game(
        self,
        game_id: str,
        home_team_id: int,
        home_team: str,
        away_team_id: int,
        away_team: str,
        our_prediction: float = None,
        analysis: Dict = None
    ):
        """Сохраняет игру текущего дня"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO recent_games (
                game_id, game_date, home_team_id, home_team,
                away_team_id, away_team, status, our_prediction,
                analysis_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            game_id, date.today().isoformat(),
            home_team_id, home_team,
            away_team_id, away_team,
            'scheduled', our_prediction,
            json.dumps(analysis) if analysis else None,
            datetime.now().isoformat()
        ))
        
        self.conn.commit()
    
    def update_game_result(
        self,
        game_id: str,
        home_score: int,
        away_score: int,
        bet_result: str = None
    ):
        """Обновляет результат игры"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            UPDATE recent_games
            SET home_score = ?, away_score = ?, status = ?, our_bet_result = ?
            WHERE game_id = ?
        """, (home_score, away_score, 'final', bet_result, game_id))
        
        self.conn.commit()
    
    # =========================================================================
    # СТАТИСТИКА БАЗЫ
    # =========================================================================
    
    def get_stats(self) -> Dict:
        """Возвращает статистику базы знаний"""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) as cnt FROM team_profiles")
        teams = cursor.fetchone()['cnt']
        
        cursor.execute("SELECT COUNT(*) as cnt FROM historical_games")
        games = cursor.fetchone()['cnt']
        
        cursor.execute("SELECT COUNT(DISTINCT season) as cnt FROM historical_games")
        seasons = cursor.fetchone()['cnt']
        
        cursor.execute("SELECT MIN(game_date) as min_date, MAX(game_date) as max_date FROM historical_games")
        dates = cursor.fetchone()
        
        return {
            'teams_with_data': teams,
            'total_historical_games': games,
            'seasons_covered': seasons,
            'date_range': f"{dates['min_date']} to {dates['max_date']}" if dates['min_date'] else "No data"
        }


# =========================================================================
# ТЕСТИРОВАНИЕ
# =========================================================================

if __name__ == "__main__":
    print("=== Team Knowledge Base Test ===\n")
    
    kb = TeamKnowledgeBase()
    
    # Показываем текущую статистику
    stats = kb.get_stats()
    print("📊 Current Knowledge Base:")
    for k, v in stats.items():
        print(f"   {k}: {v}")
    
    if not NBA_API_AVAILABLE:
        print("\n❌ nba_api not available")
    else:
        print("\n✅ nba_api available")
        
        # Тестируем сбор данных для одной команды
        # Lakers team_id = 1610612747
        test_team_id = 1610612747
        
        if not kb.has_team_data(test_team_id):
            print(f"\n🔄 Collecting Lakers history (3 years for test)...")
            games = kb.collect_team_history(test_team_id, years_back=3)
            print(f"   Collected {games} games")
        else:
            print("\n✅ Lakers data already exists")
            profile = kb.get_team_profile(test_team_id)
            if profile:
                print(f"   Total games: {profile.total_games_collected}")
                print(f"   Win %: {profile.all_time_win_pct:.1%}")
        
        # Показываем тренды
        trends = kb.get_team_trends(test_team_id, 10)
        if trends:
            print(f"\n📈 Lakers recent trends:")
            for k, v in trends.items():
                print(f"   {k}: {v}")
    
    print("\n✅ Test complete")
