"""
AutoBasket - Historical Database
================================
Сбор и хранение исторических данных NBA с 2000+ года
Источники: basketball-reference.com, nba_api
"""

import logging
import sqlite3
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path

try:
    import requests
    from bs4 import BeautifulSoup
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False

try:
    from nba_api.stats.endpoints import leaguegamefinder, teamyearbyyearstats
    from nba_api.stats.static import teams as nba_teams
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HistoricalGame:
    """Историческая игра"""
    game_id: str
    season: str  # "2023-24"
    game_date: date
    
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    
    # Детали
    home_q1: int = 0
    home_q2: int = 0
    home_q3: int = 0
    home_q4: int = 0
    home_ot: int = 0
    
    away_q1: int = 0
    away_q2: int = 0
    away_q3: int = 0
    away_q4: int = 0
    away_ot: int = 0
    
    # Контекст
    playoffs: bool = False
    home_coach: str = ""
    away_coach: str = ""
    attendance: int = 0
    arena: str = ""


@dataclass
class SeasonStats:
    """Статистика команды за сезон"""
    team: str
    season: str
    
    wins: int = 0
    losses: int = 0
    win_pct: float = 0.0
    
    # Позиция
    conference: str = ""
    division: str = ""
    conf_rank: int = 0
    playoff_result: str = ""  # "Champion", "Finals", "Conf Finals", etc.
    
    # Статистика
    ppg: float = 0.0
    opp_ppg: float = 0.0
    pace: float = 0.0
    off_rating: float = 0.0
    def_rating: float = 0.0
    net_rating: float = 0.0
    
    # Тренер
    coach: str = ""
    coach_first_season: bool = False


@dataclass
class CoachRecord:
    """Запись о тренере"""
    coach_name: str
    team: str
    season_start: str
    season_end: str
    
    total_wins: int = 0
    total_losses: int = 0
    playoff_appearances: int = 0
    championships: int = 0
    
    # Стиль
    avg_pace: float = 0.0
    avg_off_rating: float = 0.0
    avg_def_rating: float = 0.0
    style: str = ""  # "offensive", "defensive", "balanced"


class HistoricalDatabase:
    """
    База исторических данных NBA
    """
    
    def __init__(self, db_path: str = "nba_history.db"):
        self.db_path = Path(db_path)
        self._init_db()
    
    def _init_db(self):
        """Создает таблицы"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Таблица игр
        c.execute('''
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                season TEXT,
                game_date DATE,
                home_team TEXT,
                away_team TEXT,
                home_score INTEGER,
                away_score INTEGER,
                home_q1 INTEGER, home_q2 INTEGER, home_q3 INTEGER, home_q4 INTEGER, home_ot INTEGER,
                away_q1 INTEGER, away_q2 INTEGER, away_q3 INTEGER, away_q4 INTEGER, away_ot INTEGER,
                playoffs INTEGER,
                home_coach TEXT,
                away_coach TEXT,
                attendance INTEGER,
                arena TEXT
            )
        ''')
        
        # Таблица сезонов команд
        c.execute('''
            CREATE TABLE IF NOT EXISTS team_seasons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team TEXT,
                season TEXT,
                wins INTEGER,
                losses INTEGER,
                win_pct REAL,
                conference TEXT,
                division TEXT,
                conf_rank INTEGER,
                playoff_result TEXT,
                ppg REAL,
                opp_ppg REAL,
                pace REAL,
                off_rating REAL,
                def_rating REAL,
                net_rating REAL,
                coach TEXT,
                UNIQUE(team, season)
            )
        ''')
        
        # Таблица тренеров
        c.execute('''
            CREATE TABLE IF NOT EXISTS coaches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coach_name TEXT,
                team TEXT,
                season_start TEXT,
                season_end TEXT,
                total_wins INTEGER,
                total_losses INTEGER,
                playoff_appearances INTEGER,
                championships INTEGER,
                avg_pace REAL,
                avg_off_rating REAL,
                avg_def_rating REAL,
                style TEXT,
                UNIQUE(coach_name, team, season_start)
            )
        ''')
        
        # Индексы
        c.execute('CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_games_teams ON games(home_team, away_team)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_games_season ON games(season)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_seasons_team ON team_seasons(team)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_coaches_name ON coaches(coach_name)')
        
        conn.commit()
        conn.close()
    
    def get_stats(self) -> Dict:
        """Возвращает статистику базы"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('SELECT COUNT(*) FROM games')
        games_count = c.fetchone()[0]
        
        c.execute('SELECT MIN(season), MAX(season) FROM games')
        seasons = c.fetchone()
        
        c.execute('SELECT COUNT(DISTINCT coach_name) FROM coaches')
        coaches_count = c.fetchone()[0]
        
        conn.close()
        
        return {
            'total_games': games_count,
            'seasons_range': f"{seasons[0]} - {seasons[1]}" if seasons[0] else "No data",
            'coaches': coaches_count
        }
    
    # === СОХРАНЕНИЕ ДАННЫХ ===
    
    def save_game(self, game: HistoricalGame):
        """Сохраняет игру"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            INSERT OR REPLACE INTO games VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ''', (
            game.game_id, game.season, game.game_date.isoformat(),
            game.home_team, game.away_team, game.home_score, game.away_score,
            game.home_q1, game.home_q2, game.home_q3, game.home_q4, game.home_ot,
            game.away_q1, game.away_q2, game.away_q3, game.away_q4, game.away_ot,
            1 if game.playoffs else 0,
            game.home_coach, game.away_coach, game.attendance, game.arena
        ))
        
        conn.commit()
        conn.close()
    
    def save_season_stats(self, stats: SeasonStats):
        """Сохраняет статистику сезона"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            INSERT OR REPLACE INTO team_seasons 
            (team, season, wins, losses, win_pct, conference, division, conf_rank,
             playoff_result, ppg, opp_ppg, pace, off_rating, def_rating, net_rating, coach)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ''', (
            stats.team, stats.season, stats.wins, stats.losses, stats.win_pct,
            stats.conference, stats.division, stats.conf_rank, stats.playoff_result,
            stats.ppg, stats.opp_ppg, stats.pace, stats.off_rating, stats.def_rating,
            stats.net_rating, stats.coach
        ))
        
        conn.commit()
        conn.close()
    
    def save_coach(self, coach: CoachRecord):
        """Сохраняет тренера"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            INSERT OR REPLACE INTO coaches
            (coach_name, team, season_start, season_end, total_wins, total_losses,
             playoff_appearances, championships, avg_pace, avg_off_rating, avg_def_rating, style)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        ''', (
            coach.coach_name, coach.team, coach.season_start, coach.season_end,
            coach.total_wins, coach.total_losses, coach.playoff_appearances,
            coach.championships, coach.avg_pace, coach.avg_off_rating,
            coach.avg_def_rating, coach.style
        ))
        
        conn.commit()
        conn.close()
    
    # === ЗАПРОСЫ ===
    
    def get_h2h_history(
        self,
        team1: str,
        team2: str,
        seasons: int = 10
    ) -> List[HistoricalGame]:
        """Получает историю встреч"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            SELECT * FROM games
            WHERE (home_team = ? AND away_team = ?) OR (home_team = ? AND away_team = ?)
            ORDER BY game_date DESC
            LIMIT ?
        ''', (team1, team2, team2, team1, seasons * 4))
        
        games = []
        for row in c.fetchall():
            games.append(HistoricalGame(
                game_id=row[0],
                season=row[1],
                game_date=date.fromisoformat(row[2]),
                home_team=row[3],
                away_team=row[4],
                home_score=row[5],
                away_score=row[6],
                playoffs=bool(row[17]),
                home_coach=row[18],
                away_coach=row[19]
            ))
        
        conn.close()
        return games
    
    def get_team_evolution(self, team: str, years: int = 10) -> List[SeasonStats]:
        """Получает эволюцию команды"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            SELECT * FROM team_seasons
            WHERE team = ?
            ORDER BY season DESC
            LIMIT ?
        ''', (team, years))
        
        seasons = []
        for row in c.fetchall():
            seasons.append(SeasonStats(
                team=row[1],
                season=row[2],
                wins=row[3],
                losses=row[4],
                win_pct=row[5],
                conference=row[6],
                division=row[7],
                conf_rank=row[8],
                playoff_result=row[9],
                ppg=row[10],
                opp_ppg=row[11],
                pace=row[12],
                off_rating=row[13],
                def_rating=row[14],
                net_rating=row[15],
                coach=row[16]
            ))
        
        conn.close()
        return seasons
    
    def get_coach_history(self, coach_name: str) -> List[CoachRecord]:
        """Получает историю тренера"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            SELECT * FROM coaches
            WHERE coach_name = ?
            ORDER BY season_start DESC
        ''', (coach_name,))
        
        records = []
        for row in c.fetchall():
            records.append(CoachRecord(
                coach_name=row[1],
                team=row[2],
                season_start=row[3],
                season_end=row[4],
                total_wins=row[5],
                total_losses=row[6],
                playoff_appearances=row[7],
                championships=row[8],
                avg_pace=row[9],
                avg_off_rating=row[10],
                avg_def_rating=row[11],
                style=row[12]
            ))
        
        conn.close()
        return records
    
    def get_team_under_coach(self, team: str, coach: str) -> Dict:
        """Статистика команды под конкретным тренером"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            SELECT 
                COUNT(*) as seasons,
                SUM(wins) as total_wins,
                SUM(losses) as total_losses,
                AVG(win_pct) as avg_win_pct,
                AVG(off_rating) as avg_off,
                AVG(def_rating) as avg_def,
                GROUP_CONCAT(playoff_result) as playoffs
            FROM team_seasons
            WHERE team = ? AND coach = ?
        ''', (team, coach))
        
        row = c.fetchone()
        conn.close()
        
        if row and row[0] > 0:
            return {
                'seasons': row[0],
                'total_wins': row[1],
                'total_losses': row[2],
                'avg_win_pct': row[3],
                'avg_off_rating': row[4],
                'avg_def_rating': row[5],
                'playoff_results': row[6].split(',') if row[6] else []
            }
        return {}


class HistoricalDataFetcher:
    """
    Сбор исторических данных
    """
    
    BR_BASE_URL = "https://www.basketball-reference.com"
    
    def __init__(self, db: HistoricalDatabase):
        self.db = db
        self.session = requests.Session() if SCRAPING_AVAILABLE else None
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }) if self.session else None
    
    def fetch_season_games(self, season: str) -> int:
        """
        Загружает все игры сезона
        season format: "2023-24"
        """
        if not NBA_API_AVAILABLE:
            logger.warning("nba_api недоступен")
            return 0
        
        try:
            # Конвертируем формат сезона для API
            year = int(season.split('-')[0]) + 1  # 2023-24 -> 2024
            
            time.sleep(1)  # Rate limit
            
            finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=f"{year-1}-{str(year)[2:]}",
                league_id_nullable="00"
            )
            
            games_df = finder.get_data_frames()[0]
            
            # Группируем по game_id (каждая игра = 2 записи)
            game_ids = games_df['GAME_ID'].unique()
            
            count = 0
            for game_id in game_ids:
                game_data = games_df[games_df['GAME_ID'] == game_id]
                
                if len(game_data) != 2:
                    continue
                
                # Определяем home/away
                home_row = game_data[game_data['MATCHUP'].str.contains('vs.')].iloc[0]
                away_row = game_data[game_data['MATCHUP'].str.contains('@')].iloc[0]
                
                game = HistoricalGame(
                    game_id=game_id,
                    season=season,
                    game_date=datetime.strptime(home_row['GAME_DATE'], '%Y-%m-%d').date(),
                    home_team=home_row['TEAM_NAME'],
                    away_team=away_row['TEAM_NAME'],
                    home_score=int(home_row['PTS']),
                    away_score=int(away_row['PTS'])
                )
                
                self.db.save_game(game)
                count += 1
            
            logger.info(f"Загружено {count} игр за сезон {season}")
            return count
            
        except Exception as e:
            logger.error(f"Ошибка загрузки сезона {season}: {e}")
            return 0
    
    def fetch_team_history(self, team_id: int, years: int = 10) -> int:
        """Загружает историю команды"""
        if not NBA_API_AVAILABLE:
            return 0
        
        try:
            time.sleep(1)
            
            stats = teamyearbyyearstats.TeamYearByYearStats(team_id=team_id)
            df = stats.get_data_frames()[0]
            
            count = 0
            for _, row in df.head(years).iterrows():
                season_stats = SeasonStats(
                    team=row['TEAM_NAME'] if 'TEAM_NAME' in row else str(team_id),
                    season=row['YEAR'],
                    wins=int(row['WINS']),
                    losses=int(row['LOSSES']),
                    win_pct=float(row['WIN_PCT']),
                    conf_rank=int(row.get('CONF_RANK', 0)),
                    playoff_result=row.get('PO_WINS', '')
                )
                
                self.db.save_season_stats(season_stats)
                count += 1
            
            return count
            
        except Exception as e:
            logger.error(f"Ошибка загрузки истории команды: {e}")
            return 0
    
    def fetch_coaches_from_br(self, team_abbr: str) -> List[CoachRecord]:
        """
        Парсит историю тренеров с basketball-reference
        """
        if not SCRAPING_AVAILABLE:
            return []
        
        try:
            url = f"{self.BR_BASE_URL}/teams/{team_abbr}/coaches.html"
            
            time.sleep(3)  # Respect rate limits
            
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            coaches = []
            table = soup.find('table', {'id': 'coaches'})
            
            if not table:
                return []
            
            for row in table.find('tbody').find_all('tr'):
                cols = row.find_all('td')
                if len(cols) < 5:
                    continue
                
                coach_link = row.find('th').find('a')
                if not coach_link:
                    continue
                
                coach = CoachRecord(
                    coach_name=coach_link.text,
                    team=team_abbr,
                    season_start=cols[0].text if cols[0] else "",
                    season_end=cols[1].text if cols[1] else "",
                    total_wins=int(cols[3].text) if cols[3].text.isdigit() else 0,
                    total_losses=int(cols[4].text) if cols[4].text.isdigit() else 0
                )
                
                coaches.append(coach)
                self.db.save_coach(coach)
            
            return coaches
            
        except Exception as e:
            logger.error(f"Ошибка парсинга тренеров: {e}")
            return []
    
    def bulk_fetch_history(self, start_season: str = "2014-15", end_season: str = "2023-24"):
        """
        Массовая загрузка истории
        """
        logger.info(f"Начинаем загрузку истории с {start_season} по {end_season}")
        
        start_year = int(start_season.split('-')[0])
        end_year = int(end_season.split('-')[0])
        
        total_games = 0
        
        for year in range(start_year, end_year + 1):
            season = f"{year}-{str(year+1)[2:]}"
            logger.info(f"Загрузка сезона {season}...")
            
            games = self.fetch_season_games(season)
            total_games += games
            
            time.sleep(2)  # Между сезонами
        
        logger.info(f"Загрузка завершена. Всего игр: {total_games}")
        return total_games


class PatternAnalyzer:
    """
    Анализ исторических паттернов
    """
    
    def __init__(self, db: HistoricalDatabase):
        self.db = db
    
    def analyze_team_trends(self, team: str) -> Dict:
        """Анализирует многолетние тренды команды"""
        seasons = self.db.get_team_evolution(team, years=10)
        
        if not seasons:
            return {}
        
        # Тренд win%
        win_pcts = [s.win_pct for s in seasons]
        
        if len(win_pcts) >= 3:
            recent_avg = sum(win_pcts[:3]) / 3
            old_avg = sum(win_pcts[-3:]) / 3
            trend = "improving" if recent_avg > old_avg + 0.05 else \
                    "declining" if recent_avg < old_avg - 0.05 else "stable"
        else:
            trend = "unknown"
        
        # Тренд стиля игры
        paces = [s.pace for s in seasons if s.pace > 0]
        pace_trend = "faster" if paces and paces[0] > paces[-1] + 2 else \
                     "slower" if paces and paces[0] < paces[-1] - 2 else "stable"
        
        # История тренеров
        coaches = list(set(s.coach for s in seasons if s.coach))
        
        return {
            'team': team,
            'seasons_analyzed': len(seasons),
            'current_win_pct': win_pcts[0] if win_pcts else 0,
            'historical_avg_win_pct': sum(win_pcts) / len(win_pcts) if win_pcts else 0,
            'trend': trend,
            'pace_trend': pace_trend,
            'coaches_in_period': coaches,
            'playoff_appearances': sum(1 for s in seasons if s.playoff_result),
            'championships': sum(1 for s in seasons if 'Champion' in (s.playoff_result or ''))
        }
    
    def analyze_coach_impact(self, coach: str) -> Dict:
        """Анализирует влияние тренера"""
        records = self.db.get_coach_history(coach)
        
        if not records:
            return {}
        
        total_wins = sum(r.total_wins for r in records)
        total_losses = sum(r.total_losses for r in records)
        
        # Определяем стиль
        off_ratings = [r.avg_off_rating for r in records if r.avg_off_rating > 0]
        def_ratings = [r.avg_def_rating for r in records if r.avg_def_rating > 0]
        
        if off_ratings and def_ratings:
            avg_off = sum(off_ratings) / len(off_ratings)
            avg_def = sum(def_ratings) / len(def_ratings)
            
            if avg_off > avg_def + 3:
                style = "offensive"
            elif avg_def < avg_off - 3:
                style = "defensive"
            else:
                style = "balanced"
        else:
            style = "unknown"
        
        return {
            'coach': coach,
            'teams': list(set(r.team for r in records)),
            'total_seasons': len(records),
            'career_record': f"{total_wins}-{total_losses}",
            'career_win_pct': total_wins / (total_wins + total_losses) if total_wins + total_losses > 0 else 0,
            'championships': sum(r.championships for r in records),
            'style': style
        }
    
    def find_similar_matchups(
        self,
        home_team: str,
        away_team: str,
        home_record: Tuple[int, int],
        away_record: Tuple[int, int],
        limit: int = 20
    ) -> List[Dict]:
        """
        Находит похожие исторические матчапы
        (команды с похожими рекордами играли друг против друга)
        """
        conn = sqlite3.connect(self.db.db_path)
        c = conn.cursor()
        
        home_win_pct = home_record[0] / sum(home_record) if sum(home_record) > 0 else 0.5
        away_win_pct = away_record[0] / sum(away_record) if sum(away_record) > 0 else 0.5
        
        # Ищем игры где win% команд был похожим
        c.execute('''
            SELECT g.*, 
                   hs.win_pct as home_season_pct,
                   aws.win_pct as away_season_pct
            FROM games g
            LEFT JOIN team_seasons hs ON g.home_team = hs.team AND g.season = hs.season
            LEFT JOIN team_seasons aws ON g.away_team = aws.team AND g.season = aws.season
            WHERE ABS(hs.win_pct - ?) < 0.1 AND ABS(aws.win_pct - ?) < 0.1
            ORDER BY g.game_date DESC
            LIMIT ?
        ''', (home_win_pct, away_win_pct, limit))
        
        results = []
        for row in c.fetchall():
            results.append({
                'game_id': row[0],
                'date': row[2],
                'home_team': row[3],
                'away_team': row[4],
                'home_score': row[5],
                'away_score': row[6],
                'home_won': row[5] > row[6]
            })
        
        conn.close()
        
        # Статистика
        if results:
            home_wins = sum(1 for r in results if r['home_won'])
            return {
                'similar_games': results,
                'home_win_pct_in_similar': home_wins / len(results),
                'avg_total': sum(r['home_score'] + r['away_score'] for r in results) / len(results),
                'avg_margin': sum(r['home_score'] - r['away_score'] for r in results) / len(results)
            }
        
        return {'similar_games': [], 'home_win_pct_in_similar': 0.5}


# === ТЕСТИРОВАНИЕ ===

if __name__ == "__main__":
    print("=== Тест Historical Database ===\n")
    
    # Инициализация
    db = HistoricalDatabase("test_history.db")
    
    # Добавляем тестовые данные
    print("Добавление тестовых данных...")
    
    # Тестовые игры
    for i in range(50):
        game = HistoricalGame(
            game_id=f"002240{1000+i}",
            season="2023-24" if i < 25 else "2022-23",
            game_date=date(2024, 1, 1) if i < 25 else date(2023, 1, 1),
            home_team="Lakers",
            away_team="Warriors" if i % 2 == 0 else "Celtics",
            home_score=110 + (i % 20),
            away_score=105 + (i % 15),
            home_coach="Darvin Ham",
            away_coach="Steve Kerr" if i % 2 == 0 else "Joe Mazzulla"
        )
        db.save_game(game)
    
    # Тестовые сезоны
    for year in range(2015, 2025):
        season = f"{year}-{str(year+1)[2:]}"
        stats = SeasonStats(
            team="Lakers",
            season=season,
            wins=40 + (year % 10),
            losses=42 - (year % 10),
            win_pct=(40 + year % 10) / 82,
            conference="West",
            coach="Various" if year < 2022 else "Darvin Ham",
            off_rating=110 + (year % 5),
            def_rating=108 + (year % 4),
            pace=98 + (year % 6)
        )
        db.save_season_stats(stats)
    
    # Тестовые тренеры
    coaches_data = [
        ("Steve Kerr", "Warriors", "2014-15", "2024-25", 450, 230, 10, 4),
        ("Gregg Popovich", "Spurs", "1996-97", "2024-25", 1350, 700, 22, 5),
        ("Erik Spoelstra", "Heat", "2008-09", "2024-25", 650, 450, 14, 2),
    ]
    
    for name, team, start, end, wins, losses, playoffs, chips in coaches_data:
        coach = CoachRecord(
            coach_name=name,
            team=team,
            season_start=start,
            season_end=end,
            total_wins=wins,
            total_losses=losses,
            playoff_appearances=playoffs,
            championships=chips,
            avg_off_rating=112 if name == "Steve Kerr" else 108,
            avg_def_rating=106 if name == "Gregg Popovich" else 110,
            style="offensive" if name == "Steve Kerr" else "defensive"
        )
        db.save_coach(coach)
    
    # Статистика базы
    print("\nСтатистика базы:")
    stats = db.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # Тест запросов
    print("\n\nH2H Lakers vs Warriors:")
    h2h = db.get_h2h_history("Lakers", "Warriors", seasons=5)
    print(f"  Найдено игр: {len(h2h)}")
    if h2h:
        lakers_wins = sum(1 for g in h2h if (g.home_team == "Lakers" and g.home_score > g.away_score) or
                                            (g.away_team == "Lakers" and g.away_score > g.home_score))
        print(f"  Lakers wins: {lakers_wins}/{len(h2h)}")
    
    print("\n\nЭволюция Lakers:")
    evolution = db.get_team_evolution("Lakers", years=5)
    for s in evolution[:3]:
        print(f"  {s.season}: {s.wins}-{s.losses} ({s.win_pct:.1%}), Coach: {s.coach}")
    
    print("\n\nИстория Steve Kerr:")
    kerr = db.get_coach_history("Steve Kerr")
    for r in kerr:
        print(f"  {r.team} ({r.season_start}-{r.season_end}): {r.total_wins}-{r.total_losses}, 🏆x{r.championships}")
    
    # Анализ паттернов
    print("\n\nАнализ паттернов:")
    analyzer = PatternAnalyzer(db)
    
    trends = analyzer.analyze_team_trends("Lakers")
    print(f"  Lakers trend: {trends.get('trend', 'N/A')}")
    print(f"  Historical win%: {trends.get('historical_avg_win_pct', 0):.1%}")
    
    coach_impact = analyzer.analyze_coach_impact("Steve Kerr")
    print(f"\n  Steve Kerr style: {coach_impact.get('style', 'N/A')}")
    print(f"  Career win%: {coach_impact.get('career_win_pct', 0):.1%}")
    
    # Cleanup
    import os
    os.remove("test_history.db")
    print("\n✅ Тест завершен")
