"""
AutoBasket - Team & Player Analytics
=====================================
Глубокий анализ команд, игроков, травм и статистики
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from nba_api.stats.endpoints import (
        commonteamroster, playergamelog, teamdashboardbygeneralsplits,
        leaguedashteamstats, playerdashboardbygeneralsplits,
        leaguedashplayerstats, teamdashlineups, commonplayerinfo
    )
    from nba_api.stats.static import players as nba_players
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# === ENUMS ===

class InjuryStatus(Enum):
    """Статус травмы игрока"""
    HEALTHY = "healthy"
    QUESTIONABLE = "questionable"  # 50/50
    DOUBTFUL = "doubtful"          # Скорее не сыграет
    OUT = "out"                     # Точно не сыграет
    GTD = "game_time_decision"      # Решение перед игрой


class PlayerRole(Enum):
    """Роль игрока в команде"""
    STAR = "star"           # Звезда (top 2 по минутам/usage)
    STARTER = "starter"     # Стартер
    ROTATION = "rotation"   # Ротация (15+ мин)
    BENCH = "bench"         # Скамейка
    DNP = "dnp"             # Не играет


# === DATA STRUCTURES ===

@dataclass
class PlayerStats:
    """Статистика игрока"""
    player_id: int
    player_name: str
    team_id: int
    team_abbr: str
    
    # Базовая статистика
    games_played: int = 0
    minutes_per_game: float = 0.0
    points_per_game: float = 0.0
    rebounds_per_game: float = 0.0
    assists_per_game: float = 0.0
    steals_per_game: float = 0.0
    blocks_per_game: float = 0.0
    turnovers_per_game: float = 0.0
    
    # Проценты бросков
    fg_pct: float = 0.0      # Field Goal %
    fg3_pct: float = 0.0     # 3-Point %
    ft_pct: float = 0.0      # Free Throw %
    efg_pct: float = 0.0     # Effective FG%
    ts_pct: float = 0.0      # True Shooting %
    
    # Advanced
    usage_rate: float = 0.0   # % владений с участием игрока
    per: float = 0.0          # Player Efficiency Rating
    plus_minus: float = 0.0   # +/-
    vorp: float = 0.0         # Value Over Replacement
    
    # Роль
    role: PlayerRole = PlayerRole.ROTATION
    
    # Последние игры (для формы)
    last_5_ppg: float = 0.0
    last_5_fg_pct: float = 0.0


@dataclass
class InjuryReport:
    """Отчет о травме игрока"""
    player_id: int
    player_name: str
    team_abbr: str
    status: InjuryStatus
    injury_type: str = ""          # "Ankle", "Knee", etc.
    injury_detail: str = ""        # "Sprained left ankle"
    expected_return: Optional[date] = None
    games_missed: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Влияние на команду
    impact_score: float = 0.0      # 0-10, насколько важен игрок


@dataclass
class TeamShootingStats:
    """Статистика бросков команды"""
    team_id: int
    team_name: str
    
    # Общие проценты
    fg_pct: float = 0.0
    fg3_pct: float = 0.0
    ft_pct: float = 0.0
    efg_pct: float = 0.0
    ts_pct: float = 0.0
    
    # Объем бросков
    fga_per_game: float = 0.0      # Всего бросков
    fg3a_per_game: float = 0.0     # 3-очковых попыток
    fta_per_game: float = 0.0      # Штрафных попыток
    
    # По зонам
    paint_fg_pct: float = 0.0      # В краске
    midrange_fg_pct: float = 0.0   # Средняя дистанция
    corner3_pct: float = 0.0       # Угловые трехочковые
    above_break3_pct: float = 0.0  # Трехочковые сверху
    
    # Тренды (последние 10 игр)
    fg_pct_last10: float = 0.0
    fg3_pct_last10: float = 0.0
    
    # Оппозиция (что допускают)
    opp_fg_pct: float = 0.0
    opp_fg3_pct: float = 0.0
    opp_efg_pct: float = 0.0


@dataclass
class MatchupAnalysis:
    """Анализ матчапа двух команд"""
    home_team: str
    away_team: str
    
    # Стилистические факторы
    pace_diff: float = 0.0         # Разница в темпе
    style_clash: str = ""          # "fast_vs_slow", "balanced", etc.
    
    # Сильные/слабые стороны
    home_strengths: List[str] = field(default_factory=list)
    home_weaknesses: List[str] = field(default_factory=list)
    away_strengths: List[str] = field(default_factory=list)
    away_weaknesses: List[str] = field(default_factory=list)
    
    # Ключевые матчапы игроков
    key_matchups: List[Dict] = field(default_factory=list)
    
    # Исторические встречи
    h2h_record: str = ""           # "3-2 Home"
    h2h_avg_margin: float = 0.0
    h2h_avg_total: float = 0.0
    
    # Итоговая оценка
    matchup_edge: str = ""         # "home", "away", "even"
    confidence: float = 0.5


@dataclass
class TeamAdvancedStats:
    """Продвинутая статистика команды"""
    team_id: int
    team_name: str
    
    # Four Factors (ключевые факторы победы)
    efg_pct: float = 0.0           # Effective FG%
    tov_pct: float = 0.0           # Turnover %
    orb_pct: float = 0.0           # Offensive Rebound %
    ft_rate: float = 0.0           # FT Rate (FTA/FGA)
    
    # Оппозиция Four Factors
    opp_efg_pct: float = 0.0
    opp_tov_pct: float = 0.0
    opp_orb_pct: float = 0.0
    opp_ft_rate: float = 0.0
    
    # Рейтинги
    off_rating: float = 0.0
    def_rating: float = 0.0
    net_rating: float = 0.0
    
    # Clutch (последние 5 минут, разница < 5)
    clutch_net_rating: float = 0.0
    clutch_record: str = ""        # "15-8"
    
    # По периодам
    q1_net_rating: float = 0.0
    q2_net_rating: float = 0.0
    q3_net_rating: float = 0.0
    q4_net_rating: float = 0.0


# === ANALYTICS ENGINE ===

class TeamAnalytics:
    """
    Движок анализа команд
    """
    
    def __init__(self):
        self.injury_cache: Dict[str, List[InjuryReport]] = {}
        self.player_cache: Dict[int, PlayerStats] = {}
        self.team_cache: Dict[int, TeamAdvancedStats] = {}
        self._last_update = None
    
    def get_team_roster_stats(self, team_id: int, season: str = "2024-25") -> List[PlayerStats]:
        """Получает статистику всех игроков команды"""
        if not NBA_API_AVAILABLE:
            return self._get_mock_roster_stats(team_id)
        
        try:
            from time import sleep
            sleep(0.6)  # Rate limit
            
            # Получаем статистику лиги
            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                team_id_nullable=team_id
            )
            data = stats.get_normalized_dict()['LeagueDashPlayerStats']
            
            roster = []
            for p in data:
                player = PlayerStats(
                    player_id=p['PLAYER_ID'],
                    player_name=p['PLAYER_NAME'],
                    team_id=team_id,
                    team_abbr=p.get('TEAM_ABBREVIATION', ''),
                    games_played=p.get('GP', 0),
                    minutes_per_game=p.get('MIN', 0),
                    points_per_game=p.get('PTS', 0),
                    rebounds_per_game=p.get('REB', 0),
                    assists_per_game=p.get('AST', 0),
                    steals_per_game=p.get('STL', 0),
                    blocks_per_game=p.get('BLK', 0),
                    turnovers_per_game=p.get('TOV', 0),
                    fg_pct=p.get('FG_PCT', 0) * 100,
                    fg3_pct=p.get('FG3_PCT', 0) * 100,
                    ft_pct=p.get('FT_PCT', 0) * 100,
                    plus_minus=p.get('PLUS_MINUS', 0)
                )
                
                # Определяем роль
                if player.minutes_per_game >= 30:
                    player.role = PlayerRole.STAR
                elif player.minutes_per_game >= 24:
                    player.role = PlayerRole.STARTER
                elif player.minutes_per_game >= 15:
                    player.role = PlayerRole.ROTATION
                else:
                    player.role = PlayerRole.BENCH
                
                roster.append(player)
            
            return sorted(roster, key=lambda x: x.minutes_per_game, reverse=True)
            
        except Exception as e:
            logger.error(f"Ошибка получения ростера: {e}")
            return self._get_mock_roster_stats(team_id)
    
    def get_team_shooting_stats(self, team_id: int, season: str = "2024-25") -> TeamShootingStats:
        """Получает статистику бросков команды"""
        if not NBA_API_AVAILABLE:
            return self._get_mock_shooting_stats(team_id)
        
        try:
            from time import sleep
            sleep(0.6)
            
            stats = leaguedashteamstats.LeagueDashTeamStats(season=season)
            data = stats.get_normalized_dict()['LeagueDashTeamStats']
            
            for team in data:
                if team['TEAM_ID'] == team_id:
                    return TeamShootingStats(
                        team_id=team_id,
                        team_name=team['TEAM_NAME'],
                        fg_pct=team.get('FG_PCT', 0) * 100,
                        fg3_pct=team.get('FG3_PCT', 0) * 100,
                        ft_pct=team.get('FT_PCT', 0) * 100,
                        fga_per_game=team.get('FGA', 0) / max(team.get('GP', 1), 1),
                        fg3a_per_game=team.get('FG3A', 0) / max(team.get('GP', 1), 1),
                        fta_per_game=team.get('FTA', 0) / max(team.get('GP', 1), 1)
                    )
            
            return self._get_mock_shooting_stats(team_id)
            
        except Exception as e:
            logger.error(f"Ошибка получения shooting stats: {e}")
            return self._get_mock_shooting_stats(team_id)
    
    def get_team_advanced_stats(self, team_id: int, season: str = "2024-25") -> TeamAdvancedStats:
        """Получает продвинутую статистику команды"""
        if not NBA_API_AVAILABLE:
            return self._get_mock_advanced_stats(team_id)
        
        try:
            from time import sleep
            sleep(0.6)
            
            stats = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
                team_id=team_id,
                season=season
            )
            data = stats.get_normalized_dict()
            overall = data.get('OverallTeamDashboard', [{}])[0]
            
            return TeamAdvancedStats(
                team_id=team_id,
                team_name=overall.get('TEAM_NAME', ''),
                efg_pct=overall.get('EFG_PCT', 0) * 100 if overall.get('EFG_PCT') else 0,
                off_rating=overall.get('OFF_RATING', 110),
                def_rating=overall.get('DEF_RATING', 110),
                net_rating=overall.get('NET_RATING', 0)
            )
            
        except Exception as e:
            logger.error(f"Ошибка получения advanced stats: {e}")
            return self._get_mock_advanced_stats(team_id)
    
    def calculate_injury_impact(self, injuries: List[InjuryReport], roster: List[PlayerStats]) -> float:
        """
        Рассчитывает влияние травм на команду
        
        Returns:
            Impact score 0-10 (0 = нет влияния, 10 = катастрофа)
        """
        if not injuries or not roster:
            return 0.0
        
        total_impact = 0.0
        
        # Создаем карту игроков
        player_map = {p.player_name.lower(): p for p in roster}
        
        for injury in injuries:
            if injury.status in [InjuryStatus.HEALTHY]:
                continue
            
            # Находим игрока
            player = player_map.get(injury.player_name.lower())
            if not player:
                continue
            
            # Базовый impact по роли
            role_impact = {
                PlayerRole.STAR: 4.0,
                PlayerRole.STARTER: 2.5,
                PlayerRole.ROTATION: 1.5,
                PlayerRole.BENCH: 0.5,
                PlayerRole.DNP: 0.0
            }
            
            base = role_impact.get(player.role, 1.0)
            
            # Множитель по статусу
            status_mult = {
                InjuryStatus.OUT: 1.0,
                InjuryStatus.DOUBTFUL: 0.7,
                InjuryStatus.QUESTIONABLE: 0.3,
                InjuryStatus.GTD: 0.5
            }
            
            mult = status_mult.get(injury.status, 0.5)
            
            # Добавляем impact по PPG (звездные игроки важнее)
            ppg_bonus = min(player.points_per_game / 10, 2.0)
            
            impact = base * mult + ppg_bonus * mult
            total_impact += impact
            
            injury.impact_score = impact
        
        return min(total_impact, 10.0)
    
    def analyze_matchup(
        self,
        home_team_id: int,
        away_team_id: int,
        home_stats: TeamAdvancedStats,
        away_stats: TeamAdvancedStats,
        home_shooting: TeamShootingStats,
        away_shooting: TeamShootingStats
    ) -> MatchupAnalysis:
        """Анализирует матчап двух команд"""
        
        analysis = MatchupAnalysis(
            home_team=home_stats.team_name,
            away_team=away_stats.team_name
        )
        
        # Pace analysis
        # (предполагаем pace из shooting stats или используем дефолт)
        home_pace = 100  # Нужно получить из API
        away_pace = 100
        analysis.pace_diff = home_pace - away_pace
        
        if abs(analysis.pace_diff) < 2:
            analysis.style_clash = "balanced"
        elif analysis.pace_diff > 2:
            analysis.style_clash = "home_faster"
        else:
            analysis.style_clash = "away_faster"
        
        # Сильные/слабые стороны (на основе four factors)
        
        # Home strengths
        if home_stats.off_rating > 115:
            analysis.home_strengths.append("Elite offense")
        if home_stats.def_rating < 108:
            analysis.home_strengths.append("Elite defense")
        if home_shooting.fg3_pct > 37:
            analysis.home_strengths.append("Great 3PT shooting")
        
        # Home weaknesses
        if home_stats.off_rating < 110:
            analysis.home_weaknesses.append("Below average offense")
        if home_stats.def_rating > 114:
            analysis.home_weaknesses.append("Poor defense")
        if home_shooting.ft_pct < 75:
            analysis.home_weaknesses.append("Poor FT shooting")
        
        # Away strengths/weaknesses (аналогично)
        if away_stats.off_rating > 115:
            analysis.away_strengths.append("Elite offense")
        if away_stats.def_rating < 108:
            analysis.away_strengths.append("Elite defense")
        
        if away_stats.off_rating < 110:
            analysis.away_weaknesses.append("Below average offense")
        if away_stats.def_rating > 114:
            analysis.away_weaknesses.append("Poor defense")
        
        # Определяем edge
        home_score = home_stats.net_rating + 3.5  # Home advantage
        away_score = away_stats.net_rating
        
        diff = home_score - away_score
        
        if diff > 3:
            analysis.matchup_edge = "home"
            analysis.confidence = min(0.5 + diff * 0.03, 0.75)
        elif diff < -3:
            analysis.matchup_edge = "away"
            analysis.confidence = min(0.5 + abs(diff) * 0.03, 0.75)
        else:
            analysis.matchup_edge = "even"
            analysis.confidence = 0.5 + abs(diff) * 0.02
        
        return analysis
    
    # === MOCK DATA ===
    
    def _get_mock_roster_stats(self, team_id: int) -> List[PlayerStats]:
        """Тестовые данные ростера"""
        # Lakers mock data
        return [
            PlayerStats(
                player_id=2544, player_name="LeBron James", team_id=team_id, team_abbr="LAL",
                games_played=45, minutes_per_game=35.2, points_per_game=25.8,
                rebounds_per_game=7.2, assists_per_game=8.1, fg_pct=54.0, fg3_pct=38.5,
                role=PlayerRole.STAR
            ),
            PlayerStats(
                player_id=203076, player_name="Anthony Davis", team_id=team_id, team_abbr="LAL",
                games_played=42, minutes_per_game=34.5, points_per_game=24.5,
                rebounds_per_game=12.1, assists_per_game=3.2, fg_pct=55.5, fg3_pct=25.0,
                role=PlayerRole.STAR
            ),
            PlayerStats(
                player_id=1629029, player_name="Austin Reaves", team_id=team_id, team_abbr="LAL",
                games_played=48, minutes_per_game=28.5, points_per_game=15.2,
                rebounds_per_game=4.1, assists_per_game=5.5, fg_pct=45.0, fg3_pct=36.5,
                role=PlayerRole.STARTER
            ),
        ]
    
    def _get_mock_shooting_stats(self, team_id: int) -> TeamShootingStats:
        """Тестовые данные бросков"""
        return TeamShootingStats(
            team_id=team_id,
            team_name="Los Angeles Lakers",
            fg_pct=47.5,
            fg3_pct=36.2,
            ft_pct=78.5,
            efg_pct=54.2,
            fga_per_game=88.5,
            fg3a_per_game=32.5,
            fta_per_game=22.3
        )
    
    def _get_mock_advanced_stats(self, team_id: int) -> TeamAdvancedStats:
        """Тестовые продвинутые данные"""
        return TeamAdvancedStats(
            team_id=team_id,
            team_name="Los Angeles Lakers",
            efg_pct=54.2,
            tov_pct=12.5,
            orb_pct=28.5,
            ft_rate=0.25,
            off_rating=115.5,
            def_rating=112.3,
            net_rating=3.2,
            clutch_net_rating=5.5,
            clutch_record="12-8"
        )


class InjuryTracker:
    """
    Отслеживание травм игроков
    
    Источники данных:
    - ESPN API (бесплатно, но неофициально)
    - Rotowire (платно)
    - NBA официальный injury report
    """
    
    ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
    
    def __init__(self):
        self.injuries: Dict[str, List[InjuryReport]] = {}
        self._last_fetch = None
    
    def fetch_all_injuries(self) -> Dict[str, List[InjuryReport]]:
        """Получает все травмы из ESPN API"""
        if not REQUESTS_AVAILABLE:
            return self._get_mock_injuries()
        
        try:
            response = requests.get(self.ESPN_INJURIES_URL, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            injuries = {}
            
            for team_data in data.get('items', []):
                team_name = team_data.get('team', {}).get('displayName', '')
                team_abbr = team_data.get('team', {}).get('abbreviation', '')
                
                team_injuries = []
                
                for injury in team_data.get('injuries', []):
                    athlete = injury.get('athlete', {})
                    
                    status_str = injury.get('status', '').lower()
                    if 'out' in status_str:
                        status = InjuryStatus.OUT
                    elif 'doubtful' in status_str:
                        status = InjuryStatus.DOUBTFUL
                    elif 'questionable' in status_str:
                        status = InjuryStatus.QUESTIONABLE
                    elif 'day-to-day' in status_str or 'gtd' in status_str:
                        status = InjuryStatus.GTD
                    else:
                        status = InjuryStatus.QUESTIONABLE
                    
                    report = InjuryReport(
                        player_id=int(athlete.get('id', 0)),
                        player_name=athlete.get('displayName', ''),
                        team_abbr=team_abbr,
                        status=status,
                        injury_type=injury.get('type', {}).get('text', ''),
                        injury_detail=injury.get('longComment', '')
                    )
                    team_injuries.append(report)
                
                if team_injuries:
                    injuries[team_abbr] = team_injuries
            
            self.injuries = injuries
            self._last_fetch = datetime.now()
            
            return injuries
            
        except Exception as e:
            logger.error(f"Ошибка получения травм: {e}")
            return self._get_mock_injuries()
    
    def get_team_injuries(self, team_abbr: str) -> List[InjuryReport]:
        """Получает травмы конкретной команды"""
        if not self.injuries or self._should_refresh():
            self.fetch_all_injuries()
        
        return self.injuries.get(team_abbr.upper(), [])
    
    def _should_refresh(self) -> bool:
        """Проверяет, нужно ли обновить данные"""
        if not self._last_fetch:
            return True
        return (datetime.now() - self._last_fetch).seconds > 3600  # 1 час
    
    def _get_mock_injuries(self) -> Dict[str, List[InjuryReport]]:
        """Тестовые данные о травмах"""
        return {
            "LAL": [
                InjuryReport(
                    player_id=203076,
                    player_name="Anthony Davis",
                    team_abbr="LAL",
                    status=InjuryStatus.QUESTIONABLE,
                    injury_type="Ankle",
                    injury_detail="Left ankle soreness"
                )
            ],
            "GSW": [
                InjuryReport(
                    player_id=201939,
                    player_name="Stephen Curry",
                    team_abbr="GSW",
                    status=InjuryStatus.GTD,
                    injury_type="Knee",
                    injury_detail="Right knee contusion"
                ),
                InjuryReport(
                    player_id=203110,
                    player_name="Draymond Green",
                    team_abbr="GSW",
                    status=InjuryStatus.OUT,
                    injury_type="Back",
                    injury_detail="Lower back tightness"
                )
            ],
            "BOS": [],
            "MIA": [
                InjuryReport(
                    player_id=1628389,
                    player_name="Bam Adebayo",
                    team_abbr="MIA",
                    status=InjuryStatus.DOUBTFUL,
                    injury_type="Knee",
                    injury_detail="Left knee inflammation"
                )
            ]
        }


# === COMBINED ANALYSIS ===

class DeepAnalyzer:
    """
    Комбинированный глубокий анализ
    Объединяет все источники данных
    """
    
    def __init__(self):
        self.team_analytics = TeamAnalytics()
        self.injury_tracker = InjuryTracker()
    
    def full_team_analysis(self, team_id: int, team_abbr: str) -> Dict:
        """Полный анализ команды"""
        
        # Собираем все данные
        roster = self.team_analytics.get_team_roster_stats(team_id)
        shooting = self.team_analytics.get_team_shooting_stats(team_id)
        advanced = self.team_analytics.get_team_advanced_stats(team_id)
        injuries = self.injury_tracker.get_team_injuries(team_abbr)
        
        # Считаем impact травм
        injury_impact = self.team_analytics.calculate_injury_impact(injuries, roster)
        
        # Ключевые игроки
        key_players = [p for p in roster if p.role in [PlayerRole.STAR, PlayerRole.STARTER]]
        
        # Injured starters
        injured_starters = []
        for inj in injuries:
            if inj.status in [InjuryStatus.OUT, InjuryStatus.DOUBTFUL]:
                for p in key_players:
                    if p.player_name.lower() == inj.player_name.lower():
                        injured_starters.append(inj)
        
        return {
            'team_id': team_id,
            'team_abbr': team_abbr,
            'roster': roster,
            'shooting_stats': shooting,
            'advanced_stats': advanced,
            'injuries': injuries,
            'injury_impact': injury_impact,
            'key_players': key_players,
            'injured_starters': injured_starters,
            'health_status': 'healthy' if injury_impact < 2 else 'banged_up' if injury_impact < 5 else 'depleted'
        }
    
    def full_matchup_analysis(
        self,
        home_team_id: int,
        home_abbr: str,
        away_team_id: int,
        away_abbr: str
    ) -> Dict:
        """Полный анализ матчапа"""
        
        # Анализ обеих команд
        home_analysis = self.full_team_analysis(home_team_id, home_abbr)
        away_analysis = self.full_team_analysis(away_team_id, away_abbr)
        
        # Matchup analysis
        matchup = self.team_analytics.analyze_matchup(
            home_team_id, away_team_id,
            home_analysis['advanced_stats'],
            away_analysis['advanced_stats'],
            home_analysis['shooting_stats'],
            away_analysis['shooting_stats']
        )
        
        # Корректируем на травмы
        injury_adjusted_edge = matchup.confidence
        
        # Если у home больше травм - уменьшаем их шансы
        injury_diff = away_analysis['injury_impact'] - home_analysis['injury_impact']
        injury_adjusted_edge += injury_diff * 0.02  # 2% за каждый пункт разницы
        injury_adjusted_edge = max(0.3, min(0.7, injury_adjusted_edge))
        
        return {
            'home': home_analysis,
            'away': away_analysis,
            'matchup': matchup,
            'injury_adjusted_edge': injury_adjusted_edge,
            'recommendation': {
                'side': matchup.matchup_edge,
                'confidence': injury_adjusted_edge,
                'key_factors': self._get_key_factors(home_analysis, away_analysis, matchup)
            }
        }
    
    def _get_key_factors(self, home: Dict, away: Dict, matchup: MatchupAnalysis) -> List[str]:
        """Определяет ключевые факторы матча"""
        factors = []
        
        # Травмы
        if home['injury_impact'] > 3:
            factors.append(f"⚠️ {home['team_abbr']} depleted by injuries (impact: {home['injury_impact']:.1f})")
        if away['injury_impact'] > 3:
            factors.append(f"⚠️ {away['team_abbr']} depleted by injuries (impact: {away['injury_impact']:.1f})")
        
        # Сильные стороны
        if matchup.home_strengths:
            factors.append(f"✅ Home: {', '.join(matchup.home_strengths[:2])}")
        if matchup.away_strengths:
            factors.append(f"✅ Away: {', '.join(matchup.away_strengths[:2])}")
        
        # Net rating
        home_nr = home['advanced_stats'].net_rating
        away_nr = away['advanced_stats'].net_rating
        
        if abs(home_nr - away_nr) > 5:
            better = "Home" if home_nr > away_nr else "Away"
            factors.append(f"📊 {better} team has significant Net Rating advantage")
        
        return factors


# === ТЕСТИРОВАНИЕ ===

if __name__ == "__main__":
    print("=== Тест Team & Player Analytics ===\n")
    
    analyzer = DeepAnalyzer()
    
    # Тест анализа команды
    print("Тест 1: Анализ команды Lakers")
    print("-" * 50)
    
    lakers = analyzer.full_team_analysis(1610612747, "LAL")
    
    print(f"Команда: {lakers['team_abbr']}")
    print(f"Health status: {lakers['health_status']}")
    print(f"Injury impact: {lakers['injury_impact']:.1f}/10")
    
    print("\nКлючевые игроки:")
    for p in lakers['key_players'][:3]:
        print(f"  {p.player_name}: {p.points_per_game:.1f} PPG, {p.minutes_per_game:.1f} MPG")
    
    print(f"\nShooting: FG {lakers['shooting_stats'].fg_pct:.1f}%, 3P {lakers['shooting_stats'].fg3_pct:.1f}%")
    print(f"Ratings: OFF {lakers['advanced_stats'].off_rating:.1f}, DEF {lakers['advanced_stats'].def_rating:.1f}")
    
    if lakers['injuries']:
        print("\nТравмы:")
        for inj in lakers['injuries']:
            print(f"  {inj.player_name}: {inj.status.value} ({inj.injury_type})")
    
    # Тест матчапа
    print("\n\nТест 2: Анализ матчапа Lakers vs Warriors")
    print("-" * 50)
    
    matchup = analyzer.full_matchup_analysis(
        1610612747, "LAL",
        1610612744, "GSW"
    )
    
    print(f"Edge: {matchup['matchup'].matchup_edge}")
    print(f"Confidence: {matchup['matchup'].confidence:.1%}")
    print(f"Injury-adjusted: {matchup['injury_adjusted_edge']:.1%}")
    
    print("\nKey factors:")
    for factor in matchup['recommendation']['key_factors']:
        print(f"  {factor}")
    
    # Тест травм
    print("\n\nТест 3: Injury Tracker")
    print("-" * 50)
    
    tracker = InjuryTracker()
    all_injuries = tracker.fetch_all_injuries()
    
    print(f"Команд с травмами: {len(all_injuries)}")
    for team, injuries in list(all_injuries.items())[:3]:
        print(f"\n{team}:")
        for inj in injuries:
            print(f"  {inj.player_name}: {inj.status.value}")
