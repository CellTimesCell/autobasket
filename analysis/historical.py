"""
AutoBasket - Historical Analysis & Trends
==========================================
Анализ исторических данных, H2H, тренды, ATS статистика
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from collections import defaultdict
import statistics

try:
    from nba_api.stats.endpoints import (
        leaguegamefinder, teamgamelog, teamvsplayer
    )
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GameResult:
    """Результат игры"""
    game_id: str
    game_date: date
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    home_won: bool
    margin: int  # Положительный = home win
    total: int
    
    # Линии (если есть)
    spread_line: Optional[float] = None
    total_line: Optional[float] = None
    home_covered: Optional[bool] = None
    went_over: Optional[bool] = None


@dataclass
class H2HAnalysis:
    """Анализ личных встреч"""
    team1: str
    team2: str
    
    # Общий рекорд
    total_games: int = 0
    team1_wins: int = 0
    team2_wins: int = 0
    
    # Дома/на выезде
    team1_home_wins: int = 0
    team1_home_games: int = 0
    team2_home_wins: int = 0
    team2_home_games: int = 0
    
    # Средние показатели
    avg_margin: float = 0.0
    avg_total: float = 0.0
    
    # Последние встречи
    last_5_results: List[GameResult] = field(default_factory=list)
    
    # ATS (Against The Spread)
    team1_ats_record: str = ""  # "3-2"
    team2_ats_record: str = ""
    
    # Over/Under
    over_count: int = 0
    under_count: int = 0


@dataclass
class TeamTrends:
    """Тренды команды"""
    team_name: str
    
    # Общий рекорд
    record: str = ""  # "28-18"
    win_pct: float = 0.0
    
    # Последние N игр
    last_5: str = ""
    last_10: str = ""
    last_5_margin: float = 0.0
    
    # Дома/на выезде
    home_record: str = ""
    road_record: str = ""
    
    # ATS
    ats_record: str = ""
    ats_home: str = ""
    ats_road: str = ""
    ats_as_favorite: str = ""
    ats_as_underdog: str = ""
    ats_last_10: str = ""
    
    # Over/Under
    ou_record: str = ""
    ou_home: str = ""
    ou_road: str = ""
    avg_total: float = 0.0
    
    # Streak
    current_streak: str = ""  # "W3" или "L2"
    ats_streak: str = ""
    ou_streak: str = ""
    
    # По четвертям
    q1_record: str = ""  # Выиграли 1-ю четверть
    q3_record: str = ""  # 3-я четверть (после перерыва)
    
    # Situational
    after_win: str = ""
    after_loss: str = ""
    on_back_to_back: str = ""
    with_rest: str = ""  # 2+ дня отдыха


@dataclass  
class SituationalStats:
    """Ситуационная статистика"""
    team_name: str
    
    # По дням недели
    monday_record: str = ""
    friday_record: str = ""
    sunday_record: str = ""
    
    # По месяцам
    january_record: str = ""
    february_record: str = ""
    march_record: str = ""
    april_record: str = ""
    
    # По противникам
    vs_west_record: str = ""
    vs_east_record: str = ""
    vs_over500: str = ""
    vs_under500: str = ""
    
    # По отдыху
    b2b_first: str = ""     # Первая игра back-to-back
    b2b_second: str = ""    # Вторая игра back-to-back
    rest_0_days: str = ""
    rest_1_day: str = ""
    rest_2plus: str = ""
    
    # Дорожные серии
    road_trip_game1: str = ""
    road_trip_game3plus: str = ""
    
    # Revenge games
    revenge_record: str = ""  # Игра после поражения от этой же команды


class HistoricalAnalyzer:
    """
    Анализатор исторических данных
    """
    
    def __init__(self):
        self.games_cache: Dict[str, List[GameResult]] = {}
        self.h2h_cache: Dict[str, H2HAnalysis] = {}
    
    def get_h2h_analysis(
        self,
        team1: str,
        team2: str,
        seasons: List[str] = None,
        limit: int = 20
    ) -> H2HAnalysis:
        """
        Анализ личных встреч двух команд
        """
        cache_key = f"{team1}_{team2}"
        
        # Получаем игры
        games = self._get_h2h_games(team1, team2, seasons, limit)
        
        if not games:
            return H2HAnalysis(team1=team1, team2=team2)
        
        analysis = H2HAnalysis(team1=team1, team2=team2)
        analysis.total_games = len(games)
        
        margins = []
        totals = []
        
        for game in games:
            is_team1_home = game.home_team == team1
            team1_won = (is_team1_home and game.home_won) or (not is_team1_home and not game.home_won)
            
            if team1_won:
                analysis.team1_wins += 1
            else:
                analysis.team2_wins += 1
            
            # Home/away breakdown
            if is_team1_home:
                analysis.team1_home_games += 1
                if team1_won:
                    analysis.team1_home_wins += 1
            else:
                analysis.team2_home_games += 1
                if not team1_won:
                    analysis.team2_home_wins += 1
            
            # Margins and totals
            margin = game.margin if is_team1_home else -game.margin
            margins.append(margin)
            totals.append(game.total)
            
            # O/U
            if game.went_over is not None:
                if game.went_over:
                    analysis.over_count += 1
                else:
                    analysis.under_count += 1
        
        # Averages
        analysis.avg_margin = statistics.mean(margins) if margins else 0
        analysis.avg_total = statistics.mean(totals) if totals else 0
        
        # Last 5
        analysis.last_5_results = games[:5]
        
        return analysis
    
    def get_team_trends(
        self,
        team_name: str,
        season: str = "2024-25",
        include_ats: bool = True
    ) -> TeamTrends:
        """
        Получает тренды команды
        """
        games = self._get_team_games(team_name, season)
        
        if not games:
            return TeamTrends(team_name=team_name)
        
        trends = TeamTrends(team_name=team_name)
        
        # Базовые расчеты
        wins = sum(1 for g in games if self._team_won(team_name, g))
        losses = len(games) - wins
        
        trends.record = f"{wins}-{losses}"
        trends.win_pct = wins / len(games) if games else 0
        
        # Last 5, Last 10
        last_5_wins = sum(1 for g in games[:5] if self._team_won(team_name, g))
        last_10_wins = sum(1 for g in games[:10] if self._team_won(team_name, g))
        trends.last_5 = f"{last_5_wins}-{5 - last_5_wins}"
        trends.last_10 = f"{last_10_wins}-{10 - last_10_wins}"
        
        # Home/Road
        home_games = [g for g in games if g.home_team == team_name]
        road_games = [g for g in games if g.away_team == team_name]
        
        home_wins = sum(1 for g in home_games if g.home_won)
        road_wins = sum(1 for g in road_games if not g.home_won)
        
        trends.home_record = f"{home_wins}-{len(home_games) - home_wins}"
        trends.road_record = f"{road_wins}-{len(road_games) - road_wins}"
        
        # Streak
        streak = 0
        if games:
            first_result = self._team_won(team_name, games[0])
            for g in games:
                if self._team_won(team_name, g) == first_result:
                    streak += 1
                else:
                    break
            trends.current_streak = f"{'W' if first_result else 'L'}{streak}"
        
        # ATS (если есть данные)
        if include_ats:
            ats_wins = sum(1 for g in games if g.home_covered is not None and 
                          ((g.home_team == team_name and g.home_covered) or
                           (g.away_team == team_name and not g.home_covered)))
            ats_total = sum(1 for g in games if g.home_covered is not None)
            if ats_total > 0:
                trends.ats_record = f"{ats_wins}-{ats_total - ats_wins}"
        
        # O/U
        overs = sum(1 for g in games if g.went_over)
        unders = sum(1 for g in games if g.went_over is not None and not g.went_over)
        if overs + unders > 0:
            trends.ou_record = f"{overs}-{unders}"
        
        trends.avg_total = statistics.mean([g.total for g in games]) if games else 0
        
        return trends
    
    def get_situational_stats(
        self,
        team_name: str,
        games: List[GameResult]
    ) -> SituationalStats:
        """
        Рассчитывает ситуационную статистику
        """
        stats = SituationalStats(team_name=team_name)
        
        # По дням недели
        by_weekday = defaultdict(list)
        for g in games:
            weekday = g.game_date.weekday()
            by_weekday[weekday].append(self._team_won(team_name, g))
        
        if 0 in by_weekday:  # Monday
            w = sum(by_weekday[0])
            stats.monday_record = f"{w}-{len(by_weekday[0]) - w}"
        if 4 in by_weekday:  # Friday
            w = sum(by_weekday[4])
            stats.friday_record = f"{w}-{len(by_weekday[4]) - w}"
        if 6 in by_weekday:  # Sunday
            w = sum(by_weekday[6])
            stats.sunday_record = f"{w}-{len(by_weekday[6]) - w}"
        
        # After win/loss
        after_win_results = []
        after_loss_results = []
        
        for i in range(1, len(games)):
            prev_won = self._team_won(team_name, games[i])
            curr_won = self._team_won(team_name, games[i-1])
            
            if prev_won:
                after_win_results.append(curr_won)
            else:
                after_loss_results.append(curr_won)
        
        if after_win_results:
            w = sum(after_win_results)
            stats.after_win = f"{w}-{len(after_win_results) - w}"
        if after_loss_results:
            w = sum(after_loss_results)
            stats.after_loss = f"{w}-{len(after_loss_results) - w}"
        
        # Rest days
        rest_0, rest_1, rest_2plus = [], [], []
        
        for i in range(1, len(games)):
            days_rest = (games[i-1].game_date - games[i].game_date).days - 1
            won = self._team_won(team_name, games[i-1])
            
            if days_rest == 0:
                rest_0.append(won)
            elif days_rest == 1:
                rest_1.append(won)
            else:
                rest_2plus.append(won)
        
        if rest_0:
            w = sum(rest_0)
            stats.rest_0_days = f"{w}-{len(rest_0) - w}"
        if rest_1:
            w = sum(rest_1)
            stats.rest_1_day = f"{w}-{len(rest_1) - w}"
        if rest_2plus:
            w = sum(rest_2plus)
            stats.rest_2plus = f"{w}-{len(rest_2plus) - w}"
        
        return stats
    
    def calculate_ats_trends(
        self,
        games: List[GameResult],
        team_name: str
    ) -> Dict:
        """
        Детальный ATS анализ
        """
        ats_data = {
            'overall': {'wins': 0, 'losses': 0, 'pushes': 0},
            'home': {'wins': 0, 'losses': 0, 'pushes': 0},
            'road': {'wins': 0, 'losses': 0, 'pushes': 0},
            'as_favorite': {'wins': 0, 'losses': 0, 'pushes': 0},
            'as_underdog': {'wins': 0, 'losses': 0, 'pushes': 0},
            'last_10': {'wins': 0, 'losses': 0, 'pushes': 0},
            'streak': 0,
            'avg_margin_vs_spread': 0.0
        }
        
        margins = []
        streak = 0
        last_result = None
        
        for i, game in enumerate(games):
            if game.spread_line is None or game.home_covered is None:
                continue
            
            is_home = game.home_team == team_name
            covered = game.home_covered if is_home else not game.home_covered
            
            # Определяем фаворит/андердог
            if is_home:
                is_favorite = game.spread_line < 0
            else:
                is_favorite = game.spread_line > 0
            
            # Margin vs spread
            actual_margin = game.margin if is_home else -game.margin
            expected_margin = -game.spread_line if is_home else game.spread_line
            margin_diff = actual_margin - expected_margin
            margins.append(margin_diff)
            
            # Counts
            key = 'wins' if covered else 'losses'
            ats_data['overall'][key] += 1
            
            if is_home:
                ats_data['home'][key] += 1
            else:
                ats_data['road'][key] += 1
            
            if is_favorite:
                ats_data['as_favorite'][key] += 1
            else:
                ats_data['as_underdog'][key] += 1
            
            if i < 10:
                ats_data['last_10'][key] += 1
            
            # Streak
            if i == 0:
                streak = 1 if covered else -1
                last_result = covered
            elif covered == last_result:
                streak += 1 if covered else -1
            else:
                break
        
        ats_data['streak'] = streak
        ats_data['avg_margin_vs_spread'] = statistics.mean(margins) if margins else 0
        
        return ats_data
    
    def _team_won(self, team_name: str, game: GameResult) -> bool:
        """Проверяет, выиграла ли команда"""
        if game.home_team == team_name:
            return game.home_won
        return not game.home_won
    
    def _get_h2h_games(
        self,
        team1: str,
        team2: str,
        seasons: List[str],
        limit: int
    ) -> List[GameResult]:
        """Получает игры между двумя командами"""
        # В реальности здесь был бы запрос к NBA API
        # Возвращаем mock данные
        return self._get_mock_h2h_games(team1, team2, limit)
    
    def _get_team_games(self, team_name: str, season: str) -> List[GameResult]:
        """Получает все игры команды"""
        return self._get_mock_team_games(team_name)
    
    def _get_mock_h2h_games(self, team1: str, team2: str, limit: int) -> List[GameResult]:
        """Mock H2H данные"""
        base_date = date.today()
        games = []
        
        for i in range(min(limit, 10)):
            game_date = base_date - timedelta(days=30 * (i + 1))
            is_team1_home = i % 2 == 0
            home_won = i % 3 != 0  # 2/3 home wins
            
            home_score = 115 + (i % 10)
            away_score = 110 + ((i + 3) % 10)
            
            if not home_won:
                home_score, away_score = away_score, home_score
            
            games.append(GameResult(
                game_id=f"00224{1000 + i}",
                game_date=game_date,
                home_team=team1 if is_team1_home else team2,
                away_team=team2 if is_team1_home else team1,
                home_score=home_score,
                away_score=away_score,
                home_won=home_won,
                margin=home_score - away_score,
                total=home_score + away_score,
                spread_line=-3.5 if is_team1_home else 3.5,
                total_line=225.0,
                home_covered=home_score - away_score > 3.5 if is_team1_home else away_score - home_score > 3.5,
                went_over=(home_score + away_score) > 225
            ))
        
        return games
    
    def _get_mock_team_games(self, team_name: str) -> List[GameResult]:
        """Mock team games"""
        base_date = date.today()
        games = []
        opponents = ["Warriors", "Celtics", "Heat", "Nuggets", "Suns", "Bucks", "76ers", "Nets"]
        
        for i in range(50):
            game_date = base_date - timedelta(days=2 * i)
            is_home = i % 2 == 0
            opponent = opponents[i % len(opponents)]
            home_won = i % 3 != 2  # ~67% win rate
            
            home_score = 112 + (i % 15)
            away_score = 108 + ((i + 5) % 15)
            
            if not home_won:
                home_score, away_score = away_score, home_score
            
            games.append(GameResult(
                game_id=f"00224{2000 + i}",
                game_date=game_date,
                home_team=team_name if is_home else opponent,
                away_team=opponent if is_home else team_name,
                home_score=home_score,
                away_score=away_score,
                home_won=home_won,
                margin=home_score - away_score,
                total=home_score + away_score,
                spread_line=-4.5 if (is_home and home_won) else 4.5,
                total_line=222.0,
                home_covered=(home_score - away_score) > 4.5,
                went_over=(home_score + away_score) > 222
            ))
        
        return games


class PublicBettingAnalyzer:
    """
    Анализ публичных ставок и sharp money
    """
    
    def __init__(self):
        self.public_percentages: Dict[str, Dict] = {}
    
    def analyze_line_movement(
        self,
        opening_line: float,
        current_line: float,
        public_pct_home: float
    ) -> Dict:
        """
        Анализирует движение линии и определяет sharp action
        """
        line_move = current_line - opening_line
        
        analysis = {
            'opening_line': opening_line,
            'current_line': current_line,
            'line_movement': line_move,
            'public_pct_home': public_pct_home,
            'public_pct_away': 100 - public_pct_home,
            'sharp_indicator': 'none',
            'fade_public': False,
            'notes': []
        }
        
        # Reverse line movement = sharp money
        # Если публика на home (>60%), но линия движется в пользу home - sharps на away
        if public_pct_home > 60 and line_move < -1:
            analysis['sharp_indicator'] = 'away'
            analysis['fade_public'] = True
            analysis['notes'].append("Reverse line movement: sharps on away")
        
        elif public_pct_home < 40 and line_move > 1:
            analysis['sharp_indicator'] = 'home'
            analysis['fade_public'] = True
            analysis['notes'].append("Reverse line movement: sharps on home")
        
        # Steam move (резкое движение)
        if abs(line_move) > 2:
            analysis['notes'].append(f"Steam move detected: {line_move:+.1f}")
        
        # Contrarian value
        if public_pct_home > 70:
            analysis['notes'].append("Heavy public action on home - consider fade")
        elif public_pct_home < 30:
            analysis['notes'].append("Heavy public action on away - consider fade")
        
        return analysis


# === ТЕСТИРОВАНИЕ ===

if __name__ == "__main__":
    print("=== Тест Historical Analysis ===\n")
    
    analyzer = HistoricalAnalyzer()
    
    # Тест H2H
    print("Тест 1: H2H Analysis (Lakers vs Warriors)")
    print("-" * 50)
    
    h2h = analyzer.get_h2h_analysis("Lakers", "Warriors", limit=10)
    
    print(f"Total games: {h2h.total_games}")
    print(f"Lakers wins: {h2h.team1_wins}")
    print(f"Warriors wins: {h2h.team2_wins}")
    print(f"Avg margin: {h2h.avg_margin:+.1f}")
    print(f"Avg total: {h2h.avg_total:.1f}")
    print(f"O/U: {h2h.over_count}-{h2h.under_count}")
    
    # Тест трендов
    print("\n\nТест 2: Team Trends (Lakers)")
    print("-" * 50)
    
    trends = analyzer.get_team_trends("Lakers")
    
    print(f"Record: {trends.record}")
    print(f"Last 5: {trends.last_5}")
    print(f"Last 10: {trends.last_10}")
    print(f"Home: {trends.home_record}")
    print(f"Road: {trends.road_record}")
    print(f"Streak: {trends.current_streak}")
    print(f"ATS: {trends.ats_record}")
    print(f"O/U: {trends.ou_record}")
    print(f"Avg total: {trends.avg_total:.1f}")
    
    # Тест ситуационной статистики
    print("\n\nТест 3: Situational Stats")
    print("-" * 50)
    
    games = analyzer._get_mock_team_games("Lakers")
    sit_stats = analyzer.get_situational_stats("Lakers", games)
    
    print(f"After win: {sit_stats.after_win}")
    print(f"After loss: {sit_stats.after_loss}")
    print(f"0 days rest: {sit_stats.rest_0_days}")
    print(f"1 day rest: {sit_stats.rest_1_day}")
    print(f"2+ days rest: {sit_stats.rest_2plus}")
    
    # Тест ATS
    print("\n\nТест 4: ATS Analysis")
    print("-" * 50)
    
    ats = analyzer.calculate_ats_trends(games, "Lakers")
    
    print(f"Overall ATS: {ats['overall']['wins']}-{ats['overall']['losses']}")
    print(f"Home ATS: {ats['home']['wins']}-{ats['home']['losses']}")
    print(f"Road ATS: {ats['road']['wins']}-{ats['road']['losses']}")
    print(f"As favorite: {ats['as_favorite']['wins']}-{ats['as_favorite']['losses']}")
    print(f"As underdog: {ats['as_underdog']['wins']}-{ats['as_underdog']['losses']}")
    print(f"ATS streak: {ats['streak']}")
    print(f"Avg margin vs spread: {ats['avg_margin_vs_spread']:+.1f}")
    
    # Тест line movement
    print("\n\nТест 5: Line Movement Analysis")
    print("-" * 50)
    
    public_analyzer = PublicBettingAnalyzer()
    
    lm = public_analyzer.analyze_line_movement(
        opening_line=-3.5,
        current_line=-2.0,
        public_pct_home=72
    )
    
    print(f"Opening: {lm['opening_line']}")
    print(f"Current: {lm['current_line']}")
    print(f"Movement: {lm['line_movement']:+.1f}")
    print(f"Public on home: {lm['public_pct_home']}%")
    print(f"Sharp indicator: {lm['sharp_indicator']}")
    print(f"Fade public: {lm['fade_public']}")
    for note in lm['notes']:
        print(f"  - {note}")
