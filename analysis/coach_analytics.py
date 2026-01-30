"""
AutoBasket - Coach Analytics
============================
Детальный анализ тренеров, их стиля и влияния на команду
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CoachProfile:
    """Профиль тренера"""
    name: str
    current_team: str = ""
    
    # Карьера
    years_experience: int = 0
    total_wins: int = 0
    total_losses: int = 0
    career_win_pct: float = 0.0
    
    # Достижения
    championships: int = 0
    finals_appearances: int = 0
    conference_finals: int = 0
    playoff_appearances: int = 0
    coach_of_year_awards: int = 0
    
    # Стиль игры
    offensive_rating_avg: float = 0.0
    defensive_rating_avg: float = 0.0
    pace_avg: float = 0.0
    primary_style: str = ""  # "offensive", "defensive", "balanced", "pace_and_space"
    
    # Тенденции
    prefers_small_ball: bool = False
    heavy_3pt_usage: bool = False
    defensive_minded: bool = False
    player_development: bool = False  # Известен развитием игроков
    
    # История команд
    teams_coached: List[str] = field(default_factory=list)


@dataclass
class CoachMatchupHistory:
    """История встреч двух тренеров"""
    coach1: str
    coach2: str
    
    total_games: int = 0
    coach1_wins: int = 0
    coach2_wins: int = 0
    
    # Детали
    playoff_games: int = 0
    coach1_playoff_wins: int = 0
    
    avg_total_points: float = 0.0
    avg_margin: float = 0.0
    
    last_5_results: List[Dict] = field(default_factory=list)


@dataclass
class CoachSituationalStats:
    """Ситуационная статистика тренера"""
    coach_name: str
    
    # По ситуациям
    record_as_favorite: str = ""
    record_as_underdog: str = ""
    record_in_b2b: str = ""
    record_after_loss: str = ""
    record_vs_winning_teams: str = ""
    
    # Clutch
    record_in_close_games: str = ""  # Разница < 5 в последние 2 мин
    overtime_record: str = ""
    
    # По месяцам (сезонность)
    october_record: str = ""
    november_record: str = ""
    december_record: str = ""
    january_record: str = ""
    february_record: str = ""
    march_record: str = ""
    april_record: str = ""
    
    # Playoffs
    playoff_record: str = ""
    first_round_record: str = ""
    closeout_game_record: str = ""  # Игры на вылет оппонента
    elimination_game_record: str = ""  # Игры на вылет своей команды


@dataclass
class CoachTrend:
    """Тренд тренера"""
    coach_name: str
    team: str
    season: str
    
    # Текущий сезон
    current_record: str = ""
    current_win_pct: float = 0.0
    
    # Сравнение с прошлым
    last_season_win_pct: float = 0.0
    win_pct_change: float = 0.0
    
    # Изменения в стиле
    pace_change: float = 0.0
    off_rating_change: float = 0.0
    def_rating_change: float = 0.0
    
    # Тренд
    trajectory: str = ""  # "improving", "declining", "stable"
    hot_seat_risk: float = 0.0  # 0-1, вероятность увольнения


class CoachDatabase:
    """
    База данных тренеров NBA
    """
    
    # Текущие тренеры NBA (2024-25)
    CURRENT_COACHES = {
        "LAL": {"name": "JJ Redick", "since": "2024-25", "style": "offensive"},
        "GSW": {"name": "Steve Kerr", "since": "2014-15", "style": "pace_and_space"},
        "BOS": {"name": "Joe Mazzulla", "since": "2022-23", "style": "offensive"},
        "MIA": {"name": "Erik Spoelstra", "since": "2008-09", "style": "defensive"},
        "DEN": {"name": "Michael Malone", "since": "2015-16", "style": "balanced"},
        "PHX": {"name": "Mike Budenholzer", "since": "2024-25", "style": "defensive"},
        "MIL": {"name": "Doc Rivers", "since": "2023-24", "style": "balanced"},
        "PHI": {"name": "Nick Nurse", "since": "2023-24", "style": "defensive"},
        "NYK": {"name": "Tom Thibodeau", "since": "2020-21", "style": "defensive"},
        "BKN": {"name": "Jordi Fernandez", "since": "2024-25", "style": "balanced"},
        "CLE": {"name": "Kenny Atkinson", "since": "2024-25", "style": "offensive"},
        "ORL": {"name": "Jamahl Mosley", "since": "2021-22", "style": "defensive"},
        "ATL": {"name": "Quin Snyder", "since": "2023-24", "style": "balanced"},
        "CHI": {"name": "Billy Donovan", "since": "2020-21", "style": "balanced"},
        "IND": {"name": "Rick Carlisle", "since": "2021-22", "style": "offensive"},
        "DET": {"name": "JB Bickerstaff", "since": "2024-25", "style": "defensive"},
        "TOR": {"name": "Darko Rajakovic", "since": "2023-24", "style": "offensive"},
        "CHA": {"name": "Charles Lee", "since": "2024-25", "style": "balanced"},
        "WAS": {"name": "Brian Keefe", "since": "2024-25", "style": "balanced"},
        "DAL": {"name": "Jason Kidd", "since": "2021-22", "style": "balanced"},
        "OKC": {"name": "Mark Daigneault", "since": "2020-21", "style": "defensive"},
        "MIN": {"name": "Chris Finch", "since": "2020-21", "style": "offensive"},
        "LAC": {"name": "Tyronn Lue", "since": "2020-21", "style": "offensive"},
        "SAC": {"name": "Mike Brown", "since": "2022-23", "style": "defensive"},
        "NOP": {"name": "Willie Green", "since": "2021-22", "style": "balanced"},
        "HOU": {"name": "Ime Udoka", "since": "2023-24", "style": "defensive"},
        "SAS": {"name": "Gregg Popovich", "since": "1996-97", "style": "balanced"},
        "MEM": {"name": "Taylor Jenkins", "since": "2019-20", "style": "pace_and_space"},
        "UTA": {"name": "Will Hardy", "since": "2022-23", "style": "balanced"},
        "POR": {"name": "Chauncey Billups", "since": "2021-22", "style": "offensive"},
    }
    
    # Исторические данные о известных тренерах
    COACH_PROFILES = {
        "Gregg Popovich": CoachProfile(
            name="Gregg Popovich",
            current_team="SAS",
            years_experience=28,
            total_wins=1390,
            total_losses=750,
            career_win_pct=0.649,
            championships=5,
            finals_appearances=6,
            conference_finals=10,
            coach_of_year_awards=3,
            offensive_rating_avg=108.5,
            defensive_rating_avg=103.2,
            pace_avg=95.0,
            primary_style="balanced",
            defensive_minded=True,
            player_development=True,
            teams_coached=["Spurs"]
        ),
        "Steve Kerr": CoachProfile(
            name="Steve Kerr",
            current_team="GSW",
            years_experience=10,
            total_wins=480,
            total_losses=240,
            career_win_pct=0.667,
            championships=4,
            finals_appearances=6,
            conference_finals=6,
            coach_of_year_awards=1,
            offensive_rating_avg=115.5,
            defensive_rating_avg=107.0,
            pace_avg=101.5,
            primary_style="pace_and_space",
            heavy_3pt_usage=True,
            prefers_small_ball=True,
            teams_coached=["Warriors"]
        ),
        "Erik Spoelstra": CoachProfile(
            name="Erik Spoelstra",
            current_team="MIA",
            years_experience=16,
            total_wins=690,
            total_losses=480,
            career_win_pct=0.590,
            championships=2,
            finals_appearances=6,
            conference_finals=7,
            offensive_rating_avg=110.0,
            defensive_rating_avg=106.5,
            pace_avg=97.5,
            primary_style="defensive",
            defensive_minded=True,
            player_development=True,
            teams_coached=["Heat"]
        ),
        "Tom Thibodeau": CoachProfile(
            name="Tom Thibodeau",
            current_team="NYK",
            years_experience=13,
            total_wins=500,
            total_losses=350,
            career_win_pct=0.588,
            championships=0,
            finals_appearances=0,
            conference_finals=1,
            coach_of_year_awards=1,
            offensive_rating_avg=107.0,
            defensive_rating_avg=104.0,
            pace_avg=94.5,
            primary_style="defensive",
            defensive_minded=True,
            teams_coached=["Bulls", "Timberwolves", "Knicks"]
        ),
        "Nick Nurse": CoachProfile(
            name="Nick Nurse",
            current_team="PHI",
            years_experience=6,
            total_wins=250,
            total_losses=180,
            career_win_pct=0.581,
            championships=1,
            finals_appearances=1,
            conference_finals=1,
            coach_of_year_awards=1,
            offensive_rating_avg=112.0,
            defensive_rating_avg=108.0,
            pace_avg=98.0,
            primary_style="defensive",
            defensive_minded=True,
            teams_coached=["Raptors", "76ers"]
        ),
    }
    
    def __init__(self):
        self.profiles = self.COACH_PROFILES.copy()
        self.current = self.CURRENT_COACHES.copy()
    
    def get_coach_for_team(self, team_abbr: str) -> Optional[Dict]:
        """Получает текущего тренера команды"""
        return self.current.get(team_abbr)
    
    def get_coach_profile(self, name: str) -> Optional[CoachProfile]:
        """Получает профиль тренера"""
        return self.profiles.get(name)
    
    def get_coach_style(self, team_abbr: str) -> str:
        """Получает стиль игры тренера команды"""
        coach = self.current.get(team_abbr, {})
        return coach.get('style', 'balanced')


class CoachAnalyzer:
    """
    Анализатор тренеров
    """
    
    def __init__(self, db: CoachDatabase = None):
        self.db = db or CoachDatabase()
    
    def analyze_coaching_matchup(
        self,
        home_team: str,
        away_team: str
    ) -> Dict:
        """Анализирует матчап тренеров"""
        
        home_coach = self.db.get_coach_for_team(home_team)
        away_coach = self.db.get_coach_for_team(away_team)
        
        if not home_coach or not away_coach:
            return {}
        
        home_profile = self.db.get_coach_profile(home_coach['name'])
        away_profile = self.db.get_coach_profile(away_coach['name'])
        
        analysis = {
            'home_coach': home_coach['name'],
            'away_coach': away_coach['name'],
            'home_style': home_coach['style'],
            'away_style': away_coach['style'],
            'style_clash': self._analyze_style_clash(home_coach['style'], away_coach['style']),
            'experience_edge': 'home' if home_profile and away_profile and 
                             home_profile.years_experience > away_profile.years_experience else 'away',
            'factors': []
        }
        
        # Анализируем факторы
        if home_profile:
            if home_profile.championships > 0:
                analysis['factors'].append(f"{home_coach['name']} has championship experience ({home_profile.championships}x)")
            if home_profile.defensive_minded and away_coach['style'] == 'offensive':
                analysis['factors'].append(f"{home_coach['name']}'s defense could slow down {away_coach['name']}'s offense")
        
        if away_profile:
            if away_profile.career_win_pct > 0.600:
                analysis['factors'].append(f"{away_coach['name']} has elite career win% ({away_profile.career_win_pct:.1%})")
        
        # Предсказание влияния на игру
        analysis['game_style_prediction'] = self._predict_game_style(
            home_coach['style'], away_coach['style']
        )
        
        return analysis
    
    def _analyze_style_clash(self, style1: str, style2: str) -> str:
        """Анализирует столкновение стилей"""
        if style1 == style2:
            return f"Similar styles ({style1}) - expect competitive game"
        
        clashes = {
            ("offensive", "defensive"): "Offense vs Defense - style clash",
            ("defensive", "offensive"): "Defense vs Offense - style clash",
            ("pace_and_space", "defensive"): "Fast vs Slow - pace battle",
            ("defensive", "pace_and_space"): "Slow vs Fast - pace battle",
        }
        
        return clashes.get((style1, style2), f"{style1} vs {style2}")
    
    def _predict_game_style(self, style1: str, style2: str) -> Dict:
        """Предсказывает стиль игры"""
        
        # Базовые значения
        pace = 100
        total = 220
        
        # Корректировки по стилям
        if style1 == 'pace_and_space' or style2 == 'pace_and_space':
            pace += 3
            total += 8
        
        if style1 == 'defensive' and style2 == 'defensive':
            pace -= 4
            total -= 12
        elif style1 == 'defensive' or style2 == 'defensive':
            pace -= 2
            total -= 5
        
        if style1 == 'offensive' and style2 == 'offensive':
            total += 10
        elif style1 == 'offensive' or style2 == 'offensive':
            total += 4
        
        return {
            'expected_pace': pace,
            'total_adjustment': total - 220,
            'game_type': 'high_scoring' if total > 225 else 'low_scoring' if total < 215 else 'average'
        }
    
    def get_coach_trend(self, team_abbr: str, current_record: Tuple[int, int]) -> CoachTrend:
        """Получает тренд тренера"""
        coach = self.db.get_coach_for_team(team_abbr)
        
        if not coach:
            return CoachTrend(coach_name="Unknown", team=team_abbr, season="2024-25")
        
        wins, losses = current_record
        win_pct = wins / (wins + losses) if (wins + losses) > 0 else 0
        
        trend = CoachTrend(
            coach_name=coach['name'],
            team=team_abbr,
            season="2024-25",
            current_record=f"{wins}-{losses}",
            current_win_pct=win_pct
        )
        
        # Hot seat risk
        if win_pct < 0.350:
            trend.hot_seat_risk = 0.8
            trend.trajectory = "declining"
        elif win_pct < 0.450:
            trend.hot_seat_risk = 0.4
            trend.trajectory = "struggling"
        elif win_pct > 0.600:
            trend.hot_seat_risk = 0.0
            trend.trajectory = "thriving"
        else:
            trend.hot_seat_risk = 0.1
            trend.trajectory = "stable"
        
        # Первый сезон = больше терпения
        if coach['since'] == "2024-25":
            trend.hot_seat_risk *= 0.3
        
        return trend
    
    def get_coaching_edge(
        self,
        home_team: str,
        away_team: str
    ) -> Tuple[str, float]:
        """
        Определяет преимущество в тренерском матчапе
        
        Returns:
            (edge_team, edge_value) - команда с преимуществом и его величина (0-1)
        """
        home_coach = self.db.get_coach_for_team(home_team)
        away_coach = self.db.get_coach_for_team(away_team)
        
        if not home_coach or not away_coach:
            return ("none", 0.0)
        
        home_profile = self.db.get_coach_profile(home_coach['name'])
        away_profile = self.db.get_coach_profile(away_coach['name'])
        
        home_score = 0
        away_score = 0
        
        # Опыт
        if home_profile:
            home_score += min(home_profile.years_experience / 20, 1.0) * 0.3
            home_score += home_profile.championships * 0.1
            if home_profile.career_win_pct > 0.550:
                home_score += 0.2
        
        if away_profile:
            away_score += min(away_profile.years_experience / 20, 1.0) * 0.3
            away_score += away_profile.championships * 0.1
            if away_profile.career_win_pct > 0.550:
                away_score += 0.2
        
        # Новый тренер = неизвестность
        if home_coach['since'] == "2024-25":
            home_score *= 0.7
        if away_coach['since'] == "2024-25":
            away_score *= 0.7
        
        diff = home_score - away_score
        
        if diff > 0.15:
            return ("home", min(diff, 0.5))
        elif diff < -0.15:
            return ("away", min(abs(diff), 0.5))
        else:
            return ("none", 0.0)


# === ТЕСТИРОВАНИЕ ===

if __name__ == "__main__":
    print("=== Тест Coach Analytics ===\n")
    
    db = CoachDatabase()
    analyzer = CoachAnalyzer(db)
    
    # Тест профилей
    print("Профили тренеров:")
    print("-" * 50)
    
    for name, profile in db.profiles.items():
        print(f"\n{profile.name} ({profile.current_team})")
        print(f"  Experience: {profile.years_experience} years")
        print(f"  Record: {profile.total_wins}-{profile.total_losses} ({profile.career_win_pct:.1%})")
        print(f"  Championships: {profile.championships}")
        print(f"  Style: {profile.primary_style}")
    
    # Тест текущих тренеров
    print("\n\nТекущие тренеры (выборка):")
    print("-" * 50)
    
    for team in ["LAL", "GSW", "BOS", "MIA", "DEN"]:
        coach = db.get_coach_for_team(team)
        print(f"{team}: {coach['name']} ({coach['style']}) since {coach['since']}")
    
    # Тест матчапа
    print("\n\nАнализ матчапа GSW vs MIA:")
    print("-" * 50)
    
    matchup = analyzer.analyze_coaching_matchup("GSW", "MIA")
    
    print(f"Home coach: {matchup['home_coach']} ({matchup['home_style']})")
    print(f"Away coach: {matchup['away_coach']} ({matchup['away_style']})")
    print(f"Style clash: {matchup['style_clash']}")
    print(f"Experience edge: {matchup['experience_edge']}")
    
    print("\nFactors:")
    for factor in matchup['factors']:
        print(f"  - {factor}")
    
    print(f"\nGame style prediction:")
    for k, v in matchup['game_style_prediction'].items():
        print(f"  {k}: {v}")
    
    # Тест coaching edge
    print("\n\nCoaching Edge:")
    print("-" * 50)
    
    matchups = [
        ("BOS", "MIA"),
        ("LAL", "GSW"),
        ("SAS", "DET"),
        ("NYK", "PHI"),
    ]
    
    for home, away in matchups:
        edge_team, edge_val = analyzer.get_coaching_edge(home, away)
        home_coach = db.get_coach_for_team(home)['name']
        away_coach = db.get_coach_for_team(away)['name']
        print(f"{home_coach} vs {away_coach}: Edge to {edge_team} ({edge_val:.2f})")
    
    # Тест трендов
    print("\n\nCoach Trends:")
    print("-" * 50)
    
    trends_data = [
        ("LAL", (25, 20)),
        ("GSW", (22, 23)),
        ("BOS", (35, 10)),
        ("WAS", (10, 35)),
    ]
    
    for team, record in trends_data:
        trend = analyzer.get_coach_trend(team, record)
        print(f"{trend.coach_name} ({team}): {trend.current_record}")
        print(f"  Trajectory: {trend.trajectory}")
        print(f"  Hot seat risk: {trend.hot_seat_risk:.0%}")
    
    print("\n✅ Тест завершен")
