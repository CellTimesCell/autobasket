"""
AutoBasket - Elo Rating System
==============================
Собственная система рейтингов Elo для NBA/NCAA
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date
import json

import sys
sys.path.append('..')
from config.settings import config


@dataclass
class EloUpdate:
    """Запись об обновлении Elo"""
    game_id: int
    date: datetime
    home_team: str
    away_team: str
    home_won: bool
    margin: Optional[int]
    home_elo_before: float
    away_elo_before: float
    home_elo_after: float
    away_elo_after: float
    home_expected: float
    elo_change: float


class EloRatingSystem:
    """
    Система рейтингов Elo для баскетбола
    
    Особенности:
    - Учет домашнего преимущества
    - Margin of Victory adjustment
    - Сезонная регрессия к среднему
    - Автоматическое обновление после игр
    """
    
    def __init__(
        self,
        k_factor: float = None,
        home_advantage: float = None,
        default_rating: float = None
    ):
        self.k_factor = k_factor or config.elo.k_factor
        self.home_advantage = home_advantage or config.elo.home_advantage
        self.default_rating = default_rating or config.elo.default_rating
        
        # Рейтинги команд
        self.ratings: Dict[str, float] = {}
        
        # История изменений
        self.history: List[EloUpdate] = []
        
        # Статистика команд
        self.team_stats: Dict[str, Dict] = {}
        
        # NBA команды с начальными рейтингами (примерные)
        self._initialize_nba_teams()
    
    def _initialize_nba_teams(self):
        """Инициализирует рейтинги NBA команд"""
        # Примерные рейтинги на основе силы команд
        nba_teams = {
            # Топ-тир (контендеры)
            "Boston Celtics": 1680,
            "Denver Nuggets": 1660,
            "Milwaukee Bucks": 1650,
            "Phoenix Suns": 1640,
            "Philadelphia 76ers": 1630,
            
            # Второй тир (плей-офф команды)
            "Los Angeles Lakers": 1600,
            "Golden State Warriors": 1595,
            "Miami Heat": 1590,
            "Cleveland Cavaliers": 1585,
            "Sacramento Kings": 1580,
            "New York Knicks": 1575,
            "Brooklyn Nets": 1570,
            "Memphis Grizzlies": 1565,
            
            # Третий тир (борьба за плей-офф)
            "Los Angeles Clippers": 1550,
            "Minnesota Timberwolves": 1545,
            "New Orleans Pelicans": 1540,
            "Dallas Mavericks": 1535,
            "Atlanta Hawks": 1530,
            "Toronto Raptors": 1525,
            "Chicago Bulls": 1520,
            "Oklahoma City Thunder": 1515,
            
            # Четвертый тир
            "Indiana Pacers": 1500,
            "Utah Jazz": 1495,
            "Orlando Magic": 1490,
            "Washington Wizards": 1480,
            "Portland Trail Blazers": 1475,
            "Charlotte Hornets": 1470,
            "San Antonio Spurs": 1460,
            "Houston Rockets": 1450,
            "Detroit Pistons": 1440,
        }
        
        self.ratings.update(nba_teams)
        
        # Инициализируем статистику
        for team in nba_teams:
            self.team_stats[team] = {
                'games_played': 0,
                'wins': 0,
                'losses': 0,
                'peak_elo': nba_teams[team],
                'lowest_elo': nba_teams[team],
                'elo_history': []
            }
    
    def get_rating(self, team: str) -> float:
        """Возвращает рейтинг команды"""
        return self.ratings.get(team, self.default_rating)
    
    def set_rating(self, team: str, rating: float):
        """Устанавливает рейтинг команды"""
        self.ratings[team] = rating
        
        if team not in self.team_stats:
            self.team_stats[team] = {
                'games_played': 0,
                'wins': 0,
                'losses': 0,
                'peak_elo': rating,
                'lowest_elo': rating,
                'elo_history': []
            }
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Ожидаемый результат для команды A
        
        E(A) = 1 / (1 + 10^((R_B - R_A) / 400))
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def predict_game(
        self, 
        home_team: str, 
        away_team: str,
        include_home_advantage: bool = True
    ) -> Dict:
        """
        Предсказание матча на основе Elo
        
        Args:
            home_team: Домашняя команда
            away_team: Гостевая команда
            include_home_advantage: Учитывать домашнее преимущество
        
        Returns:
            Словарь с вероятностями и ожидаемым результатом
        """
        home_elo = self.get_rating(home_team)
        away_elo = self.get_rating(away_team)
        
        # Добавляем домашнее преимущество
        if include_home_advantage:
            home_elo_adjusted = home_elo + self.home_advantage
        else:
            home_elo_adjusted = home_elo
        
        home_win_prob = self.expected_score(home_elo_adjusted, away_elo)
        
        # Конвертируем в ожидаемый spread
        # Примерная формула: elo_diff / 28 ≈ expected_margin
        elo_diff = home_elo_adjusted - away_elo
        expected_margin = elo_diff / 28
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_elo': home_elo,
            'away_elo': away_elo,
            'home_elo_adjusted': home_elo_adjusted,
            'home_win_prob': home_win_prob,
            'away_win_prob': 1 - home_win_prob,
            'elo_diff': elo_diff,
            'expected_margin': expected_margin,
            'predicted_winner': home_team if home_win_prob > 0.5 else away_team,
            'confidence': max(home_win_prob, 1 - home_win_prob)
        }
    
    def _calculate_mov_multiplier(self, margin: int, elo_diff: float) -> float:
        """
        Margin of Victory multiplier
        
        Большой margin = больше изменение рейтинга
        Но с учетом того, что большой margin против слабой команды менее значим
        """
        if margin is None or margin == 0:
            return 1.0
        
        # Базовый MOV multiplier
        mov = np.log(abs(margin) + 1)
        
        # Autocorrelation adjustment
        # Если сильная команда побеждает слабую с большим margin - это ожидаемо
        # Если слабая команда побеждает сильную - это важнее
        if elo_diff != 0:
            adjustment = 2.2 / ((elo_diff * 0.001) + 2.2)
        else:
            adjustment = 1.0
        
        return mov * adjustment
    
    def update_ratings(
        self,
        game_id: int,
        home_team: str,
        away_team: str,
        home_won: bool,
        margin: Optional[int] = None,
        game_date: datetime = None
    ) -> EloUpdate:
        """
        Обновляет рейтинги после игры
        
        Args:
            game_id: ID игры
            home_team: Домашняя команда
            away_team: Гостевая команда
            home_won: True если домашняя команда выиграла
            margin: Разница в счете (опционально)
            game_date: Дата игры
        
        Returns:
            EloUpdate запись
        """
        home_elo = self.get_rating(home_team)
        away_elo = self.get_rating(away_team)
        
        # Ожидаемый результат (с учетом home advantage)
        expected_home = self.expected_score(
            home_elo + self.home_advantage, 
            away_elo
        )
        
        # Фактический результат
        actual_home = 1.0 if home_won else 0.0
        
        # K-фактор с учетом MOV
        elo_diff = home_elo + self.home_advantage - away_elo
        k = self.k_factor * self._calculate_mov_multiplier(margin, elo_diff)
        
        # Изменение рейтинга
        change = k * (actual_home - expected_home)
        
        # Обновляем рейтинги
        new_home_elo = home_elo + change
        new_away_elo = away_elo - change
        
        self.ratings[home_team] = new_home_elo
        self.ratings[away_team] = new_away_elo
        
        # Обновляем статистику
        self._update_team_stats(home_team, new_home_elo, home_won)
        self._update_team_stats(away_team, new_away_elo, not home_won)
        
        # Создаем запись
        update = EloUpdate(
            game_id=game_id,
            date=game_date or datetime.now(),
            home_team=home_team,
            away_team=away_team,
            home_won=home_won,
            margin=margin,
            home_elo_before=home_elo,
            away_elo_before=away_elo,
            home_elo_after=new_home_elo,
            away_elo_after=new_away_elo,
            home_expected=expected_home,
            elo_change=abs(change)
        )
        
        self.history.append(update)
        
        return update
    
    def _update_team_stats(self, team: str, new_elo: float, won: bool):
        """Обновляет статистику команды"""
        if team not in self.team_stats:
            self.team_stats[team] = {
                'games_played': 0,
                'wins': 0,
                'losses': 0,
                'peak_elo': new_elo,
                'lowest_elo': new_elo,
                'elo_history': []
            }
        
        stats = self.team_stats[team]
        stats['games_played'] += 1
        stats['wins'] += 1 if won else 0
        stats['losses'] += 0 if won else 1
        stats['peak_elo'] = max(stats['peak_elo'], new_elo)
        stats['lowest_elo'] = min(stats['lowest_elo'], new_elo)
        stats['elo_history'].append({
            'date': datetime.now(),
            'elo': new_elo
        })
    
    def season_reset(self, regression_factor: float = None):
        """
        Регрессия к среднему между сезонами
        
        Команды регрессируют к 1500 на X%
        """
        factor = regression_factor or config.elo.season_reset_regression
        
        for team in self.ratings:
            current = self.ratings[team]
            # Регрессия к среднему
            new_rating = current * factor + self.default_rating * (1 - factor)
            self.ratings[team] = new_rating
            
            # Обновляем пик/дно
            if team in self.team_stats:
                self.team_stats[team]['peak_elo'] = new_rating
                self.team_stats[team]['lowest_elo'] = new_rating
    
    def get_rankings(self, top_n: int = None) -> List[Tuple[str, float]]:
        """Возвращает рейтинги команд отсортированные по убыванию"""
        sorted_teams = sorted(
            self.ratings.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        if top_n:
            return sorted_teams[:top_n]
        return sorted_teams
    
    def get_team_info(self, team: str) -> Dict:
        """Подробная информация о команде"""
        rating = self.get_rating(team)
        stats = self.team_stats.get(team, {})
        
        # Позиция в рейтинге
        rankings = self.get_rankings()
        position = next(
            (i + 1 for i, (t, _) in enumerate(rankings) if t == team),
            None
        )
        
        return {
            'team': team,
            'current_elo': rating,
            'position': position,
            'games_played': stats.get('games_played', 0),
            'wins': stats.get('wins', 0),
            'losses': stats.get('losses', 0),
            'win_pct': stats.get('wins', 0) / max(stats.get('games_played', 1), 1),
            'peak_elo': stats.get('peak_elo', rating),
            'lowest_elo': stats.get('lowest_elo', rating),
            'elo_change_from_start': rating - self.default_rating
        }
    
    def export_ratings(self) -> Dict:
        """Экспортирует все рейтинги для сохранения"""
        return {
            'ratings': self.ratings.copy(),
            'team_stats': {
                team: {
                    k: v for k, v in stats.items() 
                    if k != 'elo_history'  # История может быть большой
                }
                for team, stats in self.team_stats.items()
            },
            'last_updated': datetime.now().isoformat()
        }
    
    def import_ratings(self, data: Dict):
        """Импортирует рейтинги"""
        if 'ratings' in data:
            self.ratings.update(data['ratings'])
        if 'team_stats' in data:
            for team, stats in data['team_stats'].items():
                if team not in self.team_stats:
                    self.team_stats[team] = stats
                else:
                    self.team_stats[team].update(stats)


# === Вспомогательные функции ===

def convert_odds_to_elo_diff(odds: float) -> float:
    """
    Конвертирует коэффициенты букмекера в разницу Elo
    
    odds = decimal odds (например, 1.50)
    """
    # implied_prob = 1 / odds
    # elo_diff = -400 * log10(1/implied_prob - 1)
    implied_prob = 1 / odds
    if implied_prob >= 1:
        return 400
    if implied_prob <= 0:
        return -400
    
    elo_diff = -400 * np.log10(1 / implied_prob - 1)
    return elo_diff


def convert_elo_diff_to_odds(elo_diff: float) -> float:
    """
    Конвертирует разницу Elo в примерные коэффициенты
    """
    prob = 1 / (1 + 10 ** (-elo_diff / 400))
    if prob <= 0.01:
        prob = 0.01
    if prob >= 0.99:
        prob = 0.99
    
    odds = 1 / prob
    return odds


# === ТЕСТИРОВАНИЕ ===

if __name__ == "__main__":
    print("=== Тест Elo Rating System ===\n")
    
    elo = EloRatingSystem()
    
    # Показываем топ-10
    print("Топ-10 команд NBA:")
    print("-" * 40)
    for i, (team, rating) in enumerate(elo.get_rankings(10), 1):
        print(f"{i:2}. {team:<25} {rating:.0f}")
    
    # Тест предсказания
    print("\n\nПредсказание матча:")
    print("-" * 40)
    
    prediction = elo.predict_game("Los Angeles Lakers", "Golden State Warriors")
    
    print(f"Матч: {prediction['home_team']} vs {prediction['away_team']}")
    print(f"Elo: {prediction['home_elo']:.0f} vs {prediction['away_elo']:.0f}")
    print(f"Elo с home advantage: {prediction['home_elo_adjusted']:.0f}")
    print(f"Вероятность победы Lakers: {prediction['home_win_prob']:.1%}")
    print(f"Ожидаемый margin: {prediction['expected_margin']:+.1f}")
    print(f"Прогноз: {prediction['predicted_winner']}")
    
    # Симулируем несколько игр
    print("\n\nСимуляция сезона (10 игр):")
    print("-" * 40)
    
    games = [
        ("Los Angeles Lakers", "Golden State Warriors", True, 8),
        ("Boston Celtics", "Miami Heat", True, 15),
        ("Denver Nuggets", "Phoenix Suns", False, -5),
        ("Milwaukee Bucks", "Philadelphia 76ers", True, 3),
        ("Los Angeles Lakers", "Denver Nuggets", False, -12),
        ("Golden State Warriors", "Boston Celtics", False, -7),
        ("Phoenix Suns", "Miami Heat", True, 10),
        ("Los Angeles Lakers", "Boston Celtics", False, -18),
        ("Denver Nuggets", "Milwaukee Bucks", True, 6),
        ("Golden State Warriors", "Phoenix Suns", True, 4),
    ]
    
    for home, away, home_won, margin in games:
        update = elo.update_ratings(
            game_id=1000 + games.index((home, away, home_won, margin)),
            home_team=home,
            away_team=away,
            home_won=home_won,
            margin=margin
        )
        winner = home if home_won else away
        print(f"{winner} победил ({'+' if home_won else ''}{margin}), "
              f"Elo change: {update.elo_change:.1f}")
    
    # Новые рейтинги
    print("\n\nОбновленные рейтинги:")
    print("-" * 40)
    for team in ["Los Angeles Lakers", "Golden State Warriors", 
                 "Boston Celtics", "Denver Nuggets"]:
        info = elo.get_team_info(team)
        print(f"{team}: {info['current_elo']:.0f} "
              f"(#{info['position']}, {info['wins']}-{info['losses']})")
    
    # Тест конвертации
    print("\n\nКонвертация odds <-> Elo:")
    print("-" * 40)
    
    test_odds = [1.50, 1.85, 2.00, 2.50, 3.00]
    for odds in test_odds:
        elo_diff = convert_odds_to_elo_diff(odds)
        back_odds = convert_elo_diff_to_odds(elo_diff)
        print(f"Odds {odds:.2f} -> Elo diff {elo_diff:+.0f} -> Odds {back_odds:.2f}")
