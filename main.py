"""
AutoBasket - Main Orchestrator
==============================
Главный модуль, связывающий все компоненты системы
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
import json
import logging

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import config, BetCategory, BetStatus
from core.bankroll_manager import BankrollManager
from core.prediction_engine import BasketballPredictor, GameFeatures
from core.elo_system import EloRatingSystem
from core.value_finder import ValueBetFinder, BettingPortfolioManager
from core.backtesting import StrategyBacktester, KellyStrategy
from data.database import Database
from utils.notifications import NotificationManager, AlertPriority
from utils.discipline import DisciplineManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DailyReport:
    """Дневной отчет"""
    date: date
    starting_bankroll: float
    ending_bankroll: float
    total_bets: int
    wins: int
    losses: int
    total_wagered: float
    total_profit: float
    roi: float
    best_bet: Optional[Dict] = None
    worst_bet: Optional[Dict] = None
    recommendations: List[str] = field(default_factory=list)


class AutoBasket:
    """
    Главный класс системы AutoBasket
    
    Координирует работу всех компонентов:
    - Управление банкроллом
    - ML предсказания
    - Elo рейтинги
    - Поиск value bets
    - Уведомления
    - Контроль дисциплины
    """
    
    def __init__(
        self,
        initial_bankroll: float = None,
        db_path: str = None,
        telegram_token: str = None,
        telegram_chat_id: str = None
    ):
        logger.info("Initializing AutoBasket system...")
        
        # Инициализация компонентов
        self.bankroll = BankrollManager(
            initial_bankroll=initial_bankroll or config.bankroll.initial_bankroll
        )
        
        self.predictor = BasketballPredictor(use_ml=False)  # Без ML для начала
        self.elo = EloRatingSystem()
        self.value_finder = ValueBetFinder()
        self.portfolio = BettingPortfolioManager(self.bankroll)
        
        self.db = Database(db_path=db_path or config.database.db_path)
        
        self.notifications = NotificationManager(
            telegram_token=telegram_token,
            telegram_chat_id=telegram_chat_id
        )
        
        self.discipline = DisciplineManager()
        
        # Состояние системы
        self.is_running = False
        self.today_plan = None
        self.active_bets: List[Dict] = []
        
        # Синхронизируем с БД
        self._sync_from_db()
        
        logger.info(f"AutoBasket initialized. Bankroll: ${self.bankroll.bankroll:.2f}")
    
    def _sync_from_db(self):
        """Синхронизирует состояние с базой данных"""
        # Загружаем банкролл
        db_bankroll = self.db.get_bankroll()
        if db_bankroll:
            self.bankroll.bankroll = db_bankroll['current_balance']
            self.bankroll.peak_bankroll = db_bankroll.get('peak_balance', self.bankroll.bankroll)
        else:
            # Сохраняем начальный
            self.db.update_bankroll(
                self.bankroll.bankroll,
                peak=self.bankroll.bankroll
            )
        
        # Загружаем Elo рейтинги
        elo_ratings = self.db.get_all_elo_ratings()
        for r in elo_ratings:
            self.elo.set_rating(r['team_name'], r['current_elo'])
        
        # Загружаем активные ставки
        self.active_bets = self.db.get_active_bets()
    
    def analyze_game(
        self,
        game_id: int,
        home_team: str,
        away_team: str,
        market_odds: Dict[str, float],
        additional_data: Dict = None
    ) -> Dict:
        """
        Полный анализ матча
        
        Args:
            game_id: ID игры
            home_team: Домашняя команда
            away_team: Гостевая команда
            market_odds: {'home_odds': X, 'away_odds': Y}
            additional_data: Дополнительные данные (травмы, и т.д.)
        
        Returns:
            Полный анализ с рекомендациями
        """
        logger.info(f"Analyzing game {game_id}: {home_team} vs {away_team}")
        
        # Получаем Elo рейтинги
        home_elo = self.elo.get_rating(home_team)
        away_elo = self.elo.get_rating(away_team)
        
        # Создаем features для предсказания
        features = GameFeatures(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            game_date=datetime.now(),
            home_elo=home_elo,
            away_elo=away_elo,
            # Добавляем данные из additional_data если есть
            **self._extract_features(additional_data or {})
        )
        
        # Получаем предсказание
        prediction = self.predictor.predict(features)
        
        # Elo-based prediction
        elo_prediction = self.elo.predict_game(home_team, away_team)
        
        # Комбинируем предсказания
        combined_home_prob = (
            prediction.home_win_prob * 0.6 +
            elo_prediction['home_win_prob'] * 0.4
        )
        
        # Ищем value
        predictions = {
            game_id: {
                'home_win_prob': combined_home_prob,
                'away_win_prob': 1 - combined_home_prob,
                'model_agreement': prediction.model_agreement
            }
        }
        
        value_bets = self.value_finder.find_value_bets(
            predictions,
            {game_id: market_odds},
            {game_id: {'home_team': home_team, 'away_team': away_team}}
        )
        
        # Формируем результат
        analysis = {
            'game_id': game_id,
            'home_team': home_team,
            'away_team': away_team,
            'home_elo': home_elo,
            'away_elo': away_elo,
            'elo_diff': home_elo - away_elo,
            'ml_prediction': {
                'home_win_prob': prediction.home_win_prob,
                'away_win_prob': prediction.away_win_prob,
                'confidence': prediction.confidence,
                'model_agreement': prediction.model_agreement
            },
            'elo_prediction': {
                'home_win_prob': elo_prediction['home_win_prob'],
                'expected_margin': elo_prediction['expected_margin']
            },
            'combined_prediction': {
                'home_win_prob': combined_home_prob,
                'away_win_prob': 1 - combined_home_prob,
                'predicted_winner': home_team if combined_home_prob > 0.5 else away_team
            },
            'market_odds': market_odds,
            'value_bets': [
                {
                    'bet_on': vb.bet_on,
                    'team': vb.team_name,
                    'odds': vb.odds,
                    'expected_value': vb.expected_value,
                    'edge': vb.edge,
                    'category': vb.category.value,
                    'kelly_fraction': vb.kelly_fraction
                }
                for vb in value_bets
            ],
            'has_value': len(value_bets) > 0,
            'timestamp': datetime.now().isoformat()
        }
        
        return analysis
    
    def _extract_features(self, data: Dict) -> Dict:
        """Извлекает признаки из дополнительных данных"""
        return {
            'home_win_pct_last10': data.get('home_win_pct_last10', 0.5),
            'away_win_pct_last10': data.get('away_win_pct_last10', 0.5),
            'home_net_rating': data.get('home_net_rating', 0.0),
            'away_net_rating': data.get('away_net_rating', 0.0),
            'home_rest_days': data.get('home_rest_days', 1),
            'away_rest_days': data.get('away_rest_days', 1),
            'home_injury_impact': data.get('home_injury_impact', 0.0),
            'away_injury_impact': data.get('away_injury_impact', 0.0),
        }
    
    def get_bet_recommendation(
        self,
        game_id: int,
        home_team: str,
        away_team: str,
        market_odds: Dict[str, float],
        additional_data: Dict = None
    ) -> Dict:
        """
        Получает рекомендацию по ставке
        
        Returns:
            Рекомендация с суммой и обоснованием
        """
        # Проверяем дисциплину
        can_bet, warnings = self.discipline.check_can_bet()
        
        if not can_bet:
            return {
                'recommend': False,
                'reason': 'Ставки заблокированы',
                'warnings': [w.description for w in warnings]
            }
        
        # Анализируем игру
        analysis = self.analyze_game(
            game_id, home_team, away_team, market_odds, additional_data
        )
        
        if not analysis['has_value']:
            return {
                'recommend': False,
                'reason': 'Не найдено value в этом матче',
                'analysis': analysis
            }
        
        # Берем лучший value bet
        best_value = analysis['value_bets'][0]
        
        # Рассчитываем размер ставки
        bet_amount, details = self.bankroll.calculate_optimal_bet_size(
            confidence=analysis['combined_prediction']['home_win_prob'] if best_value['bet_on'] == 'home' else analysis['combined_prediction']['away_win_prob'],
            odds=best_value['odds'],
            category=BetCategory(best_value['category'])
        )
        
        # Применяем множитель дисциплины
        discipline_mult = self.discipline.get_recommended_bet_multiplier()
        adjusted_amount = bet_amount * discipline_mult
        
        if adjusted_amount < config.bankroll.min_bet:
            return {
                'recommend': False,
                'reason': 'Размер ставки ниже минимума после корректировки дисциплины',
                'analysis': analysis,
                'discipline_multiplier': discipline_mult
            }
        
        return {
            'recommend': True,
            'game_id': game_id,
            'bet_on': best_value['bet_on'],
            'team': best_value['team'],
            'odds': best_value['odds'],
            'recommended_amount': round(adjusted_amount, 2),
            'original_amount': bet_amount,
            'discipline_multiplier': discipline_mult,
            'expected_value': best_value['expected_value'],
            'edge': best_value['edge'],
            'category': best_value['category'],
            'potential_win': round(adjusted_amount * (best_value['odds'] - 1), 2),
            'confidence': analysis['combined_prediction']['home_win_prob'] if best_value['bet_on'] == 'home' else analysis['combined_prediction']['away_win_prob'],
            'analysis': analysis,
            'warnings': [w.description for w in warnings] if warnings else []
        }
    
    def place_bet(
        self,
        game_id: int,
        team: str,
        amount: float,
        odds: float,
        home_team: str,
        away_team: str,
        confidence: float = None,
        category: str = 'value'
    ) -> Dict:
        """
        Размещает ставку
        
        Returns:
            Результат размещения
        """
        # Проверяем дисциплину
        can_bet, warnings = self.discipline.check_can_bet(proposed_amount=amount)
        
        if not can_bet:
            return {
                'success': False,
                'reason': 'Ставка заблокирована системой дисциплины',
                'warnings': [w.description for w in warnings]
            }
        
        # Размещаем через bankroll manager
        bet = self.bankroll.place_bet(
            game_id=game_id,
            team=team,
            bet_amount=amount,
            odds=odds,
            confidence=confidence or 0.55,
            category=BetCategory(category)
        )
        
        if not bet:
            return {
                'success': False,
                'reason': 'Недостаточно средств или превышен лимит'
            }
        
        # Сохраняем в БД
        bet_id = self.db.add_active_bet({
            'game_id': game_id,
            'home_team': home_team,
            'away_team': away_team,
            'team_bet_on': team,
            'bet_amount': amount,
            'odds': odds,
            'potential_win': amount * (odds - 1),
            'confidence': confidence,
            'expected_value': self.bankroll.calculate_expected_value(confidence or 0.55, odds, amount),
            'category': category
        })
        
        # Записываем в discipline manager
        self.discipline.record_bet(amount, self.bankroll.bankroll)
        
        # Обновляем банкролл в БД
        self.db.update_bankroll(self.bankroll.bankroll)
        
        # Отправляем уведомление
        self.notifications.notify_value_bet(
            team=team,
            opponent=away_team if team == home_team else home_team,
            confidence=confidence or 0.55,
            odds=odds,
            ev=(confidence or 0.55) * (odds - 1) - (1 - (confidence or 0.55)),
            bet_amount=amount
        )
        
        logger.info(f"Bet placed: ${amount:.2f} on {team} @ {odds}")
        
        return {
            'success': True,
            'bet_id': bet_id,
            'game_id': game_id,
            'team': team,
            'amount': amount,
            'odds': odds,
            'potential_win': amount * (odds - 1),
            'new_bankroll': self.bankroll.bankroll
        }
    
    def settle_bet(self, bet_id: int, won: bool) -> Dict:
        """
        Закрывает ставку с результатом
        """
        # Закрываем в bankroll manager
        bet = self.bankroll.settle_bet(bet_id, won)
        
        if not bet:
            return {'success': False, 'reason': 'Ставка не найдена'}
        
        # Обновляем в БД
        self.db.settle_bet(bet_id, 'won' if won else 'lost', bet.result_amount)
        self.db.update_bankroll(
            self.bankroll.bankroll,
            peak=self.bankroll.peak_bankroll
        )
        
        # Записываем результат для discipline
        self.discipline.record_result(won)
        
        # Проверяем stop-loss / take-profit
        daily_change = self.bankroll.get_daily_change()
        
        if daily_change <= config.bankroll.stop_loss_daily:
            self.notifications.notify_stop_loss(
                loss_amount=abs(self.bankroll.bankroll - self.bankroll.day_start_balance),
                loss_percentage=abs(daily_change)
            )
            self.discipline.lock_betting(2, "Daily stop-loss triggered")
        
        if daily_change >= config.bankroll.take_profit_daily:
            self.notifications.notify_take_profit(
                profit_amount=self.bankroll.bankroll - self.bankroll.day_start_balance,
                profit_percentage=daily_change
            )
        
        logger.info(f"Bet {bet_id} settled: {'WON' if won else 'LOST'}, P&L: ${bet.result_amount:.2f}")
        
        return {
            'success': True,
            'bet_id': bet_id,
            'result': 'won' if won else 'lost',
            'profit': bet.result_amount,
            'new_bankroll': self.bankroll.bankroll,
            'daily_change': daily_change
        }
    
    def get_status(self) -> Dict:
        """Возвращает текущий статус системы"""
        stats = self.bankroll.get_statistics()
        discipline_summary = self.discipline.get_session_summary()
        
        return {
            'bankroll': {
                'current': self.bankroll.bankroll,
                'initial': self.bankroll.initial_bankroll,
                'peak': self.bankroll.peak_bankroll,
                'daily_change': self.bankroll.get_daily_change(),
                'total_change': self.bankroll.get_total_change()
            },
            'statistics': stats,
            'discipline': discipline_summary,
            'active_bets': len(self.active_bets),
            'today_bets': self.bankroll.today_bets_count,
            'remaining_daily_bets': config.bankroll.max_bets_per_day - self.bankroll.today_bets_count,
            'system_status': 'locked' if self.discipline.global_lock else 'active'
        }
    
    def generate_daily_report(self) -> str:
        """Генерирует дневной отчет"""
        status = self.get_status()
        stats = status['statistics']
        
        lines = [
            "=" * 60,
            "🏀 AUTOBASKET - ДНЕВНОЙ ОТЧЕТ",
            f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 60,
            "",
            "💰 БАНКРОЛЛ",
            f"   Текущий: ${status['bankroll']['current']:.2f}",
            f"   Изменение за день: {status['bankroll']['daily_change']:+.1%}",
            f"   Общее изменение: {status['bankroll']['total_change']:+.1%}",
            f"   Peak: ${status['bankroll']['peak']:.2f}",
            "",
            "📊 СТАТИСТИКА",
            f"   Всего ставок: {stats['total_bets']}",
            f"   Побед: {stats['wins']}",
            f"   Поражений: {stats['losses']}",
            f"   Win Rate: {stats['win_rate']:.1%}",
            f"   ROI: {stats['roi']:.1f}%",
            f"   Всего поставлено: ${stats['total_wagered']:.2f}",
            f"   Общий профит: ${stats['total_profit']:.2f}",
            "",
            "🎯 СЕГОДНЯ",
            f"   Ставок размещено: {status['today_bets']}",
            f"   Осталось ставок: {status['remaining_daily_bets']}",
            f"   Активных ставок: {status['active_bets']}",
            "",
            "🧠 ДИСЦИПЛИНА",
            f"   Статус: {status['system_status'].upper()}",
        ]
        
        if status['discipline'].get('warnings_count', 0) > 0:
            lines.append(f"   ⚠️ Предупреждений: {status['discipline']['warnings_count']}")
        
        lines.extend(["", "=" * 60])
        
        return "\n".join(lines)


# === QUICK START FUNCTIONS ===

def create_system(
    bankroll: float = 200.0,
    telegram_token: str = None,
    telegram_chat_id: str = None
) -> AutoBasket:
    """Создает и возвращает экземпляр системы"""
    return AutoBasket(
        initial_bankroll=bankroll,
        telegram_token=telegram_token,
        telegram_chat_id=telegram_chat_id
    )


# === MAIN ===

if __name__ == "__main__":
    print("=== AutoBasket System Test ===\n")
    
    # Создаем систему
    system = create_system(bankroll=200.0)
    
    # Тестируем анализ игры
    print("Анализ матча Lakers vs Warriors:")
    print("-" * 50)
    
    analysis = system.analyze_game(
        game_id=1001,
        home_team="Los Angeles Lakers",
        away_team="Golden State Warriors",
        market_odds={'home_odds': 1.75, 'away_odds': 2.10}
    )
    
    print(f"Home Elo: {analysis['home_elo']:.0f}")
    print(f"Away Elo: {analysis['away_elo']:.0f}")
    print(f"Predicted winner: {analysis['combined_prediction']['predicted_winner']}")
    print(f"Home win prob: {analysis['combined_prediction']['home_win_prob']:.1%}")
    print(f"Has value: {analysis['has_value']}")
    
    if analysis['value_bets']:
        vb = analysis['value_bets'][0]
        print(f"\nValue bet found:")
        print(f"  Bet on: {vb['team']}")
        print(f"  Odds: {vb['odds']}")
        print(f"  EV: {vb['expected_value']:.1%}")
    
    # Получаем рекомендацию
    print("\n\nРекомендация по ставке:")
    print("-" * 50)
    
    rec = system.get_bet_recommendation(
        game_id=1001,
        home_team="Los Angeles Lakers",
        away_team="Golden State Warriors",
        market_odds={'home_odds': 1.75, 'away_odds': 2.10}
    )
    
    if rec['recommend']:
        print(f"✅ Рекомендуется ставка")
        print(f"   Команда: {rec['team']}")
        print(f"   Сумма: ${rec['recommended_amount']:.2f}")
        print(f"   Коэффициент: {rec['odds']}")
        print(f"   EV: {rec['expected_value']:.1%}")
        print(f"   Потенциальный выигрыш: ${rec['potential_win']:.2f}")
    else:
        print(f"❌ Ставка не рекомендуется: {rec['reason']}")
    
    # Генерируем отчет
    print("\n")
    print(system.generate_daily_report())
