"""
AutoBasket - Value Bet Finder & Portfolio Manager
==================================================
Поиск ставок с положительной ожидаемой ценностью
и распределение их по портфелю
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum

import sys
sys.path.append('..')
from config.settings import config, BetCategory
from core.bankroll_manager import BankrollManager


@dataclass
class ValueBet:
    """Найденная value ставка"""
    game_id: int
    home_team: str
    away_team: str
    
    # На кого ставить
    bet_on: str  # 'home' или 'away'
    team_name: str
    
    # Вероятности
    model_probability: float  # Наша оценка
    market_probability: float  # Implied probability букмекера
    
    # Коэффициенты
    odds: float
    best_odds: float  # Лучшие коэффициенты на рынке
    
    # Value метрики
    expected_value: float  # EV в %
    edge: float  # Наше преимущество
    
    # Поля с дефолтами
    best_bookmaker: Optional[str] = None
    category: BetCategory = BetCategory.VALUE
    priority_score: float = 0.0
    recommended_bet: float = 0.0
    kelly_fraction: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    model_agreement: float = 1.0
    notes: List[str] = field(default_factory=list)


class ValueBetFinder:
    """
    Находит ставки с положительной ожидаемой ценностью
    
    Value bet = когда наша оценка вероятности выше, 
    чем implied probability букмекера
    """
    
    def __init__(
        self,
        min_value_threshold: float = None,
        min_confidence: float = None,
        max_value_cap: float = 0.30  # Подозрительно если value > 30%
    ):
        self.min_value = min_value_threshold or config.portfolio.value_min_ev
        self.min_confidence = min_confidence or config.prediction.min_confidence_to_bet
        self.max_value_cap = max_value_cap
    
    def find_value_bets(
        self,
        predictions: Dict[int, Dict],
        market_odds: Dict[int, Dict],
        game_info: Dict[int, Dict] = None
    ) -> List[ValueBet]:
        """
        Находит все value bets среди доступных игр
        
        Args:
            predictions: {game_id: {'home_win_prob': X, 'away_win_prob': Y, ...}}
            market_odds: {game_id: {'home_odds': X, 'away_odds': Y, ...}}
            game_info: {game_id: {'home_team': X, 'away_team': Y, ...}} (опционально)
        
        Returns:
            Список ValueBet отсортированный по EV
        """
        value_bets = []
        
        for game_id, pred in predictions.items():
            if game_id not in market_odds:
                continue
            
            odds = market_odds[game_id]
            info = (game_info or {}).get(game_id, {})
            
            # Анализируем обе стороны
            home_vb = self._analyze_side(
                game_id=game_id,
                side='home',
                our_prob=pred.get('home_win_prob', 0.5),
                odds=odds.get('home_odds', 1.9),
                best_odds=odds.get('home_best_odds', odds.get('home_odds', 1.9)),
                best_bookmaker=odds.get('home_best_bookmaker'),
                team_name=info.get('home_team', 'Home'),
                away_team=info.get('away_team', 'Away'),
                model_agreement=pred.get('model_agreement', 1.0)
            )
            
            away_vb = self._analyze_side(
                game_id=game_id,
                side='away',
                our_prob=pred.get('away_win_prob', 0.5),
                odds=odds.get('away_odds', 1.9),
                best_odds=odds.get('away_best_odds', odds.get('away_odds', 1.9)),
                best_bookmaker=odds.get('away_best_bookmaker'),
                team_name=info.get('away_team', 'Away'),
                away_team=info.get('home_team', 'Home'),
                model_agreement=pred.get('model_agreement', 1.0)
            )
            
            # Выбираем лучший вариант (если оба value - берем с большим EV)
            if home_vb and away_vb:
                value_bets.append(home_vb if home_vb.expected_value > away_vb.expected_value else away_vb)
            elif home_vb:
                value_bets.append(home_vb)
            elif away_vb:
                value_bets.append(away_vb)
        
        # Сортируем по EV (лучшие первыми)
        value_bets.sort(key=lambda x: x.expected_value, reverse=True)
        
        return value_bets
    
    def _analyze_side(
        self,
        game_id: int,
        side: str,
        our_prob: float,
        odds: float,
        best_odds: float,
        best_bookmaker: Optional[str],
        team_name: str,
        away_team: str,
        model_agreement: float
    ) -> Optional[ValueBet]:
        """Анализирует одну сторону ставки"""
        
        # Implied probability букмекера
        market_prob = 1 / odds if odds > 1 else 1.0
        
        # Edge (наше преимущество)
        edge = our_prob - market_prob
        
        # Expected Value
        # EV = (prob * win) - ((1-prob) * loss)
        # EV = (prob * (odds-1)) - (1-prob)
        ev = (our_prob * (odds - 1)) - (1 - our_prob)
        
        # Проверяем пороги
        if ev < self.min_value:
            return None
        
        if our_prob < self.min_confidence:
            return None
        
        # Подозрительно высокий value - возможно ошибка в данных
        notes = []
        if ev > self.max_value_cap:
            notes.append(f"⚠️ Подозрительно высокий EV ({ev:.1%}), проверьте данные")
        
        # Определяем категорию
        if our_prob >= config.portfolio.safe_min_confidence:
            category = BetCategory.SAFE
        elif ev >= 0.10:  # 10%+ EV
            category = BetCategory.VALUE
        else:
            category = BetCategory.HIGH_RISK
        
        # Kelly fraction
        b = odds - 1
        kelly = ((b * our_prob) - (1 - our_prob)) / b if b > 0 else 0
        kelly = max(0, min(kelly, 0.25))  # Ограничиваем
        
        # Priority score (для ранжирования)
        # Учитываем EV, уверенность и согласие моделей
        priority = ev * our_prob * model_agreement
        
        return ValueBet(
            game_id=game_id,
            home_team=team_name if side == 'home' else away_team,
            away_team=away_team if side == 'home' else team_name,
            bet_on=side,
            team_name=team_name,
            model_probability=our_prob,
            market_probability=market_prob,
            odds=odds,
            best_odds=best_odds,
            best_bookmaker=best_bookmaker,
            expected_value=ev,
            edge=edge,
            category=category,
            priority_score=priority,
            kelly_fraction=kelly,
            model_agreement=model_agreement,
            notes=notes
        )


@dataclass
class DailyBettingPlan:
    """План ставок на день"""
    date: date
    
    # Бюджет
    total_bankroll: float
    daily_budget: float
    
    # Распределение
    safe_budget: float
    value_budget: float
    risk_budget: float
    reserve: float
    
    # Ставки
    planned_bets: List[Dict]
    
    # Метрики
    total_risk: float
    expected_profit: float
    expected_roi: float
    
    # Лимиты
    remaining_daily_bets: int
    stop_loss_triggered: bool = False
    take_profit_triggered: bool = False


class BettingPortfolioManager:
    """
    Управляет портфелем ставок как инвестиционным портфелем
    
    Распределяет бюджет по категориям и управляет рисками
    """
    
    def __init__(self, bankroll_manager: BankrollManager):
        self.bm = bankroll_manager
        self.value_finder = ValueBetFinder()
        
        # Текущий план
        self.current_plan: Optional[DailyBettingPlan] = None
        self.plans_history: List[DailyBettingPlan] = []
    
    def create_daily_plan(
        self,
        games_predictions: Dict[int, Dict],
        market_odds: Dict[int, Dict],
        game_info: Dict[int, Dict] = None
    ) -> DailyBettingPlan:
        """
        Создает план ставок на день
        
        Args:
            games_predictions: Предсказания по всем играм
            market_odds: Коэффициенты
            game_info: Информация об играх
        
        Returns:
            DailyBettingPlan с распределением ставок
        """
        today = date.today()
        bankroll = self.bm.bankroll
        
        # Дневной бюджет
        daily_budget = bankroll * config.bankroll.max_daily_risk
        
        # Распределение по категориям
        safe_budget = daily_budget * config.portfolio.safe_allocation
        value_budget = daily_budget * config.portfolio.value_allocation
        risk_budget = daily_budget * config.portfolio.high_risk_allocation
        reserve = daily_budget * config.portfolio.cash_reserve
        
        # Находим все value bets
        all_value_bets = self.value_finder.find_value_bets(
            games_predictions, market_odds, game_info
        )
        
        # Категоризируем
        safe_bets = [vb for vb in all_value_bets if vb.category == BetCategory.SAFE]
        value_bets = [vb for vb in all_value_bets if vb.category == BetCategory.VALUE]
        risky_bets = [vb for vb in all_value_bets if vb.category == BetCategory.HIGH_RISK]
        
        planned_bets = []
        
        # 1. Распределяем безопасные ставки
        remaining_safe = safe_budget
        for vb in safe_bets[:3]:  # Макс 3 безопасные
            bet_amount = self._calculate_bet_amount(vb, remaining_safe, BetCategory.SAFE)
            if bet_amount >= config.bankroll.min_bet:
                planned_bets.append(self._create_bet_entry(vb, bet_amount))
                remaining_safe -= bet_amount
        
        # 2. Распределяем value bets
        remaining_value = value_budget
        for vb in value_bets[:5]:  # Макс 5 value
            bet_amount = self._calculate_bet_amount(vb, remaining_value, BetCategory.VALUE)
            if bet_amount >= config.bankroll.min_bet:
                planned_bets.append(self._create_bet_entry(vb, bet_amount))
                remaining_value -= bet_amount
        
        # 3. Рискованные ставки (только с большим edge)
        remaining_risk = risk_budget
        for vb in risky_bets[:2]:  # Макс 2 рискованные
            if vb.edge >= config.portfolio.risk_min_edge:
                bet_amount = self._calculate_bet_amount(vb, remaining_risk, BetCategory.HIGH_RISK)
                if bet_amount >= config.bankroll.min_bet:
                    planned_bets.append(self._create_bet_entry(vb, bet_amount))
                    remaining_risk -= bet_amount
        
        # Рассчитываем метрики
        total_risk = sum(b['bet_amount'] for b in planned_bets)
        expected_profit = sum(
            b['bet_amount'] * (b['confidence'] * (b['odds'] - 1) - (1 - b['confidence']))
            for b in planned_bets
        )
        
        plan = DailyBettingPlan(
            date=today,
            total_bankroll=bankroll,
            daily_budget=daily_budget,
            safe_budget=safe_budget,
            value_budget=value_budget,
            risk_budget=risk_budget,
            reserve=reserve,
            planned_bets=planned_bets,
            total_risk=total_risk,
            expected_profit=expected_profit,
            expected_roi=(expected_profit / total_risk * 100) if total_risk > 0 else 0,
            remaining_daily_bets=config.bankroll.max_bets_per_day - len(planned_bets)
        )
        
        self.current_plan = plan
        self.plans_history.append(plan)
        
        return plan
    
    def _calculate_bet_amount(
        self,
        value_bet: ValueBet,
        available_budget: float,
        category: BetCategory
    ) -> float:
        """Рассчитывает размер ставки"""
        
        # Лимиты по категории
        max_pct = {
            BetCategory.SAFE: config.portfolio.safe_max_per_bet,
            BetCategory.VALUE: config.portfolio.value_max_per_bet,
            BetCategory.HIGH_RISK: config.portfolio.risk_max_per_bet
        }
        
        max_bet = self.bm.bankroll * max_pct.get(category, 0.05)
        
        # Kelly-based amount
        kelly_amount = self.bm.bankroll * value_bet.kelly_fraction * 0.5  # Half Kelly
        
        # Выбираем минимум из ограничений
        bet_amount = min(kelly_amount, max_bet, available_budget)
        
        return round(bet_amount, 2)
    
    def _create_bet_entry(self, value_bet: ValueBet, bet_amount: float) -> Dict:
        """Создает запись о ставке для плана"""
        return {
            'game_id': value_bet.game_id,
            'home_team': value_bet.home_team,
            'away_team': value_bet.away_team,
            'bet_on': value_bet.bet_on,
            'team_name': value_bet.team_name,
            'bet_amount': bet_amount,
            'odds': value_bet.odds,
            'best_odds': value_bet.best_odds,
            'best_bookmaker': value_bet.best_bookmaker,
            'confidence': value_bet.model_probability,
            'expected_value': value_bet.expected_value,
            'edge': value_bet.edge,
            'category': value_bet.category.value,
            'potential_win': bet_amount * (value_bet.odds - 1),
            'notes': value_bet.notes
        }
    
    def get_summary(self) -> str:
        """Возвращает текстовое резюме плана"""
        if not self.current_plan:
            return "Нет активного плана ставок"
        
        plan = self.current_plan
        
        lines = [
            f"🎯 ПЛАН СТАВОК НА {plan.date}",
            f"",
            f"💰 Банкролл: ${plan.total_bankroll:.2f}",
            f"📊 Дневной бюджет: ${plan.daily_budget:.2f}",
            f"",
            f"Распределение:",
            f"  • Безопасные: ${plan.safe_budget:.2f}",
            f"  • Value: ${plan.value_budget:.2f}",
            f"  • Рискованные: ${plan.risk_budget:.2f}",
            f"  • Резерв: ${plan.reserve:.2f}",
            f"",
            f"📋 Запланировано ставок: {len(plan.planned_bets)}",
            f"💵 Общий риск: ${plan.total_risk:.2f}",
            f"📈 Ожидаемая прибыль: ${plan.expected_profit:.2f}",
            f"📊 Ожидаемый ROI: {plan.expected_roi:.1f}%",
            f"",
            "СТАВКИ:"
        ]
        
        for i, bet in enumerate(plan.planned_bets, 1):
            category_emoji = {
                'safe': '🟢',
                'value': '🟡',
                'high_risk': '🔴'
            }.get(bet['category'], '⚪')
            
            lines.append(
                f"{i}. {category_emoji} {bet['team_name']} "
                f"(vs {bet['away_team'] if bet['bet_on'] == 'home' else bet['home_team']})"
            )
            lines.append(f"   Ставка: ${bet['bet_amount']:.2f} @ {bet['odds']:.2f}")
            lines.append(f"   Уверенность: {bet['confidence']:.0%}, EV: {bet['expected_value']:.1%}")
            lines.append(f"   Потенциальный выигрыш: ${bet['potential_win']:.2f}")
            lines.append("")
        
        return "\n".join(lines)


# === ТЕСТИРОВАНИЕ ===

if __name__ == "__main__":
    print("=== Тест Value Bet Finder & Portfolio Manager ===\n")
    
    # Создаем тестовые данные
    predictions = {
        1001: {'home_win_prob': 0.72, 'away_win_prob': 0.28, 'model_agreement': 0.95},
        1002: {'home_win_prob': 0.58, 'away_win_prob': 0.42, 'model_agreement': 0.85},
        1003: {'home_win_prob': 0.45, 'away_win_prob': 0.55, 'model_agreement': 0.90},
        1004: {'home_win_prob': 0.65, 'away_win_prob': 0.35, 'model_agreement': 0.88},
        1005: {'home_win_prob': 0.51, 'away_win_prob': 0.49, 'model_agreement': 0.75},
    }
    
    market_odds = {
        1001: {'home_odds': 1.55, 'away_odds': 2.45},  # Good value on home
        1002: {'home_odds': 1.90, 'away_odds': 1.95},  # Some value
        1003: {'home_odds': 2.20, 'away_odds': 1.70},  # Value on away
        1004: {'home_odds': 1.65, 'away_odds': 2.25},  # Good value on home
        1005: {'home_odds': 1.95, 'away_odds': 1.90},  # No value
    }
    
    game_info = {
        1001: {'home_team': 'Lakers', 'away_team': 'Warriors'},
        1002: {'home_team': 'Celtics', 'away_team': 'Heat'},
        1003: {'home_team': 'Nuggets', 'away_team': 'Suns'},
        1004: {'home_team': 'Bucks', 'away_team': '76ers'},
        1005: {'home_team': 'Clippers', 'away_team': 'Kings'},
    }
    
    # Тест ValueBetFinder
    print("Тест 1: Value Bet Finder")
    print("-" * 50)
    
    finder = ValueBetFinder(min_value_threshold=0.03, min_confidence=0.52)
    value_bets = finder.find_value_bets(predictions, market_odds, game_info)
    
    print(f"Найдено value bets: {len(value_bets)}\n")
    
    for vb in value_bets:
        print(f"{vb.team_name} ({vb.category.value})")
        print(f"  Наша вероятность: {vb.model_probability:.0%}")
        print(f"  Market probability: {vb.market_probability:.0%}")
        print(f"  Edge: {vb.edge:.1%}")
        print(f"  Expected Value: {vb.expected_value:.1%}")
        print(f"  Kelly fraction: {vb.kelly_fraction:.2%}")
        print()
    
    # Тест Portfolio Manager
    print("\nТест 2: Portfolio Manager")
    print("-" * 50)
    
    bm = BankrollManager(initial_bankroll=200.00)
    portfolio = BettingPortfolioManager(bm)
    
    plan = portfolio.create_daily_plan(predictions, market_odds, game_info)
    
    print(portfolio.get_summary())
