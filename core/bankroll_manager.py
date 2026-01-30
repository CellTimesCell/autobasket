"""
AutoBasket - Bankroll Manager
=============================
Управление банкроллом с прогрессивной системой ставок

Логика прогрессии:
- $200 → $10 ставка (5%)
- $500 → $20 ставка (4%)
- $1000 → $35 ставка (3.5%)
- $2000 → $60 ставка (3%)
- $3000+ → $100 ставка (3.3%)

Принципы:
- Чем больше банкролл, тем меньше процент риска
- Но абсолютный размер ставок растёт
- Защита от разорения через лимиты
"""

import math
from datetime import datetime, date
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

import sys
sys.path.append('..')
from config.settings import config, BetCategory


# Прогрессивная шкала ставок
BETTING_TIERS = [
    # (min_bankroll, max_bankroll, percent, min_bet, max_bet)
    (0, 200, 0.05, 5, 10),        # $0-200: 5%, $5-10
    (200, 400, 0.045, 10, 18),    # $200-400: 4.5%, $10-18
    (400, 700, 0.04, 15, 28),     # $400-700: 4%, $15-28
    (700, 1000, 0.035, 25, 35),   # $700-1000: 3.5%, $25-35
    (1000, 1500, 0.03, 30, 45),   # $1000-1500: 3%, $30-45
    (1500, 2000, 0.028, 40, 56),  # $1500-2000: 2.8%, $40-56
    (2000, 3000, 0.025, 50, 75),  # $2000-3000: 2.5%, $50-75
    (3000, 5000, 0.025, 75, 125), # $3000-5000: 2.5%, $75-125
    (5000, 10000, 0.02, 100, 200),# $5000-10000: 2%, $100-200
    (10000, float('inf'), 0.02, 150, 300),  # $10000+: 2%, $150-300
]


@dataclass
class BetRecord:
    """Запись о ставке"""
    id: int
    game_id: int
    timestamp: datetime
    team_bet_on: str  # 'home' или 'away'
    bet_amount: float
    odds: float
    potential_win: float
    confidence: float
    expected_value: float
    category: BetCategory
    status: str = 'active'  # 'active', 'won', 'lost', 'cashed_out'
    result_amount: float = 0.0
    settled_at: Optional[datetime] = None


class BankrollManager:
    """
    Управляет банкроллом с прогрессивной системой ставок
    
    Особенности:
    - Прогрессивный размер ставок (растёт с банкроллом)
    - Планирование дневного бюджета
    - Распределение между играми
    - Защитные механизмы (stop-loss, лимиты)
    """
    
    def __init__(self, initial_bankroll: float = None):
        self.initial_bankroll = initial_bankroll or config.bankroll.initial_bankroll
        self.bankroll = self.initial_bankroll
        self.peak_bankroll = self.initial_bankroll
        self.lowest_bankroll = self.initial_bankroll
        
        # Дневная статистика
        self.day_start_balance = self.initial_bankroll
        self.today_bets_count = 0
        self.today_risked = 0.0
        self.today_budget = 0.0  # Бюджет на сегодня
        self.today_remaining = 0.0  # Оставшийся бюджет
        self.current_date = date.today()
        
        # История
        self.active_bets: List[BetRecord] = []
        self.bet_history: List[BetRecord] = []
        self.bankroll_history: List[Dict] = []
        
        # Счетчик ID
        self._next_bet_id = 1
        
        # Записываем начальное состояние
        self._record_bankroll_snapshot("initial")
        
        # Рассчитываем дневной бюджет
        self._calculate_daily_budget()
    
    def _calculate_daily_budget(self):
        """Рассчитывает бюджет на день"""
        # Дневной бюджет = 15-20% от банкролла (распределяется между играми)
        daily_risk_pct = 0.15  # 15% банкролла можно рискнуть за день
        self.today_budget = self.bankroll * daily_risk_pct
        self.today_remaining = self.today_budget - self.today_risked
    
    def _check_new_day(self):
        """Проверяет, начался ли новый день"""
        today = date.today()
        if today != self.current_date:
            self.current_date = today
            self.day_start_balance = self.bankroll
            self.today_bets_count = 0
            self.today_risked = 0.0
            self._calculate_daily_budget()
            self._record_bankroll_snapshot("day_start")
    
    def reset_daily(self):
        """Сбрасывает дневную статистику"""
        self.current_date = date.today()
        self.day_start_balance = self.bankroll
        self.today_bets_count = 0
        self.today_risked = 0.0
        self._calculate_daily_budget()
        self._record_bankroll_snapshot("day_start")
    
    # =========================================================================
    # ПРОГРЕССИВНАЯ СИСТЕМА СТАВОК
    # =========================================================================
    
    def get_current_tier(self) -> Tuple[float, float, float, float]:
        """
        Возвращает текущий тир ставок на основе банкролла
        
        Returns:
            (percent, min_bet, max_bet, tier_name)
        """
        for min_br, max_br, pct, min_bet, max_bet in BETTING_TIERS:
            if min_br <= self.bankroll < max_br:
                return (pct, min_bet, max_bet, f"${min_br}-${max_br}")
        
        # Fallback для очень большого банкролла
        return (0.02, 150, 300, "$10000+")
    
    def get_base_bet_size(self) -> float:
        """
        Возвращает базовый размер ставки для текущего банкролла
        
        Это размер одной ставки БЕЗ учёта edge и confidence.
        """
        pct, min_bet, max_bet, tier = self.get_current_tier()
        
        # Базовый расчёт
        base_bet = self.bankroll * pct
        
        # Применяем лимиты тира
        base_bet = max(min_bet, min(base_bet, max_bet))
        
        return round(base_bet, 2)
    
    def get_bet_size_for_edge(self, edge: float, confidence: float) -> float:
        """
        Рассчитывает размер ставки с учётом edge и confidence
        
        Args:
            edge: Преимущество над букмекером (0.05 = 5% edge)
            confidence: Уверенность модели (0.0-1.0)
        
        Returns:
            Размер ставки в долларах
        """
        base_bet = self.get_base_bet_size()
        pct, min_bet, max_bet, tier = self.get_current_tier()
        
        # Множитель на основе edge
        # Edge 3% = 1.0x, Edge 5% = 1.25x, Edge 10% = 1.5x
        edge_multiplier = 1.0 + (edge - 0.03) * 5  # Каждый 1% edge = +5% к ставке
        edge_multiplier = max(0.75, min(edge_multiplier, 1.5))  # Ограничиваем 0.75x - 1.5x
        
        # Множитель на основе confidence
        # Confidence 55% = 1.0x, 60% = 1.1x, 70% = 1.2x
        conf_multiplier = 1.0 + (confidence - 0.55) * 1.5
        conf_multiplier = max(0.8, min(conf_multiplier, 1.3))
        
        # Финальный расчёт
        bet_size = base_bet * edge_multiplier * conf_multiplier
        
        # Применяем лимиты
        bet_size = max(min_bet, min(bet_size, max_bet))
        
        # Не больше оставшегося дневного бюджета
        bet_size = min(bet_size, self.today_remaining)
        
        return round(bet_size, 2)
    
    def plan_daily_bets(self, num_potential_bets: int) -> Dict:
        """
        Планирует распределение бюджета на день
        
        Args:
            num_potential_bets: Количество игр с потенциальным value
        
        Returns:
            План распределения бюджета
        """
        self._check_new_day()
        self._calculate_daily_budget()
        
        base_bet = self.get_base_bet_size()
        pct, min_bet, max_bet, tier = self.get_current_tier()
        
        # Сколько ставок можем сделать
        max_bets_by_budget = int(self.today_budget / min_bet)
        max_bets_by_risk = 6  # Максимум 6 ставок в день для диверсификации
        
        recommended_bets = min(num_potential_bets, max_bets_by_budget, max_bets_by_risk)
        
        # Распределяем бюджет
        if recommended_bets > 0:
            per_bet_budget = self.today_budget / recommended_bets
            per_bet_budget = max(min_bet, min(per_bet_budget, max_bet))
        else:
            per_bet_budget = base_bet
        
        return {
            'bankroll': self.bankroll,
            'tier': tier,
            'daily_budget': round(self.today_budget, 2),
            'remaining_budget': round(self.today_remaining, 2),
            'base_bet_size': base_bet,
            'potential_games': num_potential_bets,
            'recommended_bets': recommended_bets,
            'per_bet_budget': round(per_bet_budget, 2),
            'bets_made_today': self.today_bets_count,
            'already_risked': round(self.today_risked, 2)
        }
    
    def get_bankroll_status(self) -> Dict:
        """Возвращает полный статус банкролла"""
        pct, min_bet, max_bet, tier = self.get_current_tier()
        
        return {
            'current': round(self.bankroll, 2),
            'initial': self.initial_bankroll,
            'peak': round(self.peak_bankroll, 2),
            'lowest': round(self.lowest_bankroll, 2),
            'total_change': f"{self.get_total_change():+.1%}",
            'daily_change': f"{self.get_daily_change():+.1%}",
            'tier': tier,
            'bet_range': f"${min_bet}-${max_bet}",
            'base_bet': self.get_base_bet_size(),
            'daily_budget': round(self.today_budget, 2),
            'daily_remaining': round(self.today_remaining, 2),
            'active_bets': len(self.active_bets),
            'today_bets': self.today_bets_count
        }
        """Сбрасывает дневные счетчики (вызывать в начале нового дня)"""
        self.current_date = date.today()
        self.day_start_balance = self.bankroll
        self.today_bets_count = 0
        self.today_risked = 0.0
        self._record_bankroll_snapshot("day_start")
    
    def _record_bankroll_snapshot(self, event: str):
        """Записывает снимок банкролла"""
        self.bankroll_history.append({
            'timestamp': datetime.now(),
            'bankroll': self.bankroll,
            'event': event,
            'day_change': self.get_daily_change(),
            'total_change': self.get_total_change()
        })
    
    def get_daily_change(self) -> float:
        """Возвращает изменение банкролла за день в процентах"""
        if self.day_start_balance == 0:
            return 0.0
        return (self.bankroll - self.day_start_balance) / self.day_start_balance
    
    def get_total_change(self) -> float:
        """Возвращает общее изменение банкролла в процентах"""
        if self.initial_bankroll == 0:
            return 0.0
        return (self.bankroll - self.initial_bankroll) / self.initial_bankroll
    
    def calculate_kelly_fraction(self, confidence: float, odds: float) -> float:
        """
        Рассчитывает оптимальную долю по критерию Келли
        
        Формула: f* = (bp - q) / b
        где b = odds - 1, p = confidence, q = 1 - p
        
        Args:
            confidence: Вероятность выигрыша (0.0 - 1.0)
            odds: Десятичные коэффициенты (например, 1.85)
        
        Returns:
            Оптимальная доля банкролла для ставки
        """
        if odds <= 1:
            return 0.0
        
        b = odds - 1  # Чистый выигрыш на единицу ставки
        p = confidence
        q = 1 - p
        
        # Полный критерий Келли
        kelly = (b * p - q) / b if b > 0 else 0
        
        return kelly
    
    def calculate_optimal_bet_size(
        self,
        confidence: float,
        odds: float,
        category: BetCategory = BetCategory.VALUE,
        game_importance: float = 1.0,
        use_half_kelly: bool = True
    ) -> Tuple[float, Dict]:
        """
        Рассчитывает оптимальный размер ставки с учетом всех ограничений
        
        Args:
            confidence: Уверенность в прогнозе (0.0 - 1.0)
            odds: Коэффициенты букмекера
            category: Категория ставки (safe/value/high_risk)
            game_importance: Множитель важности (1.0 = обычная, 1.5 = плей-офф)
            use_half_kelly: Использовать половинный Келли (консервативнее)
        
        Returns:
            Tuple[bet_amount, details_dict]
        """
        self._check_new_day()
        
        details = {
            'confidence': confidence,
            'odds': odds,
            'category': category.value,
            'reason': None,
            'kelly_fraction': 0,
            'adjusted_fraction': 0,
            'raw_bet': 0,
            'final_bet': 0
        }
        
        # Проверка минимальной уверенности
        if confidence < config.prediction.min_confidence_to_bet:
            details['reason'] = f"Уверенность {confidence:.1%} ниже минимума {config.prediction.min_confidence_to_bet:.1%}"
            return 0, details
        
        # Проверка stop-loss
        if self.get_daily_change() <= config.bankroll.stop_loss_daily:
            details['reason'] = f"Достигнут дневной stop-loss ({config.bankroll.stop_loss_daily:.1%})"
            return 0, details
        
        if self.get_total_change() <= config.bankroll.stop_loss_total:
            details['reason'] = f"Достигнут общий stop-loss ({config.bankroll.stop_loss_total:.1%})"
            return 0, details
        
        # Проверка дневного лимита ставок
        if self.today_bets_count >= config.bankroll.max_bets_per_day:
            details['reason'] = f"Достигнут лимит ставок на день ({config.bankroll.max_bets_per_day})"
            return 0, details
        
        # Проверка дневного риска
        remaining_daily_risk = (config.bankroll.max_daily_risk * self.bankroll) - self.today_risked
        if remaining_daily_risk <= config.bankroll.min_bet:
            details['reason'] = "Исчерпан дневной лимит риска"
            return 0, details
        
        # Рассчитываем Келли
        kelly_fraction = self.calculate_kelly_fraction(confidence, odds)
        details['kelly_fraction'] = kelly_fraction
        
        if kelly_fraction <= 0:
            details['reason'] = "Отрицательный Kelly - ставка не имеет value"
            return 0, details
        
        # Половинный Келли (более консервативно)
        if use_half_kelly:
            kelly_fraction *= 0.5
        
        # Ограничиваем Келли сверху (не более 25% даже при большом edge)
        kelly_fraction = min(kelly_fraction, 0.25)
        
        # Учитываем важность игры
        adjusted_fraction = kelly_fraction * game_importance
        details['adjusted_fraction'] = adjusted_fraction
        
        # Ограничения по категории
        category_limits = {
            BetCategory.SAFE: config.portfolio.safe_max_per_bet,
            BetCategory.VALUE: config.portfolio.value_max_per_bet,
            BetCategory.HIGH_RISK: config.portfolio.risk_max_per_bet
        }
        max_for_category = category_limits.get(category, config.bankroll.max_bet_percentage)
        
        # Применяем все ограничения
        final_fraction = min(
            adjusted_fraction,
            max_for_category,
            config.bankroll.max_bet_percentage
        )
        
        # Рассчитываем сумму
        raw_bet = self.bankroll * final_fraction
        details['raw_bet'] = raw_bet
        
        # Проверка минимума
        if raw_bet < config.bankroll.min_bet:
            details['reason'] = f"Размер ставки ${raw_bet:.2f} ниже минимума ${config.bankroll.min_bet:.2f}"
            return 0, details
        
        # Ограничиваем оставшимся дневным лимитом
        final_bet = min(raw_bet, remaining_daily_risk)
        final_bet = round(final_bet, 2)
        
        details['final_bet'] = final_bet
        details['reason'] = "OK"
        
        return final_bet, details
    
    def calculate_expected_value(self, confidence: float, odds: float, bet_amount: float) -> float:
        """
        Рассчитывает ожидаемую ценность ставки
        
        EV = (win_prob * potential_profit) - (lose_prob * bet_amount)
        """
        potential_profit = bet_amount * (odds - 1)
        ev = (confidence * potential_profit) - ((1 - confidence) * bet_amount)
        return ev
    
    def place_bet(
        self,
        game_id: int,
        team: str,
        bet_amount: float,
        odds: float,
        confidence: float,
        category: BetCategory
    ) -> Optional[BetRecord]:
        """
        Размещает ставку
        
        Returns:
            BetRecord если ставка размещена, None если отклонена
        """
        self._check_new_day()
        
        # Проверяем, можем ли ставить
        if bet_amount > self.bankroll:
            return None
        
        # Создаем запись
        bet = BetRecord(
            id=self._next_bet_id,
            game_id=game_id,
            timestamp=datetime.now(),
            team_bet_on=team,
            bet_amount=bet_amount,
            odds=odds,
            potential_win=bet_amount * (odds - 1),
            confidence=confidence,
            expected_value=self.calculate_expected_value(confidence, odds, bet_amount),
            category=category
        )
        
        self._next_bet_id += 1
        
        # Обновляем счетчики
        self.active_bets.append(bet)
        self.today_bets_count += 1
        self.today_risked += bet_amount
        
        self._record_bankroll_snapshot(f"bet_placed_{bet.id}")
        
        return bet
    
    def settle_bet(self, bet_id: int, won: bool) -> Optional[BetRecord]:
        """
        Закрывает ставку с результатом
        
        Args:
            bet_id: ID ставки
            won: True если выиграла, False если проиграла
        
        Returns:
            Обновленная запись ставки
        """
        # Находим ставку
        bet = None
        for i, b in enumerate(self.active_bets):
            if b.id == bet_id:
                bet = self.active_bets.pop(i)
                break
        
        if not bet:
            return None
        
        # Обновляем результат
        bet.settled_at = datetime.now()
        
        if won:
            bet.status = 'won'
            bet.result_amount = bet.potential_win
            self.bankroll += bet.potential_win
        else:
            bet.status = 'lost'
            bet.result_amount = -bet.bet_amount
            self.bankroll -= bet.bet_amount
        
        # Обновляем пик/дно
        self.peak_bankroll = max(self.peak_bankroll, self.bankroll)
        self.lowest_bankroll = min(self.lowest_bankroll, self.bankroll)
        
        # Сохраняем в историю
        self.bet_history.append(bet)
        self._record_bankroll_snapshot(f"bet_settled_{bet.id}_{bet.status}")
        
        return bet
    
    def get_statistics(self) -> Dict:
        """Возвращает статистику по ставкам"""
        if not self.bet_history:
            return {
                'total_bets': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'total_wagered': 0,
                'total_profit': 0,
                'roi': 0,
                'current_bankroll': self.bankroll,
                'peak_bankroll': self.peak_bankroll,
                'max_drawdown': 0,
                'active_bets': len(self.active_bets),
                'total_at_risk': sum(b.bet_amount for b in self.active_bets)
            }
        
        total_bets = len(self.bet_history)
        wins = sum(1 for b in self.bet_history if b.status == 'won')
        total_wagered = sum(b.bet_amount for b in self.bet_history)
        total_profit = sum(b.result_amount for b in self.bet_history)
        
        return {
            'total_bets': total_bets,
            'wins': wins,
            'losses': total_bets - wins,
            'win_rate': wins / total_bets if total_bets > 0 else 0,
            'total_wagered': total_wagered,
            'total_profit': total_profit,
            'roi': (total_profit / total_wagered * 100) if total_wagered > 0 else 0,
            'current_bankroll': self.bankroll,
            'peak_bankroll': self.peak_bankroll,
            'max_drawdown': (self.peak_bankroll - self.lowest_bankroll) / self.peak_bankroll if self.peak_bankroll > 0 else 0,
            'active_bets': len(self.active_bets),
            'total_at_risk': sum(b.bet_amount for b in self.active_bets)
        }
    
    def get_statistics_by_category(self) -> Dict[str, Dict]:
        """Статистика по категориям ставок"""
        stats = {}
        
        for category in BetCategory:
            category_bets = [b for b in self.bet_history if b.category == category]
            
            if not category_bets:
                stats[category.value] = {'bets': 0, 'win_rate': 0, 'roi': 0}
                continue
            
            wins = sum(1 for b in category_bets if b.status == 'won')
            wagered = sum(b.bet_amount for b in category_bets)
            profit = sum(b.result_amount for b in category_bets)
            
            stats[category.value] = {
                'bets': len(category_bets),
                'wins': wins,
                'win_rate': wins / len(category_bets),
                'total_wagered': wagered,
                'profit': profit,
                'roi': (profit / wagered * 100) if wagered > 0 else 0
            }
        
        return stats
    
    def should_take_profit(self) -> bool:
        """Проверяет, нужно ли фиксировать прибыль"""
        return self.get_daily_change() >= config.bankroll.take_profit_daily
    
    def should_cash_out(self) -> bool:
        """Проверяет, нужно ли выводить часть банкролла"""
        return self.get_total_change() >= config.bankroll.cash_out_threshold
    
    def export_to_dict(self) -> Dict:
        """Экспортирует состояние в словарь для сохранения"""
        return {
            'initial_bankroll': self.initial_bankroll,
            'bankroll': self.bankroll,
            'peak_bankroll': self.peak_bankroll,
            'lowest_bankroll': self.lowest_bankroll,
            'day_start_balance': self.day_start_balance,
            'today_bets_count': self.today_bets_count,
            'today_risked': self.today_risked,
            'current_date': self.current_date.isoformat(),
            'next_bet_id': self._next_bet_id,
            'statistics': self.get_statistics()
        }


# === ТЕСТИРОВАНИЕ ===

if __name__ == "__main__":
    print("=== Тест BankrollManager ===\n")
    
    bm = BankrollManager(initial_bankroll=200.00)
    
    # Тест расчета размера ставки
    print("Тест 1: Расчет размера ставки")
    print("-" * 40)
    
    test_cases = [
        (0.65, 1.85, BetCategory.SAFE, "Высокая уверенность"),
        (0.55, 2.10, BetCategory.VALUE, "Средняя уверенность, хорошие odds"),
        (0.50, 1.90, BetCategory.VALUE, "Нет edge"),
        (0.75, 1.50, BetCategory.SAFE, "Очень уверен, низкие odds"),
    ]
    
    for conf, odds, cat, desc in test_cases:
        amount, details = bm.calculate_optimal_bet_size(conf, odds, cat)
        ev = bm.calculate_expected_value(conf, odds, amount) if amount > 0 else 0
        
        print(f"\n{desc}:")
        print(f"  Уверенность: {conf:.0%}, Odds: {odds}")
        print(f"  Kelly: {details['kelly_fraction']:.3f}")
        print(f"  Рекомендуемая ставка: ${amount:.2f}")
        print(f"  Ожидаемая прибыль: ${ev:.2f}")
        print(f"  Статус: {details['reason']}")
    
    # Тест размещения и закрытия ставок
    print("\n\nТест 2: Размещение ставок")
    print("-" * 40)
    
    # Размещаем ставку
    bet = bm.place_bet(
        game_id=1001,
        team='home',
        bet_amount=15.00,
        odds=1.85,
        confidence=0.65,
        category=BetCategory.SAFE
    )
    
    if bet:
        print(f"Ставка размещена: ID={bet.id}, ${bet.bet_amount} на {bet.team_bet_on}")
        print(f"Потенциальный выигрыш: ${bet.potential_win:.2f}")
        print(f"Активных ставок: {len(bm.active_bets)}")
    
    # Закрываем как выигрыш
    settled = bm.settle_bet(bet.id, won=True)
    if settled:
        print(f"\nСтавка закрыта: {settled.status}")
        print(f"Результат: ${settled.result_amount:.2f}")
        print(f"Новый банкролл: ${bm.bankroll:.2f}")
    
    # Статистика
    print("\n\nИтоговая статистика:")
    print("-" * 40)
    stats = bm.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
