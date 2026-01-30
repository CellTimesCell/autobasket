"""
AutoBasket - Backtesting Engine
===============================
Тестирование стратегий на исторических данных
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from abc import ABC, abstractmethod
import json

import sys
sys.path.append('..')
from config.settings import config, BetCategory


@dataclass
class BacktestBet:
    """Запись о ставке в бэктесте"""
    game_id: int
    game_date: date
    home_team: str
    away_team: str
    bet_on: str  # 'home' или 'away'
    bet_amount: float
    odds: float
    confidence: float
    category: str
    
    # Результат (заполняется после игры)
    actual_winner: Optional[str] = None
    result: Optional[str] = None  # 'win', 'loss', 'push'
    profit: float = 0.0
    bankroll_after: float = 0.0


@dataclass 
class BacktestResult:
    """Результаты бэктеста"""
    # Основные метрики
    total_bets: int
    wins: int
    losses: int
    pushes: int
    win_rate: float
    
    # Финансовые метрики
    initial_bankroll: float
    final_bankroll: float
    total_wagered: float
    total_profit: float
    roi: float  # Return on Investment %
    
    # Риск метрики
    max_drawdown: float
    max_drawdown_duration: int  # В днях
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Серии
    longest_winning_streak: int
    longest_losing_streak: int
    
    # По категориям
    results_by_category: Dict[str, Dict]
    
    # Временные ряды для графиков
    bankroll_history: List[Dict]
    daily_pnl: List[Dict]
    
    # Дополнительно
    bets: List[BacktestBet]
    period_start: date
    period_end: date
    strategy_name: str


class BettingStrategy(ABC):
    """Базовый класс для стратегий ставок"""
    
    @abstractmethod
    def should_bet(
        self, 
        game_data: Dict,
        prediction: Dict,
        odds: Dict,
        bankroll: float
    ) -> Optional[Dict]:
        """
        Решает, делать ли ставку
        
        Returns:
            None если не ставить, иначе Dict с деталями ставки
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Название стратегии"""
        pass


class KellyStrategy(BettingStrategy):
    """Стратегия на основе критерия Келли"""
    
    def __init__(
        self,
        fraction: float = 0.5,  # Половинный Келли по умолчанию
        min_confidence: float = 0.52,
        min_value: float = 0.05,
        max_bet_pct: float = 0.10
    ):
        self.fraction = fraction
        self.min_confidence = min_confidence
        self.min_value = min_value
        self.max_bet_pct = max_bet_pct
    
    @property
    def name(self) -> str:
        return f"Kelly_{self.fraction:.0%}"
    
    def should_bet(
        self,
        game_data: Dict,
        prediction: Dict,
        odds: Dict,
        bankroll: float
    ) -> Optional[Dict]:
        
        home_prob = prediction.get('home_win_prob', 0.5)
        away_prob = prediction.get('away_win_prob', 0.5)
        home_odds = odds.get('home_odds', 1.9)
        away_odds = odds.get('away_odds', 1.9)
        
        # Рассчитываем value для обеих сторон
        home_ev = (home_prob * (home_odds - 1)) - (1 - home_prob)
        away_ev = (away_prob * (away_odds - 1)) - (1 - away_prob)
        
        # Выбираем лучший вариант
        if home_ev > away_ev and home_ev > self.min_value:
            bet_on = 'home'
            confidence = home_prob
            bet_odds = home_odds
            ev = home_ev
        elif away_ev > self.min_value:
            bet_on = 'away'
            confidence = away_prob
            bet_odds = away_odds
            ev = away_ev
        else:
            return None
        
        # Проверка минимальной уверенности
        if confidence < self.min_confidence:
            return None
        
        # Kelly criterion
        b = bet_odds - 1
        p = confidence
        q = 1 - p
        kelly = (b * p - q) / b if b > 0 else 0
        
        if kelly <= 0:
            return None
        
        # Применяем фракцию Келли
        kelly *= self.fraction
        
        # Ограничиваем размер ставки
        bet_fraction = min(kelly, self.max_bet_pct)
        bet_amount = bankroll * bet_fraction
        
        # Минимальная ставка
        if bet_amount < 5:
            return None
        
        # Определяем категорию
        if confidence >= 0.70:
            category = 'safe'
        elif ev >= 0.10:
            category = 'value'
        else:
            category = 'high_risk'
        
        return {
            'bet_on': bet_on,
            'bet_amount': round(bet_amount, 2),
            'odds': bet_odds,
            'confidence': confidence,
            'expected_value': ev,
            'kelly_fraction': kelly,
            'category': category
        }


class FlatBettingStrategy(BettingStrategy):
    """Фиксированная ставка"""
    
    def __init__(
        self,
        bet_percentage: float = 0.02,  # 2% от банкролла
        min_confidence: float = 0.55,
        min_value: float = 0.03
    ):
        self.bet_percentage = bet_percentage
        self.min_confidence = min_confidence
        self.min_value = min_value
    
    @property
    def name(self) -> str:
        return f"Flat_{self.bet_percentage:.0%}"
    
    def should_bet(
        self,
        game_data: Dict,
        prediction: Dict,
        odds: Dict,
        bankroll: float
    ) -> Optional[Dict]:
        
        home_prob = prediction.get('home_win_prob', 0.5)
        away_prob = prediction.get('away_win_prob', 0.5)
        home_odds = odds.get('home_odds', 1.9)
        away_odds = odds.get('away_odds', 1.9)
        
        # Value calculation
        home_ev = (home_prob * (home_odds - 1)) - (1 - home_prob)
        away_ev = (away_prob * (away_odds - 1)) - (1 - away_prob)
        
        # Выбираем лучший вариант
        if home_ev > away_ev and home_ev > self.min_value and home_prob >= self.min_confidence:
            return {
                'bet_on': 'home',
                'bet_amount': round(bankroll * self.bet_percentage, 2),
                'odds': home_odds,
                'confidence': home_prob,
                'expected_value': home_ev,
                'category': 'flat'
            }
        elif away_ev > self.min_value and away_prob >= self.min_confidence:
            return {
                'bet_on': 'away',
                'bet_amount': round(bankroll * self.bet_percentage, 2),
                'odds': away_odds,
                'confidence': away_prob,
                'expected_value': away_ev,
                'category': 'flat'
            }
        
        return None


class StrategyBacktester:
    """
    Движок бэктестинга
    
    Прогоняет стратегию через исторические данные
    и собирает подробную статистику
    """
    
    def __init__(
        self,
        strategy: BettingStrategy,
        initial_bankroll: float = 200.0
    ):
        self.strategy = strategy
        self.initial_bankroll = initial_bankroll
    
    def run(
        self,
        historical_data: pd.DataFrame,
        predictions: Dict[int, Dict],
        odds_data: Dict[int, Dict],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> BacktestResult:
        """
        Запускает бэктест
        
        Args:
            historical_data: DataFrame с историческими играми
                Ожидаемые колонки: game_id, game_date, home_team, away_team, 
                                   home_score, away_score, home_won
            predictions: Dict {game_id: {'home_win_prob': X, 'away_win_prob': Y}}
            odds_data: Dict {game_id: {'home_odds': X, 'away_odds': Y}}
            start_date: Начало периода (опционально)
            end_date: Конец периода (опционально)
        
        Returns:
            BacktestResult с подробной статистикой
        """
        # Фильтруем данные по датам
        df = historical_data.copy()
        df['game_date'] = pd.to_datetime(df['game_date']).dt.date
        
        if start_date:
            df = df[df['game_date'] >= start_date]
        if end_date:
            df = df[df['game_date'] <= end_date]
        
        df = df.sort_values('game_date')
        
        # Инициализация
        bankroll = self.initial_bankroll
        bets: List[BacktestBet] = []
        bankroll_history = [{'date': df['game_date'].min(), 'bankroll': bankroll}]
        daily_results = {}
        
        # Проходим по каждой игре
        for _, game in df.iterrows():
            game_id = game['game_id']
            game_date = game['game_date']
            
            # Получаем предсказание и коэффициенты
            prediction = predictions.get(game_id, {
                'home_win_prob': 0.5,
                'away_win_prob': 0.5
            })
            odds = odds_data.get(game_id, {
                'home_odds': 1.90,
                'away_odds': 1.90
            })
            
            # Решаем, делать ли ставку
            game_data = {
                'game_id': game_id,
                'home_team': game['home_team'],
                'away_team': game['away_team']
            }
            
            bet_decision = self.strategy.should_bet(
                game_data, prediction, odds, bankroll
            )
            
            if bet_decision and bet_decision['bet_amount'] <= bankroll:
                # Создаем ставку
                bet = BacktestBet(
                    game_id=game_id,
                    game_date=game_date,
                    home_team=game['home_team'],
                    away_team=game['away_team'],
                    bet_on=bet_decision['bet_on'],
                    bet_amount=bet_decision['bet_amount'],
                    odds=bet_decision['odds'],
                    confidence=bet_decision['confidence'],
                    category=bet_decision['category']
                )
                
                # Определяем результат
                home_won = game['home_won']
                bet.actual_winner = 'home' if home_won else 'away'
                
                if bet.bet_on == bet.actual_winner:
                    bet.result = 'win'
                    bet.profit = bet.bet_amount * (bet.odds - 1)
                    bankroll += bet.profit
                else:
                    bet.result = 'loss'
                    bet.profit = -bet.bet_amount
                    bankroll -= bet.bet_amount
                
                bet.bankroll_after = bankroll
                bets.append(bet)
                
                # Записываем для daily P&L
                if game_date not in daily_results:
                    daily_results[game_date] = {'profit': 0, 'bets': 0}
                daily_results[game_date]['profit'] += bet.profit
                daily_results[game_date]['bets'] += 1
            
            # Обновляем историю банкролла
            bankroll_history.append({
                'date': game_date,
                'bankroll': bankroll
            })
            
            # Проверка банкротства
            if bankroll < 5:
                print(f"⚠️ Банкротство на {game_date}!")
                break
        
        # Анализируем результаты
        return self._analyze_results(
            bets=bets,
            bankroll_history=bankroll_history,
            daily_results=daily_results,
            start_date=df['game_date'].min(),
            end_date=df['game_date'].max()
        )
    
    def _analyze_results(
        self,
        bets: List[BacktestBet],
        bankroll_history: List[Dict],
        daily_results: Dict,
        start_date: date,
        end_date: date
    ) -> BacktestResult:
        """Анализирует результаты бэктеста"""
        
        if not bets:
            return BacktestResult(
                total_bets=0,
                wins=0,
                losses=0,
                pushes=0,
                win_rate=0,
                initial_bankroll=self.initial_bankroll,
                final_bankroll=self.initial_bankroll,
                total_wagered=0,
                total_profit=0,
                roi=0,
                max_drawdown=0,
                max_drawdown_duration=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                calmar_ratio=0,
                longest_winning_streak=0,
                longest_losing_streak=0,
                results_by_category={},
                bankroll_history=bankroll_history,
                daily_pnl=[],
                bets=bets,
                period_start=start_date,
                period_end=end_date,
                strategy_name=self.strategy.name
            )
        
        # Базовые метрики
        total_bets = len(bets)
        wins = sum(1 for b in bets if b.result == 'win')
        losses = sum(1 for b in bets if b.result == 'loss')
        pushes = sum(1 for b in bets if b.result == 'push')
        
        total_wagered = sum(b.bet_amount for b in bets)
        total_profit = sum(b.profit for b in bets)
        
        final_bankroll = bets[-1].bankroll_after if bets else self.initial_bankroll
        
        # Риск метрики
        bankroll_values = [h['bankroll'] for h in bankroll_history]
        max_dd, max_dd_duration = self._calculate_max_drawdown(bankroll_values)
        
        daily_returns = [d['profit'] for d in daily_results.values()]
        sharpe = self._calculate_sharpe(daily_returns)
        sortino = self._calculate_sortino(daily_returns)
        
        # Calmar ratio = Annual Return / Max Drawdown
        days = (end_date - start_date).days or 1
        annual_return = (total_profit / self.initial_bankroll) * (365 / days)
        calmar = annual_return / max_dd if max_dd > 0 else 0
        
        # Серии
        win_streak, lose_streak = self._calculate_streaks(bets)
        
        # По категориям
        by_category = self._analyze_by_category(bets)
        
        # Daily P&L
        daily_pnl = [
            {'date': str(d), 'profit': r['profit'], 'bets': r['bets']}
            for d, r in sorted(daily_results.items())
        ]
        
        return BacktestResult(
            total_bets=total_bets,
            wins=wins,
            losses=losses,
            pushes=pushes,
            win_rate=wins / total_bets if total_bets > 0 else 0,
            initial_bankroll=self.initial_bankroll,
            final_bankroll=final_bankroll,
            total_wagered=total_wagered,
            total_profit=total_profit,
            roi=(total_profit / total_wagered * 100) if total_wagered > 0 else 0,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            longest_winning_streak=win_streak,
            longest_losing_streak=lose_streak,
            results_by_category=by_category,
            bankroll_history=bankroll_history,
            daily_pnl=daily_pnl,
            bets=bets,
            period_start=start_date,
            period_end=end_date,
            strategy_name=self.strategy.name
        )
    
    def _calculate_max_drawdown(
        self, 
        bankroll_values: List[float]
    ) -> Tuple[float, int]:
        """Рассчитывает максимальную просадку"""
        if not bankroll_values:
            return 0, 0
        
        peak = bankroll_values[0]
        max_dd = 0
        max_dd_duration = 0
        current_dd_start = 0
        
        for i, value in enumerate(bankroll_values):
            if value > peak:
                peak = value
                current_dd_start = i
            
            dd = (peak - value) / peak if peak > 0 else 0
            
            if dd > max_dd:
                max_dd = dd
                max_dd_duration = i - current_dd_start
        
        return max_dd, max_dd_duration
    
    def _calculate_sharpe(self, returns: List[float], risk_free: float = 0) -> float:
        """Sharpe Ratio"""
        if len(returns) < 2:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        # Annualized (предполагаем ~250 торговых дней)
        return (mean_return - risk_free) / std_return * np.sqrt(250)
    
    def _calculate_sortino(self, returns: List[float], risk_free: float = 0) -> float:
        """Sortino Ratio (только downside deviation)"""
        if len(returns) < 2:
            return 0
        
        mean_return = np.mean(returns)
        downside_returns = [r for r in returns if r < risk_free]
        
        if not downside_returns:
            return float('inf')
        
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return 0
        
        return (mean_return - risk_free) / downside_std * np.sqrt(250)
    
    def _calculate_streaks(self, bets: List[BacktestBet]) -> Tuple[int, int]:
        """Находит самые длинные серии побед/поражений"""
        if not bets:
            return 0, 0
        
        max_win = 0
        max_lose = 0
        current_win = 0
        current_lose = 0
        
        for bet in bets:
            if bet.result == 'win':
                current_win += 1
                current_lose = 0
                max_win = max(max_win, current_win)
            elif bet.result == 'loss':
                current_lose += 1
                current_win = 0
                max_lose = max(max_lose, current_lose)
            else:
                current_win = 0
                current_lose = 0
        
        return max_win, max_lose
    
    def _analyze_by_category(self, bets: List[BacktestBet]) -> Dict:
        """Анализ по категориям ставок"""
        categories = {}
        
        for bet in bets:
            cat = bet.category
            if cat not in categories:
                categories[cat] = {
                    'bets': 0,
                    'wins': 0,
                    'wagered': 0,
                    'profit': 0
                }
            
            categories[cat]['bets'] += 1
            categories[cat]['wins'] += 1 if bet.result == 'win' else 0
            categories[cat]['wagered'] += bet.bet_amount
            categories[cat]['profit'] += bet.profit
        
        # Добавляем рассчитанные метрики
        for cat in categories:
            c = categories[cat]
            c['win_rate'] = c['wins'] / c['bets'] if c['bets'] > 0 else 0
            c['roi'] = (c['profit'] / c['wagered'] * 100) if c['wagered'] > 0 else 0
        
        return categories


class MonteCarloSimulator:
    """
    Монте-Карло симуляция для оценки рисков
    """
    
    def __init__(
        self,
        win_rate: float,
        avg_odds: float,
        avg_bet_fraction: float = 0.05
    ):
        self.win_rate = win_rate
        self.avg_odds = avg_odds
        self.avg_bet_fraction = avg_bet_fraction
    
    def simulate(
        self,
        initial_bankroll: float,
        num_bets: int,
        num_simulations: int = 10000
    ) -> Dict:
        """
        Симулирует множество возможных исходов
        
        Returns:
            Статистика по всем симуляциям
        """
        results = []
        
        for _ in range(num_simulations):
            bankroll = initial_bankroll
            
            for _ in range(num_bets):
                bet_amount = bankroll * self.avg_bet_fraction
                
                if np.random.random() < self.win_rate:
                    bankroll += bet_amount * (self.avg_odds - 1)
                else:
                    bankroll -= bet_amount
                
                # Банкротство
                if bankroll < 5:
                    break
            
            results.append(bankroll)
        
        results = np.array(results)
        
        return {
            # Центральные тенденции
            'median_outcome': np.median(results),
            'mean_outcome': np.mean(results),
            'std_outcome': np.std(results),
            
            # Вероятности
            'prob_profit': np.mean(results > initial_bankroll),
            'prob_double': np.mean(results > initial_bankroll * 2),
            'prob_bankrupt': np.mean(results < 10),
            'prob_50pct_loss': np.mean(results < initial_bankroll * 0.5),
            
            # Percentiles
            'percentile_5': np.percentile(results, 5),
            'percentile_25': np.percentile(results, 25),
            'percentile_75': np.percentile(results, 75),
            'percentile_95': np.percentile(results, 95),
            
            # Value at Risk
            'var_95': initial_bankroll - np.percentile(results, 5),
            'var_99': initial_bankroll - np.percentile(results, 1),
            
            # Метаданные
            'num_simulations': num_simulations,
            'num_bets': num_bets,
            'win_rate': self.win_rate,
            'avg_odds': self.avg_odds
        }


# === ТЕСТИРОВАНИЕ ===

if __name__ == "__main__":
    print("=== Тест Backtesting Engine ===\n")
    
    # Создаем синтетические исторические данные
    np.random.seed(42)
    
    num_games = 100
    dates = pd.date_range(start='2024-01-01', periods=num_games, freq='D')
    
    historical_data = pd.DataFrame({
        'game_id': range(1000, 1000 + num_games),
        'game_date': dates,
        'home_team': ['Lakers'] * 50 + ['Celtics'] * 50,
        'away_team': ['Warriors'] * 50 + ['Heat'] * 50,
        'home_score': np.random.randint(95, 125, num_games),
        'away_score': np.random.randint(95, 125, num_games),
    })
    historical_data['home_won'] = historical_data['home_score'] > historical_data['away_score']
    
    # Создаем предсказания (симулируем ~58% точность)
    predictions = {}
    odds_data = {}
    
    for _, row in historical_data.iterrows():
        game_id = row['game_id']
        actual_home_won = row['home_won']
        
        # Симулируем предсказания с некоторой точностью
        if actual_home_won:
            home_prob = np.clip(np.random.normal(0.62, 0.15), 0.3, 0.85)
        else:
            home_prob = np.clip(np.random.normal(0.42, 0.15), 0.15, 0.7)
        
        predictions[game_id] = {
            'home_win_prob': home_prob,
            'away_win_prob': 1 - home_prob
        }
        
        # Коэффициенты (с маржой букмекера)
        odds_data[game_id] = {
            'home_odds': round(0.95 / home_prob, 2) if home_prob > 0 else 1.90,
            'away_odds': round(0.95 / (1 - home_prob), 2) if home_prob < 1 else 1.90
        }
    
    # Тест Kelly стратегии
    print("Тест 1: Kelly Strategy (50%)")
    print("-" * 50)
    
    kelly_strategy = KellyStrategy(fraction=0.5, min_confidence=0.55, min_value=0.03)
    backtester = StrategyBacktester(kelly_strategy, initial_bankroll=200)
    
    result = backtester.run(historical_data, predictions, odds_data)
    
    print(f"Период: {result.period_start} - {result.period_end}")
    print(f"Всего ставок: {result.total_bets}")
    print(f"Win Rate: {result.win_rate:.1%}")
    print(f"ROI: {result.roi:.1f}%")
    print(f"Начальный банкролл: ${result.initial_bankroll:.2f}")
    print(f"Финальный банкролл: ${result.final_bankroll:.2f}")
    print(f"Профит: ${result.total_profit:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.1%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Longest Win Streak: {result.longest_winning_streak}")
    print(f"Longest Lose Streak: {result.longest_losing_streak}")
    
    # Тест Flat стратегии
    print("\n\nТест 2: Flat Betting Strategy (2%)")
    print("-" * 50)
    
    flat_strategy = FlatBettingStrategy(bet_percentage=0.02, min_confidence=0.55)
    backtester2 = StrategyBacktester(flat_strategy, initial_bankroll=200)
    
    result2 = backtester2.run(historical_data, predictions, odds_data)
    
    print(f"Всего ставок: {result2.total_bets}")
    print(f"Win Rate: {result2.win_rate:.1%}")
    print(f"ROI: {result2.roi:.1f}%")
    print(f"Финальный банкролл: ${result2.final_bankroll:.2f}")
    print(f"Max Drawdown: {result2.max_drawdown:.1%}")
    
    # Monte Carlo симуляция
    print("\n\nТест 3: Monte Carlo Simulation")
    print("-" * 50)
    
    mc = MonteCarloSimulator(
        win_rate=result.win_rate,
        avg_odds=1.85,
        avg_bet_fraction=0.05
    )
    
    mc_result = mc.simulate(
        initial_bankroll=200,
        num_bets=200,
        num_simulations=10000
    )
    
    print(f"Win Rate используется: {mc_result['win_rate']:.1%}")
    print(f"Симуляций: {mc_result['num_simulations']}")
    print(f"Ставок в симуляции: {mc_result['num_bets']}")
    print(f"\nРезультаты:")
    print(f"  Медианный исход: ${mc_result['median_outcome']:.2f}")
    print(f"  Средний исход: ${mc_result['mean_outcome']:.2f}")
    print(f"  P(Profit): {mc_result['prob_profit']:.1%}")
    print(f"  P(Double): {mc_result['prob_double']:.1%}")
    print(f"  P(Bankrupt): {mc_result['prob_bankrupt']:.1%}")
    print(f"  95% VaR: ${mc_result['var_95']:.2f}")
    print(f"  5th percentile: ${mc_result['percentile_5']:.2f}")
    print(f"  95th percentile: ${mc_result['percentile_95']:.2f}")
