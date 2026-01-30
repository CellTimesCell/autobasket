"""
AutoBasket - Discipline Manager
===============================
Контроль дисциплины и детекция тильта
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TiltSeverity(Enum):
    """Уровни серьезности тильта"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TiltIndicator:
    """Индикатор тильта"""
    name: str
    detected: bool
    severity: TiltSeverity
    description: str
    recommendation: str


@dataclass
class BettingSession:
    """Сессия ставок"""
    start_time: datetime
    end_time: Optional[datetime] = None
    bets_placed: int = 0
    bankroll_start: float = 0.0
    bankroll_current: float = 0.0
    wins: int = 0
    losses: int = 0
    
    # История ставок в сессии
    bet_amounts: List[float] = field(default_factory=list)
    bet_results: List[str] = field(default_factory=list)  # 'win', 'loss'
    bet_times: List[datetime] = field(default_factory=list)
    
    # Предупреждения
    tilt_warnings: List[TiltIndicator] = field(default_factory=list)
    is_locked: bool = False
    lock_until: Optional[datetime] = None
    
    @property
    def duration_minutes(self) -> float:
        """Длительность сессии в минутах"""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds() / 60
    
    @property
    def net_profit(self) -> float:
        """Чистая прибыль сессии"""
        return self.bankroll_current - self.bankroll_start
    
    @property
    def current_streak(self) -> Tuple[str, int]:
        """Текущая серия (тип, длина)"""
        if not self.bet_results:
            return ('none', 0)
        
        streak_type = self.bet_results[-1]
        count = 0
        for result in reversed(self.bet_results):
            if result == streak_type:
                count += 1
            else:
                break
        return (streak_type, count)


class DisciplineManager:
    """
    Управляет дисциплиной ставок и обнаруживает признаки тильта
    
    Индикаторы тильта:
    1. Серия проигрышей >= 3
    2. Увеличение ставок после проигрыша (chasing)
    3. Слишком частые ставки (overtrading)
    4. Длительная сессия без перерыва
    5. Большие потери за сессию
    6. Эмоциональные паттерны
    """
    
    def __init__(
        self,
        losing_streak_warning: int = 3,
        losing_streak_stop: int = 5,
        max_bets_per_hour: int = 5,
        min_minutes_between_bets: int = 10,
        max_session_hours: int = 4,
        chase_multiplier_threshold: float = 1.5,
        session_loss_warning: float = 0.10,
        session_loss_stop: float = 0.20
    ):
        # Пороги
        self.losing_streak_warning = losing_streak_warning
        self.losing_streak_stop = losing_streak_stop
        self.max_bets_per_hour = max_bets_per_hour
        self.min_minutes_between_bets = min_minutes_between_bets
        self.max_session_hours = max_session_hours
        self.chase_multiplier = chase_multiplier_threshold
        self.session_loss_warning = session_loss_warning
        self.session_loss_stop = session_loss_stop
        
        # Текущая сессия
        self.current_session: Optional[BettingSession] = None
        self.session_history: List[BettingSession] = []
        
        # Блокировка
        self.global_lock = False
        self.global_lock_until: Optional[datetime] = None
        self.lock_reasons: List[str] = []
    
    def start_session(self, bankroll: float) -> BettingSession:
        """Начинает новую сессию"""
        # Закрываем предыдущую если есть
        if self.current_session:
            self.end_session()
        
        self.current_session = BettingSession(
            start_time=datetime.now(),
            bankroll_start=bankroll,
            bankroll_current=bankroll
        )
        
        logger.info(f"Session started with bankroll ${bankroll:.2f}")
        return self.current_session
    
    def end_session(self):
        """Завершает текущую сессию"""
        if self.current_session:
            self.current_session.end_time = datetime.now()
            self.session_history.append(self.current_session)
            
            logger.info(
                f"Session ended: {self.current_session.bets_placed} bets, "
                f"P&L: ${self.current_session.net_profit:.2f}"
            )
            self.current_session = None
    
    def record_bet(
        self,
        bet_amount: float,
        bankroll_after: float
    ):
        """Записывает размещенную ставку"""
        if not self.current_session:
            self.start_session(bankroll_after + bet_amount)
        
        session = self.current_session
        session.bets_placed += 1
        session.bet_amounts.append(bet_amount)
        session.bet_times.append(datetime.now())
        session.bankroll_current = bankroll_after
    
    def record_result(self, won: bool):
        """Записывает результат ставки"""
        if not self.current_session:
            return
        
        result = 'win' if won else 'loss'
        self.current_session.bet_results.append(result)
        
        if won:
            self.current_session.wins += 1
        else:
            self.current_session.losses += 1
    
    def check_can_bet(self, proposed_amount: float = None) -> Tuple[bool, List[TiltIndicator]]:
        """
        Проверяет, можно ли делать ставку
        
        Returns:
            (can_bet, list_of_warnings)
        """
        indicators = []
        
        # Проверка глобальной блокировки
        if self.global_lock:
            if self.global_lock_until and datetime.now() < self.global_lock_until:
                indicators.append(TiltIndicator(
                    name="global_lock",
                    detected=True,
                    severity=TiltSeverity.CRITICAL,
                    description=f"Система заблокирована до {self.global_lock_until.strftime('%H:%M')}",
                    recommendation="Дождитесь окончания блокировки"
                ))
                return False, indicators
            else:
                # Блокировка истекла
                self.global_lock = False
                self.global_lock_until = None
        
        if not self.current_session:
            return True, []
        
        session = self.current_session
        
        # 1. Проверка серии проигрышей
        streak_type, streak_len = session.current_streak
        if streak_type == 'loss':
            if streak_len >= self.losing_streak_stop:
                indicators.append(TiltIndicator(
                    name="losing_streak_critical",
                    detected=True,
                    severity=TiltSeverity.CRITICAL,
                    description=f"Критическая серия проигрышей: {streak_len} подряд",
                    recommendation="НЕМЕДЛЕННО прекратите ставки. Сделайте перерыв минимум 2 часа."
                ))
            elif streak_len >= self.losing_streak_warning:
                indicators.append(TiltIndicator(
                    name="losing_streak_warning",
                    detected=True,
                    severity=TiltSeverity.HIGH,
                    description=f"Серия проигрышей: {streak_len} подряд",
                    recommendation="Уменьшите размер ставок на 50% или сделайте перерыв"
                ))
        
        # 2. Проверка chase (увеличение ставки после проигрыша)
        if proposed_amount and len(session.bet_amounts) >= 1 and len(session.bet_results) >= 1:
            last_amount = session.bet_amounts[-1]
            last_result = session.bet_results[-1] if session.bet_results else None
            
            if last_result == 'loss' and proposed_amount > last_amount * self.chase_multiplier:
                indicators.append(TiltIndicator(
                    name="chasing_losses",
                    detected=True,
                    severity=TiltSeverity.HIGH,
                    description=f"Попытка отыграться: ставка увеличена в {proposed_amount/last_amount:.1f}x после проигрыша",
                    recommendation="Не увеличивайте ставки после проигрыша. Это признак тильта."
                ))
        
        # 3. Проверка overtrading
        recent_bets = [t for t in session.bet_times if datetime.now() - t < timedelta(hours=1)]
        if len(recent_bets) >= self.max_bets_per_hour:
            indicators.append(TiltIndicator(
                name="overtrading",
                detected=True,
                severity=TiltSeverity.MEDIUM,
                description=f"Слишком много ставок: {len(recent_bets)} за последний час",
                recommendation="Замедлитесь. Качество важнее количества."
            ))
        
        # 4. Проверка времени между ставками
        if session.bet_times:
            time_since_last = (datetime.now() - session.bet_times[-1]).total_seconds() / 60
            if time_since_last < self.min_minutes_between_bets:
                indicators.append(TiltIndicator(
                    name="too_fast",
                    detected=True,
                    severity=TiltSeverity.LOW,
                    description=f"Слишком быстро: {time_since_last:.0f} мин с последней ставки",
                    recommendation=f"Подождите минимум {self.min_minutes_between_bets} минут между ставками"
                ))
        
        # 5. Проверка длительности сессии
        if session.duration_minutes > self.max_session_hours * 60:
            indicators.append(TiltIndicator(
                name="long_session",
                detected=True,
                severity=TiltSeverity.MEDIUM,
                description=f"Длинная сессия: {session.duration_minutes/60:.1f} часов",
                recommendation="Сделайте перерыв минимум 30 минут"
            ))
        
        # 6. Проверка потерь за сессию
        if session.bankroll_start > 0:
            session_loss = (session.bankroll_start - session.bankroll_current) / session.bankroll_start
            
            if session_loss >= self.session_loss_stop:
                indicators.append(TiltIndicator(
                    name="session_loss_critical",
                    detected=True,
                    severity=TiltSeverity.CRITICAL,
                    description=f"Критические потери за сессию: {session_loss:.1%}",
                    recommendation="СТОП. Завершите сессию и вернитесь завтра."
                ))
            elif session_loss >= self.session_loss_warning:
                indicators.append(TiltIndicator(
                    name="session_loss_warning",
                    detected=True,
                    severity=TiltSeverity.HIGH,
                    description=f"Значительные потери за сессию: {session_loss:.1%}",
                    recommendation="Сократите размер ставок или сделайте перерыв"
                ))
        
        # Сохраняем предупреждения в сессии
        session.tilt_warnings.extend(indicators)
        
        # Определяем, можно ли ставить
        critical = any(i.severity == TiltSeverity.CRITICAL for i in indicators)
        
        return not critical, indicators
    
    def lock_betting(self, duration_hours: float, reason: str):
        """Блокирует ставки на указанное время"""
        self.global_lock = True
        self.global_lock_until = datetime.now() + timedelta(hours=duration_hours)
        self.lock_reasons.append(f"{datetime.now()}: {reason}")
        
        if self.current_session:
            self.current_session.is_locked = True
            self.current_session.lock_until = self.global_lock_until
        
        logger.warning(f"Betting locked for {duration_hours}h: {reason}")
    
    def get_recommended_bet_multiplier(self) -> float:
        """
        Возвращает рекомендуемый множитель размера ставки
        
        1.0 = нормальный размер
        0.5 = уменьшить вдвое
        0.0 = не ставить
        """
        can_bet, indicators = self.check_can_bet()
        
        if not can_bet:
            return 0.0
        
        # Базовый множитель
        multiplier = 1.0
        
        for indicator in indicators:
            if indicator.severity == TiltSeverity.HIGH:
                multiplier *= 0.5
            elif indicator.severity == TiltSeverity.MEDIUM:
                multiplier *= 0.75
            elif indicator.severity == TiltSeverity.LOW:
                multiplier *= 0.9
        
        return max(0.25, multiplier)  # Минимум 25% от нормального
    
    def get_session_summary(self) -> Dict:
        """Возвращает саммари текущей сессии"""
        if not self.current_session:
            return {"status": "no_active_session"}
        
        session = self.current_session
        streak_type, streak_len = session.current_streak
        
        return {
            "duration_minutes": session.duration_minutes,
            "bets_placed": session.bets_placed,
            "wins": session.wins,
            "losses": session.losses,
            "win_rate": session.wins / max(session.bets_placed, 1),
            "net_profit": session.net_profit,
            "profit_percent": session.net_profit / session.bankroll_start if session.bankroll_start > 0 else 0,
            "current_streak": f"{streak_type}:{streak_len}",
            "warnings_count": len(session.tilt_warnings),
            "is_locked": session.is_locked,
            "recommended_multiplier": self.get_recommended_bet_multiplier()
        }
    
    def get_discipline_report(self) -> str:
        """Генерирует текстовый отчет о дисциплине"""
        lines = ["=" * 50, "📊 DISCIPLINE REPORT", "=" * 50, ""]
        
        if not self.current_session:
            lines.append("Нет активной сессии")
            return "\n".join(lines)
        
        summary = self.get_session_summary()
        
        lines.append(f"⏱️ Длительность: {summary['duration_minutes']:.0f} мин")
        lines.append(f"🎯 Ставок: {summary['bets_placed']} ({summary['wins']}W-{summary['losses']}L)")
        lines.append(f"📈 Win Rate: {summary['win_rate']:.0%}")
        lines.append(f"💰 P&L: ${summary['net_profit']:+.2f} ({summary['profit_percent']:+.1%})")
        lines.append(f"📊 Серия: {summary['current_streak']}")
        lines.append("")
        
        # Предупреждения
        if summary['warnings_count'] > 0:
            lines.append(f"⚠️ Предупреждений: {summary['warnings_count']}")
            for w in self.current_session.tilt_warnings[-5:]:
                lines.append(f"  • [{w.severity.value}] {w.description}")
        
        lines.append("")
        
        # Рекомендация
        mult = summary['recommended_multiplier']
        if mult >= 1.0:
            lines.append("✅ Статус: Норма")
        elif mult >= 0.5:
            lines.append(f"⚠️ Статус: Осторожность (рекомендуемый размер: {mult:.0%})")
        elif mult > 0:
            lines.append(f"🟡 Статус: Повышенный риск (рекомендуемый размер: {mult:.0%})")
        else:
            lines.append("🛑 Статус: ЗАБЛОКИРОВАНО")
        
        if self.global_lock and self.global_lock_until:
            lines.append(f"🔒 Блокировка до: {self.global_lock_until.strftime('%H:%M')}")
        
        return "\n".join(lines)


# === ТЕСТИРОВАНИЕ ===

if __name__ == "__main__":
    print("=== Тест Discipline Manager ===\n")
    
    dm = DisciplineManager(
        losing_streak_warning=2,  # Для теста уменьшим
        losing_streak_stop=4,
        max_bets_per_hour=10,
        min_minutes_between_bets=1
    )
    
    # Начинаем сессию
    dm.start_session(bankroll=200.0)
    
    # Симулируем серию ставок
    print("Симуляция ставок:")
    print("-" * 40)
    
    bets = [
        (15.0, True),   # Win
        (15.0, False),  # Loss
        (15.0, False),  # Loss
        (25.0, False),  # Loss (chasing!)
        (35.0, False),  # Loss (more chasing!)
    ]
    
    bankroll = 200.0
    
    for i, (amount, won) in enumerate(bets, 1):
        # Проверяем можно ли ставить
        can_bet, warnings = dm.check_can_bet(proposed_amount=amount)
        
        print(f"\nСтавка #{i}: ${amount:.2f}")
        
        if warnings:
            for w in warnings:
                print(f"  ⚠️ [{w.severity.value}] {w.name}: {w.description}")
        
        if not can_bet:
            print("  🛑 СТАВКА ЗАБЛОКИРОВАНА")
            break
        
        # Делаем ставку
        if won:
            bankroll += amount * 0.85  # Выигрыш
            print(f"  ✅ Выиграли")
        else:
            bankroll -= amount
            print(f"  ❌ Проиграли")
        
        dm.record_bet(amount, bankroll)
        dm.record_result(won)
        
        print(f"  Баланс: ${bankroll:.2f}")
    
    # Отчет
    print("\n")
    print(dm.get_discipline_report())
