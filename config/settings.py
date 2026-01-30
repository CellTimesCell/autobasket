"""
AutoBasket Betting Intelligence - Конфигурация
==============================================
Все настройки системы в одном месте
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import os
from dotenv import load_dotenv

load_dotenv()


class BetCategory(Enum):
    """Категории ставок"""
    SAFE = "safe"
    VALUE = "value"
    HIGH_RISK = "high_risk"


class BetStatus(Enum):
    """Статус ставки"""
    PENDING = "pending"
    WON = "won"
    LOST = "lost"
    PUSH = "push"
    CANCELLED = "cancelled"


class AlertPriority(Enum):
    """Приоритеты уведомлений"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class BankrollSettings:
    """Настройки управления банкроллом"""
    initial_bankroll: float = 200.00
    min_bet: float = 5.00
    max_bet_percentage: float = 0.10  # 10% от банкролла
    risk_tolerance: float = 0.02  # 2% максимального риска
    
    # Stop-loss и Take-profit
    stop_loss_daily: float = -0.15  # -15% за день
    stop_loss_total: float = -0.25  # -25% от начального
    take_profit_daily: float = 0.20  # +20% за день
    cash_out_threshold: float = 0.40  # Выводить при +40%
    
    # Дневные лимиты
    max_daily_risk: float = 0.30  # 30% банкролла в день
    max_bets_per_day: int = 10


@dataclass
class PortfolioSettings:
    """Настройки распределения портфеля"""
    safe_allocation: float = 0.40  # 40% на безопасные
    value_allocation: float = 0.35  # 35% на value bets
    high_risk_allocation: float = 0.15  # 15% на рискованные
    cash_reserve: float = 0.10  # 10% резерв
    
    # Лимиты по категориям
    safe_max_per_bet: float = 0.10
    safe_min_confidence: float = 0.70
    
    value_max_per_bet: float = 0.07
    value_min_ev: float = 0.05
    
    risk_max_per_bet: float = 0.03
    risk_min_edge: float = 0.15


@dataclass
class PredictionSettings:
    """Настройки ML моделей"""
    min_confidence_to_bet: float = 0.52  # Минимум 52% уверенности
    min_value_threshold: float = 0.05  # Минимум 5% value
    
    # Веса моделей в ensemble
    model_weights: Dict[str, float] = field(default_factory=lambda: {
        'xgboost': 0.45,
        'gradient_boost': 0.35,
        'random_forest': 0.20
    })
    
    # Калибровка
    calibration_factor: float = 0.85  # Сжатие к центру
    
    # Home advantage
    home_advantage_points: float = 3.5


@dataclass
class EloSettings:
    """Настройки Elo рейтингов"""
    default_rating: float = 1500.0
    k_factor: float = 20.0
    home_advantage: float = 100.0  # В Elo пунктах
    
    # Season reset
    season_reset_regression: float = 0.75  # Регрессия к среднему между сезонами


@dataclass
class LiveBettingSettings:
    """Настройки live ставок"""
    update_interval_seconds: int = 30
    auto_reduce_on_injury: bool = True
    hedge_on_momentum_shift: bool = False
    max_live_adjustment: float = 0.50  # 50% от ставки
    
    # Триггеры корректировок
    injury_impact: float = 0.15
    momentum_shift_impact: float = 0.10
    foul_trouble_impact: float = 0.08


@dataclass
class NotificationSettings:
    """Настройки уведомлений"""
    telegram_enabled: bool = True
    telegram_token: Optional[str] = field(default_factory=lambda: os.getenv('TELEGRAM_TOKEN'))
    telegram_chat_id: Optional[str] = field(default_factory=lambda: os.getenv('TELEGRAM_CHAT_ID'))
    
    discord_enabled: bool = False
    discord_webhook: Optional[str] = field(default_factory=lambda: os.getenv('DISCORD_WEBHOOK'))
    
    # Какие события отправлять
    notify_value_bets: bool = True
    notify_injuries: bool = True
    notify_bankroll_changes: bool = True
    notify_game_results: bool = True


@dataclass
class DatabaseSettings:
    """Настройки базы данных"""
    db_path: str = "autobasket.db"
    backup_enabled: bool = True
    backup_interval_hours: int = 24


@dataclass
class APISettings:
    """Настройки внешних API"""
    # NBA API
    nba_api_delay: float = 0.6  # Задержка между запросами
    
    # Odds API (the-odds-api.com)
    odds_api_key: Optional[str] = field(default_factory=lambda: os.getenv('ODDS_API_KEY'))
    odds_api_regions: List[str] = field(default_factory=lambda: ['us', 'eu'])
    
    # Bookmakers для сравнения
    bookmakers: List[str] = field(default_factory=lambda: [
        'pinnacle', 'bet365', 'draftkings', 'fanduel', 'betmgm'
    ])


@dataclass
class BacktestSettings:
    """Настройки бэктестинга"""
    default_start_date: str = "2023-01-01"
    monte_carlo_simulations: int = 10000
    train_test_split: float = 0.8
    walk_forward_window_days: int = 30


@dataclass
class DisciplineSettings:
    """Настройки дисциплины"""
    max_consecutive_bets: int = 5
    min_time_between_bets_minutes: int = 10
    mandatory_break_after_hours: int = 4
    
    # Tilt detection
    losing_streak_warning: int = 3
    losing_streak_stop: int = 5
    chase_loss_multiplier: float = 1.5  # Если ставка > 1.5x предыдущей после проигрыша


# === ГЛАВНЫЙ КОНФИГ ===

@dataclass
class Config:
    """Главный конфигурационный класс"""
    bankroll: BankrollSettings = field(default_factory=BankrollSettings)
    portfolio: PortfolioSettings = field(default_factory=PortfolioSettings)
    prediction: PredictionSettings = field(default_factory=PredictionSettings)
    elo: EloSettings = field(default_factory=EloSettings)
    live_betting: LiveBettingSettings = field(default_factory=LiveBettingSettings)
    notifications: NotificationSettings = field(default_factory=NotificationSettings)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    api: APISettings = field(default_factory=APISettings)
    backtest: BacktestSettings = field(default_factory=BacktestSettings)
    discipline: DisciplineSettings = field(default_factory=DisciplineSettings)
    
    # Режим работы
    debug_mode: bool = False
    simulation_mode: bool = True  # Если True - виртуальные деньги
    

# Глобальный экземпляр конфига
config = Config()


def load_config_from_file(filepath: str) -> Config:
    """Загрузка конфига из JSON/YAML файла"""
    import json
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Здесь можно добавить парсинг и создание Config из dict
    return Config()


def save_config_to_file(config: Config, filepath: str):
    """Сохранение конфига в файл"""
    import json
    from dataclasses import asdict
    
    with open(filepath, 'w') as f:
        json.dump(asdict(config), f, indent=2, default=str)
