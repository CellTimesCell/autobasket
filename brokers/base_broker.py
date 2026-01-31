"""
AutoBasket - Base Broker Interface
==================================
Abstract interface for all broker implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class BrokerMode(Enum):
    """Trading mode"""
    PAPER = "paper"      # Virtual money - no real bets
    REAL = "real"        # Real money - actual execution


class BetStatus(Enum):
    """Status of a placed bet"""
    PENDING = "pending"        # Bet placed, waiting for result
    CONFIRMED = "confirmed"    # Bet confirmed by broker
    WON = "won"               # Bet won
    LOST = "lost"             # Bet lost
    CANCELLED = "cancelled"    # Bet cancelled
    FAILED = "failed"         # Bet placement failed


@dataclass
class BetResult:
    """Result of a bet placement attempt"""
    success: bool
    bet_id: Optional[str] = None
    message: str = ""
    executed_amount: float = 0.0
    executed_odds: float = 0.0
    commission: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    raw_response: Optional[Dict] = None


@dataclass
class Position:
    """An open position/bet"""
    bet_id: str
    game_id: str
    home_team: str
    away_team: str
    side: str              # 'home' or 'away'
    amount: float
    odds: float
    status: BetStatus
    placed_at: datetime
    settled_at: Optional[datetime] = None
    profit: float = 0.0
    final_score: Optional[str] = None


class BaseBroker(ABC):
    """
    Abstract base class for all brokers.

    A broker is responsible for:
    1. Placing bets (virtual or real)
    2. Tracking positions
    3. Getting account balance
    4. Settling bets when games finish
    """

    def __init__(self, mode: BrokerMode, initial_balance: float = 200.0):
        self.mode = mode
        self.initial_balance = initial_balance
        self._balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.history: List[Position] = []
        self._connected = False

        logger.info(f"Broker initialized: mode={mode.value}, balance=${initial_balance:.2f}")

    @property
    def balance(self) -> float:
        """Current account balance"""
        return self._balance

    @property
    def connected(self) -> bool:
        """Whether broker is connected and ready"""
        return self._connected

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to broker/exchange.
        Returns True if connection successful.
        """
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect from broker"""
        pass

    @abstractmethod
    def get_balance(self) -> float:
        """
        Get current account balance.
        For real brokers, this fetches from API.
        """
        pass

    @abstractmethod
    def place_bet(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        side: str,
        amount: float,
        odds: float,
        **kwargs
    ) -> BetResult:
        """
        Place a bet.

        Args:
            game_id: Unique game identifier
            home_team: Home team name
            away_team: Away team name
            side: 'home' or 'away'
            amount: Bet amount in $
            odds: Decimal odds (e.g., 1.91)
            **kwargs: Additional broker-specific parameters

        Returns:
            BetResult with success status and details
        """
        pass

    @abstractmethod
    def cancel_bet(self, bet_id: str) -> bool:
        """
        Cancel an open bet if possible.
        Returns True if cancelled successfully.
        """
        pass

    @abstractmethod
    def settle_bet(
        self,
        bet_id: str,
        won: bool,
        home_score: int,
        away_score: int
    ) -> float:
        """
        Settle a bet after game finishes.

        Args:
            bet_id: The bet to settle
            won: Whether the bet won
            home_score: Final home team score
            away_score: Final away team score

        Returns:
            Profit/loss amount
        """
        pass

    def get_position(self, bet_id: str) -> Optional[Position]:
        """Get a specific position"""
        return self.positions.get(bet_id)

    def get_open_positions(self) -> List[Position]:
        """Get all open/pending positions"""
        return [p for p in self.positions.values()
                if p.status in [BetStatus.PENDING, BetStatus.CONFIRMED]]

    def get_positions_for_game(self, game_id: str) -> List[Position]:
        """Get all positions for a specific game"""
        return [p for p in self.positions.values() if p.game_id == game_id]

    def get_history(self, limit: int = 100) -> List[Position]:
        """Get settled bet history"""
        return self.history[-limit:]

    def get_stats(self) -> Dict:
        """Get performance statistics"""
        settled = [p for p in self.history if p.status in [BetStatus.WON, BetStatus.LOST]]

        if not settled:
            return {
                'total_bets': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'total_profit': 0.0,
                'roi': 0.0
            }

        wins = len([p for p in settled if p.status == BetStatus.WON])
        losses = len([p for p in settled if p.status == BetStatus.LOST])
        total_profit = sum(p.profit for p in settled)
        total_wagered = sum(p.amount for p in settled)

        return {
            'total_bets': len(settled),
            'wins': wins,
            'losses': losses,
            'win_rate': wins / len(settled) if settled else 0.0,
            'total_profit': total_profit,
            'total_wagered': total_wagered,
            'roi': total_profit / total_wagered if total_wagered > 0 else 0.0,
            'current_balance': self._balance,
            'pnl_from_start': self._balance - self.initial_balance
        }

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(mode={self.mode.value}, balance=${self._balance:.2f})"

    def __repr__(self) -> str:
        return self.__str__()
