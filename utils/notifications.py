"""
AutoBasket - Notification System
================================
Уведомления через Telegram и Discord
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging

# Настраиваем логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class AlertType(Enum):
    """Типы алертов"""
    VALUE_BET_FOUND = "value_bet"
    INJURY_ALERT = "injury"
    BANKROLL_UPDATE = "bankroll"
    GAME_RESULT = "result"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TILT_WARNING = "tilt"
    LINE_MOVEMENT = "line_move"
    SYSTEM_ERROR = "error"


class AlertPriority(Enum):
    """Приоритеты"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Alert:
    """Структура алерта"""
    type: AlertType
    priority: AlertPriority
    title: str
    message: str
    data: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    sent: bool = False
    
    def to_telegram_message(self) -> str:
        """Форматирует для Telegram"""
        emoji = {
            AlertType.VALUE_BET_FOUND: "💰",
            AlertType.INJURY_ALERT: "🚨",
            AlertType.BANKROLL_UPDATE: "📊",
            AlertType.GAME_RESULT: "🏀",
            AlertType.STOP_LOSS: "🛑",
            AlertType.TAKE_PROFIT: "🎉",
            AlertType.TILT_WARNING: "⚠️",
            AlertType.LINE_MOVEMENT: "📈",
            AlertType.SYSTEM_ERROR: "❌"
        }.get(self.type, "📢")
        
        priority_marker = "❗" * min(self.priority.value, 3)
        
        text = f"{emoji} *{self.title}* {priority_marker}\n\n"
        text += f"{self.message}\n"
        
        if self.data:
            text += "\n📋 *Детали:*\n"
            for key, value in self.data.items():
                text += f"• {key}: `{value}`\n"
        
        text += f"\n🕐 {self.timestamp.strftime('%H:%M:%S')}"
        
        return text
    
    def to_discord_embed(self) -> Dict:
        """Форматирует для Discord embed"""
        color = {
            AlertPriority.LOW: 0x808080,      # Серый
            AlertPriority.MEDIUM: 0x3498db,   # Синий
            AlertPriority.HIGH: 0xf39c12,     # Оранжевый
            AlertPriority.CRITICAL: 0xe74c3c  # Красный
        }.get(self.priority, 0x000000)
        
        fields = [
            {"name": key, "value": str(value), "inline": True}
            for key, value in self.data.items()
        ]
        
        return {
            "embeds": [{
                "title": self.title,
                "description": self.message,
                "color": color,
                "fields": fields,
                "timestamp": self.timestamp.isoformat(),
                "footer": {"text": f"AutoBasket | {self.type.value}"}
            }]
        }


class TelegramNotifier:
    """Отправка уведомлений в Telegram"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.enabled = bool(token and chat_id)
    
    def send(self, alert: Alert) -> bool:
        """Отправляет сообщение"""
        if not self.enabled or not REQUESTS_AVAILABLE:
            logger.warning("Telegram not configured or requests not available")
            return False
        
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": alert.to_telegram_message(),
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"Telegram alert sent: {alert.title}")
                return True
            else:
                logger.error(f"Telegram error: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False
    
    def send_photo(self, photo_path: str, caption: str = "") -> bool:
        """Отправляет изображение (для графиков)"""
        if not self.enabled or not REQUESTS_AVAILABLE:
            return False
        
        try:
            url = f"{self.base_url}/sendPhoto"
            with open(photo_path, 'rb') as photo:
                files = {'photo': photo}
                data = {'chat_id': self.chat_id, 'caption': caption}
                response = requests.post(url, files=files, data=data, timeout=30)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Telegram photo send failed: {e}")
            return False


class DiscordNotifier:
    """Отправка уведомлений в Discord"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.enabled = bool(webhook_url)
    
    def send(self, alert: Alert) -> bool:
        """Отправляет embed сообщение"""
        if not self.enabled or not REQUESTS_AVAILABLE:
            logger.warning("Discord not configured or requests not available")
            return False
        
        try:
            response = requests.post(
                self.webhook_url,
                json=alert.to_discord_embed(),
                timeout=10
            )
            
            if response.status_code in [200, 204]:
                logger.info(f"Discord alert sent: {alert.title}")
                return True
            else:
                logger.error(f"Discord error: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Discord send failed: {e}")
            return False


class NotificationManager:
    """
    Главный менеджер уведомлений
    
    Управляет всеми каналами и фильтрует по приоритету
    """
    
    def __init__(
        self,
        telegram_token: str = None,
        telegram_chat_id: str = None,
        discord_webhook: str = None,
        min_priority: AlertPriority = AlertPriority.MEDIUM
    ):
        self.telegram = TelegramNotifier(telegram_token, telegram_chat_id) if telegram_token else None
        self.discord = DiscordNotifier(discord_webhook) if discord_webhook else None
        self.min_priority = min_priority
        
        # История алертов
        self.alert_history: List[Alert] = []
        self.max_history = 1000
        
        # Rate limiting
        self.last_sent: Dict[AlertType, datetime] = {}
        self.cooldowns = {
            AlertType.VALUE_BET_FOUND: 60,      # 1 мин между value bets
            AlertType.INJURY_ALERT: 300,        # 5 мин между травмами
            AlertType.BANKROLL_UPDATE: 3600,    # 1 час между обновлениями
            AlertType.GAME_RESULT: 60,          # 1 мин между результатами
            AlertType.LINE_MOVEMENT: 1800,      # 30 мин между движениями
        }
    
    def send_message(self, message: str, priority: AlertPriority = AlertPriority.MEDIUM) -> bool:
        """Отправляет простое текстовое сообщение"""
        alert = Alert(
            type=AlertType.BANKROLL_UPDATE,
            priority=priority,
            title="AutoBasket",
            message=message
        )
        return self.send(alert)
    
    def _should_send(self, alert: Alert) -> bool:
        """Проверяет, нужно ли отправлять алерт"""
        # Проверка приоритета
        if alert.priority.value < self.min_priority.value:
            return False
        
        # CRITICAL всегда отправляем
        if alert.priority == AlertPriority.CRITICAL:
            return True
        
        # Rate limiting
        cooldown = self.cooldowns.get(alert.type, 0)
        if cooldown > 0:
            last = self.last_sent.get(alert.type)
            if last:
                elapsed = (datetime.now() - last).total_seconds()
                if elapsed < cooldown:
                    logger.debug(f"Rate limited: {alert.type.value}")
                    return False
        
        return True
    
    def send(self, alert: Alert) -> bool:
        """Отправляет алерт во все активные каналы"""
        if not self._should_send(alert):
            return False
        
        success = False
        
        # Telegram
        if self.telegram and self.telegram.enabled:
            if self.telegram.send(alert):
                success = True
        
        # Discord
        if self.discord and self.discord.enabled:
            if self.discord.send(alert):
                success = True
        
        if success:
            alert.sent = True
            self.last_sent[alert.type] = datetime.now()
        
        # Сохраняем в историю
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]
        
        return success
    
    # === Удобные методы для типовых алертов ===
    
    def notify_value_bet(
        self,
        team: str,
        opponent: str,
        confidence: float,
        odds: float,
        ev: float,
        bet_amount: float
    ):
        """Уведомление о найденной value bet"""
        alert = Alert(
            type=AlertType.VALUE_BET_FOUND,
            priority=AlertPriority.HIGH if ev > 0.10 else AlertPriority.MEDIUM,
            title=f"Value Bet: {team}",
            message=f"Найдена ставка с EV {ev:.1%} на {team} против {opponent}",
            data={
                "Команда": team,
                "Противник": opponent,
                "Уверенность": f"{confidence:.0%}",
                "Коэффициент": f"{odds:.2f}",
                "Expected Value": f"{ev:.1%}",
                "Рекомендуемая ставка": f"${bet_amount:.2f}"
            }
        )
        return self.send(alert)
    
    def notify_injury(
        self,
        player: str,
        team: str,
        status: str,
        impact: float,
        game_info: str = ""
    ):
        """Уведомление о травме"""
        priority = AlertPriority.CRITICAL if impact > 0.15 else AlertPriority.HIGH
        
        alert = Alert(
            type=AlertType.INJURY_ALERT,
            priority=priority,
            title=f"🚨 Травма: {player}",
            message=f"{player} ({team}) - статус: {status}",
            data={
                "Игрок": player,
                "Команда": team,
                "Статус": status,
                "Влияние на игру": f"{impact:.0%}",
                "Игра": game_info
            }
        )
        return self.send(alert)
    
    def notify_bankroll_update(
        self,
        current: float,
        change: float,
        daily_pnl: float,
        total_pnl: float
    ):
        """Уведомление об изменении банкролла"""
        emoji = "📈" if change > 0 else "📉"
        
        alert = Alert(
            type=AlertType.BANKROLL_UPDATE,
            priority=AlertPriority.LOW,
            title=f"{emoji} Банкролл: ${current:.2f}",
            message=f"Изменение: {'+' if change > 0 else ''}{change:.2f}",
            data={
                "Текущий баланс": f"${current:.2f}",
                "Изменение": f"${change:+.2f}",
                "P&L за день": f"{daily_pnl:+.1%}",
                "Общий P&L": f"{total_pnl:+.1%}"
            }
        )
        return self.send(alert)
    
    def notify_game_result(
        self,
        home_team: str,
        away_team: str,
        our_bet: str,
        won: bool,
        profit: float,
        new_bankroll: float
    ):
        """Уведомление о результате игры"""
        emoji = "✅" if won else "❌"
        
        alert = Alert(
            type=AlertType.GAME_RESULT,
            priority=AlertPriority.MEDIUM,
            title=f"{emoji} {home_team} vs {away_team}",
            message=f"Ставка на {our_bet}: {'Выиграли' if won else 'Проиграли'}",
            data={
                "Результат": "Победа" if won else "Поражение",
                "Профит": f"${profit:+.2f}",
                "Новый баланс": f"${new_bankroll:.2f}"
            }
        )
        return self.send(alert)
    
    def notify_stop_loss(self, loss_amount: float, loss_percentage: float):
        """Уведомление о срабатывании stop-loss"""
        alert = Alert(
            type=AlertType.STOP_LOSS,
            priority=AlertPriority.CRITICAL,
            title="🛑 STOP-LOSS TRIGGERED",
            message=f"Достигнут лимит потерь. Ставки приостановлены.",
            data={
                "Потеря": f"${loss_amount:.2f}",
                "Процент": f"{loss_percentage:.1%}",
                "Действие": "Ставки заблокированы"
            }
        )
        return self.send(alert)
    
    def notify_take_profit(self, profit_amount: float, profit_percentage: float):
        """Уведомление о достижении take-profit"""
        alert = Alert(
            type=AlertType.TAKE_PROFIT,
            priority=AlertPriority.HIGH,
            title="🎉 TAKE-PROFIT REACHED",
            message=f"Достигнута целевая прибыль! Рекомендуется зафиксировать.",
            data={
                "Прибыль": f"${profit_amount:.2f}",
                "Процент": f"{profit_percentage:.1%}",
                "Рекомендация": "Вывести часть прибыли"
            }
        )
        return self.send(alert)
    
    def notify_tilt_warning(self, reason: str, severity: str, recommendation: str):
        """Предупреждение о тильте"""
        priority = AlertPriority.CRITICAL if severity == "high" else AlertPriority.HIGH
        
        alert = Alert(
            type=AlertType.TILT_WARNING,
            priority=priority,
            title="⚠️ TILT WARNING",
            message=f"Обнаружены признаки тильта: {reason}",
            data={
                "Причина": reason,
                "Серьезность": severity,
                "Рекомендация": recommendation
            }
        )
        return self.send(alert)
    
    def get_recent_alerts(self, limit: int = 20) -> List[Alert]:
        """Возвращает последние алерты"""
        return self.alert_history[-limit:]


# === Консольный вывод (для режима без мессенджеров) ===

class ConsoleNotifier:
    """Вывод в консоль для тестирования"""
    
    def send(self, alert: Alert) -> bool:
        priority_colors = {
            AlertPriority.LOW: "\033[90m",      # Серый
            AlertPriority.MEDIUM: "\033[94m",   # Синий
            AlertPriority.HIGH: "\033[93m",     # Желтый
            AlertPriority.CRITICAL: "\033[91m"  # Красный
        }
        reset = "\033[0m"
        color = priority_colors.get(alert.priority, "")
        
        print(f"\n{color}{'='*50}")
        print(f"[{alert.timestamp.strftime('%H:%M:%S')}] {alert.title}")
        print(f"Priority: {alert.priority.name}")
        print(f"{'='*50}{reset}")
        print(alert.message)
        if alert.data:
            print("\nДетали:")
            for k, v in alert.data.items():
                print(f"  • {k}: {v}")
        print()
        
        return True


# === ТЕСТИРОВАНИЕ ===

if __name__ == "__main__":
    print("=== Тест Notification System ===\n")
    
    # Создаем менеджер без реальных токенов (только консоль)
    console = ConsoleNotifier()
    
    # Тест value bet
    alert1 = Alert(
        type=AlertType.VALUE_BET_FOUND,
        priority=AlertPriority.HIGH,
        title="Value Bet: Lakers",
        message="Найдена ставка с EV 12.5% на Lakers",
        data={
            "Команда": "Lakers",
            "Противник": "Warriors",
            "Уверенность": "65%",
            "Коэффициент": "1.85",
            "EV": "12.5%"
        }
    )
    console.send(alert1)
    
    # Тест травмы
    alert2 = Alert(
        type=AlertType.INJURY_ALERT,
        priority=AlertPriority.CRITICAL,
        title="🚨 Травма: LeBron James",
        message="LeBron James (Lakers) - статус: OUT",
        data={
            "Игрок": "LeBron James",
            "Влияние": "18%"
        }
    )
    console.send(alert2)
    
    # Тест stop-loss
    alert3 = Alert(
        type=AlertType.STOP_LOSS,
        priority=AlertPriority.CRITICAL,
        title="🛑 STOP-LOSS TRIGGERED",
        message="Достигнут лимит потерь -15%",
        data={
            "Потеря": "$30.00",
            "Действие": "Ставки заблокированы"
        }
    )
    console.send(alert3)
    
    print("\n✅ Notification system test complete!")
