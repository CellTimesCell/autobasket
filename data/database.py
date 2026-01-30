"""
AutoBasket - Database Module
============================
Управление базой данных SQLite для хранения ставок, рейтингов и истории
"""

import sqlite3
from contextlib import contextmanager
from typing import Dict, List, Optional, Any
from datetime import datetime, date
import json
import os

import sys
sys.path.append('..')
from config.settings import config


class Database:
    """
    Управление базой данных AutoBasket
    
    Таблицы:
    - bankroll_management: текущее состояние банкролла
    - active_bets: активные ставки
    - bet_history: история ставок
    - betting_strategies: дневные стратегии
    - team_elo_ratings: Elo рейтинги команд
    - odds_movement: история движения линий
    - model_performance: статистика моделей
    - betting_sessions: сессии ставок
    """
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.database.db_path
        self._init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager для работы с соединением"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _init_database(self):
        """Инициализирует схему базы данных"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Управление банкроллом
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS bankroll_management (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    current_balance REAL NOT NULL,
                    day_start_balance REAL,
                    initial_balance REAL,
                    total_deposits REAL DEFAULT 0,
                    total_withdrawals REAL DEFAULT 0,
                    peak_balance REAL,
                    lowest_balance REAL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Активные ставки
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS active_bets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id INTEGER NOT NULL,
                    bet_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    team_bet_on TEXT NOT NULL,
                    bet_amount REAL NOT NULL,
                    odds REAL NOT NULL,
                    potential_win REAL NOT NULL,
                    confidence_score REAL,
                    expected_value REAL,
                    bet_category TEXT,
                    status TEXT DEFAULT 'active',
                    hedge_amount REAL DEFAULT 0,
                    adjusted_confidence REAL,
                    notes TEXT
                )
            ''')
            
            # История ставок
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS bet_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bet_id INTEGER,
                    game_id INTEGER NOT NULL,
                    game_date DATE,
                    home_team TEXT,
                    away_team TEXT,
                    team_bet_on TEXT,
                    bet_amount REAL,
                    odds REAL,
                    result TEXT,
                    amount_won REAL,
                    amount_lost REAL,
                    net_profit REAL,
                    roi REAL,
                    confidence_at_bet REAL,
                    category TEXT,
                    settled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT
                )
            ''')
            
            # Дневные стратегии
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS betting_strategies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_date DATE NOT NULL,
                    total_bets_placed INTEGER DEFAULT 0,
                    total_amount_risked REAL DEFAULT 0,
                    total_won REAL DEFAULT 0,
                    total_lost REAL DEFAULT 0,
                    net_profit REAL DEFAULT 0,
                    hit_rate REAL,
                    roi REAL,
                    bankroll_start REAL,
                    bankroll_end REAL,
                    best_bet_id INTEGER,
                    worst_bet_id INTEGER,
                    strategy_summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Elo рейтинги команд
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS team_elo_ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_name TEXT UNIQUE NOT NULL,
                    current_elo REAL DEFAULT 1500,
                    peak_elo REAL,
                    lowest_elo REAL,
                    games_played INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # История движения линий
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS odds_movement (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id INTEGER NOT NULL,
                    bookmaker TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    home_odds REAL,
                    away_odds REAL,
                    spread_line REAL,
                    total_line REAL,
                    home_spread_odds REAL,
                    away_spread_odds REAL,
                    over_odds REAL,
                    under_odds REAL
                )
            ''')
            
            # Статистика моделей
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    prediction_date DATE,
                    total_predictions INTEGER DEFAULT 0,
                    correct_predictions INTEGER DEFAULT 0,
                    accuracy REAL,
                    brier_score REAL,
                    log_loss REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Сессии ставок
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS betting_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_end TIMESTAMP,
                    bets_placed INTEGER DEFAULT 0,
                    bankroll_start REAL,
                    bankroll_end REAL,
                    net_profit REAL,
                    tilt_warnings INTEGER DEFAULT 0,
                    session_notes TEXT
                )
            ''')
            
            # Бэктест результаты
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    test_period_start DATE,
                    test_period_end DATE,
                    total_bets INTEGER,
                    win_rate REAL,
                    roi REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    final_bankroll REAL,
                    parameters TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Индексы для производительности
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_active_bets_game ON active_bets(game_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_bet_history_date ON bet_history(game_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_odds_game ON odds_movement(game_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategies_date ON betting_strategies(strategy_date)')
    
    # === BANKROLL OPERATIONS ===
    
    def get_bankroll(self) -> Optional[Dict]:
        """Получает текущее состояние банкролла"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM bankroll_management ORDER BY id DESC LIMIT 1')
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def update_bankroll(
        self,
        current_balance: float,
        day_start: float = None,
        peak: float = None,
        lowest: float = None
    ):
        """Обновляет банкролл"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            existing = self.get_bankroll()
            
            if existing:
                cursor.execute('''
                    UPDATE bankroll_management 
                    SET current_balance = ?,
                        day_start_balance = COALESCE(?, day_start_balance),
                        peak_balance = COALESCE(?, peak_balance),
                        lowest_balance = COALESCE(?, lowest_balance),
                        last_updated = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (current_balance, day_start, peak, lowest, existing['id']))
            else:
                cursor.execute('''
                    INSERT INTO bankroll_management 
                    (current_balance, day_start_balance, initial_balance, peak_balance, lowest_balance)
                    VALUES (?, ?, ?, ?, ?)
                ''', (current_balance, current_balance, current_balance, current_balance, current_balance))
    
    # === BET OPERATIONS ===
    
    def add_active_bet(self, bet_data: Dict) -> int:
        """Добавляет активную ставку"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO active_bets 
                (game_id, home_team, away_team, team_bet_on, bet_amount, odds, 
                 potential_win, confidence_score, expected_value, bet_category, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                bet_data['game_id'],
                bet_data['home_team'],
                bet_data['away_team'],
                bet_data['team_bet_on'],
                bet_data['bet_amount'],
                bet_data['odds'],
                bet_data['potential_win'],
                bet_data.get('confidence'),
                bet_data.get('expected_value'),
                bet_data.get('category'),
                bet_data.get('notes')
            ))
            return cursor.lastrowid
    
    def get_active_bets(self) -> List[Dict]:
        """Получает все активные ставки"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM active_bets WHERE status = "active"')
            return [dict(row) for row in cursor.fetchall()]
    
    def settle_bet(self, bet_id: int, result: str, profit: float):
        """Закрывает ставку"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Получаем данные ставки
            cursor.execute('SELECT * FROM active_bets WHERE id = ?', (bet_id,))
            bet = cursor.fetchone()
            
            if not bet:
                return
            
            bet = dict(bet)
            
            # Обновляем статус
            cursor.execute('''
                UPDATE active_bets SET status = ? WHERE id = ?
            ''', (result, bet_id))
            
            # Добавляем в историю
            cursor.execute('''
                INSERT INTO bet_history 
                (bet_id, game_id, home_team, away_team, team_bet_on, bet_amount, 
                 odds, result, net_profit, confidence_at_bet, category)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                bet_id,
                bet['game_id'],
                bet['home_team'],
                bet['away_team'],
                bet['team_bet_on'],
                bet['bet_amount'],
                bet['odds'],
                result,
                profit,
                bet['confidence_score'],
                bet['bet_category']
            ))
    
    def get_bet_history(
        self,
        start_date: date = None,
        end_date: date = None,
        limit: int = 100
    ) -> List[Dict]:
        """Получает историю ставок"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = 'SELECT * FROM bet_history WHERE 1=1'
            params = []
            
            if start_date:
                query += ' AND game_date >= ?'
                params.append(start_date.isoformat())
            if end_date:
                query += ' AND game_date <= ?'
                params.append(end_date.isoformat())
            
            query += ' ORDER BY settled_at DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    # === ELO OPERATIONS ===
    
    def get_team_elo(self, team_name: str) -> Optional[float]:
        """Получает Elo рейтинг команды"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT current_elo FROM team_elo_ratings WHERE team_name = ?',
                (team_name,)
            )
            row = cursor.fetchone()
            return row['current_elo'] if row else None
    
    def update_team_elo(self, team_name: str, new_elo: float, won: bool):
        """Обновляет Elo рейтинг команды"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT * FROM team_elo_ratings WHERE team_name = ?',
                (team_name,)
            )
            existing = cursor.fetchone()
            
            if existing:
                existing = dict(existing)
                cursor.execute('''
                    UPDATE team_elo_ratings 
                    SET current_elo = ?,
                        peak_elo = MAX(peak_elo, ?),
                        lowest_elo = MIN(lowest_elo, ?),
                        games_played = games_played + 1,
                        wins = wins + ?,
                        losses = losses + ?,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE team_name = ?
                ''', (new_elo, new_elo, new_elo, 1 if won else 0, 0 if won else 1, team_name))
            else:
                cursor.execute('''
                    INSERT INTO team_elo_ratings 
                    (team_name, current_elo, peak_elo, lowest_elo, games_played, wins, losses)
                    VALUES (?, ?, ?, ?, 1, ?, ?)
                ''', (team_name, new_elo, new_elo, new_elo, 1 if won else 0, 0 if won else 1))
    
    def get_all_elo_ratings(self) -> List[Dict]:
        """Получает все Elo рейтинги"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM team_elo_ratings ORDER BY current_elo DESC')
            return [dict(row) for row in cursor.fetchall()]
    
    # === ODDS OPERATIONS ===
    
    def save_odds(self, game_id: int, odds_data: Dict):
        """Сохраняет коэффициенты"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO odds_movement 
                (game_id, bookmaker, home_odds, away_odds, spread_line, total_line)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                game_id,
                odds_data.get('bookmaker'),
                odds_data.get('home_odds'),
                odds_data.get('away_odds'),
                odds_data.get('spread_line'),
                odds_data.get('total_line')
            ))
    
    def get_odds_history(self, game_id: int) -> List[Dict]:
        """Получает историю коэффициентов"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM odds_movement WHERE game_id = ? ORDER BY timestamp',
                (game_id,)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    # === STRATEGY OPERATIONS ===
    
    def save_daily_strategy(self, strategy_data: Dict):
        """Сохраняет дневную стратегию"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO betting_strategies 
                (strategy_date, total_bets_placed, total_amount_risked, bankroll_start, strategy_summary)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                strategy_data['date'],
                strategy_data.get('total_bets', 0),
                strategy_data.get('total_risk', 0),
                strategy_data.get('bankroll_start', 0),
                json.dumps(strategy_data.get('summary', {}))
            ))
    
    def update_daily_strategy(self, strategy_date: date, results: Dict):
        """Обновляет результаты дневной стратегии"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE betting_strategies 
                SET total_won = ?,
                    total_lost = ?,
                    net_profit = ?,
                    hit_rate = ?,
                    roi = ?,
                    bankroll_end = ?
                WHERE strategy_date = ?
            ''', (
                results.get('total_won', 0),
                results.get('total_lost', 0),
                results.get('net_profit', 0),
                results.get('hit_rate', 0),
                results.get('roi', 0),
                results.get('bankroll_end', 0),
                strategy_date.isoformat()
            ))
    
    # === BACKTEST OPERATIONS ===
    
    def save_backtest_result(self, result: Dict):
        """Сохраняет результаты бэктеста"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO backtest_results 
                (strategy_name, test_period_start, test_period_end, total_bets,
                 win_rate, roi, max_drawdown, sharpe_ratio, final_bankroll, parameters)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result['strategy_name'],
                result.get('period_start'),
                result.get('period_end'),
                result.get('total_bets', 0),
                result.get('win_rate', 0),
                result.get('roi', 0),
                result.get('max_drawdown', 0),
                result.get('sharpe_ratio', 0),
                result.get('final_bankroll', 0),
                json.dumps(result.get('parameters', {}))
            ))
    
    # === STATISTICS ===
    
    def get_overall_stats(self) -> Dict:
        """Получает общую статистику"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Статистика ставок
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_bets,
                    SUM(CASE WHEN result = 'won' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN result = 'lost' THEN 1 ELSE 0 END) as losses,
                    SUM(bet_amount) as total_wagered,
                    SUM(net_profit) as total_profit
                FROM bet_history
            ''')
            bet_stats = dict(cursor.fetchone())
            
            # Текущий банкролл
            bankroll = self.get_bankroll()
            
            # Рассчитываем метрики
            total_bets = bet_stats['total_bets'] or 0
            wins = bet_stats['wins'] or 0
            
            return {
                'total_bets': total_bets,
                'wins': wins,
                'losses': bet_stats['losses'] or 0,
                'win_rate': wins / total_bets if total_bets > 0 else 0,
                'total_wagered': bet_stats['total_wagered'] or 0,
                'total_profit': bet_stats['total_profit'] or 0,
                'roi': (bet_stats['total_profit'] / bet_stats['total_wagered'] * 100) 
                       if bet_stats['total_wagered'] else 0,
                'current_bankroll': bankroll['current_balance'] if bankroll else 0,
                'peak_bankroll': bankroll['peak_balance'] if bankroll else 0,
                'active_bets': len(self.get_active_bets())
            }


# === ТЕСТИРОВАНИЕ ===

if __name__ == "__main__":
    print("=== Тест Database Module ===\n")
    
    # Используем тестовую БД
    db = Database(db_path=":memory:")  # В памяти для теста
    
    # Тест банкролла
    print("Тест 1: Bankroll Management")
    print("-" * 40)
    
    db.update_bankroll(200.00)
    bankroll = db.get_bankroll()
    print(f"Начальный банкролл: ${bankroll['current_balance']:.2f}")
    
    db.update_bankroll(215.50, peak=215.50)
    bankroll = db.get_bankroll()
    print(f"После выигрыша: ${bankroll['current_balance']:.2f}")
    print(f"Peak: ${bankroll['peak_balance']:.2f}")
    
    # Тест ставок
    print("\n\nТест 2: Bet Management")
    print("-" * 40)
    
    bet_id = db.add_active_bet({
        'game_id': 1001,
        'home_team': 'Lakers',
        'away_team': 'Warriors',
        'team_bet_on': 'home',
        'bet_amount': 15.00,
        'odds': 1.85,
        'potential_win': 12.75,
        'confidence': 0.65,
        'expected_value': 0.12,
        'category': 'value'
    })
    print(f"Создана ставка ID: {bet_id}")
    
    active = db.get_active_bets()
    print(f"Активных ставок: {len(active)}")
    
    db.settle_bet(bet_id, 'won', 12.75)
    print("Ставка закрыта как выигрыш")
    
    history = db.get_bet_history()
    print(f"Ставок в истории: {len(history)}")
    
    # Тест Elo
    print("\n\nТест 3: Elo Ratings")
    print("-" * 40)
    
    db.update_team_elo('Lakers', 1620, True)
    db.update_team_elo('Warriors', 1580, False)
    
    ratings = db.get_all_elo_ratings()
    for r in ratings:
        print(f"{r['team_name']}: {r['current_elo']:.0f} ({r['wins']}-{r['losses']})")
    
    # Общая статистика
    print("\n\nТест 4: Overall Stats")
    print("-" * 40)
    
    stats = db.get_overall_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
