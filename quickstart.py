#!/usr/bin/env python3
"""
AutoBasket - Quick Start Script
===============================
Быстрый старт системы с проверкой зависимостей
"""

import os
import sys
from pathlib import Path


def check_python_version():
    """Проверяет версию Python"""
    if sys.version_info < (3, 9):
        print("❌ Требуется Python 3.9 или выше")
        print(f"   Текущая версия: {sys.version}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    return True


def check_dependencies():
    """Проверяет установленные зависимости"""
    required = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'requests': 'requests',
        'sqlite3': None,  # Встроен в Python
    }
    
    optional = {
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost',
        'streamlit': 'streamlit',
        'plotly': 'plotly',
        'nba_api': 'nba_api',
    }
    
    missing_required = []
    missing_optional = []
    
    print("\n📦 Проверка зависимостей:")
    
    # Обязательные
    for module, package in required.items():
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module}")
            if package:
                missing_required.append(package)
    
    # Опциональные
    print("\n📦 Опциональные зависимости:")
    for module, package in optional.items():
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ⚪ {module} (не установлен)")
            missing_optional.append(package)
    
    if missing_required:
        print(f"\n⚠️ Установите обязательные зависимости:")
        print(f"   pip install {' '.join(missing_required)}")
        return False
    
    if missing_optional:
        print(f"\n💡 Для полной функциональности установите:")
        print(f"   pip install {' '.join(missing_optional)}")
    
    return True


def check_env_file():
    """Проверяет наличие .env файла"""
    env_path = Path(__file__).parent / '.env'
    env_example = Path(__file__).parent / '.env.example'
    
    print("\n🔑 Проверка конфигурации:")
    
    if env_path.exists():
        print("  ✅ .env файл найден")
        
        # Проверяем ключевые переменные
        with open(env_path) as f:
            content = f.read()
        
        if 'ODDS_API_KEY=' in content and 'your_' not in content.split('ODDS_API_KEY=')[1][:30]:
            print("  ✅ ODDS_API_KEY настроен")
        else:
            print("  ⚠️ ODDS_API_KEY не настроен (будут тестовые данные)")
        
        if 'TELEGRAM_TOKEN=' in content and 'your_' not in content.split('TELEGRAM_TOKEN=')[1][:30]:
            print("  ✅ Telegram настроен")
        else:
            print("  ⚠️ Telegram не настроен (уведомления отключены)")
        
        return True
    else:
        print("  ⚠️ .env файл не найден")
        if env_example.exists():
            print("  💡 Скопируйте .env.example в .env и настройте")
        return False


def check_database():
    """Проверяет базу данных"""
    db_path = Path(__file__).parent / 'autobasket.db'
    
    print("\n💾 Проверка базы данных:")
    
    if db_path.exists():
        size = db_path.stat().st_size / 1024
        print(f"  ✅ База данных найдена ({size:.1f} KB)")
    else:
        print("  ⚪ База данных будет создана при первом запуске")
    
    return True


def run_quick_test():
    """Запускает быстрый тест системы"""
    print("\n🧪 Быстрый тест системы:")
    
    try:
        # Импортируем модули
        sys.path.insert(0, str(Path(__file__).parent))
        
        from config.settings import config
        print("  ✅ Конфигурация загружена")
        
        from core.bankroll_manager import BankrollManager
        bm = BankrollManager(initial_bankroll=200)
        print(f"  ✅ Bankroll Manager (баланс: ${bm.bankroll:.2f})")
        
        from core.elo_system import EloRatingSystem
        elo = EloRatingSystem()
        print(f"  ✅ Elo System ({len(elo.ratings)} команд)")
        
        from core.prediction_engine import BasketballPredictor
        predictor = BasketballPredictor(use_ml=False)
        print("  ✅ Prediction Engine")
        
        from data.database import Database
        db = Database(db_path=":memory:")
        print("  ✅ Database")
        
        print("\n🎉 Все компоненты работают!")
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        return False


def print_usage():
    """Выводит инструкцию по использованию"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║           🏀 AutoBasket Betting Intelligence                 ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Команды запуска:                                           ║
║                                                              ║
║  1. Запуск системы:                                         ║
║     python main.py                                          ║
║                                                              ║
║  2. Веб-интерфейс:                                          ║
║     streamlit run dashboard.py                              ║
║                                                              ║
║  3. Тест конкретного модуля:                                ║
║     python -m core.bankroll_manager                         ║
║     python -m core.elo_system                               ║
║     python -m core.prediction_engine                        ║
║     python -m core.backtesting                              ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  📖 Документация: README.md                                 ║
║  ⚙️ Настройки: config/settings.py                          ║
║  🔑 API ключи: .env                                        ║
╚══════════════════════════════════════════════════════════════╝
""")


def main():
    """Главная функция"""
    print("=" * 60)
    print("🏀 AutoBasket - Проверка системы")
    print("=" * 60)
    
    # Проверки
    checks = [
        ("Python", check_python_version),
        ("Зависимости", check_dependencies),
        ("Конфигурация", check_env_file),
        ("База данных", check_database),
    ]
    
    all_passed = True
    for name, check_func in checks:
        if not check_func():
            all_passed = False
    
    # Быстрый тест
    if all_passed:
        run_quick_test()
    
    # Инструкция
    print_usage()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
