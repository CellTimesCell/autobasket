"""
AutoBasket - Prediction Engine
==============================
ML модели для предсказания исходов баскетбольных матчей
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    import xgboost as xgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: ML libraries not installed. Using simple prediction model.")

import sys
sys.path.append('..')
from config.settings import config


@dataclass
class GameFeatures:
    """Структура признаков для игры"""
    # Идентификация
    game_id: int = 0
    home_team: str = ""
    away_team: str = ""
    game_date: datetime = None
    
    # Базовая статистика (последние 10 игр)
    home_win_pct_last10: float = 0.5
    away_win_pct_last10: float = 0.5
    
    # Домашнее/выездное преимущество
    home_home_record: float = 0.5
    away_road_record: float = 0.5
    
    # Advanced metrics
    home_off_rating: float = 110.0
    home_def_rating: float = 110.0
    away_off_rating: float = 110.0
    away_def_rating: float = 110.0
    
    # Net Rating
    home_net_rating: float = 0.0
    away_net_rating: float = 0.0
    
    # Pace
    home_pace: float = 100.0
    away_pace: float = 100.0
    
    # Rest days
    home_rest_days: int = 1
    away_rest_days: int = 1
    
    # Back-to-back
    home_b2b: bool = False
    away_b2b: bool = False
    
    # Injury impact (0-10 scale)
    home_injury_impact: float = 0.0
    away_injury_impact: float = 0.0
    
    # Elo ratings
    home_elo: float = 1500.0
    away_elo: float = 1500.0
    
    # Streak (positive = wins, negative = losses)
    home_streak: int = 0
    away_streak: int = 0
    
    # ATS (Against The Spread) performance
    home_ats_pct: float = 0.5
    away_ats_pct: float = 0.5
    
    # Head-to-Head
    h2h_home_win_pct: float = 0.5
    h2h_games_count: int = 0
    
    # Average points (from knowledge base)
    avg_points_scored: float = 110.0
    avg_points_allowed: float = 110.0
    
    def to_feature_array(self) -> np.ndarray:
        """Преобразует в массив для ML модели"""
        features = [
            self.home_win_pct_last10,
            self.away_win_pct_last10,
            self.home_home_record,
            self.away_road_record,
            self.home_net_rating,
            self.away_net_rating,
            self.home_net_rating - self.away_net_rating,  # Net rating diff
            self.home_pace - self.away_pace,  # Pace diff
            self.home_rest_days,
            self.away_rest_days,
            self.home_rest_days - self.away_rest_days,  # Rest advantage
            int(self.home_b2b),
            int(self.away_b2b),
            self.home_injury_impact,
            self.away_injury_impact,
            self.home_injury_impact - self.away_injury_impact,  # Injury diff
            self.home_elo - self.away_elo,  # Elo diff
            self.home_streak,
            self.away_streak,
            self.home_ats_pct,
            self.away_ats_pct,
            self.h2h_home_win_pct,
        ]
        return np.array(features).reshape(1, -1)
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Возвращает имена признаков"""
        return [
            'home_win_pct_last10', 'away_win_pct_last10',
            'home_home_record', 'away_road_record',
            'home_net_rating', 'away_net_rating', 'net_rating_diff',
            'pace_diff',
            'home_rest_days', 'away_rest_days', 'rest_advantage',
            'home_b2b', 'away_b2b',
            'home_injury_impact', 'away_injury_impact', 'injury_diff',
            'elo_diff',
            'home_streak', 'away_streak',
            'home_ats_pct', 'away_ats_pct',
            'h2h_home_win_pct'
        ]


@dataclass
class Prediction:
    """Результат предсказания"""
    game_id: int
    home_team: str
    away_team: str
    
    home_win_prob: float
    away_win_prob: float
    
    predicted_winner: str
    confidence: float
    
    model_agreement: float  # Насколько модели согласны (0-1)
    individual_predictions: Dict[str, float] = field(default_factory=dict)
    
    # Дополнительные метрики
    predicted_margin: Optional[float] = None
    predicted_total: Optional[float] = None
    
    # Метаданные
    timestamp: datetime = field(default_factory=datetime.now)
    features_used: List[str] = field(default_factory=list)


class SimplePredictor:
    """
    Простой предиктор без ML (на случай если библиотеки не установлены)
    Использует взвешенные эвристики
    """
    
    def __init__(self):
        self.weights = {
            'elo': 0.30,
            'recent_form': 0.25,
            'home_advantage': 0.15,
            'net_rating': 0.20,
            'rest': 0.05,
            'injuries': 0.05
        }
        self.home_advantage_bonus = 0.035  # ~3.5% бонус дома
    
    def predict(self, features: GameFeatures) -> Prediction:
        """Делает предсказание на основе эвристик"""
        
        scores = {'home': 0.5, 'away': 0.5}
        
        # 1. Elo component
        elo_diff = features.home_elo - features.away_elo
        elo_prob = 1 / (1 + 10 ** (-elo_diff / 400))
        scores['home'] += (elo_prob - 0.5) * self.weights['elo']
        
        # 2. Recent form
        form_diff = features.home_win_pct_last10 - features.away_win_pct_last10
        scores['home'] += form_diff * self.weights['recent_form']
        
        # 3. Home advantage
        scores['home'] += self.home_advantage_bonus * self.weights['home_advantage']
        
        # 4. Net rating
        net_diff = features.home_net_rating - features.away_net_rating
        # Нормализуем (типичный разброс -15 до +15)
        net_factor = net_diff / 30  # Преобразуем в -0.5 до 0.5
        scores['home'] += net_factor * self.weights['net_rating']
        
        # 5. Rest advantage
        rest_diff = features.home_rest_days - features.away_rest_days
        rest_factor = min(max(rest_diff / 3, -0.1), 0.1)  # Ограничиваем влияние
        scores['home'] += rest_factor * self.weights['rest']
        
        # 6. Injuries
        injury_diff = features.away_injury_impact - features.home_injury_impact  # Инвертируем
        injury_factor = injury_diff / 20  # Нормализуем
        scores['home'] += injury_factor * self.weights['injuries']
        
        # Нормализуем вероятности
        home_prob = max(0.01, min(0.99, scores['home']))
        away_prob = 1 - home_prob
        
        # Определяем победителя и уверенность
        if home_prob > 0.5:
            winner = features.home_team
            confidence = home_prob
        else:
            winner = features.away_team
            confidence = away_prob
        
        return Prediction(
            game_id=features.game_id,
            home_team=features.home_team,
            away_team=features.away_team,
            home_win_prob=home_prob,
            away_win_prob=away_prob,
            predicted_winner=winner,
            confidence=confidence,
            model_agreement=1.0,  # Одна модель - 100% согласие
            individual_predictions={'heuristic': home_prob}
        )


class EnsemblePredictor:
    """
    Ensemble модель для предсказания исходов матчей
    Комбинирует XGBoost, Gradient Boosting и Random Forest
    """
    
    def __init__(self):
        if not ML_AVAILABLE:
            raise ImportError("ML libraries required for EnsemblePredictor")
        
        self.models = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                objective='binary:logistic',
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                random_state=42
            )
        }
        
        self.model_weights = config.prediction.model_weights
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_importance = {}
        
        # Для калибровки
        self.calibrated_models = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray, calibrate: bool = True):
        """
        Обучает все модели
        
        Args:
            X: Матрица признаков
            y: Целевая переменная (1 = home win, 0 = away win)
            calibrate: Применять ли калибровку вероятностей
        """
        # Нормализуем признаки
        X_scaled = self.scaler.fit_transform(X)
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_scaled, y)
            
            # Калибровка
            if calibrate:
                self.calibrated_models[name] = CalibratedClassifierCV(
                    model, method='isotonic', cv=3
                )
                self.calibrated_models[name].fit(X_scaled, y)
            
            # Feature importance для интерпретируемости
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
        
        self.is_fitted = True
        print("All models trained successfully!")
    
    def predict(self, features: GameFeatures, use_calibrated: bool = True) -> Prediction:
        """
        Делает предсказание с ensemble
        
        Args:
            features: Признаки игры
            use_calibrated: Использовать калиброванные модели
        
        Returns:
            Prediction объект
        """
        if not self.is_fitted:
            raise ValueError("Models not fitted. Call fit() first.")
        
        X = features.to_feature_array()
        X_scaled = self.scaler.transform(X)
        
        probabilities = {}
        models_to_use = self.calibrated_models if use_calibrated else self.models
        
        for name, model in models_to_use.items():
            try:
                proba = model.predict_proba(X_scaled)[0][1]  # P(home win)
                probabilities[name] = proba
            except Exception as e:
                print(f"Error in {name}: {e}")
                probabilities[name] = 0.5
        
        # Взвешенное среднее
        weighted_proba = sum(
            probabilities[name] * self.model_weights.get(name, 0.33)
            for name in probabilities
        )
        
        # Калибровка (сжатие к центру)
        calibrated_proba = self._calibrate_probability(weighted_proba)
        
        # Agreement между моделями
        agreement = self._calculate_agreement(probabilities)
        
        # Финальное предсказание
        if calibrated_proba > 0.5:
            winner = features.home_team
            confidence = calibrated_proba
        else:
            winner = features.away_team
            confidence = 1 - calibrated_proba
        
        return Prediction(
            game_id=features.game_id,
            home_team=features.home_team,
            away_team=features.away_team,
            home_win_prob=calibrated_proba,
            away_win_prob=1 - calibrated_proba,
            predicted_winner=winner,
            confidence=confidence,
            model_agreement=agreement,
            individual_predictions=probabilities,
            features_used=GameFeatures.get_feature_names()
        )
    
    def _calibrate_probability(self, raw_prob: float) -> float:
        """
        Калибровка вероятностей
        ML модели часто overconfident (дают 0.75 когда реально 0.65)
        """
        # Сжимаем к центру
        factor = config.prediction.calibration_factor
        calibrated = 0.5 + (raw_prob - 0.5) * factor
        return max(0.01, min(0.99, calibrated))
    
    def _calculate_agreement(self, probabilities: Dict[str, float]) -> float:
        """
        Насколько модели согласны между собой
        Высокое согласие = больше уверенности в предсказании
        """
        if len(probabilities) < 2:
            return 1.0
        
        probs = list(probabilities.values())
        variance = np.var(probs)
        
        # Преобразуем variance в agreement score
        # variance 0 = полное согласие, variance 0.1+ = сильное расхождение
        agreement = 1 - min(variance * 10, 1)
        return agreement
    
    def get_feature_importance(self) -> Dict[str, List[Tuple[str, float]]]:
        """Возвращает важность признаков для каждой модели"""
        feature_names = GameFeatures.get_feature_names()
        result = {}
        
        for model_name, importances in self.feature_importance.items():
            sorted_features = sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )
            result[model_name] = sorted_features
        
        return result


class BasketballPredictor:
    """
    Главный класс предиктора
    Автоматически выбирает между Ensemble и Simple в зависимости от доступности ML
    """
    
    def __init__(self, use_ml: bool = True):
        self.use_ml = use_ml and ML_AVAILABLE
        self.simple_predictor = SimplePredictor()  # Всегда держим как fallback
        
        if self.use_ml:
            try:
                self.predictor = EnsemblePredictor()
                self.ml_fitted = False  # Модели еще не обучены
                print("Using Ensemble ML Predictor")
            except Exception as e:
                print(f"Failed to initialize ML predictor: {e}")
                self.predictor = self.simple_predictor
                self.use_ml = False
        else:
            self.predictor = self.simple_predictor
            print("Using Simple Heuristic Predictor")
        
        self.prediction_history: List[Prediction] = []
    
    def predict(self, features: GameFeatures) -> Prediction:
        """Делает предсказание"""
        try:
            prediction = self.predictor.predict(features)
        except ValueError as e:
            # Если ML модели не обучены, используем simple predictor
            if "not fitted" in str(e):
                prediction = self.simple_predictor.predict(features)
            else:
                raise e
        
        self.prediction_history.append(prediction)
        return prediction
    
    def predict_batch(self, games: List[GameFeatures]) -> List[Prediction]:
        """Предсказания для нескольких игр"""
        return [self.predict(game) for game in games]
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Обучает модель (только для ML predictor)"""
        if self.use_ml and isinstance(self.predictor, EnsemblePredictor):
            self.predictor.fit(X, y)
            self.ml_fitted = True
        else:
            print("Training not available for simple predictor")
    
    def evaluate_accuracy(self, predictions: List[Prediction], actual_results: List[bool]) -> Dict:
        """
        Оценивает точность предсказаний
        
        Args:
            predictions: Список предсказаний
            actual_results: Список результатов (True = home won)
        
        Returns:
            Метрики качества
        """
        if len(predictions) != len(actual_results):
            raise ValueError("Length mismatch")
        
        correct = 0
        total_log_loss = 0
        total_brier = 0
        
        for pred, actual in zip(predictions, actual_results):
            predicted_home = pred.home_win_prob > 0.5
            
            if predicted_home == actual:
                correct += 1
            
            # Log loss
            prob = pred.home_win_prob if actual else pred.away_win_prob
            total_log_loss -= np.log(max(prob, 1e-10))
            
            # Brier score
            total_brier += (pred.home_win_prob - int(actual)) ** 2
        
        n = len(predictions)
        
        return {
            'accuracy': correct / n,
            'log_loss': total_log_loss / n,
            'brier_score': total_brier / n,
            'total_predictions': n,
            'correct': correct
        }


# === MARGIN & TOTAL PREDICTION ===

class MarginPredictor:
    """
    Предсказывает margin of victory (разницу в счете)
    Используется для spread betting
    """
    
    def __init__(self):
        self.home_advantage_points = config.prediction.home_advantage_points
        self.margin_std = 12.0  # Стандартное отклонение margin в NBA
    
    def predict_margin(self, features: GameFeatures) -> Dict:
        """
        Предсказывает ожидаемую разницу в счете
        
        Положительное значение = home wins by X points
        Отрицательное = away wins by X points
        """
        # Базовая формула: Net Rating Diff * ~0.4 + Home Advantage
        net_diff = features.home_net_rating - features.away_net_rating
        base_margin = net_diff * 0.4 + self.home_advantage_points
        
        # Корректировки
        rest_adjustment = (features.home_rest_days - features.away_rest_days) * 0.5
        injury_adjustment = (features.away_injury_impact - features.home_injury_impact) * 0.3
        
        predicted_margin = base_margin + rest_adjustment + injury_adjustment
        
        return {
            'predicted_margin': predicted_margin,
            'std_dev': self.margin_std,
            'confidence_interval_90': (
                predicted_margin - 1.645 * self.margin_std,
                predicted_margin + 1.645 * self.margin_std
            )
        }
    
    def calculate_spread_probability(
        self, 
        features: GameFeatures, 
        spread_line: float
    ) -> Dict:
        """
        Вероятность покрытия spread
        
        Args:
            features: Признаки игры
            spread_line: Линия spread (например, -5.5 для фаворита)
        
        Returns:
            Вероятности cover/not cover
        """
        from scipy import stats
        
        margin = self.predict_margin(features)
        predicted = margin['predicted_margin']
        
        # Z-score для spread line
        z = (predicted - spread_line) / self.margin_std
        
        # P(margin > spread_line)
        cover_prob = stats.norm.cdf(z)
        
        return {
            'cover_probability': cover_prob,
            'not_cover_probability': 1 - cover_prob,
            'predicted_margin': predicted,
            'spread_line': spread_line,
            'edge': predicted - spread_line
        }


class TotalPredictor:
    """
    Предсказывает total points (общее количество очков)
    """
    
    def __init__(self):
        self.total_std = 15.0  # Стандартное отклонение totals
    
    def predict_total(self, features: GameFeatures) -> Dict:
        """
        Предсказывает общее количество очков в игре
        """
        # Средний pace
        avg_pace = (features.home_pace + features.away_pace) / 2
        possessions = avg_pace * 0.98  # Корректировка
        
        # Expected points per team
        # Home team vs Away defense
        home_expected = possessions * (
            features.home_off_rating + (115 - features.away_def_rating)
        ) / 200
        
        # Away team vs Home defense
        away_expected = possessions * (
            features.away_off_rating + (115 - features.home_def_rating)
        ) / 200
        
        predicted_total = home_expected + away_expected
        
        return {
            'predicted_total': predicted_total,
            'home_expected': home_expected,
            'away_expected': away_expected,
            'std_dev': self.total_std
        }
    
    def calculate_over_under_probability(
        self,
        features: GameFeatures,
        total_line: float
    ) -> Dict:
        """
        Вероятность Over/Under
        """
        from scipy import stats
        
        total = self.predict_total(features)
        predicted = total['predicted_total']
        
        # Z-score
        z = (predicted - total_line) / self.total_std
        
        over_prob = 1 - stats.norm.cdf(z)  # P(actual > line)
        
        return {
            'over_probability': over_prob,
            'under_probability': 1 - over_prob,
            'predicted_total': predicted,
            'total_line': total_line,
            'recommendation': 'over' if over_prob > 0.55 else 'under' if over_prob < 0.45 else 'pass'
        }


# === ТЕСТИРОВАНИЕ ===

if __name__ == "__main__":
    print("=== Тест Prediction Engine ===\n")
    
    # Создаем тестовые данные
    test_features = GameFeatures(
        game_id=1001,
        home_team="Lakers",
        away_team="Warriors",
        game_date=datetime.now(),
        home_win_pct_last10=0.7,
        away_win_pct_last10=0.6,
        home_home_record=0.75,
        away_road_record=0.45,
        home_off_rating=115.5,
        home_def_rating=108.2,
        away_off_rating=118.0,
        away_def_rating=112.5,
        home_net_rating=7.3,
        away_net_rating=5.5,
        home_pace=100.5,
        away_pace=102.3,
        home_rest_days=2,
        away_rest_days=1,
        home_b2b=False,
        away_b2b=False,
        home_injury_impact=0.5,
        away_injury_impact=2.0,
        home_elo=1650,
        away_elo=1620,
        home_streak=3,
        away_streak=-1,
        home_ats_pct=0.55,
        away_ats_pct=0.48,
        h2h_home_win_pct=0.6,
        h2h_games_count=10
    )
    
    # Тест Simple Predictor
    print("Тест Simple Predictor:")
    print("-" * 40)
    
    simple = SimplePredictor()
    pred = simple.predict(test_features)
    
    print(f"Матч: {pred.home_team} vs {pred.away_team}")
    print(f"Вероятность победы {pred.home_team}: {pred.home_win_prob:.1%}")
    print(f"Вероятность победы {pred.away_team}: {pred.away_win_prob:.1%}")
    print(f"Прогноз: {pred.predicted_winner} ({pred.confidence:.1%})")
    
    # Тест Margin Predictor
    print("\n\nТест Margin Predictor:")
    print("-" * 40)
    
    margin_pred = MarginPredictor()
    margin = margin_pred.predict_margin(test_features)
    
    print(f"Прогнозируемый margin: {margin['predicted_margin']:+.1f}")
    print(f"90% CI: ({margin['confidence_interval_90'][0]:.1f}, {margin['confidence_interval_90'][1]:.1f})")
    
    spread = margin_pred.calculate_spread_probability(test_features, spread_line=-3.5)
    print(f"\nSpread -3.5:")
    print(f"  P(Cover): {spread['cover_probability']:.1%}")
    print(f"  Edge: {spread['edge']:+.1f}")
    
    # Тест Total Predictor
    print("\n\nТест Total Predictor:")
    print("-" * 40)
    
    total_pred = TotalPredictor()
    total = total_pred.predict_total(test_features)
    
    print(f"Прогнозируемый total: {total['predicted_total']:.1f}")
    print(f"  Home: {total['home_expected']:.1f}")
    print(f"  Away: {total['away_expected']:.1f}")
    
    ou = total_pred.calculate_over_under_probability(test_features, total_line=225.5)
    print(f"\nO/U 225.5:")
    print(f"  P(Over): {ou['over_probability']:.1%}")
    print(f"  Рекомендация: {ou['recommendation']}")
    
    # Тест BasketballPredictor (главный класс)
    print("\n\nТест BasketballPredictor:")
    print("-" * 40)
    
    predictor = BasketballPredictor(use_ml=False)  # Используем simple для теста
    full_pred = predictor.predict(test_features)
    
    print(f"Финальный прогноз: {full_pred.predicted_winner}")
    print(f"Уверенность: {full_pred.confidence:.1%}")
    print(f"Agreement моделей: {full_pred.model_agreement:.1%}")
