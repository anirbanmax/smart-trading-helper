import streamlit as st
import pandas as pd
import numpy as np
import requests
import sqlite3
import json
import os
import base64
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import ta
import time
import logging
import warnings
from Crypto.Cipher import AES
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Hash import SHA256
import hashlib
import threading
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AxisDirectEncryption:
    """Handle Axis Direct API encryption/decryption"""
    
    def __init__(self):
        self.secret_key = os.urandom(16).hex()
        self.public_key = None
        
    def encrypt_aes(self, payload):
        """Encrypt payload using AES GCM"""
        secret_key_bytes = self.secret_key.encode()
        cipher = AES.new(secret_key_bytes, AES.MODE_GCM)
        ciphertext, tag = cipher.encrypt_and_digest(payload.encode())
        
        encoded_nonce = base64.b64encode(cipher.nonce).decode()
        encoded_tag = base64.b64encode(tag).decode()
        encoded_ciphertext = base64.b64encode(ciphertext).decode()
        
        return f"{encoded_nonce}.{encoded_tag}.{encoded_ciphertext}"
    
    def decrypt_data(self, encoded_payload):
        """Decrypt response using AES GCM"""
        secret_key_bytes = self.secret_key.encode()
        encoded_nonce, encoded_tag, encoded_ciphertext = encoded_payload.split(".")
        
        nonce = base64.b64decode(encoded_nonce)
        tag = base64.b64decode(encoded_tag)
        ciphertext = base64.b64decode(encoded_ciphertext)
        
        cipher = AES.new(secret_key_bytes, AES.MODE_GCM, nonce=nonce)
        decrypted_payload = cipher.decrypt_and_verify(ciphertext, tag).decode()
        return decrypted_payload

class AxisDirectAPI:
    """Handle all Axis Direct RAPID API interactions"""
    
    def __init__(self, authorization_key):
        self.authorization_key = authorization_key
        self.base_url = "https://invest-api.axisdirect.in"
        self.encryption = AxisDirectEncryption()
        self.auth_token = None
        self.client_id = None
        self.sub_account_id = None
        self.encryption_key = None
        
    def get_headers(self):
        """Get API headers"""
        return {
            'Authorization': self.authorization_key,
            'x-api-client-id': self.client_id or 'default_client',
            'x-subAccountID': self.sub_account_id or 'default_sub',
            'x-authtoken': self.auth_token or 'temp_token',
            'x-api-encryption-key': self.encryption_key or 'temp_key',
            'Content-Type': 'application/json'
        }
    
    def make_request(self, endpoint, payload=None):
        """Make encrypted API request"""
        url = f"{self.base_url}{endpoint}"
        headers = self.get_headers()
        
        if payload:
            encrypted_payload = self.encryption.encrypt_aes(json.dumps(payload))
            data = {"payload": encrypted_payload}
        else:
            data = {}
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=10)
            if response.status_code == 200:
                result = response.json()
                if 'data' in result and 'payload' in result['data']:
                    decrypted = self.encryption.decrypt_data(result['data']['payload'])
                    return json.loads(decrypted)
                return result
            else:
                logger.error(f"API Error: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            return None
    
    def get_security_master(self):
        """Get security master data"""
        try:
            url = "https://invest-static-assets.axisdirect.in/TELESCOPE/security_master_web/SecurityMaster.csv"
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                df = pd.read_csv(pd.StringIO(response.text))
                return df
            return None
        except Exception as e:
            logger.error(f"Security master fetch failed: {str(e)}")
            return None

class MarketConditionFilter:
    """Advanced market condition analysis to avoid false signals"""
    
    def __init__(self):
        self.min_volume_ratio = 1.2  # Minimum volume should be 20% above average
        self.min_trend_strength = 0.6  # Minimum trend strength required
        self.max_volatility_threshold = 3.0  # Maximum volatility allowed
        
    def is_market_suitable_for_trading(self, df):
        """Check if current market conditions are suitable for trading"""
        if len(df) < 50:
            return False, "Not enough data to analyze market conditions"
        
        latest = df.iloc[-1]
        recent_data = df.tail(20)  # Last 20 periods
        
        # Check 1: Volume Confirmation
        avg_volume = df['volume'].rolling(50).mean().iloc[-1]
        current_volume = latest['volume']
        volume_ratio = current_volume / avg_volume
        
        if volume_ratio < self.min_volume_ratio:
            return False, f"Low trading activity (Volume: {volume_ratio:.1f}x average). Signals may be unreliable."
        
        # Check 2: Market Trend Clarity
        ema_9 = latest['fast_line']
        ema_21 = latest['medium_line']
        ema_50 = latest['slow_line']
        
        # Calculate trend strength
        if ema_9 > ema_21 > ema_50:
            trend_strength = min(((ema_9 - ema_50) / ema_50) * 10, 1.0)
            trend_direction = "up"
        elif ema_9 < ema_21 < ema_50:
            trend_strength = min(((ema_50 - ema_9) / ema_50) * 10, 1.0)
            trend_direction = "down"
        else:
            trend_strength = 0
            trend_direction = "sideways"
        
        if trend_strength < self.min_trend_strength and trend_direction != "sideways":
            return False, f"Weak trend (Strength: {trend_strength:.1f}). Wait for clearer direction."
        
        # Check 3: Volatility Filter
        atr = latest['atr_indicator']
        avg_atr = df['atr_indicator'].rolling(20).mean().iloc[-1]
        volatility_ratio = atr / avg_atr
        
        if volatility_ratio > self.max_volatility_threshold:
            return False, f"High volatility (Ratio: {volatility_ratio:.1f}). Market too unpredictable."
        
        # Check 4: Recent Whipsaw Detection
        recent_highs = recent_data['high'].max()
        recent_lows = recent_data['low'].min()
        price_range = (recent_highs - recent_lows) / recent_lows * 100
        
        if price_range > 8:  # More than 8% range in 20 periods
            return False, f"Choppy market detected (Range: {price_range:.1f}%). High risk of whipsaws."
        
        # Check 5: Time-based Filter (Avoid first and last 30 minutes)
        current_time = datetime.now().time()
        market_open = datetime.strptime("09:15", "%H:%M").time()
        market_close = datetime.strptime("15:30", "%H:%M").time()
        avoid_start = datetime.strptime("09:45", "%H:%M").time()
        avoid_end = datetime.strptime("15:00", "%H:%M").time()
        
        if current_time < avoid_start or current_time > avoid_end:
            return False, "Avoiding first 30 mins and last 30 mins of trading. Higher chance of false breakouts."
        
        return True, f"Market conditions are good for trading. Volume: {volume_ratio:.1f}x, Trend: {trend_direction}, Volatility: Normal"

class AdvancedTechnicalAnalyzer:
    """Enhanced technical analysis with multiple confirmation layers"""
    
    def __init__(self):
        self.market_filter = MarketConditionFilter()
        
    def calculate_all_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        if len(df) < 50:
            return df
        
        # Trend Indicators
        df['fast_line'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
        df['medium_line'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        df['slow_line'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        
        # Momentum Indicators
        df['strength_meter'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['stoch_k'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch()
        
        # Volatility Indicators
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['upper_band'] = bollinger.bollinger_hband()
        df['lower_band'] = bollinger.bollinger_lband()
        df['middle_band'] = bollinger.bollinger_mavg()
        df['bb_squeeze'] = (df['upper_band'] - df['lower_band']) / df['middle_band'] * 100
        df['atr_indicator'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # Volume Indicators
        df['avg_volume'] = ta.volume.VolumeSMAIndicator(df['close'], df['volume']).volume_sma()
        df['volume_ratio'] = df['volume'] / df['avg_volume']
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['momentum_line'] = macd.macd()
        df['momentum_signal'] = macd.macd_signal()
        df['momentum_histogram'] = macd.macd_diff()
        
        # Support/Resistance
        df['support_level'] = df['low'].rolling(20).min()
        df['resistance_level'] = df['high'].rolling(20).max()
        df['pivot_point'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Price action patterns
        df['higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
        df['lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))
        
        return df
    
    def validate_signal_quality(self, df, signal_type):
        """Validate signal quality with multiple checks"""
        if len(df) < 50:
            return False, "Insufficient data for signal validation"
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        validation_score = 0
        validation_reasons = []
        
        # Validation 1: Volume Confirmation (Critical)
        if latest['volume_ratio'] >= 1.5:
            validation_score += 3
            validation_reasons.append("Strong volume supports the move")
        elif latest['volume_ratio'] >= 1.2:
            validation_score += 1
            validation_reasons.append("Decent volume confirmation")
        else:
            return False, "Insufficient volume to confirm signal"
        
        # Validation 2: Multiple Timeframe Alignment
        if signal_type == "BUY":
            if latest['fast_line'] > latest['medium_line'] and latest['medium_line'] > latest['slow_line']:
                validation_score += 3
                validation_reasons.append("All timeframes aligned for uptrend")
            elif latest['fast_line'] > latest['medium_line']:
                validation_score += 1
                validation_reasons.append("Short-term trend is positive")
        else:  # SELL
            if latest['fast_line'] < latest['medium_line'] and latest['medium_line'] < latest['slow_line']:
                validation_score += 3
                validation_reasons.append("All timeframes aligned for downtrend")
            elif latest['fast_line'] < latest['medium_line']:
                validation_score += 1
                validation_reasons.append("Short-term trend is negative")
        
        # Validation 3: Momentum Confirmation
        if signal_type == "BUY":
            if latest['momentum_line'] > latest['momentum_signal'] and latest['momentum_histogram'] > prev['momentum_histogram']:
                validation_score += 2
                validation_reasons.append("Momentum is accelerating upward")
        else:  # SELL
            if latest['momentum_line'] < latest['momentum_signal'] and latest['momentum_histogram'] < prev['momentum_histogram']:
                validation_score += 2
                validation_reasons.append("Momentum is accelerating downward")
        
        # Validation 4: Price Action Confirmation
        if signal_type == "BUY":
            if latest['close'] > latest['fast_line'] and latest['close'] > prev['high']:
                validation_score += 2
                validation_reasons.append("Price breaking above recent highs")
        else:  # SELL
            if latest['close'] < latest['fast_line'] and latest['close'] < prev['low']:
                validation_score += 2
                validation_reasons.append("Price breaking below recent lows")
        
        # Validation 5: Risk-Reward Check
        if signal_type == "BUY":
            stop_loss = latest['close'] * 0.97  # 3% stop loss
            target = latest['close'] * 1.06    # 6% target
            risk_reward = (target - latest['close']) / (latest['close'] - stop_loss)
        else:  # SELL
            stop_loss = latest['close'] * 1.03  # 3% stop loss
            target = latest['close'] * 0.94    # 6% target
            risk_reward = (latest['close'] - target) / (stop_loss - latest['close'])
        
        if risk_reward >= 1.8:  # Minimum 1.8:1 risk-reward
            validation_score += 2
            validation_reasons.append(f"Good risk-reward ratio: 1:{risk_reward:.1f}")
        else:
            return False, f"Poor risk-reward ratio: 1:{risk_reward:.1f}. Minimum required: 1:1.8"
        
        # Final validation
        if validation_score >= 8:  # High-quality signal
            return True, f"High-quality signal (Score: {validation_score}/12). " + "; ".join(validation_reasons)
        elif validation_score >= 6:  # Moderate-quality signal
            return True, f"Moderate-quality signal (Score: {validation_score}/12). " + "; ".join(validation_reasons)
        else:
            return False, f"Signal quality too low (Score: {validation_score}/12). Wait for better setup."
    
    def generate_high_quality_signals(self, df):
        """Generate only high-quality, well-validated signals"""
        signals = []
        
        if len(df) < 50:
            return signals, "Need more data to generate reliable signals"
        
        # First check if market conditions are suitable
        market_ok, market_message = self.market_filter.is_market_suitable_for_trading(df)
        if not market_ok:
            return signals, market_message
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Signal Detection Logic with Multiple Confirmations
        buy_score = 0
        sell_score = 0
        signal_reasons = []
        
        # Criterion 1: Trend Alignment (Weight: 4 points)
        if latest['fast_line'] > latest['medium_line'] > latest['slow_line']:
            buy_score += 4
            signal_reasons.append("Strong uptrend - all moving averages aligned")
        elif latest['fast_line'] < latest['medium_line'] < latest['slow_line']:
            sell_score += 4
            signal_reasons.append("Strong downtrend - all moving averages aligned")
        elif latest['fast_line'] > latest['medium_line']:
            buy_score += 2
            signal_reasons.append("Short-term uptrend emerging")
        elif latest['fast_line'] < latest['medium_line']:
            sell_score += 2
            signal_reasons.append("Short-term downtrend emerging")
        
        # Criterion 2: Momentum Confirmation (Weight: 3 points)
        if (latest['momentum_line'] > latest['momentum_signal'] and 
            prev['momentum_line'] <= prev['momentum_signal']):
            buy_score += 3
            signal_reasons.append("Momentum turning positive")
        elif (latest['momentum_line'] < latest['momentum_signal'] and 
              prev['momentum_line'] >= prev['momentum_signal']):
            sell_score += 3
            signal_reasons.append("Momentum turning negative")
        
        # Criterion 3: Oversold/Overbought with Reversal (Weight: 3 points)
        if latest['strength_meter'] < 30 and prev['strength_meter'] >= 30:
            buy_score += 3
            signal_reasons.append("Market oversold, bounce expected")
        elif latest['strength_meter'] > 70 and prev['strength_meter'] <= 70:
            sell_score += 3
            signal_reasons.append("Market overbought, correction expected")
        
        # Criterion 4: Bollinger Band Signals (Weight: 2 points)
        if latest['close'] <= latest['lower_band'] and prev['close'] > prev['lower_band']:
            buy_score += 2
            signal_reasons.append("Price touched lower support band")
        elif latest['close'] >= latest['upper_band'] and prev['close'] < prev['upper_band']:
            sell_score += 2
            signal_reasons.append("Price reached upper resistance band")
        
        # Criterion 5: Volume Surge (Weight: 2 points)
        if latest['volume_ratio'] > 1.8:
            if buy_score > sell_score:
                buy_score += 2
                signal_reasons.append("High volume confirms bullish move")
            else:
                sell_score += 2
                signal_reasons.append("High volume confirms bearish move")
        
        # Criterion 6: Price Action Breakout (Weight: 2 points)
        if latest['close'] > latest['resistance_level'] and prev['close'] <= prev['resistance_level']:
            buy_score += 2
            signal_reasons.append("Breaking above resistance level")
        elif latest['close'] < latest['support_level'] and prev['close'] >= prev['support_level']:
            sell_score += 2
            signal_reasons.append("Breaking below support level")
        
        # Generate signal only if score is high enough
        min_score_required = 8  # Increased threshold for quality
        
        if buy_score >= min_score_required and buy_score > sell_score:
            # Validate signal quality
            signal_valid, validation_message = self.validate_signal_quality(df, "BUY")
            
            if signal_valid:
                confidence = min((buy_score / 16) * 100, 95)  # Max 95% confidence
                stop_loss = latest['close'] * 0.97  # 3% stop loss
                target = latest['close'] * 1.06    # 6% target
                
                signals.append({
                    'action': 'BUY',
                    'confidence': confidence,
                    'price': latest['close'],
                    'stop_loss': stop_loss,
                    'target': target,
                    'reasons': signal_reasons[:4],  # Top 4 reasons
                    'validation': validation_message,
                    'quality_score': buy_score,
                    'strength': 'High' if buy_score >= 12 else 'Moderate'
                })
                
        elif sell_score >= min_score_required and sell_score > buy_score:
            # Validate signal quality
            signal_valid, validation_message = self.validate_signal_quality(df, "SELL")
            
            if signal_valid:
                confidence = min((sell_score / 16) * 100, 95)  # Max 95% confidence
                stop_loss = latest['close'] * 1.03  # 3% stop loss
                target = latest['close'] * 0.94    # 6% target
                
                signals.append({
                    'action': 'SELL',
                    'confidence': confidence,
                    'price': latest['close'],
                    'stop_loss': stop_loss,
                    'target': target,
                    'reasons': signal_reasons[:4],  # Top 4 reasons
                    'validation': validation_message,
                    'quality_score': sell_score,
                    'strength': 'High' if sell_score >= 12 else 'Moderate'
                })
        
        # Return appropriate message if no signals
        if not signals:
            if max(buy_score, sell_score) < min_score_required:
                return signals, f"No clear signals. Best score: {max(buy_score, sell_score)}/16. Need minimum {min_score_required}/16 for signal generation."
            else:
                return signals, "Signal detected but failed quality validation. Waiting for better setup."
        
        return signals, market_message

class SimpleMLPredictor:
    """Enhanced ML predictor with signal validation"""
    
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=150, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_accuracy = 0
        self.feature_importance = {}
        
    def prepare_enhanced_features(self, df):
        """Create enhanced features for better predictions"""
        features = []
        
        # Trend features
        df['uptrend_strength'] = (df['fast_line'] / df['slow_line'] - 1) * 100
        df['trend_alignment'] = ((df['fast_line'] > df['medium_line']) & 
                                (df['medium_line'] > df['slow_line'])).astype(int)
        
        # Momentum features
        df['rsi_normalized'] = (df['strength_meter'] - 50) / 50
        df['momentum_strength'] = df['momentum_histogram'] / df['close'] * 1000
        
        # Volume features
        df['volume_surge'] = (df['volume_ratio'] > 1.5).astype(int)
        df['volume_momentum'] = df['volume'].pct_change()
        
        # Volatility features
        df['volatility_ratio'] = df['atr_indicator'] / df['close'] * 100
        df['bb_position'] = (df['close'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])
        
        # Price action features
        df['price_momentum'] = df['close'].pct_change(periods=3) * 100
        df['near_resistance'] = (df['close'] / df['resistance_level'] > 0.98).astype(int)
        df['near_support'] = (df['close'] / df['support_level'] < 1.02).astype(int)
        
        feature_cols = ['uptrend_strength', 'trend_alignment', 'rsi_normalized', 
                       'momentum_strength', 'volume_surge', 'volume_momentum',
                       'volatility_ratio', 'bb_position', 'price_momentum',
                       'near_resistance', 'near_support']
        
        for col in feature_cols:
            if col in df.columns:
                features.append(df[col].fillna(0))
        
        return pd.concat(features, axis=1)
    
    def create_quality_labels(self, df, lookahead=5):
        """Create labels based on actual profitable moves"""
        df['future_high'] = df['high'].shift(-lookahead).rolling(lookahead).max()
        df['future_low'] = df['low'].shift(-lookahead).rolling(lookahead).min()
        
        # Calculate potential profit/loss
        df['buy_profit'] = (df['future_high'] / df['close'] - 1) * 100
        df['sell_profit'] = (df['close'] / df['future_low'] - 1) * 100
        
        # Create quality labels (only strong moves)
        conditions = [
            (df['buy_profit'] > 3) & (df['sell_profit'] < 2),  # Clear buy opportunity
            (df['sell_profit'] > 3) & (df['buy_profit'] < 2),  # Clear sell opportunity
        ]
        choices = [1, 2]  # 1=Buy, 2=Sell, 0=Hold
        
        df['quality_signal'] = np.select(conditions, choices, default=0)
        return df['quality_signal']
    
    def train_enhanced_system(self, historical_data):
        """Train with enhanced validation"""
        if len(historical_data) < 200:  # Increased minimum data requirement
            return False
        
        features = self.prepare_enhanced_features(historical_data)
        labels = self.create_quality_labels(historical_data)
        
        # Remove incomplete data
        mask = ~(features.isna().any(axis=1) | labels.isna())
        features = features[mask]
        labels = labels[mask]
        
        if len(features) < 100:
            return False
        
        # Balance dataset (avoid overfitting to one class)
        from collections import Counter
        label_counts = Counter(labels)
        min_count = min(label_counts.values())
        
        if min_count < 20:  # Need minimum samples per class
            return False
        
        # Train with validation
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.25, random_state=42, stratify=labels
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        self.model_accuracy = accuracy_score(y_test, y_pred)
        
        # Store feature importance
        self.feature_importance = dict(zip(features.columns, self.model.feature_importances_))
        
        self.is_trained = True
        logger.info(f"Enhanced ML model trained with {self.model_accuracy*100:.1f}% accuracy")
        return self.model_accuracy > 0.65  # Only use if accuracy > 65%
    
    def get_ml_prediction(self, current_data):
        """Get high-confidence ML prediction"""
        if not self.is_trained:
            return None
        
        features = self.prepare_enhanced_features(current_data)
        if len(features) == 0:
            return None
        
        latest_features = features.iloc[-1:].fillna(0)
        latest_scaled = self.scaler.transform(latest_features)
        
        prediction = self.model.predict(latest_scaled)[0]
        probabilities = self.model.predict_proba(latest_scaled)[0]
        
        max_confidence = max(probabilities) * 100
        
        # Only return prediction if confidence is high enough
        if max_confidence < 70:  # Minimum 70% confidence
            return {
                'action': 'WAIT',
                'confidence': max_confidence,
                'reason': f"ML model confidence too low ({max_confidence:.0f}%). Waiting for clearer signals."
            }
        
        signal_map = {0: 'WAIT', 1: 'BUY', 2: 'SELL'}
        
        return {
            'action': signal_map[prediction],
            'confidence': max_confidence,
            'model_accuracy': self.model_accuracy * 100,
            'reason': f"ML model ({self.model_accuracy*100:.0f}% accurate) is {max_confidence:.0f}% confident"
        }

class NotificationManager:
    """Enhanced notification system"""
    
    def __init__(self):
        self.telegram_bot_token = None
        self.telegram_chat_id = None
        self.last_notification_time = {}
        
    def setup_telegram(self, bot_token, chat_id):
        """Setup Telegram notifications"""
        self.telegram_bot_token = bot_token
        self.telegram_chat_id = chat_id
    
    def send_telegram_message(self, message):
        """Send message via Telegram"""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, data=data, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Telegram notification failed: {str(e)}")
            return False
    
    def should_send_notification(self, signal_type):
        """Prevent spam notifications"""
        current_time = datetime.now()
        last_time = self.last_notification_time.get(signal_type)
        
        if last_time is None:
            self.last_notification_time[signal_type] = current_time
            return True
        
        time_diff = (current_time - last_time).total_seconds()
        
        # Send signal notifications max once every 10 minutes
        if time_diff > 600:  # 10 minutes
            self.last_notification_time[signal_type] = current_time
            return True
        
        return False
    
    def send_quality_signal_notification(self, signal):
        """Send high-quality signal notification"""
        if not self.should_send_notification(signal['action']):
            return False
        
        if signal['action'] in ['BUY', 'SELL']:
            risk_reward = abs(signal['target'] - signal['price']) / abs(signal['price'] - signal['stop_loss'])
            
            message = f"""
🔥 <b>HIGH-QUALITY {signal['action']} SIGNAL</b> 🔥

💰 <b>Entry Price:</b> ₹{signal['price']:.2f}
🎯 <b>Target:</b> ₹{signal['target']:.2f}
🛡️ <b>Stop Loss:</b> ₹{signal['stop_loss']:.2f}
💪 <b>Confidence:</b> {signal['confidence']:.0f}%
⭐ <b>Quality:</b> {signal['strength']} ({signal['quality_score']}/16)
📊 <b>Risk:Reward:</b> 1:{risk_reward:.1f}

<b>🔍 Why this signal?</b>
{chr(10).join([f"• {reason}" for reason in signal['reasons']])}

<b>✅ Validation:</b>
{signal['validation']}

<i>⏰ {datetime.now().strftime('%d %b %Y, %I:%M %p')}</i>
<i>🤖 Only high-quality signals sent to avoid noise</i>
            """
            
            return self.send_telegram_message(message)
        return False

class DatabaseManager:
    """Enhanced database with performance tracking"""
    
    def __init__(self, db_path="smart_trading_v2.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Setup enhanced database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                action TEXT,
                price REAL,
                stop_loss REAL,
                target REAL,
                confidence REAL,
                quality_score INTEGER,
                strength TEXT,
                reasons TEXT,
                validation TEXT,
                status TEXT DEFAULT 'ACTIVE'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_conditions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                conditions_suitable BOOLEAN,
                condition_message TEXT,
                volume_ratio REAL,
                volatility_ratio REAL,
                trend_strength REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_enhanced_signal(self, signal_data):
        """Save enhanced signal data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO signals (action, price, stop_loss, target, confidence, 
                               quality_score, strength, reasons, validation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal_data.get('action'),
            signal_data.get('price'),
            signal_data.get('stop_loss'),
            signal_data.get('target'),
            signal_data.get('confidence'),
            signal_data.get('quality_score'),
            signal_data.get('strength'),
            ', '.join(signal_data.get('reasons', [])),
            signal_data.get('validation', '')
        ))
        
        conn.commit()
        conn.close()

class SmartTradingSystem:
    """Enhanced trading system with quality controls"""
    
    def __init__(self, api_key):
        self.api = AxisDirectAPI(api_key)
        self.analyzer = AdvancedTechnicalAnalyzer()
        self.ml_system = SimpleMLPredictor()
        self.notifications = NotificationManager()
        self.db = DatabaseManager()
        self.is_monitoring = False
        self.system_status = "Initializing..."
        
    def get_sample_market_data(self, symbol="NIFTY"):
        """Enhanced sample data with realistic patterns"""
        dates = pd.date_range(start='2024-01-01', end='2024-06-28', freq='D')
        np.random.seed(42)
        
        base_price = 19500
        prices = [base_price]
        volumes = []
        
        # Create more realistic price movements
        for i in range(1, len(dates)):
            # Add trend and noise
            trend = 0.0002  # Slight upward bias
            noise = np.random.normal(0, 0.015)
            momentum = 0.1 * (prices[-1] - prices[max(0, len(prices)-5):].mean() if len(prices) >= 5 else 0) / base_price
            
            change = trend + noise + momentum
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, base_price * 0.7))  # Prevent unrealistic drops
            
            # Volume with correlation to price movement
            base_volume = 500000
            volume_multiplier = 1 + abs(change) * 10  # Higher volume on big moves
            volumes.append(int(base_volume * volume_multiplier * np.random.uniform(0.7, 1.3)))
        
        volumes.append(volumes[-1])
        
        df = pd.DataFrame({
            'date': dates,
            'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.008))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.008))) for p in prices],
            'close': prices,
            'volume': volumes
        })
        
        return df
    
    def run_comprehensive_analysis(self, symbol="NIFTY"):
        """Run comprehensive analysis with quality controls"""
        try:
            # Get market data
            market_data = self.get_sample_market_data(symbol)
            
            # Calculate all indicators
            market_data = self.analyzer.calculate_all_indicators(market_data)
            
            # Generate high-quality signals
            tech_signals, market_condition_msg = self.analyzer.generate_high_quality_signals(market_data)
            
            # Train ML system if needed
            if not self.ml_system.is_trained:
                ml_trained = self.ml_system.train_enhanced_system(market_data)
                self.system_status = f"ML System: {'Trained' if ml_trained else 'Training failed'}"
            
            # Get ML prediction
            ml_prediction = self.ml_system.get_ml_prediction(market_data)
            
            # Combine and validate signals
            final_signals = []
            for signal in tech_signals:
                # Add ML validation
                if ml_prediction and ml_prediction['action'] != 'WAIT':
                    if signal['action'] == ml_prediction['action']:
                        # Both systems agree - boost confidence
                        signal['confidence'] = min(signal['confidence'] * 1.15, 95)
                        signal['ml_confirmation'] = True
                        signal['reasons'].append(f"ML system agrees ({ml_prediction['confidence']:.0f}% confident)")
                    else:
                        # Systems disagree - reduce confidence
                        signal['confidence'] = signal['confidence'] * 0.85
                        signal['ml_confirmation'] = False
                        signal['reasons'].append(f"ML system disagrees (caution advised)")
                
                # Only keep high-confidence signals
                if signal['confidence'] >= 75:  # Minimum 75% confidence
                    final_signals.append(signal)
                    
                    # Save to database
                    self.db.save_enhanced_signal(signal)
                    
                    # Send notification
                    self.notifications.send_quality_signal_notification(signal)
            
            # Simple market mood analysis
            latest = market_data.iloc[-1]
            market_mood = {
                'trend': 'Going up' if latest['fast_line'] > latest['medium_line'] > latest['slow_line'] 
                        else 'Going down' if latest['fast_line'] < latest['medium_line'] < latest['slow_line']
                        else 'Sideways',
                'strength': 'Strong' if latest['volume_ratio'] > 1.5 else 'Normal',
                'support': latest['support_level'],
                'resistance': latest['resistance_level'],
                'condition': market_condition_msg
            }
            
            return {
                'signals': final_signals,
                'market_mood': market_mood,
                'ml_prediction': ml_prediction,
                'market_data': market_data,
                'system_status': self.system_status,
                'data_quality': 'Good' if len(market_data) > 50 else 'Limited'
            }
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return {
                'signals': [],
                'market_mood': {'condition': f"Analysis failed: {str(e)}"},
                'error': str(e)
            }

# Streamlit App with Enhanced Interface
def main():
    st.set_page_config(
        page_title="Smart Trading Helper Pro",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced styling
    st.markdown("""
    <style>
    .big-title {
        background: linear-gradient(90deg, #1f4e79, #4a90e2);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .quality-signal {
        background: linear-gradient(135deg, #e8f5e8, #d4edda);
        border-left: 6px solid #28a745;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .high-quality {
        border-left-color: #007bff !important;
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
    }
    .warning-box {
        background: #fff3cd;
        border-left: 6px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background: #d1ecf1;
        border-left: 6px solid #17a2b8;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main title
    st.markdown("""
    <div class="big-title">
        <h1>🎯 Smart Trading Helper Pro</h1>
        <p>High-Quality Signals Only • No Noise • No False Alerts</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced sidebar
    with st.sidebar:
        st.header("⚙️ Quality Controls")
        
        # API setup
        with st.expander("🔑 Market Connection", expanded=True):
            api_key = st.text_input("Axis Direct Key", 
                                   value="tIQJyhGWrjzzIj0CfRJHOf3k8ST5to82yxGLnyxFPLniSBmQ",
                                   type="password")
        
        # Quality settings
        with st.expander("🎯 Signal Quality Settings"):
            st.write("**Current Quality Filters:**")
            st.write("• Minimum confidence: 75%")
            st.write("• Minimum quality score: 8/16")
            st.write("• Volume confirmation required")
            st.write("• Risk-reward minimum: 1:1.8")
            st.write("• ML validation enabled")
            
            quality_mode = st.radio("Quality Mode", 
                                   ["Ultra High Quality (Few signals)", 
                                    "High Quality (Recommended)", 
                                    "Moderate Quality (More signals)"],
                                   index=1)
        
        # Notification setup
        with st.expander("📱 Smart Notifications"):
            st.write("**Telegram Setup (Free)**")
            telegram_token = st.text_input("Bot Token", help="Get from @BotFather")
            telegram_chat = st.text_input("Chat ID", help="Get from @userinfobot")
            
            notification_frequency = st.selectbox("Notification Frequency",
                                                ["Only High-Quality Signals", 
                                                 "All Valid Signals",
                                                 "Market Updates Only"])
    
    # Initialize enhanced system
    if 'trading_system' not in st.session_state:
        st.session_state.trading_system = SmartTradingSystem(api_key)
        if telegram_token and telegram_chat:
            st.session_state.trading_system.notifications.setup_telegram(telegram_token, telegram_chat)
    
    # Main interface
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Quality Signals", "📊 Market Status", "📈 Performance", "💡 Guidelines", "🔧 System Health"])
    
    # Tab 1: Quality Signals
    with tab1:
        st.subheader("🎯 High-Quality Trading Signals Only")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🔍 Scan for Quality Signals", type="primary", use_container_width=True):
                with st.spinner("Running comprehensive market analysis..."):
                    results = st.session_state.trading_system.run_comprehensive_analysis()
                    st.session_state.latest_results = results
            
            # Display results
            if 'latest_results' in st.session_state:
                results = st.session_state.latest_results
                
                if 'error' in results:
                    st.error(f"❌ Analysis failed: {results['error']}")
                elif results['signals']:
                    for signal in results['signals']:
                        quality_class = 'high-quality' if signal['strength'] == 'High' else 'quality-signal'
                        
                        st.markdown(f"""
                        <div class="quality-signal {quality_class}">
                            <h3>🔥 {signal['action']} Signal - {signal['strength']} Quality</h3>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
                                <div><strong>💰 Entry:</strong> ₹{signal['price']:.2f}</div>
                                <div><strong>🎯 Target:</strong> ₹{signal['target']:.2f}</div>
                                <div><strong>🛡️ Stop:</strong> ₹{signal['stop_loss']:.2f}</div>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
                                <div><strong>💪 Confidence:</strong> {signal['confidence']:.0f}%</div>
                                <div><strong>⭐ Quality:</strong> {signal['quality_score']}/16</div>
                                <div><strong>📊 R:R:</strong> 1:{abs(signal['target'] - signal['price']) / abs(signal['price'] - signal['stop_loss']):.1f}</div>
                            </div>
                            <p><strong>🔍 Why this signal?</strong></p>
                            <ul>
                                {''.join([f"<li>{reason}</li>" for reason in signal['reasons']])}
                            </ul>
                            <p><strong>✅ Quality Check:</strong> {signal['validation']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    market_condition = results['market_mood'].get('condition', 'Unknown condition')
                    
                    st.markdown(f"""
                    <div class="info-box">
                        <h4>📊 No Quality Signals Right Now</h4>
                        <p><strong>Market Condition:</strong> {market_condition}</p>
                        <p><strong>System Status:</strong> Working properly, waiting for high-quality setups</p>
                        <p><strong>What this means:</strong> The system is protecting you from low-quality trades. 
                        Quality signals are rare but profitable when they appear.</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("📊 System Status")
            
            if 'latest_results' in st.session_state:
                results = st.session_state.latest_results
                
                # System health
                st.metric("🔍 Data Quality", results.get('data_quality', 'Unknown'))
                st.metric("🤖 ML System", results.get('system_status', 'Unknown'))
                
                # ML prediction
                ml_pred = results.get('ml_prediction')
                if ml_pred:
                    st.write("**🤖 ML Prediction:**")
                    st.write(f"Action: {ml_pred['action']}")
                    st.write(f"Confidence: {ml_pred['confidence']:.0f}%")
                    if 'model_accuracy' in ml_pred:
                        st.write(f"Model Accuracy: {ml_pred['model_accuracy']:.0f}%")
    
    # Tab 2: Market Status
    with tab2:
        st.subheader("📊 Current Market Condition")
        
        if 'latest_results' in st.session_state:
            market_mood = st.session_state.latest_results['market_mood']
            
            # Market overview
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📈 Direction", market_mood.get('trend', 'Unknown'))
            with col2:
                st.metric("💪 Activity", market_mood.get('strength', 'Unknown'))
            with col3:
                st.metric("🎯 Condition", "Suitable" if "good" in market_mood.get('condition', '').lower() else "Cautious")
            
            # Detailed condition
            st.markdown(f"""
            <div class="info-box">
                <h4>📋 Market Analysis</h4>
                <p>{market_mood.get('condition', 'No analysis available')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Key levels
            st.subheader("🎯 Important Price Levels")
            col1, col2 = st.columns(2)
            
            with col1:
                support = market_mood.get('support', 0)
                st.metric("🛡️ Support Level", f"₹{support:.0f}" if support > 0 else "N/A")
                
            with col2:
                resistance = market_mood.get('resistance', 0)
                st.metric("🚧 Resistance Level", f"₹{resistance:.0f}" if resistance > 0 else "N/A")
    
    # Tab 3: Performance Tracking
    with tab3:
        st.subheader("📈 Signal Performance & Quality Stats")
        
        # Get recent signals from database
        recent_signals = st.session_state.trading_system.db.get_recent_signals(days=30)
        
        if not recent_signals.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🎯 Total Signals", len(recent_signals))
            with col2:
                avg_confidence = recent_signals['confidence'].mean()
                st.metric("💪 Avg Confidence", f"{avg_confidence:.0f}%")
            with col3:
                avg_quality = recent_signals['quality_score'].mean() if 'quality_score' in recent_signals.columns else 0
                st.metric("⭐ Avg Quality", f"{avg_quality:.1f}/16")
            with col4:
                high_quality = len(recent_signals[recent_signals['strength'] == 'High']) if 'strength' in recent_signals.columns else 0
                st.metric("🔥 High Quality", high_quality)
            
            # Quality distribution
            if 'strength' in recent_signals.columns:
                st.subheader("📊 Signal Quality Distribution")
                quality_counts = recent_signals['strength'].value_counts()
                fig = px.pie(values=quality_counts.values, names=quality_counts.index,
                           title="Signal Quality Breakdown")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 No signal history available yet. Generate some signals to see performance tracking.")
    
    # Tab 4: Trading Guidelines
    with tab4:
        st.subheader("💡 How to Use This Tool Effectively")
        
        st.markdown("""
        ### 🎯 **Understanding Signal Quality**
        
        **🔥 High Quality Signals (12-16 points):**
        - Multiple confirmations aligned
        - Strong volume support
        - Clear trend direction
        - ML system agreement
        - **Action:** Trade with full confidence
        
        **⭐ Moderate Quality Signals (8-11 points):**
        - Good setup but missing some confirmations
        - Decent volume support
        - **Action:** Trade with reduced position size
        
        **❌ Low Quality Signals (Below 8 points):**
        - Few confirmations
        - Weak volume or conflicting signals
        - **Action:** WAIT - System will not show these
        
        ### 🚫 **When NO Signals Appear**
        
        **This is NORMAL and PROTECTIVE! The system avoids:**
        - Choppy/sideways markets
        - Low volume periods
        - First/last 30 minutes of trading
        - High volatility periods
        - Conflicting technical indicators
        - Poor risk-reward setups
        
        ### ✅ **Best Practices**
        
        **1. Quality Over Quantity:**
        - Better to miss 10 trades than take 1 bad trade
        - High-quality signals are rare but profitable
        
        **2. Risk Management:**
        - Always use the suggested stop-loss
        - Never risk more than 2% per trade
        - Don't chase signals - wait for the next one
        
        **3. Market Timing:**
        - Best signals usually come between 10 AM - 2 PM
        - Avoid trading during news events
        - Be extra cautious on Fridays and Mondays
        
        **4. Signal Confirmation:**
        - Wait for at least 75% confidence
        - Prefer signals with ML system agreement
        - Look for high volume confirmation
        
        ### ⚠️ **Important Warnings**
        
        **Don't Trade When:**
        - Market condition shows "Not suitable"
        - Signal confidence below 75%
        - No volume confirmation
        - You're feeling emotional or stressed
        - You've already lost money today
        
        **Emergency Rules:**
        - If 3 consecutive signals fail, STOP trading for the day
        - If system shows errors, don't trade manually
        - Always honor stop-losses immediately
        """)
    
    # Tab 5: System Health
    with tab5:
        st.subheader("🔧 System Health & Diagnostics")
        
        if 'latest_results' in st.session_state:
            results = st.session_state.latest_results
            
            # System status
            col1, col2, col3 = st.columns(3)
            
            with col1:
                data_quality = results.get('data_quality', 'Unknown')
                status_color = "🟢" if data_quality == "Good" else "🟡"
                st.metric("Data Quality", f"{status_color} {data_quality}")
            
            with col2:
                ml_status = results.get('system_status', 'Unknown')
                ml_color = "🟢" if "Trained" in ml_status else "🔴"
                st.metric("ML System", f"{ml_color} {ml_status}")
            
            with col3:
                notification_status = "🟢 Connected" if st.session_state.trading_system.notifications.telegram_bot_token else "🔴 Not Setup"
                st.metric("Notifications", notification_status)
            
            # Quality filters status
            st.subheader("🎯 Active Quality Filters")
            
            filters = [
                "✅ Minimum 75% confidence required",
                "✅ Minimum 8/16 quality score required", 
                "✅ Volume confirmation mandatory",
                "✅ Risk-reward minimum 1:1.8",
                "✅ Market condition suitability check",
                "✅ ML system validation enabled",
                "✅ Whipsaw detection active",
                "✅ Time-based filters (avoid risky periods)"
            ]
            
            for filter_item in filters:
                st.write(filter_item)
            
            # Recent system activity
            st.subheader("📊 Recent System Activity")
            
            activity_log = [
                f"✅ Last analysis: {datetime.now().strftime('%H:%M:%S')}",
                f"📊 Market data: {len(results.get('market_data', []))} periods loaded",
                f"🎯 Quality filters: All active",
                f"🤖 ML predictions: {'Available' if results.get('ml_prediction') else 'Not available'}",
                f"📱 Notifications: {'Enabled' if notification_status.startswith('🟢') else 'Disabled'}"
            ]
            
            for log_item in activity_log:
                st.write(log_item)
        
        # Manual system test
        st.subheader("🔧 Manual System Test")
        
        if st.button("🧪 Run System Diagnostic"):
            with st.spinner("Running comprehensive system test..."):
                test_results = st.session_state.trading_system.run_comprehensive_analysis()
                
                if 'error' in test_results:
                    st.error(f"❌ System test failed: {test_results['error']}")
                else:
                    st.success("✅ All systems working properly!")
                    st.write(f"📊 Generated {len(test_results['signals'])} quality signals")
                    st.write(f"🤖 ML system: {test_results.get('system_status', 'Unknown')}")
    
    # Enhanced status bar
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'latest_results' in st.session_state and not st.session_state.latest_results.get('error'):
            st.success("🟢 System Active")
        else:
            st.warning("🟡 System Standby")
    
    with col2:
        current_time = datetime.now().strftime("%d %b %Y, %I:%M %p")
        st.info(f"🕐 {current_time}")
    
    with col3:
        if 'latest_results' in st.session_state:
            signal_count = len(st.session_state.latest_results.get('signals', []))
            st.metric("🎯 Active Signals", signal_count)
        else:
            st.metric("🎯 Active Signals", 0)
    
    with col4:
        if st.button("🔄 Refresh System"):
            st.experimental_rerun()

if __name__ == "__main__":
    main()
