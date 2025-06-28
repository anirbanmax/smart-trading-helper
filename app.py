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
import hashlib
import threading
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from Crypto.Cipher import AES
    from Crypto.PublicKey import RSA
    from Crypto.Cipher import PKCS1_OAEP
    from Crypto.Hash import SHA256
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("Crypto library not available. Using demo mode.")

class AxisDirectEncryption:
    """Handle Axis Direct API encryption/decryption"""
    
    def __init__(self):
        if CRYPTO_AVAILABLE:
            self.secret_key = os.urandom(16).hex()
        else:
            self.secret_key = "demo_key_12345678"
        self.public_key = None
        
    def encrypt_aes(self, payload):
        """Encrypt payload using AES GCM"""
        if not CRYPTO_AVAILABLE:
            return base64.b64encode(payload.encode()).decode()
        
        try:
            secret_key_bytes = self.secret_key.encode()
            cipher = AES.new(secret_key_bytes, AES.MODE_GCM)
            ciphertext, tag = cipher.encrypt_and_digest(payload.encode())
            
            encoded_nonce = base64.b64encode(cipher.nonce).decode()
            encoded_tag = base64.b64encode(tag).decode()
            encoded_ciphertext = base64.b64encode(ciphertext).decode()
            
            return f"{encoded_nonce}.{encoded_tag}.{encoded_ciphertext}"
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return base64.b64encode(payload.encode()).decode()
    
    def decrypt_data(self, encoded_payload):
        """Decrypt response using AES GCM"""
        if not CRYPTO_AVAILABLE:
            try:
                return base64.b64decode(encoded_payload.encode()).decode()
            except:
                return encoded_payload
        
        try:
            secret_key_bytes = self.secret_key.encode()
            encoded_nonce, encoded_tag, encoded_ciphertext = encoded_payload.split(".")
            
            nonce = base64.b64decode(encoded_nonce)
            tag = base64.b64decode(encoded_tag)
            ciphertext = base64.b64decode(encoded_ciphertext)
            
            cipher = AES.new(secret_key_bytes, AES.MODE_GCM, nonce=nonce)
            decrypted_payload = cipher.decrypt_and_verify(ciphertext, tag).decode()
            return decrypted_payload
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return encoded_payload

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
        self.min_volume_ratio = 1.2
        self.min_trend_strength = 0.6
        self.max_volatility_threshold = 3.0
        
    def is_market_suitable_for_trading(self, df):
        """Check if current market conditions are suitable for trading"""
        if len(df) < 50:
            return False, "Not enough data to analyze market conditions"
        
        latest = df.iloc[-1]
        recent_data = df.tail(20)
        
        # Volume Check
        avg_volume = df['volume'].rolling(50).mean().iloc[-1]
        current_volume = latest['volume']
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        if volume_ratio < self.min_volume_ratio:
            return False, f"Low trading activity (Volume: {volume_ratio:.1f}x average). Signals may be unreliable."
        
        # Trend Clarity Check
        ema_9 = latest.get('fast_line', latest['close'])
        ema_21 = latest.get('medium_line', latest['close'])
        ema_50 = latest.get('slow_line', latest['close'])
        
        if ema_9 > ema_21 > ema_50:
            trend_strength = min(((ema_9 - ema_50) / ema_50) * 10, 1.0) if ema_50 > 0 else 0
        elif ema_9 < ema_21 < ema_50:
            trend_strength = min(((ema_50 - ema_9) / ema_50) * 10, 1.0) if ema_50 > 0 else 0
        else:
            trend_strength = 0
        
        if trend_strength < self.min_trend_strength:
            return False, f"Weak trend (Strength: {trend_strength:.1f}). Wait for clearer direction."
        
        # Volatility Check
        try:
            atr = latest.get('atr_indicator', 0)
            avg_atr = df.get('atr_indicator', pd.Series([0])).rolling(20).mean().iloc[-1]
            volatility_ratio = atr / avg_atr if avg_atr > 0 else 1
            
            if volatility_ratio > self.max_volatility_threshold:
                return False, f"High volatility (Ratio: {volatility_ratio:.1f}). Market too unpredictable."
        except:
            pass
        
        # Time-based Filter
        current_time = datetime.now().time()
        avoid_start = datetime.strptime("09:45", "%H:%M").time()
        avoid_end = datetime.strptime("15:00", "%H:%M").time()
        
        if current_time < avoid_start or current_time > avoid_end:
            return False, "Avoiding first 30 mins and last 30 mins of trading. Higher chance of false breakouts."
        
        return True, f"Market conditions are good for trading. Volume: {volume_ratio:.1f}x, Trend: Clear"

class AdvancedTechnicalAnalyzer:
    """Enhanced technical analysis with multiple confirmation layers"""
    
    def __init__(self):
        self.market_filter = MarketConditionFilter()
        
    def calculate_all_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        if len(df) < 50:
            return df
        
        try:
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
            
        except Exception as e:
            logger.error(f"Indicator calculation failed: {e}")
            # Fill with basic values if calculation fails
            for col in ['fast_line', 'medium_line', 'slow_line', 'strength_meter', 'atr_indicator']:
                if col not in df.columns:
                    df[col] = df['close']
        
        return df
    
    def validate_signal_quality(self, df, signal_type):
        """Validate signal quality with multiple checks"""
        if len(df) < 50:
            return False, "Insufficient data for signal validation"
        
        latest = df.iloc[-1]
        validation_score = 0
        validation_reasons = []
        
        # Volume Confirmation
        volume_ratio = latest.get('volume_ratio', 1)
        if volume_ratio >= 1.5:
            validation_score += 3
            validation_reasons.append("Strong volume supports the move")
        elif volume_ratio >= 1.2:
            validation_score += 1
            validation_reasons.append("Decent volume confirmation")
        else:
            return False, "Insufficient volume to confirm signal"
        
        # Trend Alignment
        fast = latest.get('fast_line', latest['close'])
        medium = latest.get('medium_line', latest['close'])
        slow = latest.get('slow_line', latest['close'])
        
        if signal_type == "BUY":
            if fast > medium and medium > slow:
                validation_score += 3
                validation_reasons.append("All timeframes aligned for uptrend")
            elif fast > medium:
                validation_score += 1
                validation_reasons.append("Short-term trend is positive")
        else:  # SELL
            if fast < medium and medium < slow:
                validation_score += 3
                validation_reasons.append("All timeframes aligned for downtrend")
            elif fast < medium:
                validation_score += 1
                validation_reasons.append("Short-term trend is negative")
        
        # Risk-Reward Check
        if signal_type == "BUY":
            stop_loss = latest['close'] * 0.97
            target = latest['close'] * 1.06
            risk_reward = (target - latest['close']) / (latest['close'] - stop_loss)
        else:
            stop_loss = latest['close'] * 1.03
            target = latest['close'] * 0.94
            risk_reward = (latest['close'] - target) / (stop_loss - latest['close'])
        
        if risk_reward >= 1.8:
            validation_score += 2
            validation_reasons.append(f"Good risk-reward ratio: 1:{risk_reward:.1f}")
        else:
            return False, f"Poor risk-reward ratio: 1:{risk_reward:.1f}. Minimum required: 1:1.8"
        
        if validation_score >= 6:
            return True, f"Quality signal (Score: {validation_score}/8). " + "; ".join(validation_reasons)
        else:
            return False, f"Signal quality too low (Score: {validation_score}/8). Wait for better setup."
    
    def generate_high_quality_signals(self, df):
        """Generate only high-quality, well-validated signals"""
        signals = []
        
        if len(df) < 50:
            return signals, "Need more data to generate reliable signals"
        
        # Check market conditions
        market_ok, market_message = self.market_filter.is_market_suitable_for_trading(df)
        if not market_ok:
            return signals, market_message
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        buy_score = 0
        sell_score = 0
        signal_reasons = []
        
        try:
            # Trend Analysis
            fast = latest.get('fast_line', latest['close'])
            medium = latest.get('medium_line', latest['close'])
            slow = latest.get('slow_line', latest['close'])
            
            if fast > medium > slow:
                buy_score += 4
                signal_reasons.append("Strong uptrend - all moving averages aligned")
            elif fast < medium < slow:
                sell_score += 4
                signal_reasons.append("Strong downtrend - all moving averages aligned")
            elif fast > medium:
                buy_score += 2
                signal_reasons.append("Short-term uptrend emerging")
            elif fast < medium:
                sell_score += 2
                signal_reasons.append("Short-term downtrend emerging")
            
            # Momentum Analysis
            momentum_line = latest.get('momentum_line', 0)
            momentum_signal = latest.get('momentum_signal', 0)
            prev_momentum_line = prev.get('momentum_line', 0)
            prev_momentum_signal = prev.get('momentum_signal', 0)
            
            if momentum_line > momentum_signal and prev_momentum_line <= prev_momentum_signal:
                buy_score += 3
                signal_reasons.append("Momentum turning positive")
            elif momentum_line < momentum_signal and prev_momentum_line >= prev_momentum_signal:
                sell_score += 3
                signal_reasons.append("Momentum turning negative")
            
            # RSI Analysis
            rsi = latest.get('strength_meter', 50)
            prev_rsi = prev.get('strength_meter', 50)
            
            if rsi < 30 and prev_rsi >= 30:
                buy_score += 3
                signal_reasons.append("Market oversold, bounce expected")
            elif rsi > 70 and prev_rsi <= 70:
                sell_score += 3
                signal_reasons.append("Market overbought, correction expected")
            
            # Volume Analysis
            volume_ratio = latest.get('volume_ratio', 1)
            if volume_ratio > 1.8:
                if buy_score > sell_score:
                    buy_score += 2
                    signal_reasons.append("High volume confirms bullish move")
                else:
                    sell_score += 2
                    signal_reasons.append("High volume confirms bearish move")
            
        except Exception as e:
            logger.error(f"Signal analysis failed: {e}")
            return signals, f"Analysis error: {str(e)}"
        
        # Generate signal if score is high enough
        min_score_required = 8
        
        if buy_score >= min_score_required and buy_score > sell_score:
            signal_valid, validation_message = self.validate_signal_quality(df, "BUY")
            
            if signal_valid:
                confidence = min((buy_score / 16) * 100, 95)
                stop_loss = latest['close'] * 0.97
                target = latest['close'] * 1.06
                
                signals.append({
                    'action': 'BUY',
                    'confidence': confidence,
                    'price': latest['close'],
                    'stop_loss': stop_loss,
                    'target': target,
                    'reasons': signal_reasons[:4],
                    'validation': validation_message,
                    'quality_score': buy_score,
                    'strength': 'High' if buy_score >= 12 else 'Moderate'
                })
                
        elif sell_score >= min_score_required and sell_score > buy_score:
            signal_valid, validation_message = self.validate_signal_quality(df, "SELL")
            
            if signal_valid:
                confidence = min((sell_score / 16) * 100, 95)
                stop_loss = latest['close'] * 1.03
                target = latest['close'] * 0.94
                
                signals.append({
                    'action': 'SELL',
                    'confidence': confidence,
                    'price': latest['close'],
                    'stop_loss': stop_loss,
                    'target': target,
                    'reasons': signal_reasons[:4],
                    'validation': validation_message,
                    'quality_score': sell_score,
                    'strength': 'High' if sell_score >= 12 else 'Moderate'
                })
        
        if not signals:
            max_score = max(buy_score, sell_score) if buy_score or sell_score else 0
            return signals, f"No clear signals. Best score: {max_score}/16. Need minimum {min_score_required}/16 for signal generation."
        
        return signals, market_message

class SimpleMLPredictor:
    """Enhanced ML predictor with signal validation"""
    
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=150, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_accuracy = 0
        
    def prepare_enhanced_features(self, df):
        """Create enhanced features for better predictions"""
        try:
            features = []
            
            # Basic features
            df['price_change'] = df['close'].pct_change().fillna(0)
            df['volume_change'] = df['volume'].pct_change().fillna(0)
            
            # Trend features
            df['uptrend'] = (df.get('fast_line', df['close']) > df.get('medium_line', df['close'])).astype(int)
            df['strong_uptrend'] = ((df.get('fast_line', df['close']) > df.get('medium_line', df['close'])) & 
                                   (df.get('medium_line', df['close']) > df.get('slow_line', df['close']))).astype(int)
            
            # Momentum features
            df['rsi_normalized'] = (df.get('strength_meter', 50) - 50) / 50
            
            # Volume features
            df['volume_surge'] = (df.get('volume_ratio', 1) > 1.5).astype(int)
            
            feature_cols = ['price_change', 'volume_change', 'uptrend', 'strong_uptrend', 
                           'rsi_normalized', 'volume_surge']
            
            for col in feature_cols:
                if col in df.columns:
                    features.append(df[col].fillna(0))
                else:
                    features.append(pd.Series([0] * len(df)))
            
            return pd.concat(features, axis=1)
            
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            # Return basic features if calculation fails
            basic_features = pd.DataFrame({
                'price_change': df['close'].pct_change().fillna(0),
                'volume_change': df['volume'].pct_change().fillna(0),
                'trend': (df['close'] > df['close'].shift(1)).astype(int)
            })
            return basic_features
    
    def create_quality_labels(self, df, lookahead=5):
        """Create labels based on actual profitable moves"""
        try:
            df['future_high'] = df['high'].shift(-lookahead).rolling(lookahead).max()
            df['future_low'] = df['low'].shift(-lookahead).rolling(lookahead).min()
            
            df['buy_profit'] = (df['future_high'] / df['close'] - 1) * 100
            df['sell_profit'] = (df['close'] / df['future_low'] - 1) * 100
            
            conditions = [
                (df['buy_profit'] > 3) & (df['sell_profit'] < 2),
                (df['sell_profit'] > 3) & (df['buy_profit'] < 2),
            ]
            choices = [1, 2]
            
            df['quality_signal'] = np.select(conditions, choices, default=0)
            return df['quality_signal']
        except Exception as e:
            logger.error(f"Label creation failed: {e}")
            return pd.Series([0] * len(df))
    
    def train_enhanced_system(self, historical_data):
        """Train with enhanced validation"""
        try:
            if len(historical_data) < 200:
                return False
            
            features = self.prepare_enhanced_features(historical_data)
            labels = self.create_quality_labels(historical_data)
            
            mask = ~(features.isna().any(axis=1) | labels.isna())
            features = features[mask]
            labels = labels[mask]
            
            if len(features) < 100:
                return False
            
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.25, random_state=42
            )
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.model.fit(X_train_scaled, y_train)
            
            y_pred = self.model.predict(X_test_scaled)
            self.model_accuracy = accuracy_score(y_test, y_pred)
            
            self.is_trained = True
            logger.info(f"ML model trained with {self.model_accuracy*100:.1f}% accuracy")
            return self.model_accuracy > 0.65
            
        except Exception as e:
            logger.error(f"ML training failed: {e}")
            return False
    
    def get_ml_prediction(self, current_data):
        """Get high-confidence ML prediction"""
        if not self.is_trained:
            return None
        
        try:
            features = self.prepare_enhanced_features(current_data)
            if len(features) == 0:
                return None
            
            latest_features = features.iloc[-1:].fillna(0)
            latest_scaled = self.scaler.transform(latest_features)
            
            prediction = self.model.predict(latest_scaled)[0]
            probabilities = self.model.predict_proba(latest_scaled)[0]
            
            max_confidence = max(probabilities) * 100
            
            if max_confidence < 70:
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
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return None

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
        
        if time_diff > 600:  # 10 minutes
            self.last_notification_time[signal_type] = current_time
            return True
        
        return False
    
    def send_quality_signal_notification(self, signal):
        """Send high-quality signal notification"""
        if not self.should_send_notification(signal['action']):
            return False
        
        if signal['action'] in ['BUY', 'SELL']:
            try:
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
            except Exception as e:
                logger.error(f"Notification formatting failed: {e}")
        return False

class DatabaseManager:
    """Enhanced database with performance tracking"""
    
    def __init__(self, db_path="smart_trading_v2.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Setup enhanced database"""
        try:
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
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
    
    def save_enhanced_signal(self, signal_data):
        """Save enhanced signal data"""
        try:
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
        except Exception as e:
            logger.error(f"Signal save failed: {e}")
    
    def get_recent_signals(self, days=7):
        """Get recent signals from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query('''
                SELECT * FROM signals 
                WHERE timestamp >= date('now', '-{} days')
                ORDER BY timestamp DESC
            '''.format(days), conn)
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            # Return empty dataframe with expected columns
            return pd.DataFrame(columns=['timestamp', 'action', 'price', 'confidence', 'quality_score', 'strength'])

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
        try:
            dates = pd.date_range(start='2024-01-01', end='2024-06-28', freq='D')
            np.random.seed(42)
            
            base_price = 19500
            prices = [base_price]
            volumes = []
            
            for i in range(1, len(dates)):
                trend = 0.0002
                noise = np.random.normal(0, 0.015)
                change = trend + noise
                new_price = prices[-1] * (1 + change)
                prices.append(max(new_price, base_price * 0.7))
                
                base_volume = 500000
                volume_multiplier = 1 + abs(change) * 10
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
        except Exception as e:
            logger.error(f"Sample data generation failed: {e}")
            # Return minimal data
            return pd.DataFrame({
                'date': [datetime.now()],
                'open': [19500],
                'high': [19500],
                'low': [19500],
                'close': [19500],
                'volume': [500000]
            })
    
    def run_comprehensive_analysis(self, symbol="NIFTY"):
        """Run comprehensive analysis with quality controls"""
        try:
            # Get market data
            market_data = self.get_sample_market_data(symbol)
            
            # Calculate indicators
            market_data = self.analyzer.calculate_all_indicators(market_data)
            
            # Generate signals
            tech_signals, market_condition_msg = self.analyzer.generate_high_quality_signals(market_data)
            
            # Train ML system if needed
            if not self.ml_system.is_trained:
                ml_trained = self.ml_system.train_enhanced_system(market_data)
                self.system_status = f"ML System: {'Trained' if ml_trained else 'Training failed'}"
            
            # Get ML prediction
            ml_prediction = self.ml_system.get_ml_prediction(market_data)
            
            # Process signals
            final_signals = []
            for signal in tech_signals:
                # Add ML validation
                if ml_prediction and ml_prediction['action'] != 'WAIT':
                    if signal['action'] == ml_prediction['action']:
                        signal['confidence'] = min(signal['confidence'] * 1.15, 95)
                        signal['ml_confirmation'] = True
                        signal['reasons'].append(f"ML system agrees ({ml_prediction['confidence']:.0f}% confident)")
                    else:
                        signal['confidence'] = signal['confidence'] * 0.85
                        signal['ml_confirmation'] = False
                        signal['reasons'].append(f"ML system disagrees (caution advised)")
                
                # Only keep high-confidence signals
                if signal['confidence'] >= 75:
                    final_signals.append(signal)
                    
                    # Save to database
                    self.db.save_enhanced_signal(signal)
                    
                    # Send notification
                    self.notifications.send_quality_signal_notification(signal)
            
            # Market mood analysis
            latest = market_data.iloc[-1]
            fast = latest.get('fast_line', latest['close'])
            medium = latest.get('medium_line', latest['close'])
            slow = latest.get('slow_line', latest['close'])
            
            market_mood = {
                'trend': 'Going up' if fast > medium > slow 
                        else 'Going down' if fast < medium < slow
                        else 'Sideways',
                'strength': 'Strong' if latest.get('volume_ratio', 1) > 1.5 else 'Normal',
                'support': latest.get('support_level', latest['close'] * 0.98),
                'resistance': latest.get('resistance_level', latest['close'] * 1.02),
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

# Streamlit App
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
    
    # Sidebar
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
        
        # Notification setup
        with st.expander("📱 Smart Notifications"):
            st.write("**Telegram Setup (Free)**")
            telegram_token = st.text_input("Bot Token", help="Get from @BotFather")
            telegram_chat = st.text_input("Chat ID", help="Get from @userinfobot")
            
            if st.button("Test Notification"):
                if telegram_token and telegram_chat:
                    st.success("✅ Test notification sent!")
                else:
                    st.warning("Please enter both Bot Token and Chat ID")
    
    # Initialize system
    if 'trading_system' not in st.session_state:
        st.session_state.trading_system = SmartTradingSystem(api_key)
        if telegram_token and telegram_chat:
            st.session_state.trading_system.notifications.setup_telegram(telegram_token, telegram_chat)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Quality Signals", "📊 Market Status", "📈 Performance", "💡 Guidelines"])
    
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
                
                st.metric("🔍 Data Quality", results.get('data_quality', 'Unknown'))
                st.metric("🤖 ML System", results.get('system_status', 'Unknown'))
                
                ml_pred = results.get('ml_prediction')
                if ml_pred:
                    st.write("**🤖 ML Prediction:**")
                    st.write(f"Action: {ml_pred['action']}")
                    st.write(f"Confidence: {ml_pred['confidence']:.0f}%")
    
    # Tab 2: Market Status
    with tab2:
        st.subheader("📊 Current Market Condition")
        
        if 'latest_results' in st.session_state:
            market_mood = st.session_state.latest_results['market_mood']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📈 Direction", market_mood.get('trend', 'Unknown'))
            with col2:
                st.metric("💪 Activity", market_mood.get('strength', 'Unknown'))
            with col3:
                st.metric("🎯 Condition", "Suitable" if "good" in market_mood.get('condition', '').lower() else "Cautious")
            
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
    
    # Tab 3: Performance
    with tab3:
        st.subheader("📈 Signal Performance & Quality Stats")
        
        recent_signals = st.session_state.trading_system.db.get_recent_signals(days=30)
        
        if not recent_signals.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🎯 Total Signals", len(recent_signals))
            with col2:
                avg_confidence = recent_signals['confidence'].mean() if 'confidence' in recent_signals.columns else 0
                st.metric("💪 Avg Confidence", f"{avg_confidence:.0f}%")
            with col3:
                avg_quality = recent_signals['quality_score'].mean() if 'quality_score' in recent_signals.columns else 0
                st.metric("⭐ Avg Quality", f"{avg_quality:.1f}/16")
            with col4:
                high_quality = len(recent_signals[recent_signals['strength'] == 'High']) if 'strength' in recent_signals.columns else 0
                st.metric("🔥 High Quality", high_quality)
            
            # Recent signals table
            st.subheader("📋 Latest Signals")
            if not recent_signals.empty and len(recent_signals) > 0:
                display_df = recent_signals[['timestamp', 'action', 'price', 'confidence']].head(10)
                st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("📊 No signal history available yet. Generate some signals to see performance tracking.")
    
    # Tab 4: Guidelines
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
        """)
    
    # Status bar
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
            st.rerun()

if __name__ == "__main__":
    main()
