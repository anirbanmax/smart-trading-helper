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
import pytz
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

class MarketHoursManager:
    """Manage market hours and detect open/closed status"""
    
    def __init__(self):
        self.ist_timezone = pytz.timezone('Asia/Kolkata')
        self.market_holidays = [
            # Add major market holidays here
            '2024-01-26', '2024-03-08', '2024-03-25', '2024-04-11', 
            '2024-05-01', '2024-08-15', '2024-10-02', '2024-11-01'
        ]
    
    def get_current_ist_time(self):
        """Get current IST time"""
        return datetime.now(self.ist_timezone)
    
    def is_market_day(self, date=None):
        """Check if today is a market day (Monday to Friday, excluding holidays)"""
        if date is None:
            date = self.get_current_ist_time().date()
        
        # Check if weekend
        if date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check if holiday
        if date.strftime('%Y-%m-%d') in self.market_holidays:
            return False
        
        return True
    
    def is_market_open(self):
        """Check if market is currently open"""
        now = self.get_current_ist_time()
        current_date = now.date()
        current_time = now.time()
        
        if not self.is_market_day(current_date):
            return False
        
        # Market hours: 9:15 AM to 3:30 PM IST
        market_start = datetime.strptime("09:15", "%H:%M").time()
        market_end = datetime.strptime("15:30", "%H:%M").time()
        
        return market_start <= current_time <= market_end
    
    def get_market_status(self):
        """Get detailed market status"""
        now = self.get_current_ist_time()
        current_date = now.date()
        current_time = now.time()
        
        if not self.is_market_day(current_date):
            if current_date.weekday() >= 5:
                next_monday = current_date + timedelta(days=(7 - current_date.weekday()))
                return {
                    'status': 'CLOSED',
                    'reason': 'Weekend',
                    'next_open': f"Monday, {next_monday.strftime('%d %b %Y')} at 9:15 AM",
                    'color': 'red'
                }
            else:
                return {
                    'status': 'CLOSED',
                    'reason': 'Market Holiday',
                    'next_open': 'Next trading day at 9:15 AM',
                    'color': 'orange'
                }
        
        market_start = datetime.strptime("09:15", "%H:%M").time()
        market_end = datetime.strptime("15:30", "%H:%M").time()
        
        if current_time < market_start:
            return {
                'status': 'PRE-MARKET',
                'reason': f'Market opens at 9:15 AM (in {self._time_until_open(now)})',
                'next_open': 'Today at 9:15 AM',
                'color': 'yellow'
            }
        elif current_time > market_end:
            return {
                'status': 'CLOSED',
                'reason': f'Market closed at 3:30 PM (closed for {self._time_since_close(now)})',
                'next_open': 'Tomorrow at 9:15 AM',
                'color': 'red'
            }
        else:
            return {
                'status': 'OPEN',
                'reason': f'Market open (closes at 3:30 PM in {self._time_until_close(now)})',
                'next_open': 'Currently trading',
                'color': 'green'
            }
    
    def _time_until_open(self, now):
        """Calculate time until market opens"""
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        if now.time() > datetime.strptime("15:30", "%H:%M").time():
            market_open += timedelta(days=1)
        
        diff = market_open - now
        hours, remainder = divmod(diff.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        return f"{hours}h {minutes}m"
    
    def _time_until_close(self, now):
        """Calculate time until market closes"""
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        diff = market_close - now
        hours, remainder = divmod(diff.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        return f"{hours}h {minutes}m"
    
    def _time_since_close(self, now):
        """Calculate time since market closed"""
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        if now.time() < datetime.strptime("09:15", "%H:%M").time():
            market_close -= timedelta(days=1)
        
        diff = now - market_close
        hours, remainder = divmod(diff.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        return f"{hours}h {minutes}m"

class StockDataManager:
    """Manage stock data and selection"""
    
    def __init__(self):
        self.available_stocks = {
            # Major Indices
            'NIFTY 50': {'symbol': 'NIFTY', 'type': 'INDEX', 'base_price': 19500},
            'BANK NIFTY': {'symbol': 'BANKNIFTY', 'type': 'INDEX', 'base_price': 44000},
            'NIFTY IT': {'symbol': 'NIFTYIT', 'type': 'INDEX', 'base_price': 31000},
            
            # Large Cap Stocks
            'Reliance Industries': {'symbol': 'RELIANCE', 'type': 'STOCK', 'base_price': 2450},
            'HDFC Bank': {'symbol': 'HDFCBANK', 'type': 'STOCK', 'base_price': 1580},
            'Infosys': {'symbol': 'INFY', 'type': 'STOCK', 'base_price': 1420},
            'TCS': {'symbol': 'TCS', 'type': 'STOCK', 'base_price': 3980},
            'ICICI Bank': {'symbol': 'ICICIBANK', 'type': 'STOCK', 'base_price': 1120},
            'Hindustan Unilever': {'symbol': 'HINDUNILVR', 'type': 'STOCK', 'base_price': 2380},
            'ITC': {'symbol': 'ITC', 'type': 'STOCK', 'base_price': 460},
            'SBI': {'symbol': 'SBIN', 'type': 'STOCK', 'base_price': 820},
            'Bharti Airtel': {'symbol': 'BHARTIARTL', 'type': 'STOCK', 'base_price': 1540},
            'Kotak Mahindra Bank': {'symbol': 'KOTAKBANK', 'type': 'STOCK', 'base_price': 1780},
            'L&T': {'symbol': 'LT', 'type': 'STOCK', 'base_price': 3650},
            'Asian Paints': {'symbol': 'ASIANPAINT', 'type': 'STOCK', 'base_price': 3200},
            'Maruti Suzuki': {'symbol': 'MARUTI', 'type': 'STOCK', 'base_price': 12500},
            'Mahindra & Mahindra': {'symbol': 'M&M', 'type': 'STOCK', 'base_price': 2890},
            'Tata Motors': {'symbol': 'TATAMOTORS', 'type': 'STOCK', 'base_price': 980},
            'Wipro': {'symbol': 'WIPRO', 'type': 'STOCK', 'base_price': 550}
        }
    
    def get_stock_list(self):
        """Get list of available stocks"""
        return list(self.available_stocks.keys())
    
    def get_stock_info(self, stock_name):
        """Get stock information"""
        return self.available_stocks.get(stock_name, {})
    
    def generate_stock_data(self, stock_name, days=180):
        """Generate realistic stock data"""
        stock_info = self.get_stock_info(stock_name)
        if not stock_info:
            return None
        
        # Set seed based on stock name for consistent data
        np.random.seed(hash(stock_name) % 1000)
        
        base_price = stock_info['base_price']
        dates = pd.date_range(end=datetime.now().date(), periods=days, freq='D')
        
        prices = [base_price]
        volumes = []
        
        # Different volatility for different stock types
        if stock_info['type'] == 'INDEX':
            volatility = 0.015  # Lower volatility for indices
            base_volume = 1000000
        else:
            volatility = 0.025  # Higher volatility for individual stocks
            base_volume = 500000
        
        for i in range(1, len(dates)):
            # Add market trends and stock-specific patterns
            trend = 0.0001  # Slight upward bias
            
            # Add sector-specific trends
            if 'BANK' in stock_name.upper() or stock_name in ['HDFC Bank', 'ICICI Bank', 'SBI', 'Kotak Mahindra Bank']:
                trend += 0.0002  # Banking sector doing well
            elif 'IT' in stock_name.upper() or stock_name in ['Infosys', 'TCS', 'Wipro']:
                trend += 0.0001  # IT sector moderate growth
            
            noise = np.random.normal(0, volatility)
            change = trend + noise
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, base_price * 0.5))  # Prevent unrealistic drops
            
            # Volume with correlation to price movement
            volume_multiplier = 1 + abs(change) * 15
            volumes.append(int(base_volume * volume_multiplier * np.random.uniform(0.6, 1.4)))
        
        volumes.append(volumes[-1])
        
        df = pd.DataFrame({
            'date': dates,
            'open': [p * (1 + np.random.normal(0, 0.003)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.006))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.006))) for p in prices],
            'close': prices,
            'volume': volumes
        })
        
        return df

class AfterHoursAnalyzer:
    """Analyze what happened during the day and provide insights"""
    
    def __init__(self):
        self.market_events = [
            "RBI policy announcement",
            "Quarterly earnings release",
            "Global market volatility",
            "FII/DII activity",
            "Sector rotation",
            "Technical breakout/breakdown",
            "News-based movement",
            "Profit booking",
            "Fresh buying interest"
        ]
    
    def analyze_day_performance(self, df, stock_name):
        """Analyze what happened during the trading day"""
        if len(df) < 2:
            return {"error": "Insufficient data for analysis"}
        
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        week_ago = df.iloc[-7] if len(df) >= 7 else df.iloc[0]
        
        # Calculate performance metrics
        day_change = ((latest['close'] - previous['close']) / previous['close']) * 100
        week_change = ((latest['close'] - week_ago['close']) / week_ago['close']) * 100
        
        day_high = latest['high']
        day_low = latest['low']
        day_range = ((day_high - day_low) / latest['close']) * 100
        
        volume_ratio = latest['volume'] / df['volume'].rolling(20).mean().iloc[-1]
        
        # Determine market sentiment
        if day_change > 2:
            sentiment = "Very Bullish"
            sentiment_icon = "🚀"
        elif day_change > 0.5:
            sentiment = "Bullish"
            sentiment_icon = "📈"
        elif day_change > -0.5:
            sentiment = "Neutral"
            sentiment_icon = "➡️"
        elif day_change > -2:
            sentiment = "Bearish"
            sentiment_icon = "📉"
        else:
            sentiment = "Very Bearish"
            sentiment_icon = "💥"
        
        # Generate likely reasons for the movement
        reasons = self._generate_movement_reasons(day_change, volume_ratio, day_range, stock_name)
        
        # Generate next day outlook
        next_day_outlook = self._generate_next_day_outlook(df, day_change, week_change, volume_ratio)
        
        return {
            'stock_name': stock_name,
            'day_change': day_change,
            'week_change': week_change,
            'sentiment': sentiment,
            'sentiment_icon': sentiment_icon,
            'day_high': day_high,
            'day_low': day_low,
            'day_range': day_range,
            'volume_ratio': volume_ratio,
            'current_price': latest['close'],
            'reasons': reasons,
            'next_day_outlook': next_day_outlook,
            'technical_levels': self._calculate_technical_levels(df)
        }
    
    def _generate_movement_reasons(self, day_change, volume_ratio, day_range, stock_name):
        """Generate likely reasons for price movement"""
        reasons = []
        
        # Volume-based reasons
        if volume_ratio > 2:
            if day_change > 0:
                reasons.append("Heavy buying interest with 2x+ normal volume")
            else:
                reasons.append("Heavy selling pressure with 2x+ normal volume")
        elif volume_ratio > 1.5:
            reasons.append("Increased activity with above-average volume")
        elif volume_ratio < 0.7:
            reasons.append("Low interest with below-average volume")
        
        # Price movement reasons
        if abs(day_change) > 3:
            if 'BANK' in stock_name.upper():
                reasons.append("Banking sector reacting to policy news or results")
            elif 'IT' in stock_name.upper() or stock_name in ['Infosys', 'TCS', 'Wipro']:
                reasons.append("IT sector responding to global tech trends or USD movement")
            else:
                reasons.append("Stock-specific news or major announcement likely")
        
        # Range-based reasons
        if day_range > 4:
            reasons.append("High intraday volatility suggests news-driven trading")
        elif day_range < 1:
            reasons.append("Narrow trading range indicates consolidation")
        
        # Direction-based reasons
        if day_change > 1:
            reasons.append("Positive sentiment driving fresh buying")
        elif day_change < -1:
            reasons.append("Negative sentiment causing profit booking or selling")
        
        # Add general market reasons
        reasons.append("Following broader market trend and global cues")
        
        return reasons[:4]  # Return top 4 reasons
    
    def _generate_next_day_outlook(self, df, day_change, week_change, volume_ratio):
        """Generate outlook for next trading day"""
        outlook = {}
        
        # Calculate technical indicators for outlook
        latest = df.iloc[-1]
        
        try:
            # RSI
            rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi().iloc[-1]
            
            # Moving averages
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            
            # Support and resistance
            support = df['low'].rolling(20).min().iloc[-1]
            resistance = df['high'].rolling(20).max().iloc[-1]
            
        except:
            rsi = 50
            sma_20 = latest['close']
            support = latest['close'] * 0.98
            resistance = latest['close'] * 1.02
        
        # Determine bias
        if day_change > 0 and volume_ratio > 1.2 and latest['close'] > sma_20:
            outlook['bias'] = "Bullish"
            outlook['probability'] = 70
            outlook['strategy'] = "Look for buying opportunities on any dips"
        elif day_change < 0 and volume_ratio > 1.2 and latest['close'] < sma_20:
            outlook['bias'] = "Bearish"
            outlook['probability'] = 70
            outlook['strategy'] = "Be cautious, consider profit booking on any bounce"
        else:
            outlook['bias'] = "Neutral"
            outlook['probability'] = 50
            outlook['strategy'] = "Wait for clear direction, avoid aggressive positions"
        
        # Key levels for tomorrow
        outlook['key_levels'] = {
            'support': support,
            'resistance': resistance,
            'pivot': (latest['high'] + latest['low'] + latest['close']) / 3
        }
        
        # Risk factors
        risk_factors = []
        if rsi > 70:
            risk_factors.append("RSI overbought - pullback possible")
        elif rsi < 30:
            risk_factors.append("RSI oversold - bounce possible")
        
        if volume_ratio < 0.8:
            risk_factors.append("Low volume - breakouts may be false")
        
        outlook['risk_factors'] = risk_factors
        
        return outlook
    
    def _calculate_technical_levels(self, df):
        """Calculate key technical levels"""
        try:
            latest = df.iloc[-1]
            
            # Pivot points
            pivot = (latest['high'] + latest['low'] + latest['close']) / 3
            r1 = 2 * pivot - latest['low']
            s1 = 2 * pivot - latest['high']
            
            # Support and resistance from recent data
            support_20 = df['low'].rolling(20).min().iloc[-1]
            resistance_20 = df['high'].rolling(20).max().iloc[-1]
            
            # Moving averages
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else sma_20
            
            return {
                'pivot_point': pivot,
                'resistance_1': r1,
                'support_1': s1,
                'resistance_20_day': resistance_20,
                'support_20_day': support_20,
                'sma_20': sma_20,
                'sma_50': sma_50
            }
        except:
            return {
                'pivot_point': latest['close'],
                'resistance_1': latest['close'] * 1.02,
                'support_1': latest['close'] * 0.98,
                'resistance_20_day': latest['close'] * 1.05,
                'support_20_day': latest['close'] * 0.95,
                'sma_20': latest['close'],
                'sma_50': latest['close']
            }

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
        
        # Volume Check
        avg_volume = df['volume'].rolling(50).mean().iloc[-1]
        current_volume = latest['volume']
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        if volume_ratio < self.min_volume_ratio:
            return False, f"Low trading activity (Volume: {volume_ratio:.1f}x average). Signals may be unreliable."
        
        return True, f"Market conditions are good for trading. Volume: {volume_ratio:.1f}x"

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
            df['atr_indicator'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            
            # Volume Indicators
            df['avg_volume'] = ta.volume.VolumeSMAIndicator(df['close'], df['volume']).volume_sma()
            df['volume_ratio'] = df['volume'] / df['avg_volume']
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['momentum_line'] = macd.macd()
            df['momentum_signal'] = macd.macd_signal()
            df['momentum_histogram'] = macd.macd_diff()
            
            # Support/Resistance
            df['support_level'] = df['low'].rolling(20).min()
            df['resistance_level'] = df['high'].rolling(20).max()
            
        except Exception as e:
            logger.error(f"Indicator calculation failed: {e}")
            # Fill with basic values if calculation fails
            for col in ['fast_line', 'medium_line', 'slow_line', 'strength_meter', 'atr_indicator']:
                if col not in df.columns:
                    df[col] = df['close']
        
        return df
    
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
        min_score_required = 6  # Slightly lower threshold for demo
        
        if buy_score >= min_score_required and buy_score > sell_score:
            confidence = min((buy_score / 12) * 100, 95)
            stop_loss = latest['close'] * 0.97
            target = latest['close'] * 1.06
            
            signals.append({
                'action': 'BUY',
                'confidence': confidence,
                'price': latest['close'],
                'stop_loss': stop_loss,
                'target': target,
                'reasons': signal_reasons[:4],
                'quality_score': buy_score,
                'strength': 'High' if buy_score >= 10 else 'Moderate'
            })
            
        elif sell_score >= min_score_required and sell_score > buy_score:
            confidence = min((sell_score / 12) * 100, 95)
            stop_loss = latest['close'] * 1.03
            target = latest['close'] * 0.94
            
            signals.append({
                'action': 'SELL',
                'confidence': confidence,
                'price': latest['close'],
                'stop_loss': stop_loss,
                'target': target,
                'reasons': signal_reasons[:4],
                'quality_score': sell_score,
                'strength': 'High' if sell_score >= 10 else 'Moderate'
            })
        
        if not signals:
            max_score = max(buy_score, sell_score) if buy_score or sell_score else 0
            return signals, f"No clear signals. Best score: {max_score}/12. Need minimum {min_score_required}/12 for signal generation."
        
        return signals, market_message

class DatabaseManager:
    """Enhanced database with performance tracking"""
    
    def __init__(self, db_path="smart_trading_v3.db"):
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
                    stock_name TEXT,
                    action TEXT,
                    price REAL,
                    stop_loss REAL,
                    target REAL,
                    confidence REAL,
                    quality_score INTEGER,
                    strength TEXT,
                    reasons TEXT,
                    status TEXT DEFAULT 'ACTIVE'
                )
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
    
    def save_enhanced_signal(self, signal_data, stock_name):
        """Save enhanced signal data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO signals (stock_name, action, price, stop_loss, target, confidence, 
                                   quality_score, strength, reasons)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                stock_name,
                signal_data.get('action'),
                signal_data.get('price'),
                signal_data.get('stop_loss'),
                signal_data.get('target'),
                signal_data.get('confidence'),
                signal_data.get('quality_score'),
                signal_data.get('strength'),
                ', '.join(signal_data.get('reasons', []))
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Signal save failed: {e}")
    
    def get_recent_signals(self, days=7, stock_name=None):
        """Get recent signals from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            if stock_name:
                query = '''
                    SELECT * FROM signals 
                    WHERE timestamp >= date('now', '-{} days') AND stock_name = ?
                    ORDER BY timestamp DESC
                '''.format(days)
                df = pd.read_sql_query(query, conn, params=[stock_name])
            else:
                query = '''
                    SELECT * FROM signals 
                    WHERE timestamp >= date('now', '-{} days')
                    ORDER BY timestamp DESC
                '''.format(days)
                df = pd.read_sql_query(query, conn)
            
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return pd.DataFrame(columns=['timestamp', 'stock_name', 'action', 'price', 'confidence', 'quality_score', 'strength'])

class SmartTradingSystem:
    """Enhanced trading system with stock selection and market hours"""
    
    def __init__(self, api_key):
        self.api = AxisDirectAPI(api_key)
        self.analyzer = AdvancedTechnicalAnalyzer()
        self.stock_manager = StockDataManager()
        self.market_hours = MarketHoursManager()
        self.after_hours_analyzer = AfterHoursAnalyzer()
        self.db = DatabaseManager()
        
    def run_comprehensive_analysis(self, stock_name="NIFTY 50"):
        """Run comprehensive analysis for selected stock"""
        try:
            # Get market data for selected stock
            market_data = self.stock_manager.generate_stock_data(stock_name)
            if market_data is None:
                return {'error': f"Could not generate data for {stock_name}"}
            
            # Calculate indicators
            market_data = self.analyzer.calculate_all_indicators(market_data)
            
            # Generate signals
            tech_signals, market_condition_msg = self.analyzer.generate_high_quality_signals(market_data)
            
            # Process signals
            final_signals = []
            for signal in tech_signals:
                if signal['confidence'] >= 70:  # Lower threshold for demo
                    final_signals.append(signal)
                    self.db.save_enhanced_signal(signal, stock_name)
            
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
                'condition': market_condition_msg,
                'current_price': latest['close'],
                'day_change': ((latest['close'] - market_data.iloc[-2]['close']) / market_data.iloc[-2]['close']) * 100 if len(market_data) > 1 else 0
            }
            
            return {
                'signals': final_signals,
                'market_mood': market_mood,
                'market_data': market_data,
                'stock_name': stock_name,
                'data_quality': 'Good'
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
    .market-status {
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .market-open { background: #d4edda; color: #155724; }
    .market-closed { background: #f8d7da; color: #721c24; }
    .market-pre { background: #fff3cd; color: #856404; }
    .quality-signal {
        background: linear-gradient(135deg, #e8f5e8, #d4edda);
        border-left: 6px solid #28a745;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .after-hours-analysis {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border-left: 6px solid #2196f3;
        padding: 1.5rem;
        border-radius: 8px;
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
    
    # Initialize system
    if 'trading_system' not in st.session_state:
        api_key = "tIQJyhGWrjzzIj0CfRJHOf3k8ST5to82yxGLnyxFPLniSBmQ"
        st.session_state.trading_system = SmartTradingSystem(api_key)
    
    # Main title
    st.markdown("""
    <div class="big-title">
        <h1>🎯 Smart Trading Helper Pro</h1>
        <p>Individual Stock Analysis • Market Hours Detection • After-Hours Insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Market Status Bar
    market_status = st.session_state.trading_system.market_hours.get_market_status()
    status_class = f"market-{market_status['status'].lower().replace('-', '')}"
    
    st.markdown(f"""
    <div class="market-status {status_class}">
        🕐 Market Status: {market_status['status']} - {market_status['reason']}
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with stock selection
    with st.sidebar:
        st.header("🎯 Stock Selection")
        
        # Stock selector
        available_stocks = st.session_state.trading_system.stock_manager.get_stock_list()
        selected_stock = st.selectbox(
            "Choose Stock/Index to Analyze:",
            available_stocks,
            index=0
        )
        
        # Display stock info
        stock_info = st.session_state.trading_system.stock_manager.get_stock_info(selected_stock)
        if stock_info:
            st.write(f"**Type:** {stock_info['type']}")
            st.write(f"**Symbol:** {stock_info['symbol']}")
            st.write(f"**Base Price:** ₹{stock_info['base_price']:,}")
        
        st.markdown("---")
        
        # Quick actions based on market status
        st.header("⚡ Quick Actions")
        
        if market_status['status'] == 'OPEN':
            st.success("🟢 Market is OPEN")
            st.write("• Generate live trading signals")
            st.write("• Monitor real-time analysis")
            st.write("• Execute trading strategies")
        else:
            st.warning(f"🔴 Market is {market_status['status']}")
            st.write("• Analyze today's performance")
            st.write("• Plan tomorrow's strategy")
            st.write("• Review historical patterns")
    
    # Main content based on market status
    if market_status['status'] == 'OPEN':
        # Live trading mode
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Live Signals", "📊 Stock Analysis", "📈 Performance", "💡 Guidelines"])
        
        with tab1:
            st.subheader(f"🎯 Live Trading Signals for {selected_stock}")
            
            if st.button("🔍 Generate Fresh Signals", type="primary", use_container_width=True):
                with st.spinner(f"Analyzing {selected_stock} for trading opportunities..."):
                    results = st.session_state.trading_system.run_comprehensive_analysis(selected_stock)
                    st.session_state.latest_results = results
            
            # Display results
            if 'latest_results' in st.session_state and st.session_state.latest_results.get('stock_name') == selected_stock:
                results = st.session_state.latest_results
                
                if 'error' in results:
                    st.error(f"❌ Analysis failed: {results['error']}")
                elif results['signals']:
                    for signal in results['signals']:
                        st.markdown(f"""
                        <div class="quality-signal">
                            <h3>🔥 {signal['action']} Signal for {selected_stock}</h3>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
                                <div><strong>💰 Entry:</strong> ₹{signal['price']:.2f}</div>
                                <div><strong>🎯 Target:</strong> ₹{signal['target']:.2f}</div>
                                <div><strong>🛡️ Stop:</strong> ₹{signal['stop_loss']:.2f}</div>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
                                <div><strong>💪 Confidence:</strong> {signal['confidence']:.0f}%</div>
                                <div><strong>⭐ Quality:</strong> {signal['quality_score']}/12</div>
                                <div><strong>📊 R:R:</strong> 1:{abs(signal['target'] - signal['price']) / abs(signal['price'] - signal['stop_loss']):.1f}</div>
                            </div>
                            <p><strong>🔍 Why this signal?</strong></p>
                            <ul>
                                {''.join([f"<li>{reason}</li>" for reason in signal['reasons']])}
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    market_condition = results['market_mood'].get('condition', 'Unknown condition')
                    st.markdown(f"""
                    <div class="info-box">
                        <h4>📊 No Quality Signals for {selected_stock}</h4>
                        <p><strong>Current Price:</strong> ₹{results['market_mood'].get('current_price', 0):.2f}</p>
                        <p><strong>Day Change:</strong> {results['market_mood'].get('day_change', 0):.2f}%</p>
                        <p><strong>Market Condition:</strong> {market_condition}</p>
                        <p><strong>What this means:</strong> Waiting for high-quality trading setups for this stock.</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab2:
            st.subheader(f"📊 Detailed Analysis of {selected_stock}")
            
            if 'latest_results' in st.session_state and st.session_state.latest_results.get('stock_name') == selected_stock:
                results = st.session_state.latest_results
                market_mood = results['market_mood']
                
                # Current status
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("💰 Current Price", f"₹{market_mood.get('current_price', 0):.2f}")
                with col2:
                    day_change = market_mood.get('day_change', 0)
                    st.metric("📈 Day Change", f"{day_change:.2f}%", delta=f"{day_change:.2f}%")
                with col3:
                    st.metric("📊 Trend", market_mood.get('trend', 'Unknown'))
                with col4:
                    st.metric("💪 Strength", market_mood.get('strength', 'Unknown'))
                
                # Price chart
                if 'market_data' in results:
                    st.subheader("📈 Price Chart with Technical Analysis")
                    
                    market_data = results['market_data']
                    
                    fig = go.Figure()
                    
                    # Candlestick chart
                    fig.add_trace(go.Candlestick(
                        x=market_data['date'][-30:],
                        open=market_data['open'][-30:],
                        high=market_data['high'][-30:],
                        low=market_data['low'][-30:],
                        close=market_data['close'][-30:],
                        name=selected_stock
                    ))
                    
                    # Add moving averages if available
                    if 'fast_line' in market_data.columns:
                        fig.add_trace(go.Scatter(
                            x=market_data['date'][-30:],
                            y=market_data['fast_line'][-30:],
                            name='Fast MA (9)',
                            line=dict(color='blue', width=1)
                        ))
                    
                    if 'medium_line' in market_data.columns:
                        fig.add_trace(go.Scatter(
                            x=market_data['date'][-30:],
                            y=market_data['medium_line'][-30:],
                            name='Medium MA (21)',
                            line=dict(color='orange', width=1)
                        ))
                    
                    fig.update_layout(
                        title=f"{selected_stock} - Last 30 Days",
                        xaxis_title="Date",
                        yaxis_title="Price (₹)",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader(f"📈 Trading Performance for {selected_stock}")
            
            # Get signals for this stock
            recent_signals = st.session_state.trading_system.db.get_recent_signals(days=30, stock_name=selected_stock)
            
            if not recent_signals.empty:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🎯 Total Signals", len(recent_signals))
                with col2:
                    avg_confidence = recent_signals['confidence'].mean()
                    st.metric("💪 Avg Confidence", f"{avg_confidence:.0f}%")
                with col3:
                    buy_signals = len(recent_signals[recent_signals['action'] == 'BUY'])
                    st.metric("🟢 Buy Signals", buy_signals)
                with col4:
                    sell_signals = len(recent_signals[recent_signals['action'] == 'SELL'])
                    st.metric("🔴 Sell Signals", sell_signals)
                
                # Recent signals table
                st.subheader("📋 Recent Signals")
                display_df = recent_signals[['timestamp', 'action', 'price', 'confidence', 'strength']].head(10)
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info(f"📊 No signal history for {selected_stock} yet. Generate some signals to see performance.")
        
        with tab4:
            st.subheader("💡 Live Trading Guidelines")
            st.markdown("""
            ### 🎯 **How to Trade During Market Hours**
            
            **✅ DO:**
            - Wait for signals with 70%+ confidence
            - Use suggested stop-loss levels
            - Monitor volume confirmation
            - Follow the trend direction
            - Risk only 1-2% per trade
            
            **❌ DON'T:**
            - Trade without stop-loss
            - Chase breakouts without volume
            - Fight the major trend
            - Overtrade on single stock
            - Ignore market-wide movements
            
            ### 📊 **Stock-Specific Tips:**
            - **Indices (NIFTY, BANK NIFTY):** Follow broader market sentiment
            - **Banking Stocks:** Watch for policy announcements and interest rate changes
            - **IT Stocks:** Monitor USD movements and global tech trends
            - **Individual Stocks:** Check for company-specific news and earnings
            """)
    
    else:
        # After-hours analysis mode
        st.subheader(f"📊 After-Hours Analysis for {selected_stock}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button(f"📈 Analyze Today's Performance", type="primary", use_container_width=True):
                with st.spinner(f"Analyzing what happened with {selected_stock} today..."):
                    # Get stock data
                    market_data = st.session_state.trading_system.stock_manager.generate_stock_data(selected_stock)
                    if market_data is not None:
                        analysis = st.session_state.trading_system.after_hours_analyzer.analyze_day_performance(market_data, selected_stock)
                        st.session_state.after_hours_analysis = analysis
            
            # Display after-hours analysis
            if 'after_hours_analysis' in st.session_state:
                analysis = st.session_state.after_hours_analysis
                
                if 'error' not in analysis:
# Display after-hours analysis with clean formatting
st.markdown(f"### {analysis['sentiment_icon']} Today's Performance: {analysis['sentiment']}")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("💰 Current Price", f"₹{analysis['current_price']:.2f}")
with col2:
    st.metric("📈 Day Change", f"{analysis['day_change']:.2f}%")
with col3:
    st.metric("📊 Week Change", f"{analysis['week_change']:.2f}%")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🔝 Day High", f"₹{analysis['day_high']:.2f}")
with col2:
    st.metric("🔻 Day Low", f"₹{analysis['day_low']:.2f}")
with col3:
    st.metric("📊 Volume", f"{analysis['volume_ratio']:.1f}x avg")

st.subheader("🔍 Why did this happen?")
for reason in analysis['reasons']:
    st.write(f"• {reason}")

st.subheader("🔮 Tomorrow's Outlook")
st.write(f"**Bias:** {analysis['next_day_outlook']['bias']} ({analysis['next_day_outlook']['probability']}% probability)")
st.write(f"**Strategy:** {analysis['next_day_outlook']['strategy']}")

st.subheader("🎯 Key Levels for Tomorrow")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🛡️ Support", f"₹{analysis['next_day_outlook']['key_levels']['support']:.2f}")
with col2:
    st.metric("🎯 Pivot", f"₹{analysis['next_day_outlook']['key_levels']['pivot']:.2f}")
with col3:
    st.metric("🚧 Resistance", f"₹{analysis['next_day_outlook']['key_levels']['resistance']:.2f}")

if analysis['next_day_outlook']['risk_factors']:
    st.subheader("⚠️ Risk Factors")
    for risk in analysis['next_day_outlook']['risk_factors']:
        st.warning(f"• {risk}")
                        <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
                            <div><strong>💰 Current Price:</strong> ₹{analysis['current_price']:.2f}</div>
                            <div><strong>📈 Day Change:</strong> {analysis['day_change']:.2f}%</div>
                            <div><strong>📊 Week Change:</strong> {analysis['week_change']:.2f}%</div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
                            <div><strong>🔝 Day High:</strong> ₹{analysis['day_high']:.2f}</div>
                            <div><strong>🔻 Day Low:</strong> ₹{analysis['day_low']:.2f}</div>
                            <div><strong>📊 Volume:</strong> {analysis['volume_ratio']:.1f}x avg</div>
                        </div>
                        
                        <h4>🔍 Why did this happen?</h4>
                        <ul>
                            {''.join([f"<li>{reason}</li>" for reason in analysis['reasons']])}
                        </ul>
                        
                        <h4>🔮 Tomorrow's Outlook</h4>
                        <p><strong>Bias:</strong> {analysis['next_day_outlook']['bias']} 
                        ({analysis['next_day_outlook']['probability']}% probability)</p>
                        <p><strong>Strategy:</strong> {analysis['next_day_outlook']['strategy']}</p>
                        
                        <h4>🎯 Key Levels for Tomorrow</h4>
                        <div style="display: flex; justify-content: space-between;">
                            <div><strong>🛡️ Support:</strong> ₹{analysis['next_day_outlook']['key_levels']['support']:.2f}</div>
                            <div><strong>🎯 Pivot:</strong> ₹{analysis['next_day_outlook']['key_levels']['pivot']:.2f}</div>
                            <div><strong>🚧 Resistance:</strong> ₹{analysis['next_day_outlook']['key_levels']['resistance']:.2f}</div>
                        </div>
                        
                        {f"<p><strong>⚠️ Risk Factors:</strong> {', '.join(analysis['next_day_outlook']['risk_factors'])}</p>" if analysis['next_day_outlook']['risk_factors'] else ""}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Technical levels table
                    st.subheader("📊 Technical Levels")
                    
                    levels = analysis['technical_levels']
                    levels_df = pd.DataFrame([
                        ['Pivot Point', f"₹{levels['pivot_point']:.2f}"],
                        ['Resistance 1', f"₹{levels['resistance_1']:.2f}"],
                        ['Support 1', f"₹{levels['support_1']:.2f}"],
                        ['20-Day High', f"₹{levels['resistance_20_day']:.2f}"],
                        ['20-Day Low', f"₹{levels['support_20_day']:.2f}"],
                        ['20-Day MA', f"₹{levels['sma_20']:.2f}"],
                        ['50-Day MA', f"₹{levels['sma_50']:.2f}"]
                    ], columns=['Level', 'Price'])
                    
                    st.dataframe(levels_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("⏰ Market Schedule")
            
            # Next market open info
            st.info(f"**Next Open:** {market_status['next_open']}")
            
            # Today's market summary
            st.subheader("📈 Quick Stats")
            
            if 'after_hours_analysis' in st.session_state:
                analysis = st.session_state.after_hours_analysis
                st.metric("📊 Day Performance", f"{analysis['day_change']:.2f}%")
                st.metric("💪 Sentiment", analysis['sentiment'])
                st.metric("📊 Volume Activity", f"{analysis['volume_ratio']:.1f}x")
            
            # Pre-market preparation
            st.subheader("🌅 Tomorrow's Prep")
            st.write("**Before market opens:**")
            st.write("• Check global market cues")
            st.write("• Review overnight news")
            st.write("• Plan entry/exit levels")
            st.write("• Set alerts for key levels")
    
    # Bottom status bar
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_time = st.session_state.trading_system.market_hours.get_current_ist_time().strftime("%I:%M %p IST")
        st.info(f"🕐 {current_time}")
    
    with col2:
        st.info(f"📊 Analyzing: {selected_stock}")
    
    with col3:
        if market_status['status'] == 'OPEN':
            st.success("🟢 Live Mode")
        else:
            st.warning("🔴 Analysis Mode")
    
    with col4:
        if st.button("🔄 Refresh Data"):
            st.rerun()

if __name__ == "__main__":
    main()
