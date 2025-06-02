# %% [markdown]
# # FVG (Fair Value Gap) Backtesting Strategy para XAU/USD
# ## Estrategia de Trading Algorítmico Cuantitativo Basada en Desequilibrios de Liquidez
# 
# **Versión:** 1.2 - Jupyter Notebook Enhanced Edition  
# **Autor:** Experto en Trading Algorítmico Cuantitativo  
# **Fecha:** Junio 2025
# 
# ### Descripción de la Estrategia
# 
# Esta estrategia identifica y opera **Fair Value Gaps (FVGs)** en el par XAU/USD utilizando:
# - **Análisis multi-timeframe**: 4H para detección de FVGs principales y 5M para entradas precisas
# - **Filtros de tendencia**: EMA 200 en 4H
# - **Análisis de volumen avanzado**: Confirmación con volumen ascendente
# - **Gestión de riesgo dinámica**: Stop loss que se ajusta con nuevos FVGs en 5M
# - **Risk-Reward optimizado**: Ratio 1:2 por defecto

# %% [markdown]
# ## 1. Importación de Librerías y Configuración Inicial

# %%
# Importación de librerías necesarias
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime, timedelta
import warnings
import seaborn as sns
from IPython.display import display, HTML
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración de visualización
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', '{:.2f}'.format)

# %% [markdown]
# ## 2. Definición de la Clase FVGBacktester Mejorada

# %%
class FVGBacktester:
    """
    Backtester avanzado para estrategia FVG en XAU/USD
    Incluye análisis multi-timeframe y gestión dinámica de riesgo
    """
    
    def __init__(self, symbol="GC=F", start_date="2020-01-01", 
                 initial_capital=10000, risk_per_trade=0.005):
        """
        Initialize the FVG Backtester
        
        Parameters:
        - symbol: Símbolo XAU/USD para yfinance (GC=F para Gold Futures)
        - start_date: Fecha de inicio de datos históricos
        - initial_capital: Capital inicial
        - risk_per_trade: Porcentaje de riesgo por trade (0.005 = 0.5%)
        """
        self.symbol = symbol
        self.start_date = start_date
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        
        # Parámetros de la estrategia
        self.min_gap_pips = 30  # Tamaño mínimo del gap en pips
        self.ema_period = 200   # Período EMA para filtro de tendencia
        self.tp_multiple = 2.0  # Ratio riesgo-beneficio
        
        # Contenedores de datos
        self.data_4h = None
        self.data_5m = None
        self.fvg_zones_4h = []  # Zonas FVG del timeframe 4H
        self.fvg_zones_5m = []  # Zonas FVG del timeframe 5M para SL dinámico
        self.trades = []
        self.equity_curve = []
        
        # Estadísticas adicionales
        self.monthly_returns = []
        self.daily_returns = []
        
    def download_data(self):
        """Descarga datos 4H y 5M para XAU/USD"""
        print(f"📊 Descargando datos para {self.symbol}...")
        
        try:
            # Descargar datos 4H para análisis de tendencia y detección FVG
            ticker = yf.Ticker(self.symbol)
            self.data_4h = ticker.history(
                start=self.start_date,
                interval="1h",  # Usando 1h como proxy para 4h
                actions=False
            )
            
            # Convertir a 4H mediante resampling
            self.data_4h = self.data_4h.resample('4H').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            # Descargar datos 5M para entradas (últimos 60 días para rendimiento)
            start_5m = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
            self.data_5m = ticker.history(
                start=start_5m,
                interval="5m",
                actions=False
            )
            
            print(f"✅ Datos 4H descargados: {self.data_4h.shape}")
            print(f"✅ Datos 5M descargados: {self.data_5m.shape}")
            
            if self.data_4h.empty or self.data_5m.empty:
                raise ValueError("No se pudieron descargar datos")
                
        except Exception as e:
            print(f"⚠️ Error descargando datos: {e}")
            print("🔄 Generando datos de demostración...")
            self._create_demo_data()
    
    def _create_demo_data(self):
        """Crear datos de demostración si la descarga falla"""
        dates_4h = pd.date_range(start=self.start_date, end='2025-06-01', freq='4H')
        dates_5m = pd.date_range(start='2025-04-01', end='2025-06-01', freq='5T')
        
        # Generar datos realistas de precio del oro
        np.random.seed(42)
        base_price = 2000
        
        # Datos 4H con tendencia y volatilidad realista
        trend = np.linspace(0, 200, len(dates_4h))
        noise = np.cumsum(np.random.randn(len(dates_4h)) * 5)
        price_4h = base_price + trend + noise
        
        self.data_4h = pd.DataFrame({
            'Open': price_4h + np.random.randn(len(dates_4h)) * 2,
            'High': price_4h + np.abs(np.random.randn(len(dates_4h)) * 10),
            'Low': price_4h - np.abs(np.random.randn(len(dates_4h)) * 10),
            'Close': price_4h,
            'Volume': np.random.randint(1000, 10000, len(dates_4h))
        }, index=dates_4h)
        
        # Datos 5M con mayor granularidad
        trend_5m = np.linspace(0, 50, len(dates_5m))
        noise_5m = np.cumsum(np.random.randn(len(dates_5m)) * 0.5)
        price_5m = 2100 + trend_5m + noise_5m
        
        self.data_5m = pd.DataFrame({
            'Open': price_5m + np.random.randn(len(dates_5m)) * 1,
            'High': price_5m + np.abs(np.random.randn(len(dates_5m)) * 3),
            'Low': price_5m - np.abs(np.random.randn(len(dates_5m)) * 3),
            'Close': price_5m,
            'Volume': np.random.randint(100, 1000, len(dates_5m))
        }, index=dates_5m)
        
        print("✅ Datos de demostración creados exitosamente")
    
    def calculate_indicators(self):
        """Calcular indicadores EMA y volumen"""
        print("📈 Calculando indicadores técnicos...")
        
        # EMA 200 en 4H para filtro de tendencia
        self.data_4h['EMA_200'] = self.data_4h['Close'].ewm(span=self.ema_period).mean()
        self.data_4h['Bullish_Trend'] = self.data_4h['Close'] > self.data_4h['EMA_200']
        
        # FILTRO DE VOLUMEN MEJORADO - Múltiples criterios para volumen ascendente
        # 1. Media móvil de volumen (20 períodos)
        self.data_5m['Volume_MA_20'] = self.data_5m['Volume'].rolling(20).mean()
        
        # 2. Tendencia de volumen en últimas 5 barras
        self.data_5m['Volume_Trend_5'] = self.data_5m['Volume'].rolling(5).apply(
            lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0, raw=False
        )
        
        # 3. Volumen actual vs promedio reciente
        self.data_5m['Volume_Above_Avg'] = self.data_5m['Volume'] > self.data_5m['Volume_MA_20']
        
        # 4. Aceleración de volumen (actual > barra anterior)
        self.data_5m['Volume_Acceleration'] = self.data_5m['Volume'] > self.data_5m['Volume'].shift(1)
        
        # FILTRO DE VOLUMEN COMBINADO
        self.data_5m['Volume_Rising'] = (
            self.data_5m['Volume_Above_Avg'] & 
            (self.data_5m['Volume_Trend_5'] == 1) &
            self.data_5m['Volume_Acceleration']
        )
        
        # Calcular ATR para análisis de volatilidad
        self.data_4h['ATR'] = self._calculate_atr(self.data_4h, period=14)
        self.data_5m['ATR'] = self._calculate_atr(self.data_5m, period=14)
        
        print(f"✅ Indicadores calculados")
        print(f"📊 Filtro de volumen ascendente activado: {self.data_5m['Volume_Rising'].sum()} barras de {len(self.data_5m)}")
    
    def _calculate_atr(self, data, period=14):
        """Calcular Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
    
    def detect_fvg_zones(self):
        """
        Detectar zonas Fair Value Gap en ambos timeframes
        4H FVG: Para señales principales de trading
        5M FVG: Para ajustes dinámicos de stop loss
        """
        print("🔍 Detectando zonas FVG...")
        self.fvg_zones_4h = []
        
        # Detectar zonas FVG en 4H para señales principales
        for i in range(2, len(self.data_4h)):
            # FVG Alcista: máximo de hace 2 barras < mínimo de barra actual
            if (self.data_4h.iloc[i-2]['High'] < self.data_4h.iloc[i]['Low'] and
                self.data_4h.iloc[i]['Low'] - self.data_4h.iloc[i-2]['High'] >= self.min_gap_pips * 0.1):
                
                fvg = {
                    'timestamp': self.data_4h.index[i],
                    'high': self.data_4h.iloc[i-2]['High'],
                    'low': self.data_4h.iloc[i]['Low'],
                    'middle': (self.data_4h.iloc[i-2]['High'] + self.data_4h.iloc[i]['Low']) / 2,
                    'size': self.data_4h.iloc[i]['Low'] - self.data_4h.iloc[i-2]['High'],
                    'filled': False,
                    'direction': 'bullish',
                    'timeframe': '4H'
                }
                self.fvg_zones_4h.append(fvg)
        
        print(f"✅ Encontradas {len(self.fvg_zones_4h)} zonas FVG en 4H")
        
        # Detectar zonas FVG en 5M para stop loss dinámico
        self.fvg_zones_5m = []
        
        for i in range(2, len(self.data_5m)):
            # FVG Alcista en 5M: gap mínimo más pequeño (10 pips)
            if (self.data_5m.iloc[i-2]['High'] < self.data_5m.iloc[i]['Low'] and
                self.data_5m.iloc[i]['Low'] - self.data_5m.iloc[i-2]['High'] >= 10 * 0.1):
                
                fvg = {
                    'timestamp': self.data_5m.index[i],
                    'high': self.data_5m.iloc[i-2]['High'],
                    'low': self.data_5m.iloc[i]['Low'],
                    'middle': (self.data_5m.iloc[i-2]['High'] + self.data_5m.iloc[i]['Low']) / 2,
                    'size': self.data_5m.iloc[i]['Low'] - self.data_5m.iloc[i-2]['High'],
                    'direction': 'bullish',
                    'timeframe': '5M'
                }
                self.fvg_zones_5m.append(fvg)
        
        print(f"✅ Encontradas {len(self.fvg_zones_5m)} zonas FVG en 5M para SL dinámico")
    
    def get_trend_at_time(self, timestamp):
        """Obtener tendencia 4H en timestamp específico"""
        closest_4h = self.data_4h.index[self.data_4h.index <= timestamp]
        if len(closest_4h) == 0:
            return False
        
        latest_4h = closest_4h[-1]
        return self.data_4h.loc[latest_4h, 'Bullish_Trend']
    
    def backtest_strategy(self):
        """Ejecutar el backtesting de la estrategia"""
        print("🚀 Iniciando backtesting...")
        
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.current_capital = self.initial_capital
        
        position = None  # Posición actual
        trades_count = 0
        
        # Barra de progreso simple
        total_bars = len(self.data_5m)
        progress_interval = total_bars // 20
        
        for i in range(1, len(self.data_5m)):
            if i % progress_interval == 0:
                print(f"Progreso: {(i/total_bars)*100:.0f}%", end='\r')
            
            current_time = self.data_5m.index[i]
            current_bar = self.data_5m.iloc[i]
            
            # Verificar salida de posición primero
            if position:
                # Verificar stop loss
                if current_bar['Low'] <= position['stop_loss']:
                    self._close_position(position, position['stop_loss'], current_time, 'Stop Loss')
                    position = None
                    trades_count += 1
                    continue
                
                # Verificar take profit
                elif current_bar['High'] >= position['take_profit']:
                    self._close_position(position, position['take_profit'], current_time, 'Take Profit')
                    position = None
                    trades_count += 1
                    continue
                
                # Ajuste dinámico de stop loss usando FVGs de 5M
                self._update_stop_loss_5m(position, current_time)
            
            # Buscar nuevas entradas si no hay posición
            if not position and trades_count < 100:  # Limitar trades para demo
                # Verificar cada zona FVG activa de 4H
                for fvg in self.fvg_zones_4h:
                    if fvg['filled']:
                        continue
                    
                    # Verificar tendencia correcta
                    if not self.get_trend_at_time(current_time):
                        continue
                    
                    # Verificar filtro de volumen mejorado
                    if not self.data_5m.loc[current_time, 'Volume_Rising']:
                        continue
                    
                    # Lógica de entrada: precio barrió entre 50%-100% del gap y cerró sobre el gap
                    gap_50_pct = fvg['high'] + (fvg['middle'] - fvg['high']) * 0.5
                    
                    # Verificar barrido (mínimo tocó entre 50%-100% del gap)
                    if (current_bar['Low'] <= fvg['middle'] and 
                        current_bar['Low'] >= gap_50_pct and
                        current_bar['Close'] > fvg['low']):
                        
                        # Entrar en posición
                        entry_price = fvg['middle']
                        stop_loss = fvg['high'] - 0.1
                        risk_amount = self.current_capital * self.risk_per_trade
                        
                        # Calcular tamaño de posición basado en riesgo
                        risk_per_unit = abs(entry_price - stop_loss)
                        position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
                        
                        if position_size > 0:
                            take_profit = entry_price + (entry_price - stop_loss) * self.tp_multiple
                            
                            position = {
                                'entry_time': current_time,
                                'entry_price': entry_price,
                                'stop_loss': stop_loss,
                                'initial_stop_loss': stop_loss,
                                'take_profit': take_profit,
                                'position_size': position_size,
                                'fvg_id': fvg['timestamp'],
                                'fvg_size': fvg['size']
                            }
                            
                            # Marcar FVG como usada
                            fvg['filled'] = True
                            break
            
            # Actualizar curva de equity
            if position:
                unrealized_pnl = (current_bar['Close'] - position['entry_price']) * position['position_size']
                current_equity = self.current_capital + unrealized_pnl
            else:
                current_equity = self.current_capital
            
            self.equity_curve.append(current_equity)
        
        print(f"\n✅ Backtesting completado. Total de trades: {len(self.trades)}")
    
    def _close_position(self, position, exit_price, exit_time, reason):
        """Cerrar una posición y registrar el trade"""
        pnl = (exit_price - position['entry_price']) * position['position_size']
        self.current_capital += pnl
        
        trade = {
            'entry_time': position['entry_time'],
            'exit_time': exit_time,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'position_size': position['position_size'],
            'pnl': pnl,
            'return_pct': (pnl / (self.current_capital - pnl)) * 100,
            'reason': reason,
            'initial_sl': position['initial_stop_loss'],
            'final_sl': position['stop_loss'],
            'duration': exit_time - position['entry_time'],
            'fvg_size': position.get('fvg_size', 0)
        }
        
        self.trades.append(trade)
    
    def _update_stop_loss_5m(self, position, current_time):
        """
        Actualizar stop loss basado en NUEVOS FVGs de 5M que se forman después de la entrada
        """
        for fvg in self.fvg_zones_5m:
            if (fvg['timestamp'] > position['entry_time'] and
                fvg['timestamp'] <= current_time and
                fvg['direction'] == 'bullish'):
                
                # Nuevo stop loss justo debajo del máximo del nuevo FVG de 5M
                new_stop = fvg['high'] - 0.1
                
                # Solo actualizar si el nuevo stop loss es más alto (más favorable)
                if new_stop > position['stop_loss']:
                    position['stop_loss'] = new_stop
    
    def calculate_metrics(self):
        """Calcular métricas de rendimiento completas"""
        if not self.trades:
            return {}
        
        trades_df = pd.DataFrame(self.trades)
        
        # Métricas básicas
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Métricas de P&L
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Métricas de riesgo
        returns = trades_df['return_pct'].values
        sharpe_ratio = np.sqrt(252) * (np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0
        
        # Cálculo de drawdown
        equity_series = pd.Series(self.equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        # Expectativa matemática
        expectancy = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss) if total_trades > 0 else 0
        
        # Análisis de movimiento de stop loss
        sl_moved_trades = len(trades_df[trades_df['final_sl'] != trades_df['initial_sl']])
        sl_movement_rate = (sl_moved_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Métricas adicionales
        avg_trade_duration = trades_df['duration'].mean()
        best_trade = trades_df['pnl'].max()
        worst_trade = trades_df['pnl'].min()
        consecutive_wins = self._calculate_consecutive_wins(trades_df)
        consecutive_losses = self._calculate_consecutive_losses(trades_df)
        
        return {
            'Total Trades': total_trades,
            'Win Rate (%)': win_rate,
            'Total P&L ($)': total_pnl,
            'Net Return (%)': (total_pnl / self.initial_capital) * 100,
            'Average Win ($)': avg_win,
            'Average Loss ($)': avg_loss,
            'Profit Factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': max_drawdown,
            'Mathematical Expectancy ($)': expectancy,
            'Final Capital ($)': self.current_capital,
            'SL Movement Rate (%)': sl_movement_rate,
            'Trades with SL Moved': sl_moved_trades,
            'Avg Trade Duration': avg_trade_duration,
            'Best Trade ($)': best_trade,
            'Worst Trade ($)': worst_trade,
            'Max Consecutive Wins': consecutive_wins,
            'Max Consecutive Losses': consecutive_losses
        }
    
    def _calculate_consecutive_wins(self, trades_df):
        """Calcular máxima racha de trades ganadores"""
        max_consecutive = 0
        current_consecutive = 0
        
        for pnl in trades_df['pnl']:
            if pnl > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_consecutive_losses(self, trades_df):
        """Calcular máxima racha de trades perdedores"""
        max_consecutive = 0
        current_consecutive = 0
        
        for pnl in trades_df['pnl']:
            if pnl <= 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def plot_results(self):
        """Crear visualizaciones mejoradas de los resultados"""
        # Crear figura con subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Curva de Equity
        ax1 = plt.subplot(3, 3, 1)
        equity_df = pd.DataFrame({'Equity': self.equity_curve})
        ax1.plot(equity_df.index, equity_df['Equity'], linewidth=2, color='green')
        ax1.fill_between(equity_df.index, self.initial_capital, equity_df['Equity'], 
                        where=equity_df['Equity'] >= self.initial_capital, 
                        color='green', alpha=0.3, label='Profit')
        ax1.fill_between(equity_df.index, self.initial_capital, equity_df['Equity'], 
                        where=equity_df['Equity'] < self.initial_capital, 
                        color='red', alpha=0.3, label='Loss')
        ax1.axhline(y=self.initial_capital, color='black', linestyle='--', alpha=0.5)
        ax1.set_title('Curva de Equity', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Capital ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax2 = plt.subplot(3, 3, 2)
        equity_series = pd.Series(self.equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        ax2.fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.7)
        ax2.set_title('Drawdown (%)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Distribución de P&L
        if self.trades:
            ax3 = plt.subplot(3, 3, 3)
            trade_pnls = [trade['pnl'] for trade in self.trades]
            positive_pnls = [pnl for pnl in trade_pnls if pnl > 0]
            negative_pnls = [pnl for pnl in trade_pnls if pnl <= 0]
            
            bins = 30
            ax3.hist(positive_pnls, bins=bins, alpha=0.7, color='green', label='Ganadores', edgecolor='black')
            ax3.hist(negative_pnls, bins=bins, alpha=0.7, color='red', label='Perdedores', edgecolor='black')
            ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            ax3.set_title('Distribución de P&L', fontsize=12, fontweight='bold')
            ax3.set_xlabel('P&L ($)')
            ax3.set_ylabel('Frecuencia')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Precio con FVG Zones (muestra)
        ax4 = plt.subplot(3, 3, 4)
        sample_data = self.data_5m.tail(500)
        ax4.plot(sample_data.index, sample_data['Close'], linewidth=1, color='blue', label='Precio')
        
        # Mostrar últimas zonas FVG
        recent_fvgs = [fvg for fvg in self.fvg_zones_4h if fvg['timestamp'] > sample_data.index[0]][-3:]
        for fvg in recent_fvgs:
            rect = patches.Rectangle((fvg['timestamp'], fvg['high']), 
                                   timedelta(hours=16), fvg['size'],
                                   linewidth=1, edgecolor='green', 
                                   facecolor='green', alpha=0.3)
            ax4.add_patch(rect)
        
        ax4.set_title('Precio con Zonas FVG (Muestra)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Precio ($)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Análisis de Volumen
        ax5 = plt.subplot(3, 3, 5)
        volume_data = sample_data.tail(200)
        colors = ['green' if rising else 'red' for rising in volume_data['Volume_Rising']]
        ax5.bar(volume_data.index, volume_data['Volume'], alpha=0.7, color=colors)
        ax5.plot(volume_data.index, volume_data['Volume_MA_20'], 
                color='orange', linewidth=2, label='MA(20)')
        ax5.set_title('Análisis de Volumen', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Volumen')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Returns por Trade
        if self.trades:
            ax6 = plt.subplot(3, 3, 6)
            trades_df = pd.DataFrame(self.trades)
            returns = trades_df['return_pct'].values
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns <= 0]
            
            ax6.scatter(range(len(positive_returns)), positive_returns, 
                       color='green', alpha=0.6, label='Ganadores')
            ax6.scatter(range(len(positive_returns), len(returns)), negative_returns, 
                       color='red', alpha=0.6, label='Perdedores')
            ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax6.set_title('Retornos por Trade (%)', fontsize=12, fontweight='bold')
            ax6.set_xlabel('Número de Trade')
            ax6.set_ylabel('Retorno (%)')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 7. Duración de Trades
        if self.trades:
            ax7 = plt.subplot(3, 3, 7)
            trades_df = pd.DataFrame(self.trades)
            durations_hours = [d.total_seconds() / 3600 for d in trades_df['duration']]
            winning_durations = [durations_hours[i] for i in range(len(durations_hours)) 
                               if trades_df.iloc[i]['pnl'] > 0]
            losing_durations = [durations_hours[i] for i in range(len(durations_hours)) 
                              if trades_df.iloc[i]['pnl'] <= 0]
            
            ax7.hist(winning_durations, bins=20, alpha=0.7, color='green', 
                    label='Ganadores', edgecolor='black')
            ax7.hist(losing_durations, bins=20, alpha=0.7, color='red', 
                    label='Perdedores', edgecolor='black')
            ax7.set_title('Duración de Trades', fontsize=12, fontweight='bold')
            ax7.set_xlabel('Duración (horas)')
            ax7.set_ylabel('Frecuencia')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # 8. Relación FVG Size vs P&L
        if self.trades:
            ax8 = plt.subplot(3, 3, 8)
            trades_df = pd.DataFrame(self.trades)
            fvg_sizes = trades_df['fvg_size'].values
            pnls = trades_df['pnl'].values
            
            scatter = ax8.scatter(fvg_sizes, pnls, c=pnls, cmap='RdYlGn', 
                                 alpha=0.6, edgecolors='black', linewidth=1)
            ax8.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax8.set_title('Tamaño FVG vs P&L', fontsize=12, fontweight='bold')
            ax8.set_xlabel('Tamaño FVG')
            ax8.set_ylabel('P&L ($)')
            plt.colorbar(scatter, ax=ax8)
            ax8.grid(True, alpha=0.3)
        
        # 9. Tabla de Métricas
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('tight')
        ax9.axis('off')
        
        metrics = self.calculate_metrics()
        metrics_table = []
        for key, value in list(metrics.items())[:10]:  # Mostrar primeras 10 métricas
            if isinstance(value, float):
                metrics_table.append([key, f"{value:.2f}"])
            elif isinstance(value, pd.Timedelta):
                hours = value.total_seconds() / 3600
                metrics_table.append([key, f"{hours:.1f} hours"])
            else:
                metrics_table.append([key, str(value)])
        
        table = ax9.table(cellText=metrics_table, 
                         colLabels=['Métrica', 'Valor'],
                         cellLoc='left',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax9.set_title('Métricas Principales', fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_chart(self):
        """Crear gráfico interactivo con Plotly"""
        if not self.trades:
            print("No hay trades para mostrar")
            return
        
        # Preparar datos
        trades_df = pd.DataFrame(self.trades)
        
        # Crear subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Precio y Señales de Trading', 'Curva de Equity', 'Volumen'),
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # 1. Gráfico de precio con trades
        sample_data = self.data_5m.tail(1000)
        
        # Candlesticks
        fig.add_trace(
            go.Candlestick(
                x=sample_data.index,
                open=sample_data['Open'],
                high=sample_data['High'],
                low=sample_data['Low'],
                close=sample_data['Close'],
                name='XAU/USD'
            ),
            row=1, col=1
        )
        
        # Marcar entradas y salidas
        for trade in self.trades[-20:]:  # Últimos 20 trades
            if trade['entry_time'] in sample_data.index:
                # Entrada
                fig.add_trace(
                    go.Scatter(
                        x=[trade['entry_time']],
                        y=[trade['entry_price']],
                        mode='markers',
                        marker=dict(symbol='triangle-up', size=12, color='green'),
                        name='Entrada',
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                # Salida
                if trade['exit_time'] in sample_data.index:
                    color = 'blue' if trade['pnl'] > 0 else 'red'
                    fig.add_trace(
                        go.Scatter(
                            x=[trade['exit_time']],
                            y=[trade['exit_price']],
                            mode='markers',
                            marker=dict(symbol='triangle-down', size=12, color=color),
                            name='Salida',
                            showlegend=False
                        ),
                        row=1, col=1
                    )
        
        # 2. Curva de equity
        equity_df = pd.DataFrame({
            'index': range(len(self.equity_curve)),
            'equity': self.equity_curve
        })
        
        fig.add_trace(
            go.Scatter(
                x=equity_df['index'],
                y=equity_df['equity'],
                mode='lines',
                name='Equity',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        # Línea de capital inicial
        fig.add_hline(
            y=self.initial_capital, 
            line_dash="dash", 
            line_color="black",
            annotation_text="Capital Inicial",
            row=2, col=1
        )
        
        # 3. Volumen
        fig.add_trace(
            go.Bar(
                x=sample_data.index,
                y=sample_data['Volume'],
                name='Volumen',
                marker_color='lightblue'
            ),
            row=3, col=1
        )
        
        # Actualizar diseño
        fig.update_layout(
            title='Análisis Interactivo de Trading FVG - XAU/USD',
            xaxis_title='Tiempo',
            yaxis_title='Precio',
            height=1000,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Configurar rangeslider solo para el primer subplot
        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
        fig.update_xaxes(rangeslider_visible=False, row=2, col=1)
        fig.update_xaxes(rangeslider_visible=False, row=3, col=1)
        
        fig.show()
    
    def generate_report(self):
        """Generar reporte HTML completo"""
        metrics = self.calculate_metrics()
        
        html_report = f"""
        <html>
        <head>
            <title>Reporte de Backtesting FVG - XAU/USD</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th {{ background-color: #3498db; color: white; padding: 12px; text-align: left; }}
                td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .positive {{ color: green; font-weight: bold; }}
                .negative {{ color: red; font-weight: bold; }}
                .section {{ margin: 30px 0; }}
            </style>
        </head>
        <body>
            <h1>📊 Reporte de Backtesting - Estrategia FVG XAU/USD</h1>
            <p><strong>Fecha del reporte:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>📈 Resumen de Rendimiento</h2>
                <table>
                    <tr>
                        <th>Métrica</th>
                        <th>Valor</th>
                    </tr>
                    <tr>
                        <td>Capital Inicial</td>
                        <td>${self.initial_capital:,.2f}</td>
                    </tr>
                    <tr>
                        <td>Capital Final</td>
                        <td class="{'positive' if metrics['Final Capital ($)'] > self.initial_capital else 'negative'}">
                            ${metrics['Final Capital ($)']:,.2f}
                        </td>
                    </tr>
                    <tr>
                        <td>Retorno Neto</td>
                        <td class="{'positive' if metrics['Net Return (%)'] > 0 else 'negative'}">
                            {metrics['Net Return (%)']:.2f}%
                        </td>
                    </tr>
                    <tr>
                        <td>Total de Trades</td>
                        <td>{metrics['Total Trades']}</td>
                    </tr>
                    <tr>
                        <td>Win Rate</td>
                        <td>{metrics['Win Rate (%)']:.2f}%</td>
                    </tr>
                    <tr>
                        <td>Profit Factor</td>
                        <td>{metrics['Profit Factor']:.2f}</td>
                    </tr>
                    <tr>
                        <td>Sharpe Ratio</td>
                        <td>{metrics['Sharpe Ratio']:.2f}</td>
                    </tr>
                    <tr>
                        <td>Max Drawdown</td>
                        <td class="negative">{metrics['Max Drawdown (%)']:.2f}%</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>💰 Análisis de P&L</h2>
                <table>
                    <tr>
                        <th>Métrica</th>
                        <th>Valor</th>
                    </tr>
                    <tr>
                        <td>P&L Total</td>
                        <td class="{'positive' if metrics['Total P&L ($)'] > 0 else 'negative'}">
                            ${metrics['Total P&L ($)']:,.2f}
                        </td>
                    </tr>
                    <tr>
                        <td>Ganancia Promedio</td>
                        <td class="positive">${metrics['Average Win ($)']:,.2f}</td>
                    </tr>
                    <tr>
                        <td>Pérdida Promedio</td>
                        <td class="negative">${metrics['Average Loss ($)']:,.2f}</td>
                    </tr>
                    <tr>
                        <td>Mejor Trade</td>
                        <td class="positive">${metrics['Best Trade ($)']:,.2f}</td>
                    </tr>
                    <tr>
                        <td>Peor Trade</td>
                        <td class="negative">${metrics['Worst Trade ($)']:,.2f}</td>
                    </tr>
                    <tr>
                        <td>Expectativa Matemática</td>
                        <td>${metrics['Mathematical Expectancy ($)']:,.2f}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>⚙️ Gestión de Riesgo</h2>
                <table>
                    <tr>
                        <th>Métrica</th>
                        <th>Valor</th>
                    </tr>
                    <tr>
                        <td>Riesgo por Trade</td>
                        <td>{self.risk_per_trade * 100:.1f}%</td>
                    </tr>
                    <tr>
                        <td>Ratio Risk/Reward</td>
                        <td>1:{self.tp_multiple}</td>
                    </tr>
                    <tr>
                        <td>Trades con SL Movido</td>
                        <td>{metrics['Trades with SL Moved']} ({metrics['SL Movement Rate (%)']:.1f}%)</td>
                    </tr>
                    <tr>
                        <td>Racha Máxima Ganadora</td>
                        <td>{metrics['Max Consecutive Wins']}</td>
                    </tr>
                    <tr>
                        <td>Racha Máxima Perdedora</td>
                        <td>{metrics['Max Consecutive Losses']}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>🔍 Parámetros de la Estrategia</h2>
                <table>
                    <tr>
                        <th>Parámetro</th>
                        <th>Valor</th>
                    </tr>
                    <tr>
                        <td>Período EMA (Filtro de Tendencia)</td>
                        <td>{self.ema_period}</td>
                    </tr>
                    <tr>
                        <td>Tamaño Mínimo de Gap</td>
                        <td>{self.min_gap_pips} pips</td>
                    </tr>
                    <tr>
                        <td>Timeframe Principal</td>
                        <td>4H</td>
                    </tr>
                    <tr>
                        <td>Timeframe de Entrada</td>
                        <td>5M</td>
                    </tr>
                    <tr>
                        <td>Filtro de Volumen</td>
                        <td>Volumen Ascendente Mejorado</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>📝 Conclusiones</h2>
                <p>La estrategia FVG en XAU/USD muestra los siguientes resultados:</p>
                <ul>
                    <li>Win Rate: {metrics['Win Rate (%)']:.1f}% con un total de {metrics['Total Trades']} trades</li>
                    <li>Retorno neto: {metrics['Net Return (%)']:.2f}% con un drawdown máximo de {abs(metrics['Max Drawdown (%)']):.2f}%</li>
                    <li>Profit Factor de {metrics['Profit Factor']:.2f} indica {'rentabilidad positiva' if metrics['Profit Factor'] > 1 else 'necesidad de optimización'}</li>
                    <li>El sistema de stop loss dinámico se activó en {metrics['SL Movement Rate (%)']:.1f}% de los trades</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Guardar reporte
        with open('FVG_Backtest_Report.html', 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        print("✅ Reporte HTML generado: FVG_Backtest_Report.html")
        
        # Mostrar preview en Jupyter
        display(HTML(html_report))
    
    def export_trades_to_csv(self):
        """Exportar todos los trades a CSV para análisis adicional"""
        if not self.trades:
            print("No hay trades para exportar")
            return
        
        trades_df = pd.DataFrame(self.trades)
        
        # Convertir timedelta a horas
        trades_df['duration_hours'] = trades_df['duration'].apply(lambda x: x.total_seconds() / 3600)
        
        # Agregar columnas adicionales
        trades_df['risk_reward_achieved'] = abs(trades_df['pnl'] / (trades_df['entry_price'] - trades_df['initial_sl']))
        trades_df['sl_moved'] = trades_df['final_sl'] != trades_df['initial_sl']
        
        # Guardar a CSV
        filename = f'FVG_Trades_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        trades_df.to_csv(filename, index=False)
        
        print(f"✅ Trades exportados a: {filename}")
        
        # Mostrar preview
        print("\nPreview de los primeros 5 trades:")
        display(trades_df.head())
    
    def run_backtest(self):
        """Ejecutar proceso completo de backtesting con reportes"""
        print("=" * 60)
        print("🚀 FVG BACKTESTING STRATEGY - XAU/USD")
        print("=" * 60)
        print("📋 CARACTERÍSTICAS DE LA ESTRATEGIA:")
        print("✓ Detección de Fair Value Gaps en 4H")
        print("✓ Entradas con confirmación en 5M")
        print("✓ Stop loss dinámico con FVGs de 5M")
        print("✓ Filtro de volumen ascendente avanzado")
        print("✓ Gestión de riesgo del 0.5% por trade")
        print("=" * 60)
        
        # Paso 1: Descargar datos
        self.download_data()
        
        # Paso 2: Calcular indicadores
        self.calculate_indicators()
        
        # Paso 3: Detectar zonas FVG
        self.detect_fvg_zones()
        
        # Paso 4: Ejecutar backtest
        self.backtest_strategy()
        
        # Paso 5: Calcular y mostrar métricas
        metrics = self.calculate_metrics()
        
        print("\n" + "=" * 30)
        print("📊 RESULTADOS DEL BACKTEST")
        print("=" * 30)
        
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            elif isinstance(value, pd.Timedelta):
                hours = value.total_seconds() / 3600
                print(f"{key}: {hours:.1f} horas")
            else:
                print(f"{key}: {value}")
        
        # Paso 6: Generar visualizaciones
        self.plot_results()
        
        # Paso 7: Crear gráfico interactivo
        self.create_interactive_chart()
        
        # Paso 8: Generar reporte HTML
        self.generate_report()
        
        # Paso 9: Exportar trades
        self.export_trades_to_csv()
        
        return metrics

# %% [markdown]
# ## 3. Ejecución del Backtest

# %%
# Crear instancia del backtester con parámetros optimizados
backtester = FVGBacktester(
    symbol="GC=F",           # Gold Futures (XAU/USD)
    start_date="2023-01-01", # Fecha de inicio
    initial_capital=10000,   # Capital inicial: $10,000
    risk_per_trade=0.005     # Riesgo por trade: 0.5%
)

# Ejecutar el backtest completo
results = backtester.run_backtest()

# %% [markdown]
# ## 4. Análisis Detallado de Trades

# %%
# Análisis adicional de los primeros trades
if backtester.trades:
    print("\n📋 ANÁLISIS DETALLADO DE LOS PRIMEROS 10 TRADES:")
    print("=" * 80)
    
    for i, trade in enumerate(backtester.trades[:10]):
        sl_moved = "✓ SL MOVIDO" if trade['final_sl'] != trade['initial_sl'] else "✗ SL ESTÁTICO"
        duration_hours = trade['duration'].total_seconds() / 3600
        
        print(f"\nTrade #{i+1}:")
        print(f"  Entrada: {trade['entry_time'].strftime('%Y-%m-%d %H:%M')} @ ${trade['entry_price']:.2f}")
        print(f"  Salida:  {trade['exit_time'].strftime('%Y-%m-%d %H:%M')} @ ${trade['exit_price']:.2f}")
        print(f"  Duración: {duration_hours:.1f} horas")
        print(f"  P&L: ${trade['pnl']:.2f} ({trade['return_pct']:.2f}%)")
        print(f"  SL Inicial: ${trade['initial_sl']:.2f} | SL Final: ${trade['final_sl']:.2f} | {sl_moved}")
        print(f"  Motivo de salida: {trade['reason']}")
        print(f"  Tamaño FVG: {trade['fvg_size']:.2f}")

# %% [markdown]
# ## 5. Optimización de Parámetros

# %%
def optimize_parameters():
    """Función para optimizar parámetros de la estrategia"""
    
    # Parámetros a optimizar
    risk_levels = [0.003, 0.005, 0.007, 0.01]
    tp_multiples = [1.5, 2.0, 2.5, 3.0]
    min_gap_sizes = [20, 30, 40, 50]
    
    results_matrix = []
    
    print("🔄 Iniciando optimización de parámetros...")
    print("=" * 60)
    
    for risk in risk_levels:
        for tp in tp_multiples:
            for gap in min_gap_sizes:
                # Crear backtester con parámetros específicos
                bt = FVGBacktester(
                    symbol="GC=F",
                    start_date="2023-01-01",
                    initial_capital=10000,
                    risk_per_trade=risk
                )
                
                # Modificar parámetros
                bt.tp_multiple = tp
                bt.min_gap_pips = gap
                
                # Ejecutar backtest silenciosamente
                bt.download_data()
                bt.calculate_indicators()
                bt.detect_fvg_zones()
                bt.backtest_strategy()
                
                # Obtener métricas
                metrics = bt.calculate_metrics()
                
                results_matrix.append({
                    'Risk %': risk * 100,
                    'TP Multiple': tp,
                    'Min Gap': gap,
                    'Net Return %': metrics.get('Net Return (%)', 0),
                    'Win Rate %': metrics.get('Win Rate (%)', 0),
                    'Sharpe Ratio': metrics.get('Sharpe Ratio', 0),
                    'Max DD %': metrics.get('Max Drawdown (%)', 0),
                    'Total Trades': metrics.get('Total Trades', 0)
                })
    
    # Convertir a DataFrame para análisis
    optimization_df = pd.DataFrame(results_matrix)
    
    # Ordenar por Sharpe Ratio
    optimization_df = optimization_df.sort_values('Sharpe Ratio', ascending=False)
    
    print("\n🏆 TOP 10 COMBINACIONES DE PARÁMETROS:")
    display(optimization_df.head(10))
    
    # Crear heatmap de resultados
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Preparar datos para heatmaps
    for idx, (risk, tp) in enumerate([(0.005, 2.0), (0.005, 2.5), (0.007, 2.0), (0.007, 2.5)]):
        ax = axes[idx // 2, idx % 2]
        
        subset = optimization_df[(optimization_df['Risk %'] == risk * 100) & 
                                (optimization_df['TP Multiple'] == tp)]
        
        if not subset.empty:
            pivot = subset.pivot_table(
                values='Sharpe Ratio',
                index='Min Gap',
                columns='TP Multiple',
                aggfunc='first'
            )
            
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax)
            ax.set_title(f'Sharpe Ratio - Risk: {risk*100}%, TP: {tp}')
    
    plt.tight_layout()
    plt.show()
    
    return optimization_df

# Ejecutar optimización (comentado por defecto para no ralentizar)
# optimization_results = optimize_parameters()

# %% [markdown]
# ## 6. Conclusiones y Recomendaciones
# 
# ### 📊 Resumen de la Estrategia FVG
# 
# La estrategia Fair Value Gap (FVG) implementada para XAU/USD presenta las siguientes características clave:
# 
# 1. **Detección Multi-Timeframe**: 
#    - Identifica FVGs en gráficos de 4H para señales principales
#    - Utiliza 5M para timing preciso de entradas y gestión dinámica de stops
# 
# 2. **Filtros de Calidad**:
#    - Filtro de tendencia con EMA 200
#    - Análisis de volumen ascendente avanzado
#    - Tamaño mínimo de gap configurable
# 
# 3. **Gestión de Riesgo**:
#    - Risk per trade del 0.5% (configurable)
#    - Stop loss dinámico que se ajusta con nuevos FVGs
#    - Risk-Reward ratio de 1:2 (optimizable)
# 
# ### 🎯 Recomendaciones para Trading Real
# 
# 1. **Validación Forward**: Ejecutar la estrategia en paper trading durante al menos 3 meses
# 2. **Diversificación**: Considerar aplicar la estrategia en otros pares de metales preciosos
# 3. **Monitoreo**: Revisar métricas semanalmente y ajustar parámetros según condiciones del mercado
# 4. **Risk Management**: Nunca exceder el 2% de riesgo total en trades simultáneos
# 
# ### 🔧 Próximos Pasos
# 
# 1. Implementar filtros adicionales de market regime
# 2. Añadir machine learning para predicción de FVG quality
# 3. Desarrollar API para ejecución automática
# 4. Crear dashboard en tiempo real para monitoreo

# %%
print("\n✅ Notebook de Backtesting FVG completado exitosamente!")
print("📁 Archivos generados:")
print("   - FVG_Backtest_Report.html")
print("   - FVG_Trades_[timestamp].csv")
print("\n💡 Para análisis adicional, revise los archivos exportados y el reporte HTML generado.")