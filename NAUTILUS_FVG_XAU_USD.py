# %% [markdown]
# # Estrategia FVG (Fair Value Gap) para XAU/USD en NautilusTrader
# ## Migración de Estrategia de Trading Algorítmico a NautilusTrader
#
# Este notebook demuestra cómo implementar la estrategia FVG usando NautilusTrader,
# aprovechando su arquitectura event-driven de alto rendimiento.

# %%
# Importaciones necesarias para NautilusTrader
from decimal import Decimal
from typing import Optional, Dict, List
import pandas as pd
import numpy as np

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.config import StrategyConfig, LoggingConfig
from nautilus_trader.core.data import Data
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType, QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue, TraderId
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.model.objects import Price, Quantity, Money
from nautilus_trader.model.orders import MarketOrder, StopMarketOrder
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.indicators.ema import ExponentialMovingAverage
from nautilus_trader.common.enums import LogLevel
from nautilus_trader.backtest.data import BacktestDataConfig
from nautilus_trader.backtest.venue import BacktestVenueConfig
from nautilus_trader.model.book import OrderBook
from nautilus_trader.model.enums import BookType
from nautilus_trader.model.identifiers import ClientId
from nautilus_trader.common.clock import TestClock
from nautilus_trader.model.currencies import USD

# %% [markdown]
# ## 1. Configuración de la Estrategia FVG

# %%
class FVGStrategyConfig(StrategyConfig):
    """
    Configuración para la estrategia Fair Value Gap en NautilusTrader
    """
    instrument_id: InstrumentId
    bar_type_4h: BarType
    bar_type_5m: BarType
    ema_period: int = 200
    min_gap_pips: float = 30.0
    risk_per_trade: float = 0.005  # 0.5%
    tp_multiple: float = 2.0
    initial_capital: float = 10000.0
    volume_ma_period: int = 20
    volume_trend_period: int = 5
    max_concurrent_trades: int = 1
    use_dynamic_sl: bool = True

# %%
class FVGStrategy(Strategy):
    """
    Estrategia Fair Value Gap (FVG) para XAU/USD en NautilusTrader

    Esta estrategia:
    - Detecta Fair Value Gaps en timeframe 4H
    - Ejecuta entradas en 5M con confirmación de volumen
    - Gestiona stop loss dinámico basado en nuevos FVGs
    """

    def __init__(self, config: FVGStrategyConfig):
        super().__init__(config)

        # Configuración
        self.instrument_id = config.instrument_id
        self.bar_type_4h = config.bar_type_4h
        self.bar_type_5m = config.bar_type_5m
        self.min_gap_pips = config.min_gap_pips
        self.risk_per_trade = config.risk_per_trade
        self.tp_multiple = config.tp_multiple
        self.initial_capital = config.initial_capital
        self.use_dynamic_sl = config.use_dynamic_sl

        # Indicadores
        self.ema_4h = ExponentialMovingAverage(config.ema_period)

        # Estado interno
        self.fvg_zones_4h: List[Dict] = []
        self.fvg_zones_5m: List[Dict] = []
        self.current_position = None
        self.bars_4h_cache: List[Bar] = []
        self.bars_5m_cache: List[Bar] = []
        self.volume_data: List[float] = []

        # Métricas
        self.trades_executed = 0
        self.sl_adjustments = 0

    def on_start(self):
        """
        Llamado cuando la estrategia inicia
        """
        self.log.info("Iniciando estrategia FVG")

        # Suscribirse a los datos necesarios
        self.subscribe_bars(self.bar_type_4h)
        self.subscribe_bars(self.bar_type_5m)

        # Configurar el tamaño de posición
        self._calculate_position_size()

    def on_bar(self, bar: Bar):
        """
        Procesar nuevas barras
        """
        # Actualizar cache de barras
        if bar.bar_type == self.bar_type_4h:
            self._process_4h_bar(bar)
        elif bar.bar_type == self.bar_type_5m:
            self._process_5m_bar(bar)

    def _process_4h_bar(self, bar: Bar):
        """
        Procesar barra de 4H para detectar FVGs y actualizar tendencia
        """
        # Actualizar cache
        self.bars_4h_cache.append(bar)
        if len(self.bars_4h_cache) > 200:  # Mantener últimas 200 barras
            self.bars_4h_cache.pop(0)

        # Actualizar EMA
        self.ema_4h.update_raw(bar.close.as_double())

        # Detectar nuevos FVGs
        if len(self.bars_4h_cache) >= 3:
            self._detect_fvg_4h()

        self.log.debug(f"Procesada barra 4H: {bar}")

    def _process_5m_bar(self, bar: Bar):
        """
        Procesar barra de 5M para entradas y gestión de posiciones
        """
        # Actualizar cache
        self.bars_5m_cache.append(bar)
        if len(self.bars_5m_cache) > 100:
            self.bars_5m_cache.pop(0)

        # Actualizar datos de volumen
        self.volume_data.append(float(bar.volume))
        if len(self.volume_data) > 20:
            self.volume_data.pop(0)

        # Gestionar posición existente
        if self.current_position:
            self._manage_position(bar)

        # Buscar nuevas entradas
        elif self._should_enter_trade(bar):
            self._enter_trade(bar)

        # Detectar FVGs en 5M para SL dinámico
        if self.use_dynamic_sl and len(self.bars_5m_cache) >= 3:
            self._detect_fvg_5m()

    def _detect_fvg_4h(self):
        """
        Detectar Fair Value Gaps en timeframe 4H
        """
        bars = self.bars_4h_cache[-3:]  # Últimas 3 barras

        # FVG Alcista: high[0] < low[2]
        gap_size = float(bars[2].low - bars[0].high)
        if gap_size >= self.min_gap_pips * 0.1:  # Convertir pips
            fvg = {
                'timestamp': bars[2].ts_event,
                'high': float(bars[0].high),
                'low': float(bars[2].low),
                'middle': (float(bars[0].high) + float(bars[2].low)) / 2,
                'size': gap_size,
                'filled': False,
                'direction': 'bullish'
            }
            self.fvg_zones_4h.append(fvg)
            self.log.info(f"Nuevo FVG 4H detectado: {fvg}")

    def _detect_fvg_5m(self):
        """
        Detectar Fair Value Gaps en 5M para ajuste dinámico de SL
        """
        bars = self.bars_5m_cache[-3:]

        gap_size = float(bars[2].low - bars[0].high)
        if gap_size >= 10 * 0.1:  # Gap mínimo más pequeño para 5M
            fvg = {
                'timestamp': bars[2].ts_event,
                'high': float(bars[0].high),
                'low': float(bars[2].low),
                'middle': (float(bars[0].high) + float(bars[2].low)) / 2,
                'size': gap_size,
                'direction': 'bullish'
            }
            self.fvg_zones_5m.append(fvg)

    def _is_volume_rising(self) -> bool:
        """
        Verificar si el volumen está ascendiendo
        """
        if len(self.volume_data) < 20:
            return False

        # Media móvil de volumen
        volume_ma = np.mean(self.volume_data)
        current_volume = self.volume_data[-1]

        # Tendencia de volumen en últimas 5 barras
        if len(self.volume_data) >= 5:
            volume_trend = self.volume_data[-1] > self.volume_data[-5]
        else:
            volume_trend = False

        # Aceleración de volumen
        if len(self.volume_data) >= 2:
            volume_acceleration = self.volume_data[-1] > self.volume_data[-2]
        else:
            volume_acceleration = False

        return (current_volume > volume_ma and volume_trend and volume_acceleration)

    def _is_bullish_trend(self) -> bool:
        """
        Verificar si estamos en tendencia alcista (precio > EMA 200)
        """
        if not self.ema_4h.initialized:
            return False

        if len(self.bars_4h_cache) == 0:
            return False

        current_price = float(self.bars_4h_cache[-1].close)
        ema_value = self.ema_4h.value

        return current_price > ema_value

    def _should_enter_trade(self, bar: Bar) -> bool:
        """
        Evaluar si se deben cumplir las condiciones de entrada
        """
        # Verificar tendencia
        if not self._is_bullish_trend():
            return False

        # Verificar volumen
        if not self._is_volume_rising():
            return False

        # Buscar FVG activo
        current_price = float(bar.close)
        current_low = float(bar.low)

        for fvg in self.fvg_zones_4h:
            if fvg['filled']:
                continue

            # Verificar barrido del FVG
            gap_50_pct = fvg['high'] + (fvg['middle'] - fvg['high']) * 0.5

            if (current_low <= fvg['middle'] and
                current_low >= gap_50_pct and
                current_price > fvg['low']):

                # Marcar para entrada
                self.entry_fvg = fvg
                return True

        return False

    def _calculate_position_size(self):
        """
        Calcular el tamaño de posición basado en el riesgo
        """
        account_balance = self.portfolio.account(self.instrument_id.venue).balance_total(USD)
        if account_balance is None:
            account_balance = Money(self.initial_capital, USD)

        self.risk_amount = float(account_balance) * self.risk_per_trade

    def _enter_trade(self, bar: Bar):
        """
        Ejecutar entrada en posición
        """
        if not hasattr(self, 'entry_fvg'):
            return

        fvg = self.entry_fvg

        # Calcular niveles
        entry_price = fvg['middle']
        stop_loss = fvg['high'] - 0.1
        take_profit = entry_price + (entry_price - stop_loss) * self.tp_multiple

        # Calcular tamaño de posición
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit > 0:
            position_size = self.risk_amount / risk_per_unit
        else:
            return

        # Crear orden de mercado
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY,
            quantity=Quantity.from_float(position_size),
            time_in_force=TimeInForce.GTC,
        )

        # Enviar orden
        self.submit_order(order)

        # Guardar información de la posición
        self.current_position = {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'initial_stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'entry_time': bar.ts_event,
            'fvg': fvg
        }

        # Marcar FVG como usado
        fvg['filled'] = True
        self.trades_executed += 1

        self.log.info(
            f"Posición abierta - Entry: {entry_price:.2f}, "
            f"SL: {stop_loss:.2f}, TP: {take_profit:.2f}, "
            f"Size: {position_size:.4f}"
        )

    def _manage_position(self, bar: Bar):
        """
        Gestionar posición abierta
        """
        if not self.current_position:
            return

        current_price = float(bar.close)

        # Verificar Stop Loss
        if float(bar.low) <= self.current_position['stop_loss']:
            self._close_position('Stop Loss', self.current_position['stop_loss'])
            return

        # Verificar Take Profit
        if float(bar.high) >= self.current_position['take_profit']:
            self._close_position('Take Profit', self.current_position['take_profit'])
            return

        # Actualizar Stop Loss dinámico
        if self.use_dynamic_sl:
            self._update_dynamic_stop_loss(bar)

    def _update_dynamic_stop_loss(self, bar: Bar):
        """
        Actualizar stop loss basado en nuevos FVGs de 5M
        """
        if not self.current_position:
            return

        # Buscar FVGs de 5M formados después de la entrada
        for fvg in self.fvg_zones_5m:
            if (fvg['timestamp'] > self.current_position['entry_time'] and
                fvg['direction'] == 'bullish'):

                new_stop = fvg['high'] - 0.1

                # Solo actualizar si el nuevo SL es más favorable
                if new_stop > self.current_position['stop_loss']:
                    old_sl = self.current_position['stop_loss']
                    self.current_position['stop_loss'] = new_stop
                    self.sl_adjustments += 1

                    self.log.info(
                        f"Stop Loss actualizado de {old_sl:.2f} a {new_stop:.2f}"
                    )

    def _close_position(self, reason: str, exit_price: float):
        """
        Cerrar posición actual
        """
        if not self.current_position:
            return

        # Calcular P&L
        pnl = (exit_price - self.current_position['entry_price']) * \
              self.current_position['position_size']

        # Crear orden de cierre
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.SELL,
            quantity=Quantity.from_float(self.current_position['position_size']),
            time_in_force=TimeInForce.GTC,
        )

        self.submit_order(order)

        self.log.info(
            f"Posición cerrada - Razón: {reason}, "
            f"Exit: {exit_price:.2f}, P&L: ${pnl:.2f}"
        )

        # Limpiar posición
        self.current_position = None

    def on_stop(self):
        """
        Llamado cuando la estrategia se detiene
        """
        self.log.info(
            f"Estrategia FVG detenida - "
            f"Trades ejecutados: {self.trades_executed}, "
            f"Ajustes de SL: {self.sl_adjustments}"
        )

        # Cerrar posición abierta si existe
        if self.current_position:
            if len(self.bars_5m_cache) > 0:
                last_price = float(self.bars_5m_cache[-1].close)
                self._close_position('Strategy Stop', last_price)

    def on_reset(self):
        """
        Resetear el estado de la estrategia
        """
        # Resetear indicadores
        self.ema_4h.reset()

        # Limpiar estado
        self.fvg_zones_4h.clear()
        self.fvg_zones_5m.clear()
        self.bars_4h_cache.clear()
        self.bars_5m_cache.clear()
        self.volume_data.clear()
        self.current_position = None

        # Resetear métricas
        self.trades_executed = 0
        self.sl_adjustments = 0

# %% [markdown]
# ## 2. Configuración del Motor de Backtesting

# %%
def setup_backtest_engine():
    """
    Configurar el motor de backtesting de NautilusTrader
    """
    # Configuración del motor
    config = BacktestEngineConfig(
        trader_id=TraderId("BACKTESTER-001"),
        logging=LoggingConfig(
            log_level=LogLevel.INFO,
            log_to_console=True,
        ),
    )

    # Crear motor
    engine = BacktestEngine(config=config)

    # Configurar venue simulado
    venue = Venue("SIM")
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.HEDGING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money(10_000, USD)],
        fill_model=FillModel(
            prob_fill_on_limit=0.2,
            prob_fill_on_stop=0.95,
            prob_slippage=0.5,
            random_seed=42,
        ),
    )

    return engine, venue

# %%
def create_xauusd_instrument(venue: Venue) -> CurrencyPair:
    """
    Crear instrumento XAU/USD
    """
    return CurrencyPair(
        instrument_id=InstrumentId(
            symbol=Symbol("XAUUSD"),
            venue=venue,
        ),
        raw_symbol=Symbol("XAUUSD"),
        base_currency=USD,  # Simplificación: usando USD como base
        quote_currency=USD,
        price_precision=2,
        size_precision=2,
        price_increment=Price.from_str("0.01"),
        size_increment=Quantity.from_str("0.01"),
        lot_size=Quantity.from_str("1"),
        max_quantity=Quantity.from_str("1000"),
        min_quantity=Quantity.from_str("0.01"),
        max_price=Price.from_str("5000.00"),
        min_price=Price.from_str("100.00"),
        margin_init=Decimal("0.05"),
        margin_maint=Decimal("0.03"),
        maker_fee=Decimal("0.0002"),
        taker_fee=Decimal("0.0002"),
    )

# %% [markdown]
# ## 3. Generación de Datos de Prueba

# %%
def generate_test_data(instrument: CurrencyPair, num_days: int = 30):
    """
    Generar datos de prueba para XAU/USD
    """
    # Generar timestamps
    start_date = pd.Timestamp('2024-01-01', tz='UTC')

    # Datos 4H
    timestamps_4h = pd.date_range(
        start=start_date,
        periods=num_days * 6,  # 6 barras de 4H por día
        freq='4H',
        tz='UTC'
    )

    # Datos 5M
    timestamps_5m = pd.date_range(
        start=start_date,
        periods=num_days * 288,  # 288 barras de 5M por día
        freq='5T',
        tz='UTC'
    )

    # Generar precios con tendencia y volatilidad
    np.random.seed(42)
    base_price = 2000

    # Precio 4H con tendencia alcista
    trend_4h = np.linspace(0, 100, len(timestamps_4h))
    noise_4h = np.cumsum(np.random.randn(len(timestamps_4h)) * 5)
    prices_4h = base_price + trend_4h + noise_4h

    # Crear barras 4H con FVGs ocasionales
    bars_4h = []
    for i, (ts, price) in enumerate(zip(timestamps_4h, prices_4h)):
        # Crear gap ocasional
        if i > 2 and np.random.random() < 0.1:  # 10% de probabilidad
            gap_size = np.random.uniform(3, 6)  # Gap de 3-6 dólares
            open_price = price + gap_size
        else:
            open_price = price + np.random.randn() * 2

        high = max(open_price, price) + abs(np.random.randn() * 3)
        low = min(open_price, price) - abs(np.random.randn() * 3)
        volume = np.random.randint(1000, 5000)

        bar = Bar(
            bar_type=BarType.from_str(f"XAUUSD.SIM-4-HOUR-BID-INTERNAL"),
            open=Price.from_str(f"{open_price:.2f}"),
            high=Price.from_str(f"{high:.2f}"),
            low=Price.from_str(f"{low:.2f}"),
            close=Price.from_str(f"{price:.2f}"),
            volume=Quantity.from_int(volume),
            ts_event=ts,
            ts_init=ts,
        )
        bars_4h.append(bar)

    # Precio 5M con mayor granularidad
    trend_5m = np.linspace(0, 100, len(timestamps_5m))
    noise_5m = np.cumsum(np.random.randn(len(timestamps_5m)) * 0.5)
    prices_5m = base_price + trend_5m + noise_5m

    # Crear barras 5M
    bars_5m = []
    for ts, price in zip(timestamps_5m, prices_5m):
        open_price = price + np.random.randn() * 0.5
        high = max(open_price, price) + abs(np.random.randn() * 1)
        low = min(open_price, price) - abs(np.random.randn() * 1)

        # Volumen con patrones ascendentes ocasionales
        base_volume = np.random.randint(50, 200)
        if np.random.random() < 0.2:  # 20% con volumen alto
            volume = base_volume * 3
        else:
            volume = base_volume

        bar = Bar(
            bar_type=BarType.from_str(f"XAUUSD.SIM-5-MINUTE-BID-INTERNAL"),
            open=Price.from_str(f"{open_price:.2f}"),
            high=Price.from_str(f"{high:.2f}"),
            low=Price.from_str(f"{low:.2f}"),
            close=Price.from_str(f"{price:.2f}"),
            volume=Quantity.from_int(volume),
            ts_event=ts,
            ts_init=ts,
        )
        bars_5m.append(bar)

    return bars_4h, bars_5m

# %% [markdown]
# ## 4. Ejecutar Backtest

# %%
def run_fvg_backtest():
    """
    Ejecutar backtest completo de la estrategia FVG
    """
    print("=" * 60)
    print("BACKTESTING ESTRATEGIA FVG EN NAUTILUSTRADER")
    print("=" * 60)

    # 1. Configurar motor
    engine, venue = setup_backtest_engine()

    # 2. Crear instrumento
    instrument = create_xauusd_instrument(venue)
    engine.add_instrument(instrument)

    # 3. Generar datos de prueba
    print("Generando datos de prueba...")
    bars_4h, bars_5m = generate_test_data(instrument, num_days=60)

    # Agregar datos al motor
    engine.add_data(bars_4h)
    engine.add_data(bars_5m)

    print(f"Datos cargados: {len(bars_4h)} barras 4H, {len(bars_5m)} barras 5M")

    # 4. Configurar estrategia
    strategy_config = FVGStrategyConfig(
        instrument_id=instrument.id,
        bar_type_4h=BarType.from_str("XAUUSD.SIM-4-HOUR-BID-INTERNAL"),
        bar_type_5m=BarType.from_str("XAUUSD.SIM-5-MINUTE-BID-INTERNAL"),
        ema_period=200,
        min_gap_pips=30.0,
        risk_per_trade=0.005,
        tp_multiple=2.0,
        initial_capital=10000.0,
    )

    # 5. Inicializar estrategia
    strategy = FVGStrategy(config=strategy_config)
    engine.add_strategy(strategy)

    # 6. Ejecutar backtest
    print("\nEjecutando backtest...")
    engine.run()

    # 7. Obtener resultados
    print("\n" + "=" * 30)
    print("RESULTADOS DEL BACKTEST")
    print("=" * 30)

    # Imprimir estadísticas del portfolio
    engine.trader.generate_order_fills_report()
    engine.trader.generate_positions_report()
    engine.trader.generate_account_report(venue)

    return engine

# %%
# Ejecutar el backtest
if __name__ == "__main__":
    engine = run_fvg_backtest()

    print("\n✅ Backtest completado exitosamente!")
    print("\nNota: Esta es una implementación básica de demostración.")
    print("Para usar con datos reales, necesitarás:")
    print("1. Conectar un proveedor de datos real (Databento, Interactive Brokers, etc.)")
    print("2. Ajustar los parámetros del instrumento")
    print("3. Implementar gestión de riesgo más sofisticada")
    print("4. Agregar más métricas y análisis de rendimiento")

# %% [markdown]
# ## 5. Análisis de Resultados y Visualización
#
# NautilusTrader proporciona herramientas integradas para analizar resultados:

# %%
def analyze_results(engine: BacktestEngine):
    """
    Analizar y visualizar resultados del backtest
    """
    from nautilus_trader.analysis.statistics import PortfolioStatistics
    from nautilus_trader.analysis.performance import PerformanceAnalyzer
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Obtener datos del portfolio
    account = engine.trader.portfolio.account(Venue("SIM"))

    # Crear analizador de rendimiento
    analyzer = PerformanceAnalyzer()

    # Obtener posiciones y órdenes
    positions = engine.cache.positions()
    orders = engine.cache.orders()

    # Estadísticas básicas
    stats = {
        "Total Trades": len(positions),
        "Total Orders": len(orders),
        "Starting Balance": 10000.0,
        "Final Balance": float(account.balance_total(USD)),
        "Net P&L": float(account.balance_total(USD)) - 10000.0,
        "Return %": ((float(account.balance_total(USD)) - 10000.0) / 10000.0) * 100
    }

    # Calcular métricas adicionales
    winning_trades = [p for p in positions if p.realized_pnl > 0]
    losing_trades = [p for p in positions if p.realized_pnl <= 0]

    if positions:
        stats["Win Rate %"] = (len(winning_trades) / len(positions)) * 100
        stats["Average Win"] = np.mean([float(p.realized_pnl) for p in winning_trades]) if winning_trades else 0
        stats["Average Loss"] = np.mean([float(p.realized_pnl) for p in losing_trades]) if losing_trades else 0
        stats["Profit Factor"] = abs(stats["Average Win"] / stats["Average Loss"]) if stats["Average Loss"] != 0 else 0

    # Imprimir estadísticas
    print("\n📊 ESTADÍSTICAS DE RENDIMIENTO:")
    print("=" * 40)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    # Crear visualizaciones
    create_performance_charts(engine, positions)

    return stats

# %%
def create_performance_charts(engine: BacktestEngine, positions):
    """
    Crear gráficos de rendimiento
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle

    # Configurar estilo
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(16, 12))

    # 1. Curva de Equity
    ax1 = plt.subplot(3, 2, 1)

    # Obtener balance histórico
    account_states = engine.cache.account_state(Venue("SIM"))
    if account_states:
        timestamps = [state.ts_event for state in account_states]
        balances = [float(state.balance_total(USD)) for state in account_states]

        ax1.plot(timestamps, balances, 'g-', linewidth=2)
        ax1.axhline(y=10000, color='k', linestyle='--', alpha=0.5)
        ax1.fill_between(timestamps, 10000, balances,
                        where=[b >= 10000 for b in balances],
                        color='green', alpha=0.3)
        ax1.fill_between(timestamps, 10000, balances,
                        where=[b < 10000 for b in balances],
                        color='red', alpha=0.3)

    ax1.set_title('Curva de Equity', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Balance ($)')
    ax1.grid(True, alpha=0.3)

    # 2. Drawdown
    ax2 = plt.subplot(3, 2, 2)

    if account_states and len(balances) > 0:
        # Calcular drawdown
        running_max = np.maximum.accumulate(balances)
        drawdown = ((balances - running_max) / running_max) * 100

        ax2.fill_between(timestamps, 0, drawdown, color='red', alpha=0.7)
        ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)

    # 3. Distribución de P&L
    ax3 = plt.subplot(3, 2, 3)

    if positions:
        pnls = [float(p.realized_pnl) for p in positions]
        positive_pnls = [pnl for pnl in pnls if pnl > 0]
        negative_pnls = [pnl for pnl in pnls if pnl <= 0]

        bins = 20
        if positive_pnls:
            ax3.hist(positive_pnls, bins=bins, alpha=0.7, color='green',
                    label=f'Ganadores ({len(positive_pnls)})', edgecolor='black')
        if negative_pnls:
            ax3.hist(negative_pnls, bins=bins, alpha=0.7, color='red',
                    label=f'Perdedores ({len(negative_pnls)})', edgecolor='black')

        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_title('Distribución de P&L', fontsize=14, fontweight='bold')
        ax3.set_xlabel('P&L ($)')
        ax3.set_ylabel('Frecuencia')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. Rendimientos por Trade
    ax4 = plt.subplot(3, 2, 4)

    if positions:
        returns = [(float(p.realized_pnl) / 10000) * 100 for p in positions]
        trade_numbers = range(1, len(returns) + 1)

        colors = ['green' if r > 0 else 'red' for r in returns]
        ax4.bar(trade_numbers, returns, color=colors, alpha=0.7, edgecolor='black')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_title('Rendimientos por Trade', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Número de Trade')
        ax4.set_ylabel('Rendimiento (%)')
        ax4.grid(True, alpha=0.3)

    # 5. Análisis de Duración de Trades
    ax5 = plt.subplot(3, 2, 5)

    if positions:
        durations = []
        for p in positions:
            if p.ts_closed and p.ts_opened:
                duration = (p.ts_closed - p.ts_opened).total_seconds() / 3600  # En horas
                durations.append(duration)

        if durations:
            winning_durations = [durations[i] for i in range(len(durations))
                               if pnls[i] > 0]
            losing_durations = [durations[i] for i in range(len(durations))
                              if pnls[i] <= 0]

            bins = 15
            if winning_durations:
                ax5.hist(winning_durations, bins=bins, alpha=0.7, color='green',
                        label='Ganadores', edgecolor='black')
            if losing_durations:
                ax5.hist(losing_durations, bins=bins, alpha=0.7, color='red',
                        label='Perdedores', edgecolor='black')

            ax5.set_title('Duración de Trades', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Duración (horas)')
            ax5.set_ylabel('Frecuencia')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

    # 6. Resumen de Métricas
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')

    # Calcular métricas adicionales
    if positions:
        total_trades = len(positions)
        winning_trades = len([p for p in positions if p.realized_pnl > 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

        avg_win = np.mean([float(p.realized_pnl) for p in positions if p.realized_pnl > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([float(p.realized_pnl) for p in positions if p.realized_pnl <= 0]) if losing_trades else 0

        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        # Crear tabla de métricas
        metrics_text = f"""
        RESUMEN DE MÉTRICAS

        Total de Trades: {total_trades}
        Trades Ganadores: {winning_trades}
        Trades Perdedores: {total_trades - winning_trades}

        Win Rate: {win_rate:.1f}%
        Profit Factor: {profit_factor:.2f}

        Ganancia Promedio: ${avg_win:.2f}
        Pérdida Promedio: ${avg_loss:.2f}

        Máximo Drawdown: {min(drawdown):.2f}% (si está disponible)
        """

        ax6.text(0.1, 0.5, metrics_text, fontsize=12,
                verticalalignment='center', fontfamily='monospace')

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 6. Integración con Proveedores de Datos Reales
#
# Para usar NautilusTrader con datos reales, necesitas configurar adaptadores:

# %%
def setup_databento_adapter():
    """
    Ejemplo de configuración para Databento (proveedor de datos)
    """
    from nautilus_trader.adapters.databento.config import DatabentoDataClientConfig
    from nautilus_trader.adapters.databento.factories import DatabentoLiveDataClientFactory

    # Configuración de Databento
    config = DatabentoDataClientConfig(
        api_key="YOUR_DATABENTO_API_KEY",  # Reemplazar con tu API key
        http_timeout=20,
        http_retry_count=3,
        instrument_ids=[
            InstrumentId.from_str("XAUUSD.DATABENTO"),
        ],
    )

    # Crear factory
    factory = DatabentoLiveDataClientFactory(
        loop=None,  # Se asignará automáticamente
        config=config,
    )

    return factory

# %%
def setup_interactive_brokers_adapter():
    """
    Ejemplo de configuración para Interactive Brokers
    """
    from nautilus_trader.adapters.interactive_brokers.config import InteractiveBrokersDataClientConfig
    from nautilus_trader.adapters.interactive_brokers.config import InteractiveBrokersExecClientConfig

    # Configuración de datos
    data_config = InteractiveBrokersDataClientConfig(
        ibg_host="127.0.0.1",
        ibg_port=7497,
        ibg_client_id=1,
        subscribe_bars=True,
        subscribe_quotes=True,
        subscribe_trades=True,
    )

    # Configuración de ejecución
    exec_config = InteractiveBrokersExecClientConfig(
        ibg_host="127.0.0.1",
        ibg_port=7497,
        ibg_client_id=2,
        account_id="YOUR_ACCOUNT_ID",  # Reemplazar con tu ID de cuenta
    )

    return data_config, exec_config

# %% [markdown]
# ## 7. Optimización de Parámetros con NautilusTrader

# %%
def optimize_fvg_parameters():
    """
    Optimización de parámetros usando múltiples backtests
    """
    import itertools
    from concurrent.futures import ProcessPoolExecutor

    # Parámetros a optimizar
    param_grid = {
        'ema_period': [150, 200, 250],
        'min_gap_pips': [20, 30, 40],
        'risk_per_trade': [0.003, 0.005, 0.007],
        'tp_multiple': [1.5, 2.0, 2.5],
    }

    # Generar todas las combinaciones
    param_combinations = list(itertools.product(*param_grid.values()))
    param_keys = list(param_grid.keys())

    results = []

    print(f"Optimizando {len(param_combinations)} combinaciones de parámetros...")

    for i, combo in enumerate(param_combinations):
        params = dict(zip(param_keys, combo))

        # Configurar y ejecutar backtest
        engine, venue = setup_backtest_engine()
        instrument = create_xauusd_instrument(venue)
        engine.add_instrument(instrument)

        # Generar datos (reusar los mismos para comparación justa)
        bars_4h, bars_5m = generate_test_data(instrument, num_days=60)
        engine.add_data(bars_4h)
        engine.add_data(bars_5m)

        # Configurar estrategia con parámetros específicos
        strategy_config = FVGStrategyConfig(
            instrument_id=instrument.id,
            bar_type_4h=BarType.from_str("XAUUSD.SIM-4-HOUR-BID-INTERNAL"),
            bar_type_5m=BarType.from_str("XAUUSD.SIM-5-MINUTE-BID-INTERNAL"),
            **params
        )

        strategy = FVGStrategy(config=strategy_config)
        engine.add_strategy(strategy)

        # Ejecutar
        engine.run()

        # Obtener resultados
        account = engine.trader.portfolio.account(venue)
        final_balance = float(account.balance_total(USD))
        net_return = ((final_balance - 10000) / 10000) * 100

        # Guardar resultados
        result = {
            **params,
            'final_balance': final_balance,
            'net_return': net_return,
            'trades': strategy.trades_executed,
        }
        results.append(result)

        print(f"Progreso: {i+1}/{len(param_combinations)} - Return: {net_return:.2f}%")

    # Crear DataFrame con resultados
    import pandas as pd
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('net_return', ascending=False)

    print("\n🏆 TOP 10 MEJORES COMBINACIONES:")
    print(results_df.head(10))

    # Visualizar resultados
    visualize_optimization_results(results_df)

    return results_df

# %%
def visualize_optimization_results(results_df):
    """
    Visualizar resultados de optimización
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Heatmap de retornos por EMA y Min Gap
    ax1 = axes[0, 0]
    pivot1 = results_df.pivot_table(
        values='net_return',
        index='ema_period',
        columns='min_gap_pips',
        aggfunc='mean'
    )
    sns.heatmap(pivot1, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax1)
    ax1.set_title('Retorno Promedio por EMA Period y Min Gap Pips')

    # 2. Heatmap de retornos por Risk y TP Multiple
    ax2 = axes[0, 1]
    pivot2 = results_df.pivot_table(
        values='net_return',
        index='risk_per_trade',
        columns='tp_multiple',
        aggfunc='mean'
    )
    sns.heatmap(pivot2, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax2)
    ax2.set_title('Retorno Promedio por Risk per Trade y TP Multiple')

    # 3. Scatter plot de Trades vs Return
    ax3 = axes[1, 0]
    scatter = ax3.scatter(
        results_df['trades'],
        results_df['net_return'],
        c=results_df['risk_per_trade'],
        cmap='viridis',
        s=100,
        alpha=0.6
    )
    ax3.set_xlabel('Número de Trades')
    ax3.set_ylabel('Retorno Neto (%)')
    ax3.set_title('Trades vs Retorno (color = risk per trade)')
    plt.colorbar(scatter, ax=ax3)

    # 4. Box plot de retornos por parámetro
    ax4 = axes[1, 1]

    # Preparar datos para box plot
    param_returns = []
    for param in ['ema_period', 'min_gap_pips', 'risk_per_trade', 'tp_multiple']:
        for value in results_df[param].unique():
            subset = results_df[results_df[param] == value]
            for ret in subset['net_return']:
                param_returns.append({
                    'parameter': f"{param}\n{value}",
                    'return': ret
                })

    param_returns_df = pd.DataFrame(param_returns)

    # Crear box plot
    param_returns_df.boxplot(column='return', by='parameter', ax=ax4, rot=45)
    ax4.set_title('Distribución de Retornos por Parámetro')
    ax4.set_xlabel('Parámetro')
    ax4.set_ylabel('Retorno (%)')

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 8. Configuración para Trading en Vivo
#
# Para usar la estrategia en trading en vivo con NautilusTrader:

# %%
def setup_live_trading():
    """
    Configuración para trading en vivo
    """
    from nautilus_trader.live.node import LiveNode
    from nautilus_trader.live.config import LiveNodeConfig

    # Configuración del nodo en vivo
    config = LiveNodeConfig(
        trader_id=TraderId("LIVE-TRADER-001"),
        log_level="INFO",
        cache_database=CacheDatabaseConfig(
            type="redis",
            host="localhost",
            port=6379,
        ),
        message_bus=MessageBusConfig(
            database=MessageBusDatabaseConfig(
                type="redis",
                host="localhost",
                port=6379,
            ),
        ),
        data_clients={
            "DATABENTO": DatabentoDataClientConfig(
                api_key="YOUR_API_KEY",
                instrument_ids=[
                    InstrumentId.from_str("XAUUSD.DATABENTO"),
                ],
            ),
        },
        exec_clients={
            "IB": InteractiveBrokersExecClientConfig(
                ibg_host="127.0.0.1",
                ibg_port=7497,
                account_id="YOUR_ACCOUNT",
            ),
        },
        strategies=[
            ImportableStrategyConfig(
                strategy_path="strategies.fvg_strategy:FVGStrategy",
                config_path="strategies.fvg_strategy:FVGStrategyConfig",
                config={
                    "instrument_id": "XAUUSD.DATABENTO",
                    "bar_type_4h": "XAUUSD.DATABENTO-4-HOUR-BID-INTERNAL",
                    "bar_type_5m": "XAUUSD.DATABENTO-5-MINUTE-BID-INTERNAL",
                    "ema_period": 200,
                    "min_gap_pips": 30.0,
                    "risk_per_trade": 0.005,
                    "tp_multiple": 2.0,
                },
            ),
        ],
    )

    # Crear nodo en vivo
    node = LiveNode(config=config)

    return node

# %% [markdown]
# ## 9. Monitoreo en Tiempo Real

# %%
class FVGMonitor:
    """
    Monitor en tiempo real para la estrategia FVG
    """
    def __init__(self, strategy: FVGStrategy):
        self.strategy = strategy
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        plt.ion()  # Modo interactivo

    def update(self):
        """
        Actualizar visualizaciones en tiempo real
        """
        # Limpiar axes
        for ax in self.axes.flat:
            ax.clear()

        # 1. Precio y FVG zones
        ax1 = self.axes[0, 0]
        if self.strategy.bars_5m_cache:
            prices = [float(bar.close) for bar in self.strategy.bars_5m_cache[-100:]]
            ax1.plot(prices, 'b-', linewidth=1)

            # Mostrar FVG zones activos
            for fvg in self.strategy.fvg_zones_4h[-5:]:
                if not fvg['filled']:
                    ax1.axhspan(fvg['high'], fvg['low'], alpha=0.3, color='green')
                    ax1.axhline(fvg['middle'], color='green', linestyle='--', alpha=0.5)

            ax1.set_title('Precio y Zonas FVG Activas')
            ax1.set_ylabel('Precio')
            ax1.grid(True, alpha=0.3)

        # 2. Indicador de Volumen
        ax2 = self.axes[0, 1]
        if self.strategy.volume_data:
            ax2.bar(range(len(self.strategy.volume_data)),
                   self.strategy.volume_data,
                   color='blue', alpha=0.7)
            if len(self.strategy.volume_data) >= 20:
                ma = np.mean(self.strategy.volume_data)
                ax2.axhline(ma, color='orange', linestyle='--', label='MA(20)')

            ax2.set_title('Análisis de Volumen')
            ax2.set_ylabel('Volumen')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 3. Estado de la Posición
        ax3 = self.axes[1, 0]
        ax3.text(0.1, 0.9, "ESTADO DE LA POSICIÓN", fontsize=14, fontweight='bold')

        if self.strategy.current_position:
            pos = self.strategy.current_position
            status_text = f"""
            Estado: POSICIÓN ABIERTA
            Entry: ${pos['entry_price']:.2f}
            Stop Loss: ${pos['stop_loss']:.2f}
            Take Profit: ${pos['take_profit']:.2f}
            Size: {pos['position_size']:.4f}

            SL Inicial: ${pos['initial_stop_loss']:.2f}
            SL Ajustes: {self.strategy.sl_adjustments}
            """
            color = 'green'
        else:
            status_text = """
            Estado: SIN POSICIÓN

            Esperando señal de entrada...
            """
            color = 'gray'

        ax3.text(0.1, 0.5, status_text, fontsize=11,
                verticalalignment='center', color=color)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')

        # 4. Métricas de Rendimiento
        ax4 = self.axes[1, 1]
        ax4.text(0.1, 0.9, "MÉTRICAS DE RENDIMIENTO", fontsize=14, fontweight='bold')

        metrics_text = f"""
        Trades Ejecutados: {self.strategy.trades_executed}
        FVG Zones 4H: {len(self.strategy.fvg_zones_4h)}
        FVG Zones 5M: {len(self.strategy.fvg_zones_5m)}

        Tendencia: {'ALCISTA' if self.strategy._is_bullish_trend() else 'BAJISTA'}
        Volumen: {'ASCENDENTE' if self.strategy._is_volume_rising() else 'NORMAL'}
        """

        ax4.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

# %% [markdown]
# ## 10. Conclusiones y Próximos Pasos
#
# ### Ventajas de NautilusTrader para la Estrategia FVG:
#
# 1. **Alto Rendimiento**: Componentes core en Rust garantizan velocidad
# 2. **Event-Driven**: Simula perfectamente el comportamiento real del mercado
# 3. **Paridad Backtest-Live**: El mismo código funciona en ambos entornos
# 4. **Escalabilidad**: Puede manejar múltiples estrategias y venues simultáneamente
#
# ### Próximos Pasos:
#
# 1. **Integración de Datos Reales**: Configurar Databento o Interactive Brokers
# 2. **Optimización Avanzada**: Usar algoritmos genéticos o machine learning
# 3. **Risk Management**: Implementar controles de riesgo más sofisticados
# 4. **Alertas y Notificaciones**: Integrar con servicios de mensajería
# 5. **Deployment**: Configurar en servidor con Redis para producción

# %%
print("✅ Migración completa a NautilusTrader!")
print("\nPara comenzar con datos reales:")
print("1. Instala NautilusTrader: pip install nautilus-trader")
print("2. Configura tu proveedor de datos (Databento, IB, etc.)")
print("3. Ajusta los parámetros del instrumento XAU/USD")
print("4. Ejecuta backtests con datos históricos reales")
print("5. Optimiza y despliega en producción")