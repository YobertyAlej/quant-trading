# 📊 FVG Backtesting Strategy para XAU/USD

**Estrategia de Trading Algorítmico Cuantitativo Basada en Fair Value Gaps (FVGs)**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

## 🚀 Descripción

Este proyecto implementa una **estrategia avanzada de trading algorítmico** que identifica y opera **Fair Value Gaps (FVGs)** en el mercado XAU/USD (Oro). La estrategia utiliza análisis multi-timeframe, gestión dinámica de riesgo y filtros técnicos avanzados para optimizar las entradas y salidas del mercado.

### ✨ Características Principales

- 🎯 **Detección Automática de FVGs**: Identifica gaps de liquidez en timeframes de 4H
- ⏰ **Análisis Multi-Timeframe**: Combina 4H para señales y 5M para entradas precisas
- 📈 **Filtros de Tendencia**: EMA 200 para confirmar dirección del mercado
- 📊 **Análisis de Volumen Avanzado**: Filtro de volumen ascendente multicriteria
- 🛡️ **Gestión de Riesgo Dinámica**: Stop loss que se ajusta con nuevos FVGs
- 📋 **Backtesting Completo**: Sistema completo de simulación histórica
- 📊 **Visualizaciones Interactivas**: Gráficos con Plotly y Matplotlib
- 📑 **Reportes HTML**: Informes detallados con métricas de rendimiento

## 🛠️ Instalación

### Requisitos Previos

- Python 3.8 o superior
- Jupyter Notebook
- Conexión a internet (para descargar datos financieros)

### Instalación Rápida

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/fvg-xau-usd-backtest.git
cd fvg-xau-usd-backtest

# Instalar dependencias
pip install -r requirements.txt

# Iniciar Jupyter Notebook
jupyter notebook
```

### Dependencias

```txt
yfinance>=0.2.61
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.0.0
jupyter>=1.0.0
```

## 🔧 Uso

### Ejecución Básica

1. **Abrir el notebook**: `FVG_XAU_USD_Backtest.ipynb`
2. **Ejecutar todas las celdas** o ejecutar paso a paso:

```python
# Crear instancia del backtester
backtester = FVGBacktester(
    symbol="GC=F",           # Gold Futures (XAU/USD)
    start_date="2023-01-01", # Fecha de inicio
    initial_capital=10000,   # Capital inicial: $10,000
    risk_per_trade=0.005     # Riesgo por trade: 0.5%
)

# Ejecutar backtesting completo
results = backtester.run_backtest()
```

### Configuración de Parámetros

```python
# Parámetros principales
backtester.min_gap_pips = 30      # Tamaño mínimo del gap (pips)
backtester.ema_period = 200       # Período EMA para filtro
backtester.tp_multiple = 2.0      # Ratio riesgo-beneficio
backtester.risk_per_trade = 0.005 # 0.5% de riesgo por trade
```

## 📁 Estructura del Proyecto

```
fvg-xau-usd-backtest/
│
├── README.md                      # Este archivo
├── FVG_XAU_USD_Backtest.ipynb    # Notebook principal
├── FVG_XAU_USD_Backtest.py       # Versión en Python
├── requirements.txt               # Dependencias
│
├── outputs/                       # Archivos generados
│   ├── FVG_Backtest_Report.html  # Reporte HTML
│   └── FVG_Trades_*.csv          # Historial de trades
│
└── docs/                          # Documentación adicional
    ├── strategy_explanation.md    # Explicación de la estrategia
    └── api_reference.md           # Referencia de la API
```

## 🧮 Metodología de la Estrategia

### 1. Detección de Fair Value Gaps (4H)

```python
# FVG Alcista: máximo de hace 2 barras < mínimo de barra actual
if (data_4h.iloc[i-2]['High'] < data_4h.iloc[i]['Low'] and
    data_4h.iloc[i]['Low'] - data_4h.iloc[i-2]['High'] >= min_gap_pips * 0.1):
    # Crear zona FVG
```

### 2. Filtros de Entrada

- ✅ **Tendencia Alcista**: Precio > EMA 200 en 4H
- ✅ **Volumen Ascendente**: Múltiples criterios de volumen
- ✅ **Barrido del Gap**: Precio toca entre 50%-100% del FVG
- ✅ **Cierre sobre el Gap**: Confirmación de fuerza alcista

### 3. Gestión de Riesgo

- 🎯 **Entrada**: En el punto medio del FVG
- 🛑 **Stop Loss Inicial**: Justo debajo del máximo del FVG
- 📈 **Take Profit**: 2x el riesgo (ratio 1:2)
- 🔄 **Stop Loss Dinámico**: Se ajusta con nuevos FVGs de 5M

## 📊 Métricas y Resultados

El sistema genera un reporte completo con las siguientes métricas:

### Métricas de Rendimiento
- **Win Rate**: Porcentaje de trades ganadores
- **Profit Factor**: Relación ganancia/pérdida
- **Sharpe Ratio**: Rendimiento ajustado por riesgo
- **Maximum Drawdown**: Pérdida máxima sostenida

### Métricas de Gestión de Riesgo
- **Average Win/Loss**: Ganancia y pérdida promedio
- **Risk-Reward Ratio**: Relación riesgo-beneficio
- **Mathematical Expectancy**: Expectativa matemática por trade

### Visualizaciones Generadas
- 📈 Curva de equity
- 📉 Análisis de drawdown
- 📊 Distribución de P&L
- 🕐 Duración de trades
- 📍 Señales de entrada/salida en el precio

## 🔬 Optimización de Parámetros

El notebook incluye una función de optimización para encontrar los mejores parámetros:

```python
# Ejecutar optimización automática
optimization_results = optimize_parameters()

# Parámetros a optimizar:
# - Niveles de riesgo: [0.3%, 0.5%, 0.7%, 1.0%]
# - Múltiplos de TP: [1.5x, 2.0x, 2.5x, 3.0x]
# - Tamaños mínimos de gap: [20, 30, 40, 50 pips]
```

## 📄 Archivos de Salida

### 1. Reporte HTML (`FVG_Backtest_Report.html`)
- Resumen completo de rendimiento
- Métricas de P&L y gestión de riesgo
- Parámetros de la estrategia
- Conclusiones y recomendaciones

### 2. Historial de Trades CSV (`FVG_Trades_YYYYMMDD_HHMMSS.csv`)
- Detalles de cada trade ejecutado
- Timestamps de entrada y salida
- P&L y métricas por trade
- Información de FVG utilizado

## ⚠️ Consideraciones Importantes

### Limitaciones
- ⚡ **Datos de Demostración**: Se generan datos sintéticos si falla la descarga
- 🌐 **Dependencia de Internet**: Requiere conexión para datos de yfinance
- 📊 **Limitación de Trades**: Demo limitado a 100 trades para rendimiento

### Advertencias de Riesgo
- 🚨 **Solo para Backtesting**: No es asesoramiento financiero
- 📈 **Rendimiento Pasado**: No garantiza resultados futuros
- 💰 **Trading Real**: Ejecutar en paper trading primero
- ⚖️ **Gestión de Riesgo**: Nunca arriesgar más del 2% del capital

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Próximas Mejoras

- [ ] Filtros adicionales de market regime
- [ ] Machine learning para predicción de FVG quality
- [ ] API para ejecución automática
- [ ] Dashboard en tiempo real
- [ ] Soporte para múltiples instrumentos
- [ ] Integración con brokers

## 📞 Soporte

Si tienes preguntas o problemas:

1. Revisa la documentación en `/docs`
2. Busca en los [Issues](https://github.com/tu-usuario/fvg-xau-usd-backtest/issues)
3. Crea un nuevo issue si es necesario

## 📜 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 🙏 Reconocimientos

- **yfinance**: Para la descarga de datos financieros
- **Plotly**: Para visualizaciones interactivas
- **Pandas**: Para manipulación de datos
- **Comunidad de Trading Algorítmico**: Por inspiración y feedback

---

**⚠️ Disclaimer**: Este software es solo para fines educativos y de investigación. No es asesoramiento financiero. El trading conlleva riesgos y puede resultar en pérdidas. Siempre consulte con un asesor financiero profesional antes de tomar decisiones de inversión.

---

🌟 **¡Si este proyecto te ha sido útil, no olvides darle una estrella!** ⭐ 