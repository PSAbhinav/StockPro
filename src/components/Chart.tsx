"use client";

import { useEffect, useRef, useState } from 'react';
import { createChart, ColorType } from 'lightweight-charts';

interface OHLCVData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

interface ChartProps {
  ticker: string;
  currency?: string;
}

const TIMEFRAMES = [
  { label: '1D', period: '1d', interval: '5m' },
  { label: '1W', period: '5d', interval: '15m' },
  { label: '1M', period: '1mo', interval: '1h' },
  { label: '3M', period: '3mo', interval: '1d' },
  { label: '6M', period: '6mo', interval: '1d' },
  { label: '1Y', period: '1y', interval: '1d' },
  { label: '3Y', period: '3y', interval: '1wk' },
  { label: '5Y', period: '5y', interval: '1wk' },
  { label: 'MAX', period: 'max', interval: '1mo' },
];

type ChartType = 'area' | 'candlestick' | 'line' | 'bar';

const CHART_TYPES: { label: string; value: ChartType; icon: string }[] = [
  { label: 'Area', value: 'area', icon: '📈' },
  { label: 'Candle', value: 'candlestick', icon: '🕯️' },
  { label: 'Line', value: 'line', icon: '📉' },
  { label: 'Bar', value: 'bar', icon: '📊' },
];

export default function AdvancedChart({ ticker, currency = 'INR' }: ChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<ReturnType<typeof createChart> | null>(null);
  const [activeTimeframe, setActiveTimeframe] = useState(TIMEFRAMES[4]); // 6M default
  const [chartType, setChartType] = useState<ChartType>('area');
  const [data, setData] = useState<OHLCVData[]>([]);
  const [loading, setLoading] = useState(true);
  const [currentPrice, setCurrentPrice] = useState<string>('');

  // Fetch data whenever ticker or timeframe changes
  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      try {
        const res = await fetch(
          `/api/python/history?ticker=${ticker}&period=${activeTimeframe.period}&interval=${activeTimeframe.interval}`
        );
        const json = await res.json();
        if (json.status === 'success' && json.data && json.data.length > 0) {
          setData(json.data);
        }
      } catch (err) {
        console.error('Chart data error:', err);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, [ticker, activeTimeframe]);

  // Render chart whenever data or chart type changes
  useEffect(() => {
    if (!chartContainerRef.current || data.length === 0) return;

    // Clean up previous chart
    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
    }

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#94A3B8',
      },
      width: chartContainerRef.current.clientWidth,
      height: 420,
      grid: {
        vertLines: { color: 'rgba(255, 255, 255, 0.03)' },
        horzLines: { color: 'rgba(255, 255, 255, 0.03)' },
      },
      timeScale: {
        borderVisible: false,
        timeVisible: activeTimeframe.interval.includes('m') || activeTimeframe.interval.includes('h'),
      },
      rightPriceScale: {
        borderVisible: false,
      },
      crosshair: {
        vertLine: { color: 'rgba(59, 130, 246, 0.3)', width: 1, style: 2 },
        horzLine: { color: 'rgba(59, 130, 246, 0.3)', width: 1, style: 2 },
      },
    });

    chartRef.current = chart;

    let mainSeries: any;

    // Use v4 API: chart.addCandlestickSeries(), chart.addLineSeries(), etc.
    if (chartType === 'candlestick') {
      mainSeries = chart.addCandlestickSeries({
        upColor: '#34D399',
        downColor: '#F87171',
        borderDownColor: '#F87171',
        borderUpColor: '#34D399',
        wickDownColor: '#F87171',
        wickUpColor: '#34D399',
      });
      mainSeries.setData(data.map(d => ({
        time: d.time as any,
        open: d.open,
        high: d.high,
        low: d.low,
        close: d.close,
      })));
    } else if (chartType === 'bar') {
      mainSeries = chart.addBarSeries({
        upColor: '#34D399',
        downColor: '#F87171',
      });
      mainSeries.setData(data.map(d => ({
        time: d.time as any,
        open: d.open,
        high: d.high,
        low: d.low,
        close: d.close,
      })));
    } else if (chartType === 'line') {
      mainSeries = chart.addLineSeries({
        color: '#3B82F6',
        lineWidth: 2,
      });
      mainSeries.setData(data.map(d => ({ time: d.time as any, value: d.close })));
    } else {
      // Area (default)
      mainSeries = chart.addAreaSeries({
        lineColor: '#3B82F6',
        topColor: 'rgba(59, 130, 246, 0.4)',
        bottomColor: 'rgba(59, 130, 246, 0.0)',
        lineWidth: 2,
      });
      mainSeries.setData(data.map(d => ({ time: d.time as any, value: d.close })));
    }

    // Volume histogram
    if (data[0]?.volume) {
      const volumeSeries = chart.addHistogramSeries({
        priceFormat: { type: 'volume' },
        priceScaleId: 'volume',
      });
      chart.priceScale('volume').applyOptions({
        scaleMargins: { top: 0.85, bottom: 0 },
      });
      volumeSeries.setData(data.map(d => ({
        time: d.time as any,
        value: d.volume || 0,
        color: d.close >= d.open ? 'rgba(52, 211, 153, 0.3)' : 'rgba(248, 113, 113, 0.3)',
      })));
    }

    // Crosshair tooltip
    chart.subscribeCrosshairMove((param: any) => {
      if (param.time && param.seriesData) {
        const val = param.seriesData.get(mainSeries);
        if (val) {
          const price = (val as any).close || (val as any).value || 0;
          const sym = currency === 'INR' ? '₹' : '$';
          setCurrentPrice(`${sym}${price.toLocaleString(undefined, { minimumFractionDigits: 2 })}`);
        }
      } else {
        setCurrentPrice('');
      }
    });

    chart.timeScale().fitContent();

    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
      chartRef.current = null;
    };
  }, [data, chartType, currency, activeTimeframe.interval]);

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        {/* Timeframe Selector */}
        <div className="flex items-center space-x-1 bg-white/5 rounded-xl p-1">
          {TIMEFRAMES.map((tf) => (
            <button
              key={tf.label}
              onClick={() => setActiveTimeframe(tf)}
              className={`px-3 py-1.5 text-xs font-bold uppercase tracking-wider rounded-lg transition-all ${
                activeTimeframe.label === tf.label
                  ? 'bg-accent text-white shadow-lg shadow-accent/20'
                  : 'text-gray-500 hover:text-white hover:bg-white/5'
              }`}
            >
              {tf.label}
            </button>
          ))}
        </div>

        {/* Chart Type Selector */}
        <div className="flex items-center space-x-1 bg-white/5 rounded-xl p-1">
          {CHART_TYPES.map((ct) => (
            <button
              key={ct.value}
              onClick={() => setChartType(ct.value)}
              className={`px-3 py-1.5 text-xs font-bold rounded-lg transition-all flex items-center space-x-1 ${
                chartType === ct.value
                  ? 'bg-accent text-white shadow-lg shadow-accent/20'
                  : 'text-gray-500 hover:text-white hover:bg-white/5'
              }`}
              title={ct.label}
            >
              <span>{ct.icon}</span>
              <span className="hidden sm:inline">{ct.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Crosshair Price */}
      {currentPrice && (
        <div className="text-right text-sm font-mono text-accent font-bold">
          {currentPrice}
        </div>
      )}

      {/* Chart Container */}
      <div className="relative bg-black/20 rounded-2xl overflow-hidden border border-white/5">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/40 z-20">
            <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-accent" />
          </div>
        )}
        <div ref={chartContainerRef} className="w-full" />
      </div>
    </div>
  );
}
