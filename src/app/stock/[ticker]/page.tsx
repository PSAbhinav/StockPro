"use client";

import { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { ArrowLeft, Cpu, Activity, TrendingUp, TrendingDown, Shield, Loader2, BarChart3, Newspaper, Building2 } from "lucide-react";
import AdvancedChart from "@/components/Chart";
import NewsCard from "@/components/NewsCard";

const formatPrice = (val: number, currency: string) => {
  const sym = currency === 'INR' ? '₹' : '$';
  const locale = currency === 'INR' ? 'en-IN' : 'en-US';
  return `${sym}${val.toLocaleString(locale, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
};

const formatLargeNumber = (num: number | null | undefined) => {
  if (!num) return 'N/A';
  if (num >= 1e12) return `${(num / 1e12).toFixed(2)}T`;
  if (num >= 1e9) return `${(num / 1e9).toFixed(2)}B`;
  if (num >= 1e7) return `${(num / 1e7).toFixed(2)}Cr`;
  if (num >= 1e5) return `${(num / 1e5).toFixed(2)}L`;
  return num.toLocaleString();
};

interface NewsArticle {
  title: string;
  publisher: string;
  link: string;
  published: string | number;
  thumbnail?: string;
}

export default function StockDetail() {
  const params = useParams();
  const router = useRouter();
  const ticker = (params.ticker as string).toUpperCase();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [analysis, setAnalysis] = useState<any>(null);
  const [news, setNews] = useState<NewsArticle[]>([]);
  const [newsLoading, setNewsLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        const res = await fetch(`/api/python/predict?ticker=${ticker}`);
        const data = await res.json();
        if (data.status === "success") {
          setAnalysis(data.analysis);
        } else {
          setError(data.message || "Analysis failed");
        }
      } catch {
        setError("Could not connect to the backend.");
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, [ticker]);

  useEffect(() => {
    async function fetchNews() {
      try {
        const res = await fetch(`/api/python/news?ticker=${ticker}`);
        const data = await res.json();
        if (data.status === "success") {
          setNews(data.articles || []);
        }
      } catch {
        console.error("News fetch failed");
      } finally {
        setNewsLoading(false);
      }
    }
    fetchNews();
  }, [ticker]);

  if (loading) {
    return (
      <div className="flex flex-col h-[80vh] items-center justify-center space-y-4">
        <Loader2 className="animate-spin text-accent" size={48} />
        <p className="text-gray-400 font-bold tracking-widest text-sm uppercase">AI analyzing {ticker}...</p>
        <p className="text-gray-600 text-xs">Training on 5 years of data • Calculating technical indicators</p>
      </div>
    );
  }

  if (error || !analysis) {
    return (
      <div className="flex flex-col h-[80vh] items-center justify-center space-y-4">
        <Shield className="text-danger" size={48} />
        <p className="text-white font-bold text-xl">{error || "Analysis not available"}</p>
        <p className="text-gray-400 text-sm">Try: RELIANCE, TCS, GOOGLE, APPLE, TESLA, NVIDIA</p>
        <button onClick={() => router.back()} className="glass-button mt-4">← Back to Dashboard</button>
      </div>
    );
  }

  const currency = analysis.currency || 'INR';
  const isUp = analysis.expected_change >= 0;
  const stats = analysis.key_stats || {};
  const signals = analysis.signals || {};
  const tech = analysis.technical_indicators || {};

  const signalColor = (s: string) => s === 'BUY' ? 'text-success' : s === 'SELL' ? 'text-danger' : 'text-yellow-400';
  const signalBg = (s: string) => s === 'BUY' ? 'bg-success/20' : s === 'SELL' ? 'bg-danger/20' : 'bg-yellow-400/20';

  return (
    <div className="max-w-7xl mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
      {/* Back Button */}
      <button onClick={() => router.back()} className="flex items-center space-x-2 text-gray-400 hover:text-white transition-colors group">
        <ArrowLeft size={18} className="group-hover:-translate-x-1 transition-transform" />
        <span className="font-bold tracking-widest text-xs uppercase">Back to Dashboard</span>
      </button>

      {/* Price Header */}
      <div className="glass-card p-8 relative overflow-hidden">
        <div className="flex flex-col md:flex-row justify-between items-start gap-4">
          <div>
            <div className="flex items-center space-x-3 mb-1">
              <h1 className="text-5xl font-black text-white tracking-tighter">{analysis.ticker}</h1>
              <span className="text-[10px] font-black tracking-widest uppercase bg-white/5 px-2 py-1 rounded text-gray-400">{analysis.exchange}</span>
            </div>
            <p className="text-gray-400 font-medium text-lg">{analysis.display_name || stats.name || ticker}</p>
            {stats.sector && (
              <div className="flex items-center space-x-2 mt-2 text-xs text-gray-500">
                <Building2 size={12} />
                <span>{stats.sector} • {stats.industry}</span>
              </div>
            )}
          </div>
          <div className="text-right">
            <p className="text-4xl font-mono font-black text-white">{formatPrice(analysis.current_price, currency)}</p>
            <p className={`text-lg font-bold ${isUp ? 'text-success' : 'text-danger'}`}>
              {isUp ? '↑' : '↓'} {Math.abs(analysis.expected_change).toFixed(2)}%
            </p>
          </div>
        </div>
        <div className="absolute top-0 right-0 w-64 h-64 bg-accent/5 blur-[80px] rounded-full" />
      </div>

      {/* Chart Section */}
      <div className="glass-card p-6">
        <AdvancedChart ticker={ticker} currency={currency} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column */}
        <div className="lg:col-span-2 space-y-8">
          {/* Key Statistics */}
          <div className="glass-card p-6">
            <h3 className="font-black text-white uppercase tracking-widest text-xs mb-6 flex items-center space-x-2">
              <BarChart3 size={14} className="text-accent" />
              <span>Key Statistics</span>
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {[
                { label: "Market Cap", value: formatLargeNumber(stats.market_cap) },
                { label: "P/E Ratio", value: stats.pe_ratio ? stats.pe_ratio.toFixed(2) : 'N/A' },
                { label: "52W High", value: stats.week52_high ? formatPrice(stats.week52_high, currency) : 'N/A' },
                { label: "52W Low", value: stats.week52_low ? formatPrice(stats.week52_low, currency) : 'N/A' },
                { label: "Avg Volume", value: formatLargeNumber(stats.avg_volume) },
                { label: "Dividend Yield", value: stats.dividend_yield ? `${(stats.dividend_yield * 100).toFixed(2)}%` : 'N/A' },
                { label: "Support", value: formatPrice(tech.support || 0, currency) },
                { label: "Resistance", value: formatPrice(tech.resistance || 0, currency) },
              ].map((item) => (
                <div key={item.label} className="bg-white/5 rounded-xl p-4">
                  <p className="text-[10px] text-gray-500 font-black tracking-widest uppercase mb-1">{item.label}</p>
                  <p className="text-sm font-mono font-bold text-white">{item.value}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Technical Indicators */}
          <div className="glass-card p-6">
            <h3 className="font-black text-white uppercase tracking-widest text-xs mb-6 flex items-center space-x-2">
              <Activity size={14} className="text-accent" />
              <span>Technical Indicators</span>
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div className="bg-white/5 rounded-xl p-4">
                <p className="text-[10px] text-gray-500 font-black tracking-widest uppercase mb-1">RSI (14)</p>
                <p className={`text-2xl font-mono font-bold ${tech.rsi > 70 ? 'text-danger' : tech.rsi < 30 ? 'text-success' : 'text-white'}`}>
                  {tech.rsi?.toFixed(1)}
                </p>
                <p className="text-[10px] text-gray-600 mt-1">{tech.rsi > 70 ? 'Overbought' : tech.rsi < 30 ? 'Oversold' : 'Neutral'}</p>
              </div>
              <div className="bg-white/5 rounded-xl p-4">
                <p className="text-[10px] text-gray-500 font-black tracking-widest uppercase mb-1">MACD</p>
                <p className={`text-2xl font-mono font-bold ${tech.macd >= 0 ? 'text-success' : 'text-danger'}`}>
                  {tech.macd?.toFixed(2)}
                </p>
                <p className="text-[10px] text-gray-600 mt-1">Signal: {tech.macd_signal?.toFixed(2)}</p>
              </div>
              <div className="bg-white/5 rounded-xl p-4">
                <p className="text-[10px] text-gray-500 font-black tracking-widest uppercase mb-1">SMA (20)</p>
                <p className="text-xl font-mono font-bold text-white">{formatPrice(tech.sma_20 || 0, currency)}</p>
              </div>
              <div className="bg-white/5 rounded-xl p-4">
                <p className="text-[10px] text-gray-500 font-black tracking-widest uppercase mb-1">SMA (50)</p>
                <p className="text-xl font-mono font-bold text-white">{formatPrice(tech.sma_50 || 0, currency)}</p>
              </div>
              <div className="bg-white/5 rounded-xl p-4">
                <p className="text-[10px] text-gray-500 font-black tracking-widest uppercase mb-1">Bollinger Upper</p>
                <p className="text-xl font-mono font-bold text-white">{formatPrice(tech.bollinger_upper || 0, currency)}</p>
              </div>
              <div className="bg-white/5 rounded-xl p-4">
                <p className="text-[10px] text-gray-500 font-black tracking-widest uppercase mb-1">Bollinger Lower</p>
                <p className="text-xl font-mono font-bold text-white">{formatPrice(tech.bollinger_lower || 0, currency)}</p>
              </div>
            </div>
          </div>

          {/* News Feed */}
          <div className="space-y-4">
            <h3 className="font-black text-white uppercase tracking-widest text-xs flex items-center space-x-2">
              <Newspaper size={14} className="text-accent" />
              <span>Latest News</span>
            </h3>
            {newsLoading ? (
              <div className="space-y-3">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="glass-card p-5 animate-pulse">
                    <div className="h-4 bg-white/10 rounded w-3/4 mb-3" />
                    <div className="h-3 bg-white/5 rounded w-1/3" />
                  </div>
                ))}
              </div>
            ) : news.length > 0 ? (
              <div className="space-y-3">
                {news.map((article, i) => (
                  <motion.div key={i} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: i * 0.05 }}>
                    <NewsCard {...article} />
                  </motion.div>
                ))}
              </div>
            ) : (
              <div className="glass-card p-8 text-center text-gray-500">
                <p>No recent news available for {analysis.ticker}</p>
              </div>
            )}
          </div>
        </div>

        {/* Right Sidebar */}
        <div className="space-y-6">
          {/* AI Forecast */}
          <div className="glass-card p-8 bg-gradient-to-br from-accent/20 to-transparent relative overflow-hidden group">
            <div className="relative z-10">
              <div className="p-3 bg-white/5 w-fit rounded-xl mb-6 group-hover:bg-accent/20 transition-colors">
                <Cpu className="text-accent" size={32} />
              </div>
              <h3 className="text-2xl font-black text-white mb-1 uppercase tracking-tighter">AI Forecast</h3>
              <p className="text-gray-400 text-xs mb-8">{analysis.model_info.type} • {analysis.model_info.training_period} of data</p>

              <div className="space-y-6">
                <div>
                  <p className="text-[10px] text-gray-500 font-black tracking-widest uppercase mb-1">Target (24h)</p>
                  <p className="text-3xl font-black text-white font-mono">{formatPrice(analysis.predicted_price, currency)}</p>
                  <p className={`text-sm font-bold ${isUp ? 'text-success' : 'text-danger'}`}>
                    {isUp ? '↑' : '↓'} {Math.abs(analysis.expected_change).toFixed(2)}%
                  </p>
                </div>

                {/* Sentiment Gauge */}
                <div className="p-4 bg-black/40 rounded-xl border border-white/10">
                  <p className="text-[10px] text-gray-400 font-black tracking-widest uppercase mb-2">AI Sentiment</p>
                  <div className="flex items-center space-x-3">
                    <div className="flex-1 bg-white/5 h-3 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${analysis.sentiment_score || 50}%` }}
                        transition={{ duration: 1.5, ease: "easeOut" }}
                        className={`h-full rounded-full ${
                          (analysis.sentiment_score || 50) > 60 ? 'bg-success' : 
                          (analysis.sentiment_score || 50) < 40 ? 'bg-danger' : 'bg-yellow-400'
                        }`}
                      />
                    </div>
                    <span className="text-lg font-black text-white">{analysis.sentiment_score || 50}</span>
                  </div>
                </div>

                {/* Multi-Timeframe Signals */}
                <div className="space-y-2">
                  <p className="text-[10px] text-gray-500 font-black tracking-widest uppercase">Multi-Timeframe Signals</p>
                  {[
                    { label: "Short Term", signal: signals.short_term },
                    { label: "Medium Term", signal: signals.medium_term },
                    { label: "Long Term", signal: signals.long_term },
                  ].map((s) => (
                    <div key={s.label} className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                      <span className="text-xs text-gray-400">{s.label}</span>
                      <span className={`text-sm font-black ${signalColor(s.signal || 'HOLD')} ${signalBg(s.signal || 'HOLD')} px-3 py-1 rounded-lg`}>
                        {s.signal || 'HOLD'}
                      </span>
                    </div>
                  ))}
                </div>

                {/* Main Recommendation */}
                <div className={`text-center py-4 rounded-xl font-black tracking-widest text-xl ${
                  analysis.recommendation === 'BUY' ? 'bg-success text-black' : 
                  analysis.recommendation === 'SELL' ? 'bg-danger text-white' : 'bg-yellow-400 text-black'
                }`}>
                  {analysis.recommendation} SIGNAL
                </div>
              </div>
            </div>
            <div className="absolute -bottom-20 -right-20 w-64 h-64 border border-accent/20 rounded-full group-hover:scale-110 transition-transform duration-1000" />
          </div>
        </div>
      </div>
    </div>
  );
}
