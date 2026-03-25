"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Activity, User, AlertTriangle, Globe, IndianRupee } from "lucide-react";
import StockCard from "@/components/StockCard";
import TickerSearch from "@/components/TickerSearch";

interface StockData {
  ticker: string;
  name: string;
  price: number;
  change: number;
  currency: string;
  exchange: string;
}

export default function Dashboard() {
  const [watchlistLoading, setWatchlistLoading] = useState(true);
  const [indianStocks, setIndianStocks] = useState<StockData[]>([]);
  const [globalStocks, setGlobalStocks] = useState<StockData[]>([]);

  useEffect(() => {
    async function fetchWatchlist() {
      try {
        const res = await fetch("/api/python/watchlist");
        const data = await res.json();
        if (data.status === "success") {
          setIndianStocks(data.indian || []);
          setGlobalStocks(data.global || []);
        }
      } catch (err) {
        console.error("Watchlist fetch failed:", err);
      } finally {
        setWatchlistLoading(false);
      }
    }
    fetchWatchlist();
  }, []);

  const SkeletonCards = ({ count }: { count: number }) => (
    <>
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="glass-card p-6 h-44 animate-pulse">
          <div className="h-4 bg-white/10 rounded w-24 mb-3" />
          <div className="h-3 bg-white/5 rounded w-40 mb-6" />
          <div className="h-8 bg-white/10 rounded w-32 mb-4" />
          <div className="h-3 bg-white/5 rounded w-20" />
        </div>
      ))}
    </>
  );

  return (
    <div className="max-w-7xl mx-auto space-y-10 animate-in fade-in duration-700">
      {/* Header */}
      <header className="flex justify-between items-center glass-card p-6">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-accent rounded-lg flex items-center justify-center shadow-lg shadow-accent/20">
            <Activity className="text-white" />
          </div>
          <h1 className="text-2xl font-black tracking-tighter italic uppercase text-white">
            StockPro <span className="text-accent underline decoration-2 underline-offset-4">AI</span>
          </h1>
        </div>
        <div className="flex items-center space-x-4">
          <div className="hidden md:flex items-center space-x-2 text-xs text-gray-500">
            <div className="w-2 h-2 rounded-full bg-success animate-pulse" />
            <span className="font-bold tracking-widest uppercase">Markets Connected</span>
          </div>
          <button className="glass-button flex items-center space-x-2 border-white/5">
            <User size={18} />
            <span className="hidden sm:inline">Account</span>
          </button>
        </div>
      </header>

      {/* Hero Search */}
      <section className="relative z-20 p-12">
        <div className="absolute inset-0 glass-card bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-blue-900/20 via-transparent to-transparent -z-10" />
        <div className="relative z-10 max-w-2xl">
          <h2 className="text-4xl lg:text-5xl font-black text-white mb-4 tracking-tight leading-tight">
            Predict the <span className="text-accent">Future</span> of Markets.
          </h2>
          <p className="text-gray-400 mb-8 text-lg font-medium">
            AI-powered analysis with interactive charts, real-time news, and multi-timeframe signals. Search any stock globally.
          </p>
          <TickerSearch />
        </div>
        <div className="absolute top-0 right-0 w-96 h-96 bg-accent/10 blur-[100px] rounded-full -mr-48 -mt-48 pointer-events-none" />
        <div className="absolute bottom-0 left-0 w-64 h-64 bg-blue-500/10 blur-[80px] rounded-full -ml-32 -mb-32 pointer-events-none" />
      </section>

      {/* Indian Market */}
      <section className="space-y-6">
        <div className="flex justify-between items-end">
          <div className="flex items-center space-x-3">
            <IndianRupee size={20} className="text-accent" />
            <h3 className="text-2xl font-black text-white tracking-tighter">NSE TOP MOVERS</h3>
          </div>
          <p className={`text-sm font-bold uppercase tracking-widest ${watchlistLoading ? 'text-yellow-400 animate-pulse' : 'text-success'}`}>
            {watchlistLoading ? 'Fetching...' : '● Live from NSE'}
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-5">
          {watchlistLoading ? (
            <SkeletonCards count={8} />
          ) : indianStocks.length > 0 ? (
            indianStocks.map((stock, i) => (
              <motion.div key={stock.ticker} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }}>
                <StockCard {...stock} />
              </motion.div>
            ))
          ) : (
            <div className="col-span-full glass-card p-8 text-center">
              <AlertTriangle className="mx-auto text-yellow-500 mb-4" size={32} />
              <p className="text-gray-400">Could not load Indian market data.</p>
            </div>
          )}
        </div>
      </section>

      {/* Global Markets */}
      <section className="space-y-6">
        <div className="flex justify-between items-end">
          <div className="flex items-center space-x-3">
            <Globe size={20} className="text-accent" />
            <h3 className="text-2xl font-black text-white tracking-tighter">GLOBAL TECH GIANTS</h3>
          </div>
          <p className={`text-sm font-bold uppercase tracking-widest ${watchlistLoading ? 'text-yellow-400 animate-pulse' : 'text-success'}`}>
            {watchlistLoading ? 'Fetching...' : '● Live from NASDAQ'}
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-5">
          {watchlistLoading ? (
            <SkeletonCards count={4} />
          ) : globalStocks.length > 0 ? (
            globalStocks.map((stock, i) => (
              <motion.div key={stock.ticker} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }}>
                <StockCard {...stock} />
              </motion.div>
            ))
          ) : (
            <div className="col-span-full glass-card p-8 text-center">
              <AlertTriangle className="mx-auto text-yellow-500 mb-4" size={32} />
              <p className="text-gray-400">Could not load global market data.</p>
            </div>
          )}
        </div>
      </section>

      {/* Footer */}
      <footer className="text-center py-8 text-gray-600 text-xs space-y-1">
        <p>StockPro AI — Random Forest Ensemble • RSI • MACD • Bollinger Bands • Multi-Timeframe Signals</p>
        <p>AI predictions are for informational purposes only. Not financial advice.</p>
      </footer>
    </div>
  );
}
