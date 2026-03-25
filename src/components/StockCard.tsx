"use client";

import { motion } from "framer-motion";
import { useRouter } from "next/navigation";
import { TrendingUp, TrendingDown, ChevronRight } from "lucide-react";

interface StockCardProps {
  ticker: string;
  name: string;
  price: number;
  change: number;
  currency?: string;
}

export default function StockCard({ ticker, name, price, change, currency = "INR" }: StockCardProps) {
  const router = useRouter();
  const isUp = change >= 0;
  const sym = currency === "INR" ? "₹" : "$";
  const locale = currency === "INR" ? "en-IN" : "en-US";

  return (
    <motion.div 
      whileHover={{ 
        scale: 1.02, 
        y: -4,
        boxShadow: "0 0 25px rgba(59, 130, 246, 0.2)",
        borderColor: "rgba(59, 130, 246, 0.4)"
      }}
      whileTap={{ scale: 0.98 }}
      onClick={() => router.push(`/stock/${ticker}`)}
      className="glass-card p-6 futuristic-border group cursor-pointer border-white/5"
    >
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-lg font-bold text-accent group-hover:text-blue-400 transition-colors uppercase">
            {ticker}
          </h3>
          <p className="text-[11px] text-gray-400 line-clamp-1">{name}</p>
        </div>
        <div className={`p-2 rounded-lg ${isUp ? 'bg-success/20' : 'bg-danger/20'}`}>
          {isUp ? <TrendingUp className="text-success" size={16} /> : <TrendingDown className="text-danger" size={16} />}
        </div>
      </div>

      <div className="space-y-1">
        <p className="text-xl font-mono font-bold">{sym}{price.toLocaleString(locale, { minimumFractionDigits: 2 })}</p>
        <p className={`text-sm font-bold ${isUp ? 'text-success' : 'text-danger'}`}>
          {isUp ? '+' : ''}{change.toFixed(2)}%
        </p>
      </div>

      <div className="mt-3 pt-3 border-t border-white/5 flex items-center justify-between">
        <span className="text-[10px] text-gray-500 font-black tracking-widest uppercase">AI Analysis →</span>
        <ChevronRight size={14} className="text-gray-500 group-hover:text-accent group-hover:translate-x-1 transition-all" />
      </div>
    </motion.div>
  );
}
