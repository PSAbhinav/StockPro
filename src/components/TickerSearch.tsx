"use client";

import { useState, useEffect, useRef } from "react";
import { Search, Loader2 } from "lucide-react";
import { useRouter } from "next/navigation";

interface SearchResult {
  ticker: string;
  symbol: string;
  name: string;
  exchange: string;
}

export default function TickerSearch() {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const debounceRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    if (!query.trim()) {
      setResults([]);
      setShowDropdown(false);
      return;
    }

    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(async () => {
      setLoading(true);
      try {
        const res = await fetch(`/api/python/search-ticker?q=${encodeURIComponent(query)}`);
        const data = await res.json();
        setResults(data.results || []);
        setShowDropdown(true);
      } catch {
        setResults([]);
      } finally {
        setLoading(false);
      }
    }, 250);

    return () => clearTimeout(debounceRef.current);
  }, [query]);

  // Close dropdown on outside click
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setShowDropdown(false);
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const handleSelect = (ticker: string) => {
    setShowDropdown(false);
    setQuery("");
    router.push(`/stock/${ticker}`);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    setShowDropdown(false);
    router.push(`/stock/${query.toUpperCase().trim()}`);
    setQuery("");
  };

  return (
    <div ref={dropdownRef} className="relative w-full">
      <form onSubmit={handleSubmit} className="relative group">
        <div className="absolute inset-y-0 left-4 flex items-center pointer-events-none text-gray-400 transition-colors group-focus-within:text-accent">
          {loading ? <Loader2 className="animate-spin" size={24} /> : <Search size={24} />}
        </div>
        <input
          type="text"
          placeholder="Search stock (e.g. Google, Reliance...)"
          className="w-full bg-white/5 border border-white/10 rounded-2xl py-4 pl-12 pr-40 text-white placeholder-white/40 focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all font-medium"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => results.length > 0 && setShowDropdown(true)}
        />
        <button
          type="submit"
          disabled={!query.trim()}
          className="absolute right-3 top-1/2 -translate-y-1/2 px-8 py-3 bg-accent hover:bg-blue-600 text-white rounded-xl font-black tracking-widest uppercase transition-all shadow-xl shadow-accent/20 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Analyze
        </button>
      </form>

      {/* Autocomplete Dropdown */}
      {showDropdown && results.length > 0 && (
        <div className="absolute top-full left-0 right-0 mt-2 z-[100] glass-card border border-white/10 overflow-hidden">
          {results.map((r) => (
            <button
              key={r.symbol}
              onClick={() => handleSelect(r.ticker)}
              className="w-full text-left px-6 py-4 hover:bg-white/5 transition-colors flex items-center justify-between border-b border-white/5 last:border-0"
            >
              <div>
                <span className="text-accent font-bold text-lg">{r.ticker}</span>
                <span className="text-gray-400 ml-3 text-sm">{r.name}</span>
              </div>
              <span className="text-[10px] text-gray-600 font-bold tracking-widest uppercase bg-white/5 px-2 py-1 rounded">
                {r.exchange}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
