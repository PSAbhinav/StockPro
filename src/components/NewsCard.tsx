"use client";

import { motion } from "framer-motion";
import { ExternalLink, Clock } from "lucide-react";

interface NewsArticle {
  title: string;
  publisher: string;
  link: string;
  published: string | number;
  thumbnail?: string;
}

export default function NewsCard({ title, publisher, link, published, thumbnail }: NewsArticle) {
  const formatTime = (ts: string | number) => {
    if (!ts) return "";
    try {
      const date = typeof ts === "number" ? new Date(ts * 1000) : new Date(ts);
      const now = new Date();
      const diffMs = now.getTime() - date.getTime();
      const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
      if (diffHours < 1) return "Just now";
      if (diffHours < 24) return `${diffHours}h ago`;
      const diffDays = Math.floor(diffHours / 24);
      if (diffDays < 7) return `${diffDays}d ago`;
      return date.toLocaleDateString('en-IN', { day: 'numeric', month: 'short' });
    } catch {
      return "";
    }
  };

  return (
    <motion.a
      href={link}
      target="_blank"
      rel="noopener noreferrer"
      whileHover={{ scale: 1.01, y: -2 }}
      className="glass-card p-5 flex gap-4 group cursor-pointer futuristic-border"
    >
      {thumbnail && (
        <div className="w-24 h-20 flex-shrink-0 rounded-xl overflow-hidden bg-white/5">
          <img src={thumbnail} alt="" className="w-full h-full object-cover" />
        </div>
      )}
      <div className="flex-1 min-w-0">
        <h4 className="text-sm font-bold text-white group-hover:text-accent transition-colors line-clamp-2 leading-snug mb-2">
          {title}
        </h4>
        <div className="flex items-center space-x-3 text-[10px] text-gray-500">
          <span className="font-bold uppercase tracking-widest">{publisher}</span>
          {published && (
            <>
              <Clock size={10} />
              <span>{formatTime(published)}</span>
            </>
          )}
        </div>
      </div>
      <div className="flex-shrink-0 self-center opacity-0 group-hover:opacity-100 transition-opacity">
        <ExternalLink size={14} className="text-accent" />
      </div>
    </motion.a>
  );
}
