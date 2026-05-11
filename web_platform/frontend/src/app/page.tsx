'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Sparkles, Zap, Bot, Package, ThumbsUp, ShoppingBag, Camera, Star, Tag, Gift, Cpu, Layers, ShieldCheck, MessageCircle, TrendingUp } from 'lucide-react';
import { useRouter } from 'next/navigation';

export default function MultimodalDashboard() {
  const [url, setUrl] = useState('');
  const router = useRouter();

  const handleAnalyze = (e: React.FormEvent) => {
    e.preventDefault();
    if (!url) return;
    router.push(`/analyze?url=${encodeURIComponent(url)}`);
  };

  return (
    <div className="w-full relative z-10">
      <AnimatePresence mode="wait">
        <motion.div key="idle" initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, scale: 0.9 }} className="w-full max-w-4xl mx-auto relative z-10 mt-10">

          {/* BACKGROUND DECORATIONS (The scattered cards) */}
          <div className="absolute inset-0 pointer-events-none hidden md:block">
            {/* ShoppingBag (hồng) */}
            <motion.div animate={{ y: [0, -10, 0] }} transition={{ repeat: Infinity, duration: 4, ease: "easeInOut" }} className="absolute -left-10 top-0 w-16 h-16 bg-pink-100/80 backdrop-blur-md border border-white rounded-2xl shadow-xl flex items-center justify-center rotate-[-12deg]">
              <ShoppingBag className="w-8 h-8 text-pink-500" />
            </motion.div>
            {/* Package (cam) */}
            <motion.div animate={{ y: [0, 15, 0] }} transition={{ repeat: Infinity, duration: 5, ease: "easeInOut", delay: 1 }} className="absolute -left-24 top-40 w-24 h-24 bg-orange-100/80 backdrop-blur-md border border-white rounded-3xl shadow-xl flex items-center justify-center rotate-[15deg]">
              <Package className="w-12 h-12 text-orange-500" />
            </motion.div>
            {/* MessageCircle (xanh cyan) */}
            <motion.div animate={{ y: [0, -15, 0] }} transition={{ repeat: Infinity, duration: 6, ease: "easeInOut", delay: 2 }} className="absolute left-10 bottom-10 w-20 h-20 bg-cyan-100/80 backdrop-blur-md border border-white rounded-2xl shadow-xl flex items-center justify-center rotate-[-25deg]">
              <MessageCircle className="w-10 h-10 text-cyan-500" />
            </motion.div>
            {/* Bot (tím) */}
            <motion.div animate={{ y: [0, 10, 0] }} transition={{ repeat: Infinity, duration: 4.5, ease: "easeInOut", delay: 0.5 }} className="absolute left-48 -bottom-10 w-16 h-16 bg-purple-100/80 backdrop-blur-md border border-white rounded-xl shadow-xl flex items-center justify-center rotate-[10deg]">
              <Bot className="w-8 h-8 text-purple-500" />
            </motion.div>

            {/* ThumbsUp (xanh lá) */}
            <motion.div animate={{ y: [0, -12, 0] }} transition={{ repeat: Infinity, duration: 5.5, ease: "easeInOut", delay: 1.5 }} className="absolute -right-10 top-10 w-20 h-20 bg-green-100/80 backdrop-blur-md border border-white rounded-3xl shadow-xl flex items-center justify-center rotate-[18deg]">
              <ThumbsUp className="w-10 h-10 text-green-500" />
            </motion.div>
            {/* TrendingUp (vàng) */}
            <motion.div animate={{ y: [0, 14, 0] }} transition={{ repeat: Infinity, duration: 4.2, ease: "easeInOut", delay: 0.8 }} className="absolute -right-24 top-48 w-24 h-24 bg-yellow-100/80 backdrop-blur-md border border-white rounded-3xl shadow-xl flex items-center justify-center rotate-[-15deg]">
              <TrendingUp className="w-12 h-12 text-yellow-500" />
            </motion.div>
            {/* Sparkles (tím) */}
            <motion.div animate={{ y: [0, -10, 0] }} transition={{ repeat: Infinity, duration: 5, ease: "easeInOut", delay: 2.5 }} className="absolute right-10 -bottom-5 w-16 h-16 bg-fuchsia-100/80 backdrop-blur-md border border-white rounded-2xl shadow-xl flex items-center justify-center rotate-[22deg]">
              <Sparkles className="w-8 h-8 text-fuchsia-500" />
            </motion.div>
            {/* Zap (xanh) */}
            <motion.div animate={{ y: [0, 10, 0] }} transition={{ repeat: Infinity, duration: 4.8, ease: "easeInOut", delay: 1.2 }} className="absolute right-48 -bottom-16 w-16 h-16 bg-blue-100/80 backdrop-blur-md border border-white rounded-xl shadow-xl flex items-center justify-center rotate-[-10deg]">
              <Zap className="w-8 h-8 text-blue-500" />
            </motion.div>

            {/* -- NEW CARDS -- */}
            {/* Star (vàng ngọc) */}
            <motion.div animate={{ y: [0, -15, 0] }} transition={{ repeat: Infinity, duration: 5.2, ease: "easeInOut", delay: 0.3 }} className="absolute -left-44 top-20 w-16 h-16 bg-amber-100/80 backdrop-blur-md border border-white rounded-2xl shadow-xl flex items-center justify-center rotate-[-20deg]">
              <Star className="w-8 h-8 text-amber-500" />
            </motion.div>
            {/* Camera (chàm) */}
            <motion.div animate={{ y: [0, 12, 0] }} transition={{ repeat: Infinity, duration: 4.7, ease: "easeInOut", delay: 1.8 }} className="absolute -left-40 top-72 w-20 h-20 bg-indigo-100/80 backdrop-blur-md border border-white rounded-3xl shadow-xl flex items-center justify-center rotate-[25deg]">
              <Camera className="w-10 h-10 text-indigo-500" />
            </motion.div>
            {/* Tag (xanh cổ vịt) */}
            <motion.div animate={{ y: [0, -8, 0] }} transition={{ repeat: Infinity, duration: 5.8, ease: "easeInOut", delay: 2.1 }} className="absolute -left-20 -bottom-20 w-16 h-16 bg-teal-100/80 backdrop-blur-md border border-white rounded-2xl shadow-xl flex items-center justify-center rotate-[15deg]">
              <Tag className="w-8 h-8 text-teal-500" />
            </motion.div>
            {/* Gift (đỏ) */}
            <motion.div animate={{ y: [0, 14, 0] }} transition={{ repeat: Infinity, duration: 4.6, ease: "easeInOut", delay: 0.7 }} className="absolute -right-48 top-20 w-20 h-20 bg-red-100/80 backdrop-blur-md border border-white rounded-3xl shadow-xl flex items-center justify-center rotate-[12deg]">
              <Gift className="w-10 h-10 text-red-500" />
            </motion.div>
            {/* Cpu (xám) */}
            <motion.div animate={{ y: [0, -10, 0] }} transition={{ repeat: Infinity, duration: 5.4, ease: "easeInOut", delay: 1.4 }} className="absolute -right-56 top-72 w-16 h-16 bg-slate-200/80 backdrop-blur-md border border-white rounded-2xl shadow-xl flex items-center justify-center rotate-[-18deg]">
              <Cpu className="w-8 h-8 text-slate-600" />
            </motion.div>
            {/* Layers (tím nhạt) */}
            <motion.div animate={{ y: [0, -12, 0] }} transition={{ repeat: Infinity, duration: 6.2, ease: "easeInOut", delay: 0.9 }} className="absolute left-80 -top-20 w-14 h-14 bg-violet-100/80 backdrop-blur-md border border-white rounded-xl shadow-xl flex items-center justify-center rotate-[10deg]">
              <Layers className="w-6 h-6 text-violet-500" />
            </motion.div>
            {/* ShieldCheck (ngọc lục bảo) */}
            <motion.div animate={{ y: [0, 12, 0] }} transition={{ repeat: Infinity, duration: 4.9, ease: "easeInOut", delay: 2.2 }} className="absolute right-80 -top-16 w-16 h-16 bg-emerald-100/80 backdrop-blur-md border border-white rounded-2xl shadow-xl flex items-center justify-center rotate-[-15deg]">
              <ShieldCheck className="w-8 h-8 text-emerald-500" />
            </motion.div>
          </div>

          <div className="relative z-10 text-center mt-20">
            <h2 className="text-4xl md:text-5xl lg:text-6xl font-extrabold mb-6 tracking-tight leading-tight">
              <span className="bg-gradient-to-r from-blue-500 to-indigo-600 dark:from-indigo-400 dark:to-cyan-400 bg-clip-text text-transparent block pb-2 text-5xl md:text-6xl lg:text-7xl font-momo">e-commerce</span>
              <span className="text-slate-800 dark:text-slate-100 block text-4xl md:text-5xl lg:text-6xl">Review Analytics</span>
            </h2>
            <p className="font-quicksand text-center text-slate-500 dark:text-slate-400 mb-12 text-lg md:text-xl max-w-2xl mx-auto">
              Dán URL sản phẩm từ Shopee, Tiki, Lazada để AI phân tích cảm xúc và tình trạng hàng hóa qua hình ảnh
            </p>
            <form onSubmit={handleAnalyze} className="relative group max-w-3xl mx-auto">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-200 to-indigo-200 dark:from-indigo-500 dark:to-purple-500 rounded-full blur-xl opacity-50 dark:opacity-20 group-hover:opacity-100 dark:group-hover:opacity-40 transition duration-500"></div>
              <div className="relative flex items-center bg-white/90 dark:bg-slate-900/90 backdrop-blur-xl border border-white dark:border-slate-700/50 rounded-full p-2 pl-6 shadow-[0_8px_30px_rgb(0,0,0,0.08)] dark:shadow-2xl hover:shadow-[0_8px_30px_rgb(0,0,0,0.12)] transition-shadow">
                <Search className="w-6 h-6 text-blue-500 dark:text-indigo-400" />
                <input
                  type="url" required placeholder="https://shopee.vn/product-url..."
                  className="flex-1 bg-transparent border-none outline-none text-slate-700 dark:text-slate-200 px-4 py-4 text-base md:text-lg placeholder:text-slate-400 dark:placeholder:text-slate-600"
                  value={url} onChange={(e) => setUrl(e.target.value)}
                />
                <button type="submit" className="cursor-pointer bg-gradient-to-r from-cyan-500 to-blue-600 dark:from-indigo-600 dark:to-purple-600 hover:from-cyan-400 hover:to-blue-500 dark:hover:from-indigo-500 dark:hover:to-purple-500 transition-colors text-white font-semibold py-3 px-8 md:py-4 md:px-10 rounded-full shadow-lg shadow-blue-500/30 text-base md:text-lg">
                  Phân Tích
                </button>
              </div>
              <p className="font-quicksand text-sm text-slate-500 dark:text-slate-400 mt-6 text-center">Hỗ trợ các sàn: Shopee, Tiki, Lazada, TGDĐ...</p>
            </form>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
