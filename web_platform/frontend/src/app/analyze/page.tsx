'use client';

import { useState, useMemo, useEffect, Suspense, useRef } from 'react';
import { io, Socket } from 'socket.io-client';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Loader2, CheckCircle, AlertTriangle, MessageSquare, TrendingUp, Bot, Sparkles,
  ShieldAlert, ShieldCheck, AlertOctagon, Filter, Image as ImageIcon, List, ShoppingCart,
  X, ArrowLeftIcon, Download, Smile, Frown
} from 'lucide-react';
import {
  PieChart, Pie, Cell, ResponsiveContainer, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid, Legend,
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, LineChart, Line
} from 'recharts';
import { useRouter, useSearchParams } from 'next/navigation';
import { supabase } from '@/lib/supabase';

type Status = 'PROCESSING' | 'COMPLETED' | 'ERROR';

// ─── Word Cloud ────────────────────────────────────────────────────────────
type Kw = { text: string; value: number };
type PlacedWord = {
  text: string; cx: number; cy: number;
  ww: number; wh: number;
  fontSize: number; fontWeight: number; fill: string; vertical: boolean;
};

function placeWords(words: Kw[], W: number, H: number, colors: string[]): PlacedWord[] {
  const sorted = [...words].sort((a, b) => b.value - a.value);
  const maxV = sorted[0]?.value || 60;
  const placed: PlacedWord[] = [];

  sorted.forEach((word, idx) => {
    const ratio = (word.value || 10) / maxV;
    const fontSize = Math.round(11 + ratio * 38);
    const fontWeight = ratio > 0.72 ? 900 : ratio > 0.48 ? 700 : ratio > 0.28 ? 600 : 500;
    const cIdx = Math.min(Math.floor((1 - ratio) * colors.length), colors.length - 1);
    const fill = colors[cIdx];
    const vertical = idx === 2 || idx === 6 || idx === 10;

    const charW = fontSize * 0.56;
    const charH = fontSize * 1.25;
    const ww = vertical ? charH * 1.05 : word.text.length * charW;
    const wh = vertical ? word.text.length * charW : charH;

    let ok = false;
    for (let step = 0; step < 800; step++) {
      const t = step * 0.15;
      const r = 0.85 * t;
      const cx = W / 2 + r * Math.cos(t);
      const cy = H / 2 + r * Math.sin(t) * 0.72;
      if (cx - ww / 2 < 3 || cy - wh / 2 < 3 || cx + ww / 2 > W - 3 || cy + wh / 2 > H - 3) continue;
      let collide = false;
      for (const p of placed) {
        if (Math.abs(cx - p.cx) < (ww + p.ww) / 2 + 3 && Math.abs(cy - p.cy) < (wh + p.wh) / 2 + 3) {
          collide = true; break;
        }
      }
      if (!collide) { placed.push({ text: word.text, cx, cy, ww, wh, fontSize, fontWeight, fill, vertical }); ok = true; break; }
    }
    if (!ok) {
      for (let a = 0; a < 30; a++) {
        const cx = ww / 2 + 4 + Math.random() * (W - ww - 8);
        const cy = wh / 2 + 4 + Math.random() * (H - wh - 8);
        let collide = false;
        for (const p of placed) {
          if (Math.abs(cx - p.cx) < (ww + p.ww) / 2 + 2 && Math.abs(cy - p.cy) < (wh + p.wh) / 2 + 2) { collide = true; break; }
        }
        if (!collide) { placed.push({ text: word.text, cx, cy, ww, wh, fontSize, fontWeight, fill, vertical }); break; }
      }
    }
  });
  return placed;
}

const BLUE_SHADES = [
  "fill-blue-800 dark:fill-blue-200",
  "fill-blue-700 dark:fill-blue-300",
  "fill-blue-600 dark:fill-blue-400",
  "fill-blue-500 dark:fill-blue-400",
  "fill-blue-500 dark:fill-blue-500",
  "fill-blue-400 dark:fill-blue-500",
  "fill-blue-400 dark:fill-blue-600",
];
const RED_SHADES = [
  "fill-rose-800 dark:fill-rose-200",
  "fill-rose-700 dark:fill-rose-300",
  "fill-rose-600 dark:fill-rose-400",
  "fill-rose-500 dark:fill-rose-400",
  "fill-rose-500 dark:fill-rose-500",
  "fill-rose-400 dark:fill-rose-500",
  "fill-rose-400 dark:fill-rose-600",
];

function WordCloudSVG({ words, colors }: { words: Kw[]; colors: string[] }) {
  const W = 520, H = 270;
  const placed = useMemo(() => placeWords(words, W, H, colors), [words, colors]);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block", overflow: "visible" }}>
      {placed.map((w, i) => (
        <text
          key={i}
          x={w.cx}
          y={w.cy}
          textAnchor="middle"
          dominantBaseline="central"
          fontSize={w.fontSize}
          fontWeight={w.fontWeight}
          className={w.fill}
          transform={w.vertical ? `rotate(90,${w.cx},${w.cy})` : undefined}
          style={{ cursor: "default", userSelect: "none", fontFamily: "'Quicksand',sans-serif", transition: "opacity .15s" }}
          onMouseEnter={e => ((e.target as SVGTextElement).style.opacity = "0.65")}
          onMouseLeave={e => ((e.target as SVGTextElement).style.opacity = "1")}
        >
          {w.text}
        </text>
      ))}
    </svg>
  );
}
// ───────────────────────────────────────────────────────────────────────────

function AnalyzeContent() {
  const [status, setStatus] = useState<Status>('PROCESSING');
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('');
  const [result, setResult] = useState<any>(null);

  // For Epic 2: Dynamic Filters & Tabs
  const [activeTab, setActiveTab] = useState<'table' | 'images'>('table');
  const [mainTab, setMainTab] = useState<'overview' | 'details' | 'recommendations'>('overview');
  const [filterSentiment, setFilterSentiment] = useState('all');
  const [filterLabel, setFilterLabel] = useState('all');
  const [filterRating, setFilterRating] = useState('all');
  const [selectedImage, setSelectedImage] = useState<any>(null);
  const [isExporting, setIsExporting] = useState(false);

  const router = useRouter();
  const searchParams = useSearchParams();
  const analyzedUrlRef = useRef<string | null>(null);

  useEffect(() => {
    const url = searchParams.get('url');
    const historyId = searchParams.get('historyId');

    if (historyId) {
      if (analyzedUrlRef.current === `history-${historyId}`) return;
      analyzedUrlRef.current = `history-${historyId}`;
      loadHistoryDetail(historyId);
    } else if (url) {
      if (analyzedUrlRef.current === `url-${url}`) return;
      analyzedUrlRef.current = `url-${url}`;
      handleAnalyze(url);
    } else {
      router.push('/');
    }
  }, [searchParams, router]);

  const loadHistoryDetail = async (productId: string) => {
    try {
      setStatus('PROCESSING');
      setProgress(100);
      setMessage('Đang trích xuất dữ liệu từ Database...');
      setMainTab('overview');

      const { data: { session } } = await supabase.auth.getSession();
      const res = await axios.get(`${process.env.NEXT_PUBLIC_API_URL}/history/${productId}`, {
        headers: { Authorization: `Bearer ${session?.access_token}` }
      });

      const product = res.data.data;
      setResult({
        productId: product.id,
        productData: {
          name: product.name,
          thumbnail: product.thumbnail,
          price: 'Giá lấy từ DB'
        },
        reviews: product.reviews,
        summary: product.report?.summary_text || 'Không có tóm tắt.',
        metadata: product.report?.metadata || {}
      });
      setStatus('COMPLETED');
    } catch (error) {
      alert('Không có quyền tải chi tiết sản phẩm này');
      router.push('/');
    }
  };

  const handleAnalyze = async (urlStr: string) => {
    setStatus('PROCESSING');
    setProgress(5);
    setMessage('Đang kết nối AI Engine...');
    setMainTab('overview');

    try {
      const { data: { session } } = await supabase.auth.getSession();
      const res = await axios.post(`${process.env.NEXT_PUBLIC_API_URL}/analyze`, {
        url: urlStr,
        userId: session?.user?.id
      });
      const { productId } = res.data;

      const socket: Socket = io(process.env.NEXT_PUBLIC_SOCKET_URL || 'http://localhost:3000');
      socket.emit('join-room', productId);

      socket.on('progress', (data) => {
        setProgress(data.progress);
        setMessage(data.message);
      });

      socket.on('finished', (data) => {
        setStatus('COMPLETED');
        setResult(data);
        socket.disconnect();
      });
    } catch (error) {
      console.error(error);
      setStatus('ERROR');
      alert('Lỗi kết nối với Backend. Đảm bảo Node.js đang chạy.');
      router.push('/');
    }
  };

  const handleExportPDF = async () => {
    try {
      const targetId = result.id || result.productId;
      if (!targetId) {
        alert('Không tìm thấy ID sản phẩm.');
        return;
      }
      setIsExporting(true);
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) return alert('Vui lòng đăng nhập lại');

      const res = await axios.get(`${process.env.NEXT_PUBLIC_API_URL}/history/${targetId}/export`, {
        headers: { Authorization: `Bearer ${session.access_token}` },
        responseType: 'text'
      });

      const printWindow = window.open('', '_blank');
      if (printWindow) {
        printWindow.document.open();
        printWindow.document.write(res.data);
        printWindow.document.close();

        // Đợi các tài nguyên (ảnh, font) tải xong rồi tự động in
        printWindow.onload = () => {
          setTimeout(() => {
            printWindow.print();
          }, 500);
        };
      } else {
        alert('Vui lòng cho phép mở popup để có thể in báo cáo!');
      }
    } catch (error) {
      alert('Lỗi xuất báo cáo PDF!');
      console.error(error);
    } finally {
      setIsExporting(false);
    }
  };

  const chartData = useMemo(() => {
    if (!result?.reviews) return { sentimentData: [], labelData: [] };
    const sentiments = { positive: 0, neutral: 0, negative: 0 };
    const labels = { intact: 0, damaged: 0, wrong_item: 0, irrelevant: 0 };

    result.reviews.forEach((r: any) => {
      if (sentiments[r.sentiment as keyof typeof sentiments] !== undefined) sentiments[r.sentiment as keyof typeof sentiments]++;
      if (labels[r.label as keyof typeof labels] !== undefined) labels[r.label as keyof typeof labels]++;
    });

    return {
      sentimentData: [
        { name: 'Tích cực', value: sentiments.positive, color: '#10b981' },
        { name: 'Trung lập', value: sentiments.neutral, color: '#8b5cf6' },
        { name: 'Tiêu cực', value: sentiments.negative, color: '#ef4444' }
      ],
      labelData: [
        { name: 'Nguyên vẹn', value: labels.intact },
        { name: 'Móp méo', value: labels.damaged },
        { name: 'Sai hàng', value: labels.wrong_item },
        { name: 'Không liên quan', value: labels.irrelevant }
      ]
    };
  }, [result]);

  const filteredReviews = useMemo(() => {
    if (!result?.reviews) return [];
    return result.reviews.filter((r: any) => {
      if (filterSentiment !== 'all' && r.sentiment !== filterSentiment) return false;
      if (filterLabel !== 'all' && r.label !== filterLabel) return false;
      if (filterRating !== 'all' && r.rating.toString() !== filterRating) return false;
      return true;
    });
  }, [result, filterSentiment, filterLabel, filterRating]);

  // Cross-Modal Alert Logic
  const crossModalAlerts = useMemo(() => {
    if (!result?.reviews) return 0;
    return result.reviews.filter((r: any) => (r.rating >= 4 || r.sentiment === 'positive') && r.label === 'damaged').length;
  }, [result]);

  const isDark = typeof document !== 'undefined' ? document.documentElement.classList.contains('dark') : false;
  const metadata = result?.metadata || {};

  // Map ABSA Data
  const aspectData = useMemo(() => {
    if (!metadata.aspectSentiment) return [];
    return [
      { subject: 'Sản phẩm', A: metadata.aspectSentiment.Product || 0, fullMark: 5 },
      { subject: 'Đóng gói', A: metadata.aspectSentiment.Packaging || 0, fullMark: 5 },
      { subject: 'Vận chuyển', A: metadata.aspectSentiment.Shipping || 0, fullMark: 5 },
    ];
  }, [metadata]);

  return (
    <div className="w-full relative z-10 px-6 pb-6">
      <AnimatePresence mode="wait">
        {status === 'PROCESSING' && (
          <motion.div key="processing" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="w-full max-w-lg mx-auto bg-white dark:bg-slate-900/40 dark:backdrop-blur-xl rounded-3xl shadow-xl border border-slate-100 dark:border-white/10 p-10 text-center relative overflow-hidden mt-20">
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-64 h-64 bg-blue-100 dark:bg-indigo-500/20 rounded-full blur-[80px]"></div>
            <Loader2 className="w-16 h-16 text-blue-500 dark:text-indigo-400 animate-spin mx-auto mb-8 relative z-10" />
            <h3 className="text-2xl font-bold mb-3 relative z-10 text-slate-800 dark:text-slate-100 font-quicksand">AI Đang Xử Lý Dữ Liệu</h3>
            <p className="text-slate-500 dark:text-slate-400 mb-10 relative z-10 font-quicksand">{message}</p>
            <div className="w-full bg-slate-100 dark:bg-slate-800/50 rounded-full h-4 mb-3 overflow-hidden border border-transparent dark:border-slate-700/50 relative z-10 p-1">
              <motion.div
                className="bg-blue-500 dark:bg-gradient-to-r dark:from-indigo-500 dark:via-purple-500 dark:to-cyan-400 h-full rounded-full shadow-sm dark:shadow-[0_0_10px_rgba(99,102,241,0.5)]"
                initial={{ width: 0 }} animate={{ width: `${progress}%` }} transition={{ duration: 0.5 }}
              />
            </div>
            <p className="text-right text-sm text-blue-600 dark:text-indigo-300 font-mono font-semibold relative z-10">{progress}%</p>
          </motion.div>
        )}

        {status === 'COMPLETED' && result && (
          <motion.div key="completed" initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} className="w-full max-w-6xl mx-auto space-y-8 mt-10">

            {/* Tóm tắt Quét */}
            <div className="flex items-center justify-center mt-6 mb-10">
              <div className="w-full flex items-center justify-center gap-3 self-center">
                <CheckCircle className="w-10 h-10 text-emerald-600 dark:text-emerald-400" />
                <div className="text-2xl font-quicksand font-semibold dark:text-slate-100">Đã quét <span className="font-mono text-blue-500 text-4xl">{result.reviews?.length || 0}</span> reviews</div>
              </div>

              <div className="text-center flex items-center justify-center">
                <button onClick={() => router.push('/')} className="p-1 rounded-full border dark:border-white text-black dark:text-white cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-700" title="Quay về trang chủ">
                  <ArrowLeftIcon />
                </button>
              </div>

            </div>

            {/* Main Navigation Tabs */}
            <div className="flex flex-col md:flex-row items-center justify-between mb-8 relative z-10 w-full gap-4">
              <div className="flex-1 hidden md:block"></div>

              <div className="inline-flex gap-2 bg-white/50 dark:bg-slate-800/50 p-1.5 px-2 rounded-[20px] shadow-sm border border-slate-200/50 dark:border-slate-700/50 backdrop-blur-xl shrink-0">
                <button
                  onClick={() => setMainTab('overview')}
                  className={`cursor-pointer flex items-center gap-2 px-6 py-3 rounded-2xl font-bold transition-all duration-300 font-quicksand ${mainTab === 'overview' ? 'bg-white dark:bg-slate-700 shadow-[0_2px_10px_rgba(0,0,0,0.05)] text-blue-600 dark:text-blue-400 scale-105' : 'text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200 hover:bg-white/40 dark:hover:bg-slate-700/40'}`}
                >
                  <TrendingUp className="w-5 h-5" />
                  Tổng Quan
                </button>
                <button
                  onClick={() => setMainTab('details')}
                  className={`cursor-pointer flex items-center gap-2 px-6 py-3 rounded-2xl font-bold transition-all duration-300 font-quicksand ${mainTab === 'details' ? 'bg-white dark:bg-slate-700 shadow-[0_2px_10px_rgba(0,0,0,0.05)] text-blue-600 dark:text-blue-400 scale-105' : 'text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200 hover:bg-white/40 dark:hover:bg-slate-700/40'}`}
                >
                  <List className="w-5 h-5" />
                  Chi Tiết
                </button>
                <button
                  onClick={() => setMainTab('recommendations')}
                  className={`cursor-pointer flex items-center gap-2 px-6 py-3 rounded-2xl font-bold transition-all duration-300 font-quicksand ${mainTab === 'recommendations' ? 'bg-white dark:bg-slate-700 shadow-[0_2px_10px_rgba(0,0,0,0.05)] text-blue-600 dark:text-blue-400 scale-105' : 'text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200 hover:bg-white/40 dark:hover:bg-slate-700/40'}`}
                >
                  <ShoppingCart className="w-5 h-5" />
                  Đề Xuất
                </button>
              </div>

              <div className="flex-1 flex justify-center md:justify-end w-full md:w-auto">
                <button
                  onClick={handleExportPDF}
                  disabled={isExporting}
                  className="font-quicksand cursor-pointer flex items-center gap-2 bg-white/40 dark:bg-slate-800/40 backdrop-blur-xl border border-blue-500 dark:border-white/50 hover:bg-white/60 dark:hover:bg-slate-800/60 text-blue-700 dark:text-blue-400 px-5 py-2.5 rounded-md shadow-md font-bold text-sm hover:scale-[1.02] transition-all disabled:opacity-50 disabled:cursor-not-allowed group relative overflow-hidden ring-1 ring-white/20 whitespace-nowrap"
                >
                  <div className="absolute inset-0 w-full h-full bg-gradient-to-r from-transparent via-white/40 dark:via-white/5 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000 ease-in-out"></div>
                  {isExporting ? <Loader2 className="w-5 h-5 animate-spin" /> : <Download className="w-5 h-5 text-blue-500" />}
                  {isExporting ? 'Đang xuất PDF...' : 'Xuất Báo Cáo'}
                </button>
              </div>
            </div>

            {mainTab === 'overview' && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} className="space-y-8">
                {/* Smart Alerts Row */}
                {(crossModalAlerts > 0 || metadata.smartAdvice) && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {crossModalAlerts > 0 && (
                      <div className="bg-rose-50 dark:bg-rose-500/10 border border-rose-200 dark:border-rose-500/30 rounded-2xl p-5 flex items-center gap-4 text-rose-700 dark:text-rose-400 shadow-sm animate-pulse">
                        <AlertOctagon className="w-8 h-8 flex-shrink-0" />
                        <p className="text-sm font-quicksand m-0 leading-relaxed">
                          Phát hiện <strong>{crossModalAlerts}</strong> đánh giá có nội dung hoặc rating tích cực nhưng hình ảnh cho thấy sản phẩm bị hư hỏng/móp méo.
                        </p>
                      </div>
                    )}

                    {metadata.smartAdvice && (
                      <div className="bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-500/10 dark:to-orange-500/10 border border-amber-200 dark:border-amber-500/30 rounded-2xl p-5 flex items-center gap-4 text-amber-800 dark:text-amber-300 shadow-sm">
                        <Sparkles className="w-8 h-8 flex-shrink-0" />
                        <p className="text-sm font-quicksand leading-relaxed m-0">
                          {metadata.smartAdvice.replace('💡 Gợi ý mua hàng: ', '')}
                        </p>
                      </div>
                    )}
                  </div>
                )}

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                  {/* Product Info */}
                  <div className="bg-white dark:bg-slate-900/40 dark:backdrop-blur-xl rounded-3xl shadow-lg border border-slate-100 dark:border-white/10 p-6 flex flex-col justify-center items-center text-center lg:col-span-1 relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-32 h-32 bg-cyan-100 dark:bg-cyan-500/10 rounded-full blur-3xl"></div>
                    <img src={result.productData?.thumbnail} alt="Product" className="w-32 h-32 object-cover rounded-2xl shadow-md border border-slate-100 dark:border-white/10 mb-4 z-10 bg-white" />
                    <h3 className="font-bold text-xl line-clamp-2 z-10 text-slate-800 dark:text-slate-100">{result.productData?.name}</h3>
                    <div className="mt-4 flex gap-4 w-full justify-center z-10">
                      {/* UC2.2 Trust Score & Spam */}
                      <div className="bg-slate-50 dark:bg-slate-800/50 rounded-xl p-3 flex-1 border border-slate-100 dark:border-slate-700 font-quicksand">
                        <div className="text-xs text-slate-500 dark:text-slate-400 mb-1 flex justify-center items-center gap-1"><ShieldCheck className="w-3 h-3" /> Trust Score</div>
                        <div className={`font-bold text-xl ${metadata.trustScore >= 70 ? 'text-emerald-500' : metadata.trustScore >= 50 ? 'text-amber-500' : 'text-rose-500'}`}>{metadata.trustScore || 0}/100</div>
                      </div>
                      <div className="bg-slate-50 dark:bg-slate-800/50 rounded-xl p-3 flex-1 border border-slate-100 dark:border-slate-700 font-quicksand">
                        <div className="text-xs text-slate-500 dark:text-slate-400 mb-1 flex justify-center items-center gap-1"><ShieldAlert className="w-3 h-3" /> Tỷ lệ Spam</div>
                        <div className={`font-bold text-xl ${metadata.spamPercentage <= 20 ? 'text-emerald-500' : metadata.spamPercentage <= 40 ? 'text-amber-500' : 'text-rose-500'}`}>{metadata.spamPercentage || 0}%</div>
                      </div>
                    </div>
                  </div>

                  {/* LLM Summary */}
                  <div className="bg-gradient-to-br from-violet-50 to-purple-50 dark:from-indigo-950/40 dark:to-purple-950/40 backdrop-blur-xl rounded-3xl lg:col-span-2 shadow-xl border border-violet-100 dark:border-purple-800/50 p-8 relative flex flex-col justify-center">
                    <div className="absolute -bottom-10 -right-10 w-40 h-40 bg-purple-200 dark:bg-purple-600/20 rounded-full blur-3xl"></div>
                    <Bot className="absolute top-6 right-6 w-24 h-24 text-purple-200 dark:text-purple-800/30 -rotate-12" />
                    <h3 className="flex items-center gap-3 text-xl font-bold mb-6 text-purple-700 dark:text-purple-300 z-10 relative font-quicksand">
                      <Sparkles className="w-6 h-6 text-purple-500 dark:text-purple-400 self-start" /> Tổng Quan Cảm Xúc (AI Summary)
                    </h3>
                    <div className="relative z-10">
                      <MessageSquare className="w-8 h-8 text-purple-300 dark:text-purple-600 absolute -top-2 -left-3 opacity-50" />
                      <p className="text-slate-700 dark:text-slate-200 leading-relaxed text-lg italic pl-6 font-quicksand">
                        "{result.summary}"
                      </p>
                    </div>
                  </div>
                </div>

                {/* CHARTS ROW 1 */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  {/* Sentiment Pie */}
                  <div className="bg-white dark:bg-slate-900/40 dark:backdrop-blur-xl rounded-3xl shadow-lg border border-slate-100 dark:border-white/10 p-8">
                    <h3 className="text-xl font-bold mb-6 flex items-center gap-3 text-slate-800 dark:text-slate-100 font-quicksand">
                      <TrendingUp className="w-6 h-6 text-blue-500 dark:text-indigo-400" /> Cảm Xúc Chung (PhoBERT)
                    </h3>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie data={chartData.sentimentData} innerRadius={60} outerRadius={90} paddingAngle={8} dataKey="value" stroke="none">
                            {chartData.sentimentData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.color} />
                            ))}
                          </Pie>
                          <Tooltip contentStyle={{ backgroundColor: isDark ? '#0f172a' : '#ffffff', border: isDark ? '1px solid #1e293b' : '1px solid #e2e8f0', borderRadius: '12px' }} itemStyle={{ color: isDark ? '#e2e8f0' : '#0f172a' }} />
                          <Legend verticalAlign="bottom" height={36} iconType="circle" />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* ABSA Radar Chart UC4.1 */}
                  <div className="bg-white dark:bg-slate-900/40 dark:backdrop-blur-xl rounded-3xl shadow-lg border border-slate-100 dark:border-white/10 p-8">
                    <h3 className="text-xl font-bold mb-6 flex items-center gap-3 text-slate-800 dark:text-slate-100 font-quicksand">
                      <Bot className="w-6 h-6 text-fuchsia-500 dark:text-fuchsia-400" /> Phân Tích Khía Cạnh (ABSA)
                    </h3>
                    <div className="h-64">
                      {aspectData.length > 0 ? (
                        <ResponsiveContainer width="100%" height="100%">
                          <RadarChart cx="50%" cy="50%" outerRadius="70%" data={aspectData}>
                            <PolarGrid stroke={isDark ? "#334155" : "#e2e8f0"} />
                            <PolarAngleAxis dataKey="subject" tick={{ fill: isDark ? '#94a3b8' : '#64748b', fontSize: 13 }} />
                            <PolarRadiusAxis angle={90} domain={[0, 5]} axisLine={false} tick={false} />
                            <Radar name="Điểm Đánh Giá" dataKey="A" stroke="#d946ef" fill="#d946ef" fillOpacity={0.5} />
                            <Tooltip contentStyle={{ backgroundColor: isDark ? '#0f172a' : '#ffffff', border: isDark ? '1px solid #1e293b' : '1px solid #e2e8f0', borderRadius: '12px' }} itemStyle={{ color: isDark ? '#e2e8f0' : '#0f172a' }} />
                          </RadarChart>
                        </ResponsiveContainer>
                      ) : (
                        <div className="flex items-center justify-center h-full text-slate-500">Chưa có dữ liệu khía cạnh</div>
                      )}
                    </div>
                  </div>
                </div>

                {/* CHARTS ROW 2 */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  {/* Image Labels Bar Chart */}
                  <div className="bg-white dark:bg-slate-900/40 dark:backdrop-blur-xl rounded-3xl shadow-lg border border-slate-100 dark:border-white/10 p-8">
                    <h3 className="text-xl font-bold mb-6 flex items-center gap-3 text-slate-800 dark:text-slate-100 font-quicksand">
                      <AlertTriangle className="w-6 h-6 text-orange-500 dark:text-rose-400" /> Tình Trạng Hình Ảnh (ResNet)
                    </h3>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={chartData.labelData}>
                          <CartesianGrid strokeDasharray="3 3" stroke={isDark ? "#334155" : "#e2e8f0"} vertical={false} />
                          <XAxis dataKey="name" stroke="#94a3b8" fontSize={13} tickLine={false} axisLine={false} />
                          <YAxis stroke="#94a3b8" fontSize={13} tickLine={false} axisLine={false} />
                          <Tooltip cursor={{ fill: isDark ? '#1e293b' : '#f8fafc' }} contentStyle={{ backgroundColor: isDark ? '#0f172a' : '#ffffff', border: isDark ? '1px solid #1e293b' : '1px solid #e2e8f0', borderRadius: '12px' }} itemStyle={{ color: isDark ? '#e2e8f0' : '#0f172a' }} />
                          <Bar dataKey="value" fill={isDark ? "#6366f1" : "#3b82f6"} radius={[6, 6, 0, 0]}>
                            {chartData.labelData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={['#10b981', '#ef4444', '#f59e0b', '#64748b'][index % 4]} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Time Series UC4.2 */}
                  <div className="bg-white dark:bg-slate-900/40 dark:backdrop-blur-xl rounded-3xl shadow-lg border border-slate-100 dark:border-white/10 p-8">
                    <h3 className="text-xl font-bold mb-6 flex items-center gap-3 text-slate-800 dark:text-slate-100 font-quicksand">
                      <TrendingUp className="w-6 h-6 text-cyan-500 dark:text-cyan-400" /> Biến Động Cảm Xúc
                    </h3>
                    <div className="h-64">
                      {metadata.sentimentTimeSeries && metadata.sentimentTimeSeries.length > 0 ? (
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={metadata.sentimentTimeSeries}>
                            <CartesianGrid strokeDasharray="3 3" stroke={isDark ? "#334155" : "#e2e8f0"} vertical={false} />
                            <XAxis dataKey="date" stroke="#94a3b8" fontSize={11} tickLine={false} axisLine={false} />
                            <YAxis stroke="#94a3b8" fontSize={11} tickLine={false} axisLine={false} />
                            <Tooltip contentStyle={{ backgroundColor: isDark ? '#0f172a' : '#ffffff', border: isDark ? '1px solid #1e293b' : '1px solid #e2e8f0', borderRadius: '12px' }} />
                            <Legend verticalAlign="top" height={36} iconType="circle" wrapperStyle={{ fontSize: '12px' }} />
                            <Line type="monotone" dataKey="positive" name="Tích cực" stroke="#10b981" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />
                            <Line type="monotone" dataKey="negative" name="Tiêu cực" stroke="#ef4444" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />
                          </LineChart>
                        </ResponsiveContainer>
                      ) : (
                        <div className="flex items-center justify-center h-full text-slate-500">Chưa có dữ liệu chuỗi thời gian</div>
                      )}
                    </div>
                  </div>
                </div>

                {/* UC4.3 Word Cloud / Keywords */}
                {metadata.keywords && (
                  <div className="bg-white dark:bg-slate-900 rounded-3xl shadow-sm border border-slate-100 dark:border-slate-800 pt-6 overflow-hidden">
                    <div className="px-6 border-b border-slate-100 dark:border-slate-800 pb-4">
                      <h3 className="text-xl font-bold flex items-center gap-3 text-slate-800 dark:text-slate-100 font-quicksand">
                        <MessageSquare className="w-5 h-5 text-blue-500" /> Từ Khóa Nổi Bật
                      </h3>
                    </div>
                    <div className="flex flex-col md:flex-row overflow-hidden border border-slate-100 dark:border-slate-800 shadow-sm">
                      {/* Left: Tích cực */}
                      <div className="flex-1 bg-gradient-to-br from-slate-50 to-blue-50/80 dark:from-slate-800/50 dark:to-blue-900/20">
                        <div className="flex justify-center items-center pt-6 pb-2">
                          <div className="inline-flex items-center gap-2 px-6 py-2.5 bg-white/60 dark:bg-slate-800/60 backdrop-blur-md border border-blue-200 dark:border-blue-500/30 shadow-[0_4px_12px_rgba(59,130,246,0.15)] rounded-full hover:scale-105 transition-transform cursor-default">
                            <Smile className="w-5 h-5 text-blue-500" />
                            <span className="font-bold text-blue-700 dark:text-blue-400 text-lg font-quicksand tracking-wide">Tích cực</span>
                          </div>
                        </div>
                        <div className="px-2 pb-3">
                          <WordCloudSVG words={metadata.keywords.positive} colors={BLUE_SHADES} />
                        </div>
                      </div>

                      {/* Right: Tiêu cực */}
                      <div className="flex-1 bg-gradient-to-br from-slate-50 to-rose-50/80 dark:from-slate-800/50 dark:to-rose-900/20 border-t md:border-t-0 md:border-l border-white dark:border-slate-800">
                        <div className="flex justify-center items-center pt-6 pb-2">
                          <div className="inline-flex items-center gap-2 px-6 py-2.5 bg-white/60 dark:bg-slate-800/60 backdrop-blur-md border border-rose-200 dark:border-rose-500/30 shadow-[0_4px_12px_rgba(244,63,94,0.15)] rounded-full hover:scale-105 transition-transform cursor-default">
                            <Frown className="w-5 h-5 text-rose-500" />
                            <span className="font-bold text-rose-700 dark:text-rose-400 text-lg font-quicksand tracking-wide">Tiêu cực</span>
                          </div>
                        </div>
                        <div className="px-2 pb-3">
                          <WordCloudSVG words={metadata.keywords.negative} colors={RED_SHADES} />
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </motion.div>
            )}

            {mainTab === 'details' && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} className="space-y-8">
                {/* UC2.3 & UC2.4 Dynamic Tabs */}
                <div className="bg-white dark:bg-slate-900/40 dark:backdrop-blur-xl rounded-3xl shadow-lg border border-slate-100 dark:border-white/10 overflow-hidden">
                  <div className="font-quicksand flex border-b border-slate-200 dark:border-slate-700 bg-slate-50/50 dark:bg-slate-800/30">
                    <button
                      onClick={() => setActiveTab('table')}
                      className={`flex-1 flex justify-center items-center gap-2 py-4 font-semibold text-sm transition-colors ${activeTab === 'table' ? 'text-blue-500 border-b-[3px] border-blue-500 dark:text-blue-400 dark:border-blue-400' : 'text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200 border-b-[3px] border-transparent'}`}
                    >
                      <List className="w-4 h-4" /> Chi Tiết Đánh Giá
                    </button>
                    <button
                      onClick={() => setActiveTab('images')}
                      className={`flex-1 flex justify-center items-center gap-2 py-4 font-semibold text-sm transition-colors ${activeTab === 'images' ? 'text-blue-500 border-b-[3px] border-blue-500 dark:text-blue-400 dark:border-blue-400' : 'text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200 border-b-[3px] border-transparent'}`}
                    >
                      <ImageIcon className="w-4 h-4" /> Thư Viện Ảnh ({result.reviews?.filter((r: any) => r.image_path).length || 0})
                    </button>
                  </div>

                  {activeTab === 'table' && (
                    <div className="p-6">
                      {/* Filters */}
                      <div className="flex flex-wrap gap-4 mb-6 bg-slate-50 dark:bg-slate-800/50 p-4 rounded-xl border border-slate-100 dark:border-slate-700">
                        <div className="flex items-center gap-2 text-slate-600 dark:text-slate-300 font-medium">
                          <Filter className="w-4 h-4" /> Lọc:
                        </div>
                        <select className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-1.5 text-sm dark:text-white outline-none focus:border-blue-500" value={filterSentiment} onChange={e => setFilterSentiment(e.target.value)}>
                          <option value="all">Mọi trạng thái</option>
                          <option value="positive">Tích cực</option>
                          <option value="neutral">Trung lập</option>
                          <option value="negative">Tiêu cực</option>
                        </select>
                        <select className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-1.5 text-sm dark:text-white outline-none focus:border-blue-500" value={filterLabel} onChange={e => setFilterLabel(e.target.value)}>
                          <option value="all">Mọi tình trạng ảnh</option>
                          <option value="intact">Nguyên vẹn</option>
                          <option value="damaged">Móp méo</option>
                          <option value="wrong_item">Sai hàng</option>
                          <option value="irrelevant">Không liên quan</option>
                        </select>
                        <select className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-1.5 text-sm dark:text-white outline-none focus:border-blue-500" value={filterRating} onChange={e => setFilterRating(e.target.value)}>
                          <option value="all">Mọi rating</option>
                          <option value="5">5 ⭐</option>
                          <option value="4">4 ⭐</option>
                          <option value="3">3 ⭐</option>
                          <option value="2">2 ⭐</option>
                          <option value="1">1 ⭐</option>
                        </select>
                      </div>

                      <div className="overflow-x-auto">
                        {filteredReviews.length > 20 && <div className="text-center text-md text-black dark:text-white mb-2 font-quicksand font-semibold">Hiển thị 20 đánh giá tiêu biểu</div>}

                        <table className="w-full text-left text-sm text-slate-600 dark:text-slate-300">
                          <thead className="text-xs uppercase bg-slate-50 dark:bg-slate-800/50 text-slate-500 dark:text-slate-400">
                            <tr>
                              <th className="px-4 py-3 rounded-tl-lg">Đánh giá</th>
                              <th className="px-4 py-3">Ảnh</th>
                              <th className="px-4 py-3">Nhãn Text</th>
                              <th className="px-4 py-3 rounded-tr-lg">Nhãn Ảnh</th>
                            </tr>
                          </thead>
                          <tbody>
                            {filteredReviews.length === 0 ? (
                              <tr><td colSpan={4} className="text-center py-8">Không có đánh giá nào phù hợp với bộ lọc.</td></tr>
                            ) : (
                              filteredReviews.slice(0, 20).map((r: any, idx: number) => (
                                <tr key={idx} className="font-quicksand border-b border-slate-100 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors">
                                  <td className="px-4 py-4 max-w-xs">
                                    <div className="text-amber-400 text-xs mb-1">{"★".repeat(r.rating)}{"☆".repeat(5 - r.rating)}</div>
                                    <p className="truncate" title={r.review_text}>{r.review_text}</p>
                                  </td>
                                  <td className="px-4 py-4">
                                    {r.image_path ? (
                                      <img src={r.image_path} alt="review" className="w-10 h-10 object-cover rounded border border-slate-200 dark:border-slate-700 cursor-pointer hover:opacity-80" onClick={() => setSelectedImage(r)} />
                                    ) : <span className="text-slate-400 italic text-xs">Không có</span>}
                                  </td>
                                  <td className="px-4 py-4">
                                    <span className={`px-2 py-1 rounded font-semibold ${r.sentiment === 'positive' ? 'text-emerald-700 dark:text-emerald-400' : r.sentiment === 'negative' ? 'text-rose-700 dark:text-rose-400' : 'text-purple-700 dark:text-purple-400'}`}>
                                      {r.sentiment === 'positive' ? 'Tích cực' : r.sentiment === 'negative' ? 'Tiêu cực' : 'Trung lập'}
                                    </span>
                                  </td>
                                  <td className="px-4 py-4">
                                    <span className={`px-2 py-1 rounded font-semibold ${r.label === 'intact' ? 'text-emerald-700 dark:text-emerald-400' : r.label === 'damaged' ? 'text-rose-700 dark:text-rose-400' : r.label === 'wrong_item' ? 'text-amber-700 dark:text-amber-400' : 'text-slate-700 dark:text-slate-300'}`}>
                                      {r.label === 'intact' ? 'Nguyên vẹn' : r.label === 'damaged' ? 'Móp méo' : r.label === 'wrong_item' ? 'Sai hàng' : 'Không liên quan'}
                                    </span>
                                  </td>
                                </tr>
                              ))
                            )}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {activeTab === 'images' && (
                    <div className="p-6 grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-4">
                      {result.reviews?.filter((r: any) => r.image_path).map((r: any, idx: number) => (
                        <div key={idx} className="relative aspect-square rounded-xl overflow-hidden cursor-pointer group border-2 border-transparent hover:border-blue-500 transition-all" onClick={() => setSelectedImage(r)}>
                          <img src={r.image_path} alt="review" className="w-full h-full object-cover" />
                          <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-80 transition-opacity flex items-center justify-center">
                            <span className={`px-2 py-1 rounded text-xs font-bold ${r.label === 'intact' ? 'bg-emerald-500 text-white' : r.label === 'damaged' ? 'bg-rose-500 text-white' : r.label === 'wrong_item' ? 'bg-amber-500 text-white' : 'bg-slate-500 text-white'}`}>
                              {r.label === 'intact' ? 'OK' : r.label === 'damaged' ? 'Hỏng' : r.label === 'wrong_item' ? 'Sai' : 'Không liên quan'}
                            </span>
                          </div>
                        </div>
                      ))}
                      {result.reviews?.filter((r: any) => r.image_path).length === 0 && (
                        <div className="col-span-full text-center py-10 text-slate-500">Không có hình ảnh nào trong dữ liệu.</div>
                      )}
                    </div>
                  )}
                </div>
              </motion.div>
            )}

            {mainTab === 'recommendations' && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} className="space-y-8">
                {/* UC8 Alternative Products */}
                {(metadata.trustScore < 50 || metadata.spamPercentage > 40) && metadata.alternativeProducts ? (
                  <div className="bg-white dark:bg-slate-900/40 dark:backdrop-blur-xl rounded-3xl shadow-lg border border-rose-200 dark:border-rose-900/50 p-8 overflow-hidden relative">
                    <h3 className="text-xl font-bold mb-2 flex items-center gap-3 text-slate-800 dark:text-slate-100 font-quicksand">
                      <ShieldAlert className="w-6 h-6 text-rose-500" /> Sản phẩm thay thế
                    </h3>
                    <p className="text-sm text-slate-500 dark:text-slate-400 mb-6">Sản phẩm hiện tại có rủi ro cao. Hệ thống đề xuất các sản phẩm tương tự có độ tin cậy tốt hơn:</p>

                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                      {metadata.alternativeProducts.map((alt: any, idx: number) => (
                        <div key={idx} className="bg-slate-50 dark:bg-slate-800/50 border border-slate-100 dark:border-slate-700 rounded-2xl p-4 hover:shadow-md transition-shadow cursor-default group flex flex-col">
                          <img src={alt.thumbnail} alt={alt.name} className="w-full aspect-square object-cover rounded-xl mb-3 group-hover:scale-[1.03] transition-transform" />
                          <h4 className="font-semibold text-sm line-clamp-2 mb-1 dark:text-slate-200 flex-1">{alt.name}</h4>
                          <div className="flex items-center gap-1 text-xs text-emerald-600 dark:text-emerald-400 font-medium bg-emerald-50 dark:bg-emerald-500/10 w-fit px-2 py-1 rounded mt-auto">
                            <ShieldCheck className="w-3 h-3" /> Trust: {alt.trustScore}
                          </div>
                          <div className="grid grid-cols-2 gap-2 mt-3">
                            <button onClick={(e) => { e.stopPropagation(); window.open(alt.url, '_blank'); }} className="cursor-pointer flex items-center justify-center text-xs font-semibold bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-200 py-2 rounded-xl transition-colors w-full">Truy cập</button>
                            <button onClick={(e) => { e.stopPropagation(); router.push('/analyze?url=' + encodeURIComponent(alt.url)); }} className="cursor-pointer flex items-center justify-center text-xs font-semibold bg-blue-100 dark:bg-blue-500/20 hover:bg-blue-200 dark:hover:bg-blue-500/30 text-blue-700 dark:text-blue-400 py-2 rounded-xl transition-colors w-full">Phân tích</button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="bg-white dark:bg-slate-900/40 dark:backdrop-blur-xl rounded-3xl shadow-lg border border-slate-100 dark:border-white/10 p-16 text-center text-slate-500 dark:text-slate-400 font-quicksand flex flex-col items-center justify-center">
                    <CheckCircle className="w-16 h-16 text-emerald-500 mb-4 opacity-50" />
                    <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-2">Sản phẩm Uy tín</h3>
                    <p>Sản phẩm này khá an toàn nên hệ thống không đề xuất thay thế.</p>
                  </div>
                )}
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Image Modal */}
      <AnimatePresence>
        {selectedImage && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/80 backdrop-blur-sm"
            onClick={() => setSelectedImage(null)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.95, opacity: 0 }}
              className="bg-white dark:bg-slate-900 rounded-3xl max-w-4xl w-full overflow-hidden shadow-2xl flex flex-col md:flex-row"
              onClick={e => e.stopPropagation()}
            >
              <div className="md:w-2/3 bg-black relative flex items-center justify-center min-h-[300px]">
                <img src={selectedImage.image_path} alt="Review full" className="max-w-full max-h-[70vh] object-contain" />
                {/* Mock bounding box if damaged */}
                {selectedImage.label === 'damaged' && (
                  <div className="absolute border-2 border-rose-500 bg-rose-500/20" style={{ top: '30%', left: '30%', width: '40%', height: '40%' }}>
                    <div className="absolute -top-6 left-0 bg-rose-500 text-white text-xs px-2 py-1 font-bold">Damaged 98%</div>
                  </div>
                )}
              </div>
              <div className="md:w-1/3 p-6 flex flex-col font-quicksand">
                <div className="flex justify-between items-start mb-4">
                  <h3 className="text-lg font-bold dark:text-white">Chi tiết hình ảnh</h3>
                  <button onClick={() => setSelectedImage(null)} className="p-1 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-500"><X className="w-5 h-5" /></button>
                </div>

                <div className="space-y-4 flex-1">
                  <div>
                    <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Nhãn hình ảnh</div>
                    <span className={`px-3 py-1.5 rounded-lg text-sm font-bold inline-flex items-center gap-2 ${selectedImage.label === 'intact' ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-400' : selectedImage.label === 'damaged' ? 'bg-rose-100 text-rose-700 dark:bg-rose-500/20 dark:text-rose-400' : selectedImage.label === 'wrong_item' ? 'bg-amber-100 text-amber-700 dark:bg-amber-500/20 dark:text-amber-400' : 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300'}`}>
                      {selectedImage.label === 'intact' ? 'Nguyên vẹn' : selectedImage.label === 'damaged' ? 'Móp méo / Hư hỏng' : selectedImage.label === 'wrong_item' ? 'Sai sản phẩm' : 'Không liên quan'}
                    </span>
                  </div>

                  <div>
                    <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Cảm xúc đánh giá</div>
                    <span className={`px-3 py-1.5 rounded-lg text-sm font-bold inline-flex items-center gap-2 ${selectedImage.sentiment === 'positive' ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-400' : selectedImage.sentiment === 'negative' ? 'bg-rose-100 text-rose-700 dark:bg-rose-500/20 dark:text-rose-400' : 'bg-purple-100 text-purple-700 dark:bg-purple-500/20 dark:text-purple-400'}`}>
                      {selectedImage.sentiment === 'positive' ? 'Tích cực' : selectedImage.sentiment === 'negative' ? 'Tiêu cực' : 'Trung lập'}
                    </span>
                  </div>

                  <div className="pt-6 border-t border-slate-100 dark:border-slate-800 mt-4">
                    <div className="text-xs text-slate-500 dark:text-slate-400 mb-2">Đánh giá gốc</div>
                    <div className="text-amber-400 text-sm mb-2">{"★".repeat(selectedImage.rating)}{"☆".repeat(5 - selectedImage.rating)}</div>
                    <p className="text-sm dark:text-slate-300 italic">"{selectedImage.review_text}"</p>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function AnalyzePage() {
  return (
    <Suspense fallback={<div className="text-center mt-20"><Loader2 className="w-10 h-10 animate-spin mx-auto text-blue-500" /></div>}>
      <AnalyzeContent />
    </Suspense>
  );
}
