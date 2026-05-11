'use client';

import { useState, useMemo, useEffect, Suspense } from 'react';
import { io, Socket } from 'socket.io-client';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { Loader2, CheckCircle, AlertTriangle, MessageSquare, TrendingUp, Bot, Sparkles } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid, Legend } from 'recharts';
import { useRouter, useSearchParams } from 'next/navigation';
import { supabase } from '@/lib/supabase';

type Status = 'PROCESSING' | 'COMPLETED' | 'ERROR';

function AnalyzeContent() {
  const [status, setStatus] = useState<Status>('PROCESSING');
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('');
  const [result, setResult] = useState<any>(null);

  const router = useRouter();
  const searchParams = useSearchParams();

  useEffect(() => {
    const url = searchParams.get('url');
    const historyId = searchParams.get('historyId');

    if (historyId) {
      loadHistoryDetail(historyId);
    } else if (url) {
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
        summary: product.report?.summary_text || 'Không có tóm tắt.'
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

  const isDark = typeof document !== 'undefined' ? document.documentElement.classList.contains('dark') : false;

  return (
    <div className="w-full relative z-10 px-6">
      <AnimatePresence mode="wait">
        {status === 'PROCESSING' && (
          <motion.div key="processing" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="w-full max-w-lg mx-auto bg-white dark:bg-slate-900/40 dark:backdrop-blur-xl rounded-3xl shadow-xl border border-slate-100 dark:border-white/10 p-10 mt-20 text-center relative overflow-hidden mt-10">
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
            <div className="w-full flex items-center justify-center gap-3">
              <CheckCircle className="w-10 h-10 text-emerald-600 dark:text-emerald-400" />
              <div className="text-2xl font-quicksand font-semibold">Đã quét <span className="font-momo text-blue-500 text-4xl">{result.reviews?.length || 0}</span> reviews</div>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              <div className="bg-white dark:bg-slate-900/40 dark:backdrop-blur-xl rounded-3xl shadow-lg border border-slate-100 dark:border-white/10 p-6 flex flex-col justify-center items-center text-center lg:col-span-1 relative overflow-hidden">
                <div className="absolute top-0 right-0 w-32 h-32 bg-cyan-100 dark:bg-cyan-500/10 rounded-full blur-3xl"></div>
                <img src={result.productData?.thumbnail} alt="Product" className="w-32 h-32 object-cover rounded-2xl shadow-md border border-slate-100 dark:border-white/10 mb-4 z-10 bg-white" />
                <h3 className="font-bold text-xl line-clamp-2 z-10 text-slate-800 dark:text-slate-100">{result.productData?.name}</h3>
              </div>

              {/* <div className="bg-white dark:bg-slate-900/40 dark:backdrop-blur-xl rounded-3xl shadow-lg border border-slate-100 dark:border-white/10 p-8 lg:col-span-2 relative overflow-hidden flex flex-col justify-center">
                <div className="absolute bottom-0 left-0 w-48 h-48 bg-purple-100 dark:bg-purple-500/10 rounded-full blur-3xl"></div>
                <h3 className="flex items-center gap-3 text-xl font-bold mb-4 text-purple-600 dark:text-purple-400 z-10">
                  <MessageSquare className="w-6 h-6" /> Nhận định từ LLM
                </h3>
                <p className="text-slate-600 dark:text-slate-300 leading-relaxed text-xl italic z-10 border-l-4 border-purple-300 dark:border-purple-500/50 pl-4 font-quicksand">
                  "{result.summary}"
                </p>
              </div> */}

              {/* LLM Summary */}
              <div className="bg-gradient-to-br from-violet-50 to-purple-50 dark:from-indigo-950/40 dark:to-purple-950/40 backdrop-blur-xl rounded-3xl lg:col-span-2 shadow-xl border border-violet-100 dark:border-purple-800/50 p-8 relative flex flex-col justify-center">
                <div className="absolute -bottom-10 -right-10 w-40 h-40 bg-purple-200 dark:bg-purple-600/20 rounded-full blur-3xl"></div>
                <Bot className="absolute top-6 right-6 w-24 h-24 text-purple-200 dark:text-purple-800/30 -rotate-12" />
                <h3 className="flex items-center gap-3 text-xl font-bold mb-6 text-purple-700 dark:text-purple-300 z-10 relative font-quicksand">
                  <Sparkles className="w-6 h-6 text-purple-500 dark:text-purple-400 self-start" /> Nhận Định AI
                </h3>
                <div className="relative z-10">
                  <MessageSquare className="w-8 h-8 text-purple-300 dark:text-purple-600 absolute -top-2 -left-3 opacity-50" />
                  <p className="text-slate-700 dark:text-slate-200 leading-relaxed text-lg italic pl-6 font-quicksand">
                    "{result.summary}"
                  </p>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="bg-white dark:bg-slate-900/40 dark:backdrop-blur-xl rounded-3xl shadow-lg border border-slate-100 dark:border-white/10 p-8">
                <h3 className="text-xl font-bold mb-8 flex items-center gap-3 text-slate-800 dark:text-slate-100 font-quicksand">
                  <TrendingUp className="w-6 h-6 text-blue-500 dark:text-indigo-400" /> Cảm Xúc Khách Hàng (PhoBERT)
                </h3>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={chartData.sentimentData} innerRadius={70} outerRadius={100} paddingAngle={8} dataKey="value" stroke="none">
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

              <div className="bg-white dark:bg-slate-900/40 dark:backdrop-blur-xl rounded-3xl shadow-lg border border-slate-100 dark:border-white/10 p-8">
                <h3 className="text-xl font-bold mb-8 flex items-center gap-3 text-slate-800 dark:text-slate-100 font-quicksand">
                  <AlertTriangle className="w-6 h-6 text-orange-500 dark:text-rose-400" /> Tình Trạng Đóng Gói (ResNet & CLIP)
                </h3>
                <div className="h-72">
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
            </div>

            <div className="text-center pt-6 pb-10">
              <button onClick={() => router.push('/')} className="text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-white transition-colors decoration-slate-300 dark:decoration-slate-600 underline-offset-8 font-medium font-quicksand">
                ← Phân tích sản phẩm khác
              </button>
            </div>
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
