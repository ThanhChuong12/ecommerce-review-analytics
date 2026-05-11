'use client';

import { useState, useEffect } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import { History, Trash2, ArrowLeft, Loader2 } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { supabase } from '@/lib/supabase';

export default function HistoryPage() {
  const router = useRouter();
  const [historyList, setHistoryList] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) {
        router.push('/');
        return;
      }

      const res = await axios.get(`${process.env.NEXT_PUBLIC_API_URL}/history`, {
        headers: { Authorization: `Bearer ${session.access_token}` }
      });
      setHistoryList(res.data.data);
    } catch (error) {
      alert('Lỗi tải lịch sử');
    } finally {
      setLoading(false);
    }
  };

  const deleteHistory = async (productId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm('Chắc chắn xoá báo cáo phân tích này?')) return;
    try {
      const { data: { session } } = await supabase.auth.getSession();
      await axios.delete(`${process.env.NEXT_PUBLIC_API_URL}/history/${productId}`, {
        headers: { Authorization: `Bearer ${session?.access_token}` }
      });
      setHistoryList(prev => prev.filter(p => p.id !== productId));
    } catch (error) {
      alert('Lỗi xoá lịch sử (Có thể Token đã hết hạn)');
    }
  };

  const viewDetail = (productId: string) => {
    // Navigate to the shared analyze page
    router.push(`/analyze?historyId=${productId}`);
  };

  return (
    <div className="w-full max-w-5xl mx-auto">
      <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}>
        {/* <div className="w-full flex justify-center py-4 mb-4">
          <div className="w-1/3 border-b"></div>
        </div> */}
        <div className="flex items-center justify-center mb-8 mt-4">
          <h2 className="text-3xl font-bold flex items-center gap-3 text-slate-800 dark:text-slate-100 font-quicksand">
            <History className="text-blue-500 dark:text-indigo-400 w-8 h-8" /> Lịch sử
          </h2>
        </div>

        {loading ? (
          <div className="flex justify-center items-center py-20">
            <Loader2 className="w-10 h-10 animate-spin text-indigo-500 dark:text-indigo-400" />
          </div>
        ) : historyList.length === 0 ? (
          <div className="bg-white dark:bg-slate-900/40 dark:backdrop-blur-xl rounded-3xl shadow-sm border border-slate-200 dark:border-white/10 p-16 text-center text-slate-500 dark:text-slate-400 border-dashed dark:border-slate-700/50 font-quicksand">
            Chưa có báo cáo nào. Hãy phân tích thử 1 sản phẩm nhé!
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {historyList.map(item => (
              <div key={item.id} className="bg-white dark:bg-slate-900/40 dark:backdrop-blur-xl rounded-3xl shadow-sm border border-slate-100 dark:border-white/10 p-5 flex gap-5 relative group cursor-pointer hover:shadow-md hover:border-blue-200 dark:hover:border-indigo-500/50 transition-all duration-300" onClick={() => viewDetail(item.id)}>
                <img src={item.thumbnail || '/placeholder.png'} className="w-24 h-24 object-cover rounded-2xl shadow-sm border border-slate-100 dark:border-slate-700 bg-white dark:bg-slate-800" />
                <div className="flex-1 pr-8">
                  <h3 className="font-bold text-lg line-clamp-2 leading-snug text-slate-800 dark:text-slate-200 font-quicksand">{item.name || item.url}</h3>
                  <p className="text-sm text-slate-500 dark:text-slate-400 mt-2 font-quicksand">Trạng thái: <span className={`font-semibold ${item.status === 'COMPLETED' ? 'text-emerald-500 dark:text-emerald-400' : 'text-yellow-500 dark:text-yellow-400'}`}>{item.status}</span></p>
                  <p className="text-xs text-slate-400 dark:text-slate-500 mt-1 font-quicksand">{new Date(item.createdAt).toLocaleString('vi-VN')}</p>
                </div>
                <button
                  onClick={(e) => deleteHistory(item.id, e)}
                  className="absolute top-4 right-4 p-2 bg-rose-50 dark:bg-rose-500/10 text-rose-500 dark:text-rose-400 rounded-lg opacity-0 group-hover:opacity-100 hover:bg-rose-500 dark:hover:bg-rose-500 hover:text-white transition-all shadow-sm"
                  title="Xóa báo cáo"
                >
                  <Trash2 className="w-5 h-5" />
                </button>
              </div>
            ))}
          </div>
        )}
      </motion.div>
    </div>
  );
}
