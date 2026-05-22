'use client';

import { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { History, Trash2, Loader2, Search, Filter, ChevronLeft, ChevronRight } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { supabase } from '@/lib/supabase';

export default function HistoryPage() {
  const router = useRouter();
  const [historyList, setHistoryList] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  // States cho Lọc, Tìm kiếm, Phân trang
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('ALL');
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 6;

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
    router.push(`/analyze?historyId=${productId}`);
  };

  // Logic Lọc & Tìm kiếm
  const filteredList = useMemo(() => {
    return historyList.filter(item => {
      const matchSearch = item.name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.url?.toLowerCase().includes(searchQuery.toLowerCase());
      const matchStatus = statusFilter === 'ALL' || item.status === statusFilter;
      return matchSearch && matchStatus;
    });
  }, [historyList, searchQuery, statusFilter]);

  // Logic Phân trang
  const totalPages = Math.ceil(filteredList.length / itemsPerPage);
  const currentList = useMemo(() => {
    const start = (currentPage - 1) * itemsPerPage;
    return filteredList.slice(start, start + itemsPerPage);
  }, [filteredList, currentPage]);

  // Reset trang về 1 khi thay đổi bộ lọc
  useEffect(() => {
    setCurrentPage(1);
  }, [searchQuery, statusFilter]);

  return (
    <div className="w-full max-w-5xl mx-auto px-4 pb-20">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex flex-col items-center mb-8 mt-4">
          <h2 className="text-3xl font-bold flex items-center gap-3 text-slate-800 dark:text-slate-100 font-quicksand mb-6">
            <History className="text-blue-500 dark:text-indigo-400 w-8 h-8" /> Lịch sử
          </h2>

          {/* Công cụ Lọc & Tìm kiếm */}
          <div className="flex flex-wrap items-center justify-center gap-4 w-full max-w-2xl">
            <div className="relative flex-1 min-w-[250px]">
              <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
              <input
                type="text"
                placeholder="Tìm tên sản phẩm hoặc URL..."
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                className="w-full pl-9 pr-4 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-white/50 dark:bg-slate-800/50 focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm dark:text-slate-200 transition-all"
              />
            </div>

            <div className="relative min-w-[180px]">
              <Filter className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" />
              <select
                value={statusFilter}
                onChange={e => setStatusFilter(e.target.value)}
                className="w-full pl-9 pr-8 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-white/50 dark:bg-slate-800/50 focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm dark:text-slate-200 appearance-none cursor-pointer transition-all"
              >
                <option value="ALL">Tất cả Trạng thái</option>
                <option value="COMPLETED">Hoàn thành (COMPLETED)</option>
                <option value="PROCESSING">Đang xử lý (PROCESSING)</option>
                <option value="ERROR">Lỗi (ERROR)</option>
              </select>
            </div>
          </div>
        </div>

        {loading ? (
          <div className="flex justify-center items-center py-20">
            <Loader2 className="w-10 h-10 animate-spin text-blue-500 dark:text-indigo-400" />
          </div>
        ) : filteredList.length === 0 ? (
          <div className="bg-white/80 dark:bg-slate-900/40 backdrop-blur-xl rounded-3xl shadow-sm border border-slate-200 dark:border-white/10 p-16 text-center text-slate-500 dark:text-slate-400 font-quicksand flex flex-col items-center justify-center min-h-[300px]">
            <Search className="w-12 h-12 mb-4 text-slate-300 dark:text-slate-600" />
            <p className="text-lg">Không tìm thấy báo cáo nào phù hợp.</p>
            {historyList.length === 0 && <p className="text-sm mt-2">Hãy bắt đầu phân tích thử 1 sản phẩm nhé!</p>}
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <AnimatePresence>
                {currentList.map(item => (
                  <motion.div
                    layout
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    transition={{ duration: 0.2 }}
                    key={item.id}
                    className="bg-white/80 dark:bg-slate-900/60 backdrop-blur-xl rounded-3xl shadow-sm border border-slate-100 dark:border-white/10 p-5 flex gap-5 relative group cursor-pointer hover:shadow-lg hover:border-blue-300 dark:hover:border-indigo-500/50 transition-all duration-300"
                    onClick={() => viewDetail(item.id)}
                  >
                    <div className="w-24 h-24 flex-shrink-0 relative rounded-2xl overflow-hidden shadow-sm border border-slate-100 dark:border-slate-700 bg-white dark:bg-slate-800">
                      <img src={item.thumbnail || '/placeholder.png'} className="w-full h-full object-cover" alt="thumbnail" />
                    </div>

                    <div className="flex-1 pr-8 overflow-hidden">
                      <h3 className="font-bold text-lg line-clamp-2 leading-snug text-slate-800 dark:text-slate-200 font-quicksand">{item.name || item.url}</h3>

                      <div className="mt-3 flex items-center gap-2">
                        <span className={`px-2.5 py-1 rounded-full text-xs font-bold inline-flex items-center gap-1 ${item.status === 'COMPLETED' ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-400' :
                          item.status === 'PROCESSING' ? 'bg-blue-100 text-blue-700 dark:bg-blue-500/20 dark:text-blue-400' :
                            'bg-rose-100 text-rose-700 dark:bg-rose-500/20 dark:text-rose-400'
                          }`}>
                          <div className={`w-1.5 h-1.5 rounded-full ${item.status === 'COMPLETED' ? 'bg-emerald-500' : item.status === 'PROCESSING' ? 'bg-blue-500 animate-pulse' : 'bg-rose-500'}`}></div>
                          {item.status}
                        </span>
                      </div>

                      <p className="text-sm font-semibold text-slate-400 dark:text-slate-500 mt-2 font-quicksand flex items-center gap-1">
                        ⏱ {new Date(item.createdAt).toLocaleString('vi-VN')}
                      </p>
                    </div>

                    <button
                      onClick={(e) => deleteHistory(item.id, e)}
                      className="absolute cursor-pointer top-4 right-4 p-2 bg-rose-50 dark:bg-rose-500/10 text-rose-500 dark:text-rose-400 rounded-xl opacity-0 group-hover:opacity-100 hover:bg-rose-500 dark:hover:bg-rose-500 hover:text-white transition-all shadow-sm"
                      title="Xóa báo cáo"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>

            {/* Pagination Controls */}
            {totalPages > 1 && (
              <div className="flex justify-center items-center gap-4 mt-10">
                <button
                  onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                  className="p-2 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors text-slate-600 dark:text-slate-300"
                >
                  <ChevronLeft className="w-5 h-5" />
                </button>

                <div className="flex items-center gap-2">
                  {Array.from({ length: totalPages }).map((_, idx) => (
                    <button
                      key={idx}
                      onClick={() => setCurrentPage(idx + 1)}
                      className={`w-10 h-10 rounded-xl font-bold transition-all ${currentPage === idx + 1
                        ? 'bg-blue-500 text-white shadow-md shadow-blue-500/30'
                        : 'bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700'
                        }`}
                    >
                      {idx + 1}
                    </button>
                  ))}
                </div>

                <button
                  onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                  disabled={currentPage === totalPages}
                  className="p-2 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors text-slate-600 dark:text-slate-300"
                >
                  <ChevronRight className="w-5 h-5" />
                </button>
              </div>
            )}
          </>
        )}
      </motion.div>
    </div>
  );
}
