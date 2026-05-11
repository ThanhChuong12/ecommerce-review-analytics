'use client';

import { useState, useEffect } from 'react';
import { BarChart3, Sun, Moon, History, LogOut, LogIn } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { supabase } from '@/lib/supabase';
import axios from 'axios';

export default function Header() {
  const [user, setUser] = useState<any>(null);
  const router = useRouter();
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    // Check initial theme
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark' || (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
      setIsDark(true);
      document.documentElement.classList.add('dark');
    } else {
      setIsDark(false);
      document.documentElement.classList.remove('dark');
    }

    // Auth
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (session?.user) handleUserLogin(session.user);
    });

    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      if (session?.user) handleUserLogin(session.user);
      else setUser(null);
    });

    return () => subscription.unsubscribe();
  }, []);

  const handleUserLogin = async (supabaseUser: any) => {
    const userData = {
      id: supabaseUser.id,
      email: supabaseUser.email,
      name: supabaseUser.user_metadata?.full_name || supabaseUser.email.split('@')[0],
      avatar: supabaseUser.user_metadata?.avatar_url || ''
    };
    setUser(userData);
    try {
      await axios.post(`${process.env.NEXT_PUBLIC_API_URL}/auth/sync`, userData);
    } catch (error) {
      console.error('Lỗi đồng bộ user:', error);
    }
  };

  const toggleTheme = () => {
    if (isDark) {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
      setIsDark(false);
    } else {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
      setIsDark(true);
    }
  };

  const loginWithGoogle = async () => {
    await supabase.auth.signInWithOAuth({ provider: 'google', options: { redirectTo: window.location.origin } });
  };

  const logout = async () => {
    await supabase.auth.signOut();
    setUser(null);
    router.push('/');
  };

  return (
    <header className="w-full max-w-6xl mx-auto p-6 pb-2 md:p-12 md:pb-4 flex items-center justify-between">
      <div className="flex items-center gap-4 cursor-pointer" onClick={() => router.push('/')}>
        <BarChart3 className="w-10 h-10 text-blue-500 dark:text-indigo-400" />
        <div>
          <h1 className="text-3xl font-quicksand bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 dark:from-indigo-400 dark:via-purple-400 dark:to-cyan-400 bg-clip-text text-transparent">
            Review Analytics
          </h1>
          <p className="font-quicksand text-sm text-slate-500 dark:text-slate-400 tracking-wide">Multimodal AI System</p>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <button onClick={toggleTheme} className="cursor-pointer p-2.5 rounded-full bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 shadow-sm text-slate-500 dark:text-slate-400 hover:text-blue-500 dark:hover:text-indigo-400 transition-colors">
          {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
        </button>

        {user ? (
          <div className="flex items-center gap-3 bg-white/80 dark:bg-slate-800/80 p-1.5 pr-4 rounded-full border border-white dark:border-slate-700 shadow">
            <img src={user.avatar || '/placeholder.png'} alt="avatar" className="w-9 h-9 rounded-full border border-slate-200 dark:border-slate-600" />
            <span className="text-sm font-medium hidden md:block text-slate-700 dark:text-slate-200">{user.name}</span>
            <button onClick={() => router.push('/history')} className="cursor-pointer text-blue-500 dark:text-indigo-400 hover:text-blue-600 dark:hover:text-indigo-300 ml-2 p-1.5 hover:bg-blue-50 dark:hover:bg-slate-700 rounded-full transition-colors" title="Lịch sử">
              <History className="w-5 h-5" />
            </button>
            <button onClick={logout} className="cursor-pointer text-rose-500 dark:text-rose-400 hover:text-rose-600 dark:hover:text-rose-300 ml-1 p-1.5 hover:bg-rose-50 dark:hover:bg-slate-700 rounded-full transition-colors" title="Đăng xuất">
              <LogOut className="w-5 h-5" />
            </button>
          </div>
        ) : (
          <button onClick={loginWithGoogle} className="cursor-pointer flex items-center gap-2 bg-blue-400 dark:bg-indigo-500 text-white px-5 py-2.5 rounded-full font-bold hover:bg-blue-700 dark:hover:bg-indigo-500 transition-colors shadow-md">
            <LogIn className="w-4 h-4" /> Đăng nhập
            <img
              src="https://www.svgrepo.com/show/475656/google-color.svg"
              alt="Google"
              className="w-5 h-5"
            />
          </button>
        )}
      </div>
    </header>
  );
}
