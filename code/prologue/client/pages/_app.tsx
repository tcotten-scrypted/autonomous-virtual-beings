import type { AppProps } from 'next/app';
import '@/styles/globals.css';

function MyApp({ Component, pageProps }: AppProps) {
  return (
    <div className="bg-red-500"> {/* Test wrapper */}
      <Component {...pageProps} />
    </div>
  );
}

export default MyApp;
