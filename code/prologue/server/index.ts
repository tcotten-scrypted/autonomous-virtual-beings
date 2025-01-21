// **index.ts**

// Middleware for request logging
import { NextResponse } from 'next/server';

export function middleware(request: Request) {
  const start = Date.now();
  const url = new URL(request.url);

  const response = NextResponse.next();

  response.headers.set('X-Response-Time', `${Date.now() - start}ms`);

  if (url.pathname.startsWith('/api')) {
    console.log(`${request.method} ${url.pathname} processed in ${Date.now() - start}ms`);
  }

  return response;
}

// API route example
import { NextApiRequest, NextApiResponse } from 'next';

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method === 'GET') {
    res.status(200).json({ message: 'This is an example route!' });
  } else {
    res.setHeader('Allow', ['GET']);
    res.status(405).end(`Method ${req.method} Not Allowed`);
  }
}
