import React from 'react'

export default function TailwindTest() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen">
      <div className="p-6 bg-blue-500 text-white rounded-lg shadow-lg">
        <h1 className="text-2xl font-bold mb-4">Tailwind Test</h1>
        <p className="text-lg">Testing basic Tailwind classes</p>
      </div>
      
      {/* Test utility classes individually */}
      <div className="mt-4 space-y-2">
        <div className="bg-red-500 p-2">Background Color Test</div>
        <div className="border-2 border-green-500">Border Test</div>
        <div className="shadow-xl p-2">Shadow Test</div>
        <div className="hover:bg-purple-500 p-2">Hover Test</div>
      </div>
    </div>
  )
}
