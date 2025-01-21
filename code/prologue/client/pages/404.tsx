import React from "react";

export default function NotFound() {
  return (
    <div className="relative w-full h-screen overflow-hidden">
      {/* Full-Screen Background Video */}
      <video
        autoPlay
        loop
        muted
        playsInline
        className="absolute top-0 left-0 w-full h-full object-cover"
      >
        <source src="/404-loop-720.mp4" type="video/mp4" />
        Your browser does not support the video tag.
      </video>

      {/* Overlay Text */}
      <div className="relative z-10 flex flex-col items-center justify-center h-full text-center text-white">
        <h1 className="text-6xl md:text-8xl font-bold drop-shadow-md">
          404
        </h1>
        <p className="mt-4 text-lg md:text-2xl drop-shadow-md">
          No idea what you're looking for, anon.
        </p>
      </div>

      {/* Overlay for contrast */}
      <div className="absolute top-0 left-0 w-full h-full bg-black bg-opacity-30"></div>
    </div>
  );
}
