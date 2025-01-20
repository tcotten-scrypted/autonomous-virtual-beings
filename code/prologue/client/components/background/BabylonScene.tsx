import { useEffect, useRef } from "react";
import { Engine, Scene } from "@babylonjs/core";
import { createSlideShowScene } from "./scenes/SlideShowScene";
import { createInteractiveScene } from "./scenes/InteractiveScene";

interface BabylonSceneProps {
  type: 'image' | 'video' | 'gif' | '3d';
  source: string | string[];
  options?: {
    opacity?: number;
    blur?: string;
    scale?: number;
    interval?: number;
  };
}

export function BabylonScene({ type, source, options = {} }: BabylonSceneProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<Engine | null>(null);
  const sceneRef = useRef<Scene | null>(null);
  const cleanupRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    try {
      // Create engine and scene
      engineRef.current = new Engine(canvasRef.current, true, { 
        preserveDrawingBuffer: true, 
        stencil: true 
      });
      sceneRef.current = new Scene(engineRef.current);

      const scene = sceneRef.current;
      const engine = engineRef.current;

      // Initialize appropriate scene based on type
      if (type === '3d') {
        cleanupRef.current = createInteractiveScene(
          canvasRef.current,
          engine,
          scene,
          { enablePanning: true }
        );
      } else {
        const sources = Array.isArray(source) ? source : [source];
        cleanupRef.current = createSlideShowScene(
          canvasRef.current,
          engine,
          scene,
          {
            sources,
            interval: options.interval || 5000,
            opacity: options.opacity,
            blur: options.blur
          }
        );
      }

      // Start rendering
      engine.runRenderLoop(() => {
        scene.render();
      });

      return () => {
        cleanupRef.current?.();
        if (sceneRef.current) {
          sceneRef.current.dispose();
        }
        if (engineRef.current) {
          engineRef.current.dispose();
        }
      };
    } catch (error) {
      console.error("Error in BabylonScene:", error);
    }
  }, [type, source, options.opacity, options.interval, options.blur]);

  return (
    <canvas 
      ref={canvasRef}
      className="fixed inset-0 w-full h-full -z-10"
    />
  );
}
