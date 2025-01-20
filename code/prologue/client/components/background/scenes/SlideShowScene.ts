import { 
  Scene,
  Engine,
  FreeCamera,
  Vector3,
  MeshBuilder,
  StandardMaterial,
  Texture,
  VideoTexture,
  Color3,
  Camera,
  Observable
} from "@babylonjs/core";

interface SlideShowOptions {
  sources: string[];
  interval?: number;
  opacity?: number;
  blur?: string;
}

export function createSlideShowScene(
  canvas: HTMLCanvasElement,
  engine: Engine,
  scene: Scene,
  options: SlideShowOptions
) {
  // Create camera for 2D viewing
  const camera = new FreeCamera("camera", new Vector3(0, 0, -1), scene);
  camera.mode = Camera.ORTHOGRAPHIC_CAMERA;

  const distance = 1;
  const aspect = window.innerWidth / window.innerHeight;

  camera.orthoLeft = -distance * aspect;
  camera.orthoRight = distance * aspect;
  camera.orthoBottom = -distance;
  camera.orthoTop = distance;
  camera.setTarget(Vector3.Zero());

  // Create a billboard plane that fills the viewport
  const plane = MeshBuilder.CreatePlane(
    "billboard",
    { width: 2 * aspect, height: 2 },
    scene
  );
  plane.position = Vector3.Zero();

  // Create material
  const material = new StandardMaterial("billboardMaterial", scene);
  material.alpha = options.opacity ?? 1;
  material.backFaceCulling = false;
  material.emissiveColor = Color3.White();
  material.disableLighting = true;
  plane.material = material;

  let currentIndex = 0;
  const textures: (Texture | VideoTexture)[] = [];

  // Load all textures
  options.sources.forEach((source, index) => {
    const isVideo = source.endsWith('.mp4') || source.endsWith('.webm');
    const texture = isVideo 
      ? new VideoTexture(`video${index}`, source, scene, true)
      : new Texture(source, scene);

    // Use the base Texture class's observable for both types
    (texture as Texture).onLoadObservable.add(() => {
      textures.push(texture);
      if (index === 0) {
        material.diffuseTexture = texture;
        material.emissiveTexture = texture;
      }
    });

    const errorObservable = new Observable<any>();
    errorObservable.add((error) => {
      console.error(`Failed to load texture ${source}:`, error);
    });
    (texture as any).onLoadErrorObservable = errorObservable;
  });

  // Set up slideshow interval
  const interval = options.interval || 5000;
  let timeoutId: NodeJS.Timeout;

  const nextSlide = () => {
    if (textures.length === 0) return;
    currentIndex = (currentIndex + 1) % textures.length;
    material.diffuseTexture = textures[currentIndex];
    material.emissiveTexture = textures[currentIndex];

    // If it's a video, play it
    if (textures[currentIndex] instanceof VideoTexture) {
      (textures[currentIndex] as VideoTexture).video.play();
    }

    timeoutId = setTimeout(nextSlide, interval);
  };

  timeoutId = setTimeout(nextSlide, interval);

  // Handle window resize
  const handleResize = () => {
    const newAspect = window.innerWidth / window.innerHeight;
    plane.scaling.x = newAspect;

    camera.orthoLeft = -distance * newAspect;
    camera.orthoRight = distance * newAspect;
    
    engine.resize();
  };

  window.addEventListener("resize", handleResize);

  // Return cleanup function
  return () => {
    window.removeEventListener("resize", handleResize);
    clearTimeout(timeoutId);
    textures.forEach(texture => texture.dispose());
  };
}

