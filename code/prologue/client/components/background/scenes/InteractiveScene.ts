import {
  Scene,
  Engine,
  ArcRotateCamera,
  Vector3,
  HemisphericLight,
  MeshBuilder,
  StandardMaterial,
  Color3
} from "@babylonjs/core";

interface InteractiveSceneOptions {
  cameraPosition?: Vector3;
  cameraTarget?: Vector3;
  enablePanning?: boolean;
}

export function createInteractiveScene(
  canvas: HTMLCanvasElement,
  engine: Engine,
  scene: Scene,
  options: InteractiveSceneOptions = {}
) {
  // Create camera
  const camera = new ArcRotateCamera(
    "camera",
    0,
    Math.PI / 3,
    10,
    options.cameraTarget || Vector3.Zero(),
    scene
  );

  // Set camera limits
  camera.lowerRadiusLimit = 5;
  camera.upperRadiusLimit = 20;
  camera.lowerBetaLimit = 0.1;
  camera.upperBetaLimit = Math.PI / 2;

  // Enable camera controls
  camera.attachControl(canvas, true);
  camera.panningSensibility = options.enablePanning ? 1000 : 0;

  if (options.cameraPosition) {
    camera.setPosition(options.cameraPosition);
  }

  // Add basic lighting
  const light = new HemisphericLight("light", new Vector3(0, 1, 0), scene);
  light.intensity = 0.7;

  // Add a ground plane for reference
  const ground = MeshBuilder.CreateGround("ground", {
    width: 20,
    height: 20,
    subdivisions: 2
  }, scene);

  const groundMaterial = new StandardMaterial("groundMaterial", scene);
  groundMaterial.diffuseColor = new Color3(0.2, 0.2, 0.2);
  groundMaterial.specularColor = new Color3(0.1, 0.1, 0.1);
  ground.material = groundMaterial;

  // Handle window resize
  const handleResize = () => {
    engine.resize();
  };

  window.addEventListener("resize", handleResize);

  // Return cleanup function
  return () => {
    window.removeEventListener("resize", handleResize);
  };
}

