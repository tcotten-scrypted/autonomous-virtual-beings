import { BackgroundConfig } from "@/types/config";
import { BabylonScene } from "./BabylonScene";

interface BackgroundManagerProps {
  config: BackgroundConfig;
}

export function BackgroundManager({ config }: BackgroundManagerProps) {
  return (
    <BabylonScene 
      type={config.type}
      source={config.source}
      options={config.options}
    />
  );
}
