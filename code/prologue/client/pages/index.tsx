import siteConfig from "@/config/site.json";
import { Header } from "@/components/layout/Header";
import { ChatPlaceholder } from "@/components/layout/ChatPlaceholder";
import { BackgroundManager } from "@/components/background/BackgroundManager";

export default function Index() {
  return (
    <div className="min-h-screen w-full">
      <Header config={siteConfig.header} />
      <BackgroundManager config={siteConfig.background} />
      
      <main className="container mx-auto pt-24 px-4">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-4xl md:text-6xl font-bold text-foreground mb-4 drop-shadow-2xl">
            autonomous virtual beings
          </h1>
          <p className="text-lg md:text-xl text-muted-foreground drop-shadow-2xl">
            Experience new life, new civilizations with AI.
          </p>
        </div>
      </main>

      <ChatPlaceholder />
    </div>
  );
}

