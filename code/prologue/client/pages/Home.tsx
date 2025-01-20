import siteConfig from "@/config/site.json";
import { Header } from "@/components/layout/Header";
import { ChatPlaceholder } from "@/components/layout/ChatPlaceholder";
import { BackgroundManager } from "@/components/background/BackgroundManager";

export default function Home() {
  return (
    <div className="min-h-screen w-full">
      <Header config={siteConfig.header} />
      <BackgroundManager config={siteConfig.background} />
      
      <main className="container mx-auto pt-24 px-4">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-4xl md:text-6xl font-bold text-foreground mb-4">
            your dreams. unraveled, remixed, and alive
          </h1>
          <p className="text-lg md:text-xl text-muted-foreground">
            Experience your imagination in ways you never thought possible.
          </p>
        </div>
      </main>

      <ChatPlaceholder />
    </div>
  );
}

