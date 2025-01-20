import { Card } from "@/components/ui/card";
import { MessageCircle } from "lucide-react";

export function ChatPlaceholder() {
  return (
    <Card className="fixed bottom-4 right-4 p-4 w-[350px] backdrop-blur bg-background/80">
      <div className="flex items-center space-x-2 text-muted-foreground">
        <MessageCircle className="h-5 w-5" />
        <p>Chat integration coming soon...</p>
      </div>
    </Card>
  );
}

