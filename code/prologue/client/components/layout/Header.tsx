import { Link } from "wouter";
import { HeaderConfig } from "@/types/config";

interface HeaderProps {
  config: HeaderConfig;
}

export function Header({ config }: HeaderProps) {
  return (
    <header className={`fixed top-0 left-0 right-0 z-50 px-4 py-3 ${config.colors.background}`}>
      <div className="container mx-auto flex items-center justify-between">
        <Link href={config.logo.url}>
          <span className={`text-2xl font-bold cursor-pointer ${config.colors.text}`}>
            {config.logo.text}
          </span>
        </Link>

        <nav className="hidden md:flex items-center space-x-6">
          {config.links.map((link) => (
            <Link key={link.text} href={link.url}>
              <span className={`cursor-pointer hover:opacity-80 transition-opacity ${config.colors.text}`}>
                {link.text}
              </span>
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
