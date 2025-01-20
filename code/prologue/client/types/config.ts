export type HeaderConfig = {
  logo: {
    text: string;
    url: string;
  };
  links: {
    text: string;
    url: string;
  }[];
  colors: {
    background: string;
    text: string;
  };
};

export type BackgroundConfig = {
  type: 'image' | 'video' | 'gif' | '3d';
  source: string;
  options?: {
    opacity?: number;
    blur?: string;
    scale?: number;
  };
};

export type SiteConfig = {
  header: HeaderConfig;
  background: BackgroundConfig;
};

