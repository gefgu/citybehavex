/// <reference types="vite/client" />

declare module "echarts-gl";

interface ImportMetaEnv {
  readonly VITE_MAPBOX_TOKEN?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
