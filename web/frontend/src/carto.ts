const CARTODB_TOKEN = import.meta.env.CARTODB_TOKEN;

const CARTO_VOYAGER_TILES =
  "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png";

export const CARTO_VOYAGER_TILES_URL = CARTODB_TOKEN
  ? `${CARTO_VOYAGER_TILES}?key=${encodeURIComponent(CARTODB_TOKEN)}`
  : CARTO_VOYAGER_TILES;

export const CARTO_ATTRIBUTION = "&copy; OpenStreetMap &copy; CARTO";
