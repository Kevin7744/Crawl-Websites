import { Config } from "./src/config";

export const defaultConfig: Config = {
  url: "https://www.jumia.co.ke/",
  match: "https://www.jumia.co.ke/**",
  maxPagesToCrawl: 50,
  outputFileName: "hired.json",
};
