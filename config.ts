import { Config } from "./src/config";

export const defaultConfig: Config = {
  url: "https://nexusbubble.io/",
  match: "https://nexusbubble.io/**",
  maxPagesToCrawl: 100,
  outputFileName: "nexus.json",
};
