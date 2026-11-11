import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource/inter/latin-400.css";
import "@fontsource/inter/latin-500.css";
import "@fontsource/inter/latin-600.css";
import "@fontsource/space-grotesk/latin-500.css";
import "@fontsource/space-grotesk/latin-600.css";
import "@fontsource/jetbrains-mono/latin-400.css";
import "@fontsource/jetbrains-mono/latin-500.css";
import App from "./App";
import "./styles.css";

// Phones: keep the app exactly as tall as the visible area, so the composer sits just above the on-screen
// keyboard instead of underneath it. Browsers that resize the page themselves simply report the same height.
const viewport = window.visualViewport;
if (viewport) {
  const fit = () => {
    document.documentElement.style.setProperty("--app-h", `${viewport.height}px`);
    window.scrollTo(0, 0); // iOS pans the page when the keyboard opens; put it back
  };
  viewport.addEventListener("resize", fit);
  fit();
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
