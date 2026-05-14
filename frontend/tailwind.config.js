/** @type {import('tailwindcss').Config} */

import typography from "@tailwindcss/typography";

export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    container: {
      center: true,
    },
    extend: {
      fontFamily: {
        display: ["var(--font-display)"],
        body: ["var(--font-body)"],
      },
      colors: {
        bg0: "var(--bg-0)",
        bg1: "var(--bg-1)",
        fg0: "var(--fg-0)",
        fg1: "var(--fg-1)",
        muted: "var(--muted)",
        amber: "var(--amber)",
        cyan: "var(--cyan)",
        danger: "var(--danger)",
      },
    },
  },
  plugins: [typography],
};
