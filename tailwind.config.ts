import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "gradient-conic":
          "conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))",
      },
      colors: {
        background: {
          primary: "#0F172A",
          secondary: "#1E293B",
          card: "rgba(30, 41, 59, 0.7)",
        },
        accent: "#3B82F6",
        success: "#34D399",
        danger: "#F87171",
      },
    },
  },
  plugins: [],
};
export default config;
