/**
 * SentinelAI Theme - Flat Yellow Palette with WCAG AA Contrast
 * Design constraints:
 * - Flat fills only (NO gradients)
 * - Yellow primary palette with neutral greys
 * - All text meets WCAG AA contrast requirements
 * - Rounded corners, soft shadows, consistent spacing
 */

import { Platform } from 'react-native';

export const theme = {
  colors: {
    // Primary yellows (flat)
    primary: "#FFD600",        // primary - bright yellow (buttons, accents)
    primary700: "#F2C200",    // for pressed / dark variant
    primary900: "#C89A00",    // for strong contrast backgrounds / borders

    // Surface / backgrounds
    background: "#FFF9E6",    // very light warm background
    surface: "#FFFFFF",       // card background
    mutedSurface: "#FFF2CC",  // subtle filled surface

    // Text & neutrals
    text: "#111827",          // dark text (for contrast on yellow/white)
    subText: "#6B7280",       // muted text
    border: "#E6D87A",        // soft border color derived from yellow

    // Semantic
    success: "#16A34A",
    danger: "#DC2626",
    info: "#0EA5E9"
  },
  radii: {
    sm: 8,
    md: 12,
    lg: 20
  },
  spacing: {
    xs: 6,
    sm: 8,
    md: 16,
    lg: 24,
    xl: 32
  },
  typography: {
    heading: 20,
    subheading: 16,
    body: 14,
    caption: 12
  },
  shadows: {
    sm: {
      shadowColor: "#000",
      shadowOffset: { width: 0, height: 1 },
      shadowOpacity: 0.05,
      shadowRadius: 2,
      elevation: 2,
    },
    md: {
      shadowColor: "#000",
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.08,
      shadowRadius: 4,
      elevation: 4,
    },
    lg: {
      shadowColor: "#000",
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 0.1,
      shadowRadius: 8,
      elevation: 8,
    }
  }
} as const;

export type Theme = typeof theme;

export const Fonts = Platform.select({
  ios: {
    /** iOS `UIFontDescriptorSystemDesignDefault` */
    sans: 'system-ui',
    /** iOS `UIFontDescriptorSystemDesignSerif` */
    serif: 'ui-serif',
    /** iOS `UIFontDescriptorSystemDesignRounded` */
    rounded: 'ui-rounded',
    /** iOS `UIFontDescriptorSystemDesignMonospaced` */
    mono: 'ui-monospace',
  },
  default: {
    sans: 'normal',
    serif: 'serif',
    rounded: 'normal',
    mono: 'monospace',
  },
  web: {
    sans: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
    serif: "Georgia, 'Times New Roman', serif",
    rounded: "'SF Pro Rounded', 'Hiragino Maru Gothic ProN', Meiryo, 'MS PGothic', sans-serif",
    mono: "SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace",
  },
});
