# 🚀 SentinelAI - Quick Reference

**Flat Yellow Design System • WCAG AA Compliant • React Native + TypeScript**

---

## 🎨 Theme Quick Access

```tsx
import { useAppTheme } from '../contexts/ThemeContext';
const { theme } = useAppTheme();

// Colors
theme.colors.primary        // #FFD600 - Main yellow
theme.colors.text           // #111827 - Dark text
theme.colors.background     // #FFF9E6 - Page BG
theme.colors.surface        // #FFFFFF - Cards

// Spacing
theme.spacing.md            // 16px
theme.spacing.lg            // 24px

// Typography
theme.typography.heading    // 20px
theme.typography.body       // 14px

// Radii
theme.radii.md              // 12px
theme.radii.lg              // 20px
```

---

## 🧩 Component Imports

```tsx
// UI Components
import { Button } from '../components/ui/Button';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Modal } from '../components/ui/Modal';

// Detection
import { DetectionModal } from '../components/detection/DetectionModal';
import { BlurredMessage } from '../components/detection/BlurredMessage';

// Context
import { useAppTheme } from '../contexts/ThemeContext';

// Navigation
import { useRouter } from 'expo-router';

// Mock Data
import { mockIncidents, mockStats } from '../data/mockData';
```

---

## 📱 Navigation

```tsx
const router = useRouter();

router.push('/(tabs)');              // Dashboard
router.push('/settings');            // Settings
router.push('/onboarding');          // Onboarding
router.push(`/incident/${id}`);      // Incident detail
router.back();                       // Go back
```

---

## 🎯 Common Patterns

### Button Usage
```tsx
<Button 
  title="Continue"
  onPress={handlePress}
  variant="primary"          // or "secondary", "danger"
  disabled={false}
  loading={false}
/>
```

### Card Layout
```tsx
<Card variant="default">     {/* or "muted" */}
  <Text>Content</Text>
</Card>
```

### Badge Display
```tsx
<Badge 
  label="High Risk"
  variant="danger"           // or "primary", "success", "info", "neutral"
  size="small"              // or "medium"
/>
```

### Screen Template
```tsx
export default function MyScreen() {
  const { theme } = useAppTheme();
  
  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: theme.colors.background }}>
      <ScrollView contentContainerStyle={{ padding: theme.spacing.lg }}>
        {/* Content */}
      </ScrollView>
    </SafeAreaView>
  );
}
```

---

## ✅ Design Rules

1. ✅ **No gradients** - Flat fills only
2. ✅ **Use theme constants** - Never hardcode
3. ✅ **WCAG AA contrast** - Dark text on yellow
4. ✅ **48dp touch targets** - Minimum size
5. ✅ **Accessibility labels** - All interactive elements
6. ✅ **Rounded corners** - 8/12/20px
7. ✅ **Consistent spacing** - 6/8/16/24/32px

---

## 🎨 Color Contrast

```tsx
// ✅ GOOD - High contrast
<Text style={{ color: theme.colors.text }}>        // On yellow/white
<Text style={{ color: '#FFFFFF' }}>                // On danger/dark

// ❌ BAD - Low contrast
<Text style={{ color: theme.colors.subText }}>     // On yellow
<Text style={{ color: theme.colors.primary }}>     // On white
```

---

## 📦 File Structure

```
app/                    # Screens (file-based routing)
components/
  ├── ui/              # Reusable primitives
  └── detection/       # Feature components
contexts/              # React contexts
constants/             # Theme config
data/                  # Mock data
```

---

## 🚀 Commands

```bash
npm install            # Install dependencies
npx expo start         # Start dev server
npx expo start --ios   # iOS simulator
npx expo start -a      # Android emulator
npx expo start --web   # Web browser
```

---

## 🐛 Common Issues

**Import paths wrong?**
- From app/*: `../components/...`
- From components/*: `../../contexts/...`

**Theme not available?**
- Wrap in `<ThemeProvider>` (check `app/_layout.tsx`)

**Shadows not showing?**
- Use `...theme.shadows.md` (includes elevation for Android)

**Navigation not working?**
- Check route is in `app/_layout.tsx`
- Use correct path: `/settings` not `settings`

---

## 📚 Full Docs

- [README.md](README.md) - Complete documentation
- [USAGE.md](USAGE.md) - Developer guide with examples
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Project overview

---

**Built with ❤️ for safer communities**
