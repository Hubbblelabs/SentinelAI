# 🛡️ SentinelAI - Cyberbullying Detection Frontend

An Expo (React Native) mobile application for real-time cyberbullying detection and prevention. Features a **flat yellow design system** with WCAG AA contrast compliance and accessible, compassionate UI/UX.

---

## 🎨 Design System

### Theme Colors
SentinelAI uses a **flat yellow palette** with no gradients:

```typescript
colors: {
  // Primary yellows (flat)
  primary: "#FFD600",        // Bright yellow (buttons, accents)
  primary700: "#F2C200",    // Pressed/dark variant
  primary900: "#C89A00",    // Strong contrast borders

  // Surfaces
  background: "#FFF9E6",    // Warm background
  surface: "#FFFFFF",       // Card background
  mutedSurface: "#FFF2CC",  // Subtle fills

  // Text & neutrals
  text: "#111827",          // Dark text (WCAG AA compliant)
  subText: "#6B7280",       // Muted text
  border: "#E6D87A",        // Soft borders

  // Semantic
  success: "#16A34A",
  danger: "#DC2626",
  info: "#0EA5E9"
}
```

### Visual Language
- ✅ **Flat fills only** - No gradients
- 🔲 **Rounded corners** - 8px (sm), 12px (md), 20px (lg)
- 🎯 **Soft shadows** - Subtle elevation (2-8dp)
- 📏 **Consistent spacing** - 6, 8, 16, 24, 32px scale
- 🔤 **System fonts** + geometric display (Inter/Poppins)
- ♿ **WCAG AA contrast** - All text meets accessibility standards

---

## 📦 Project Structure

```
Frontend/
├── app/                          # File-based routing
│   ├── _layout.tsx              # Root layout with ThemeProvider
│   ├── onboarding.tsx           # Welcome & permissions flow
│   ├── settings.tsx             # Configuration screen
│   ├── (tabs)/                  # Tab navigation
│   │   ├── _layout.tsx         # Tab bar setup
│   │   ├── index.tsx           # Home Dashboard
│   │   └── explore.tsx         # (Settings redirect)
│   └── incident/
│       └── [id].tsx            # Incident details (dynamic route)
├── components/
│   ├── ui/                      # Reusable UI primitives
│   │   ├── Button.tsx          # Primary/Secondary/Danger variants
│   │   ├── Card.tsx            # Surface container
│   │   ├── Badge.tsx           # Status indicators
│   │   └── Modal.tsx           # Accessible dialog
│   └── detection/               # Feature-specific components
│       ├── DetectionModal.tsx  # Alert for outgoing harmful content
│       └── BlurredMessage.tsx  # Incoming message protection
├── contexts/
│   └── ThemeContext.tsx        # Central theme provider
├── constants/
│   └── theme.ts                # Theme configuration
└── data/
    └── mockData.ts             # Static demo data
```

---

## 🚀 Getting Started

### Installation

```bash
# Install dependencies
npm install

# Start development server
npx expo start
```

### Running the App

- **iOS Simulator**: Press `i`
- **Android Emulator**: Press `a`
- **Web**: Press `w`
- **Physical Device**: Scan QR with Expo Go app

---

## 🧩 Component Usage

### Button Component

```tsx
import { Button } from '../components/ui/Button';

// Primary button (bright yellow)
<Button 
  title="Get Started"
  onPress={() => {}}
  variant="primary"
  accessibilityLabel="Start using SentinelAI"
/>

// Secondary button (outline)
<Button 
  title="Cancel"
  onPress={() => {}}
  variant="secondary"
/>

// Danger button (for destructive actions)
<Button 
  title="Block User"
  onPress={() => {}}
  variant="danger"
  disabled={false}
  loading={false}
/>
```

**States**: default, pressed, disabled (0.5 opacity), loading

---

### Card Component

```tsx
import { Card } from '../components/ui/Card';

// Default surface card
<Card>
  <Text>Card content here</Text>
</Card>

// Muted surface variant
<Card variant="muted">
  <Text>Subtle background card</Text>
</Card>

// Custom styles
<Card style={{ marginBottom: 16 }}>
  <Text>Custom styled card</Text>
</Card>
```

**Features**: Soft shadows, rounded corners (12px), responsive padding

---

### Badge Component

```tsx
import { Badge } from '../components/ui/Badge';

// Status badges
<Badge label="New" variant="primary" size="small" />
<Badge label="High Risk" variant="danger" size="medium" />
<Badge label="Verified" variant="success" size="small" />
<Badge label="Info" variant="info" />
<Badge label="Archived" variant="neutral" />
```

**Variants**: primary (yellow), success (green), danger (red), info (blue), neutral (grey)

---

### Modal Component

```tsx
import { Modal } from '../components/ui/Modal';
import { Button } from '../components/ui/Button';

<Modal
  visible={isVisible}
  onClose={() => setIsVisible(false)}
  title="Confirm Action"
  actions={
    <>
      <Button title="Cancel" onPress={handleCancel} variant="secondary" />
      <Button title="Confirm" onPress={handleConfirm} variant="primary" />
    </>
  }
>
  <Text>Are you sure you want to proceed?</Text>
</Modal>
```

**Features**: Accessible, screen reader support, auto-announcement, tap-outside to close

---

### Detection Components

#### DetectionModal
Alerts users when they're about to send harmful content:

```tsx
import { DetectionModal } from '../components/detection/DetectionModal';

<DetectionModal
  visible={showAlert}
  onClose={() => setShowAlert(false)}
  detection={{
    type: 'harassment',
    severity: 'high',
    confidence: 0.87,
    message: 'Your message content here'
  }}
  onEdit={() => console.log('Edit message')}
  onSendAnyway={() => console.log('Send anyway')}
/>
```

#### BlurredMessage
Protects users from viewing incoming harmful content:

```tsx
import { BlurredMessage } from '../components/detection/BlurredMessage';

<BlurredMessage
  message={{
    id: '1',
    sender: 'unknown_user',
    preview: 'Tap to reveal...',
    fullMessage: 'Hidden harmful content',
    type: 'threat',
    severity: 'high',
    timestamp: new Date(),
    app: 'Instagram'
  }}
  onReveal={() => console.log('Message revealed')}
  onReport={() => console.log('Reported')}
  onBlock={() => console.log('Blocked sender')}
/>
```

---

## 🎯 Theme Usage

### Accessing Theme in Components

```tsx
import { useAppTheme } from '../contexts/ThemeContext';

function MyComponent() {
  const { theme } = useAppTheme();

  return (
    <View style={{ 
      backgroundColor: theme.colors.background,
      padding: theme.spacing.lg,
      borderRadius: theme.radii.md 
    }}>
      <Text style={{ 
        color: theme.colors.text,
        fontSize: theme.typography.heading 
      }}>
        Hello World
      </Text>
    </View>
  );
}
```

### Common Style Patterns

```tsx
// Button styles
{
  backgroundColor: theme.colors.primary,
  paddingVertical: theme.spacing.md,
  paddingHorizontal: theme.spacing.lg,
  borderRadius: theme.radii.lg,
  ...theme.shadows.md
}

// Card styles
{
  backgroundColor: theme.colors.surface,
  padding: theme.spacing.md,
  borderRadius: theme.radii.md,
  ...theme.shadows.sm
}

// Text hierarchy
{
  // Headings
  fontSize: theme.typography.heading,      // 20px
  fontWeight: '700',
  color: theme.colors.text,

  // Body
  fontSize: theme.typography.body,         // 14px
  color: theme.colors.text,

  // Caption/Meta
  fontSize: theme.typography.caption,      // 12px
  color: theme.colors.subText,
}
```

---

## 📱 Screens Overview

### 1. Onboarding (`app/onboarding.tsx`)
- Multi-step welcome flow
- Permission requests (notifications)
- Platform-specific instructions
- Skip option available

### 2. Home Dashboard (`app/(tabs)/index.tsx`)
- **Safety Score**: Circular badge showing protection level
- **Stats Grid**: Messages scanned, threats blocked, active days, new alerts
- **Recent Incidents**: Scrollable list with type badges and severity indicators
- **Pull to Refresh**: Update metrics

### 3. Incident Details (`app/incident/[id].tsx`)
- Full message content with warning styling
- Metadata: sender, app, timestamp
- Actions: Report, Block, Delete
- Accessible navigation

### 4. Settings (`app/settings.tsx`)
- **Detection Sensitivity**: Low/Medium/High toggle
- **Notifications**: Enable/disable alerts
- **Privacy**: Auto-block, anonymous reporting
- **About**: App info and version

---

## ♿ Accessibility

All components follow WCAG AA guidelines:

- ✅ **Contrast Ratios**: 
  - Dark text (#111827) on yellow (#FFD600) = 7.8:1
  - Subtext (#6B7280) on white = 4.7:1
- 🎯 **Touch Targets**: Minimum 48x48dp
- 🔊 **Screen Readers**: Proper labels and hints
- ⌨️ **Keyboard Navigation**: Full support on web
- 🎬 **Reduced Motion**: Respects system preferences

---

## 🔧 Customization

### Adding New Colors

```typescript
// constants/theme.ts
export const theme = {
  colors: {
    ...existingColors,
    customColor: "#YOUR_HEX",
  }
}
```

### Creating Custom Components

```tsx
import { useAppTheme } from '../contexts/ThemeContext';

export const CustomComponent = () => {
  const { theme } = useAppTheme();
  
  return (
    <View style={{
      backgroundColor: theme.colors.mutedSurface,
      borderRadius: theme.radii.sm,
      padding: theme.spacing.md,
    }}>
      {/* Your content */}
    </View>
  );
};
```

---

## 📊 Mock Data

Static demo data is in `data/mockData.ts`:

```typescript
// Access mock incidents
import { mockIncidents, mockStats } from '../data/mockData';

// Use in components
const incidents = mockIncidents.filter(i => !i.read);
const { safetyScore } = mockStats;
```

**Production**: Replace with API calls to backend services.

---

## 🧪 Testing Checklist

- [ ] All text meets WCAG AA contrast
- [ ] Touch targets are 48dp minimum
- [ ] Screen reader announces modals
- [ ] No gradients in design
- [ ] Buttons show pressed state
- [ ] Cards have subtle shadows
- [ ] Spacing is consistent (6/8/16/24/32)
- [ ] Colors use theme constants
- [ ] Navigation flows work on all platforms

---

## 📝 Microcopy Guidelines

Use compassionate, non-judgmental language:

✅ **Good**: "This message may hurt someone. Want help?"  
❌ **Bad**: "You're being mean. Stop it."

✅ **Good**: "Tap to reveal this message"  
❌ **Bad**: "Warning: Bad content"

---

## 🚢 Deployment

```bash
# Build for production
eas build --platform all

# Submit to stores
eas submit --platform ios
eas submit --platform android
```

[Expo Application Services (EAS) docs](https://docs.expo.dev/eas/)

---

## 📚 Resources

- [Expo Documentation](https://docs.expo.dev/)
- [React Navigation](https://reactnavigation.org/)
- [WCAG Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [Expo Router](https://docs.expo.dev/router/introduction/)

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

1. Follow the flat yellow design system
2. Ensure WCAG AA contrast compliance
3. Add accessibility labels to all interactive elements
4. Test on iOS, Android, and web
5. Use TypeScript for type safety

---

**Built with ❤️ for safer online communities**
