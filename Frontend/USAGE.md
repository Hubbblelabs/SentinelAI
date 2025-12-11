# 🚀 SentinelAI Usage Guide

Quick start guide for developers working with SentinelAI components and features.

---

## Quick Navigation

- [Theme System](#theme-system)
- [Navigation](#navigation)
- [Components](#components)
- [Common Patterns](#common-patterns)
- [Mock Data](#mock-data)
- [Troubleshooting](#troubleshooting)

---

## Theme System

### Accessing the Theme

```tsx
import { useAppTheme } from '../contexts/ThemeContext';

function MyComponent() {
  const { theme } = useAppTheme();
  
  // Use theme values
  const styles = {
    container: {
      backgroundColor: theme.colors.background,
      padding: theme.spacing.lg,
    }
  };
}
```

### Theme Structure

```typescript
theme.colors.primary          // #FFD600 - Main yellow
theme.colors.primary700       // #F2C200 - Darker yellow
theme.colors.primary900       // #C89A00 - Darkest yellow
theme.colors.background       // #FFF9E6 - Page background
theme.colors.surface          // #FFFFFF - Card surface
theme.colors.text             // #111827 - Primary text
theme.colors.subText          // #6B7280 - Secondary text

theme.spacing.xs              // 6
theme.spacing.sm              // 8
theme.spacing.md              // 16
theme.spacing.lg              // 24
theme.spacing.xl              // 32

theme.radii.sm                // 8
theme.radii.md                // 12
theme.radii.lg                // 20

theme.typography.heading      // 20
theme.typography.subheading   // 16
theme.typography.body         // 14
theme.typography.caption      // 12

theme.shadows.sm              // Subtle shadow
theme.shadows.md              // Medium shadow
theme.shadows.lg              // Strong shadow
```

---

## Navigation

### Navigating Between Screens

```tsx
import { useRouter } from 'expo-router';

function MyComponent() {
  const router = useRouter();
  
  // Navigate to home
  router.push('/(tabs)');
  
  // Navigate to settings
  router.push('/settings');
  
  // Navigate to incident detail
  router.push(`/incident/${incidentId}`);
  
  // Navigate to onboarding
  router.push('/onboarding');
  
  // Go back
  router.back();
  
  // Replace current screen
  router.replace('/(tabs)');
}
```

### Available Routes

- `/(tabs)` - Home Dashboard
- `/settings` - Settings screen
- `/onboarding` - Onboarding flow
- `/demo` - Component demo/showcase
- `/incident/[id]` - Incident details (dynamic)

---

## Components

### Button

```tsx
import { Button } from '../components/ui/Button';

// Primary action
<Button 
  title="Continue"
  onPress={handlePress}
  variant="primary"
/>

// Secondary action
<Button 
  title="Cancel"
  onPress={handlePress}
  variant="secondary"
/>

// Destructive action
<Button 
  title="Delete"
  onPress={handlePress}
  variant="danger"
/>

// Loading state
<Button 
  title="Saving..."
  onPress={handlePress}
  loading={true}
/>

// Disabled
<Button 
  title="Unavailable"
  onPress={handlePress}
  disabled={true}
/>
```

### Card

```tsx
import { Card } from '../components/ui/Card';

// Default card
<Card>
  <Text>Content here</Text>
</Card>

// Muted background
<Card variant="muted">
  <Text>Subtle background</Text>
</Card>

// Custom styling
<Card style={{ marginBottom: 16, padding: 24 }}>
  <Text>Custom styled</Text>
</Card>
```

### Badge

```tsx
import { Badge } from '../components/ui/Badge';

<Badge label="New" variant="primary" size="small" />
<Badge label="High Risk" variant="danger" />
<Badge label="Success" variant="success" />
<Badge label="Info" variant="info" size="medium" />
```

### Modal

```tsx
import { Modal } from '../components/ui/Modal';
import { Button } from '../components/ui/Button';

const [visible, setVisible] = useState(false);

<Modal
  visible={visible}
  onClose={() => setVisible(false)}
  title="Confirm Action"
  actions={
    <>
      <Button 
        title="Cancel" 
        onPress={() => setVisible(false)} 
        variant="secondary" 
      />
      <Button 
        title="Confirm" 
        onPress={handleConfirm} 
        variant="primary" 
      />
    </>
  }
>
  <Text>Are you sure you want to continue?</Text>
</Modal>
```

### DetectionModal

```tsx
import { DetectionModal } from '../components/detection/DetectionModal';

const [showAlert, setShowAlert] = useState(false);

<DetectionModal
  visible={showAlert}
  onClose={() => setShowAlert(false)}
  detection={{
    type: 'harassment',        // or 'threat', 'hate-speech', 'sexual-content'
    severity: 'high',          // or 'low', 'medium'
    confidence: 0.87,          // 0-1
    message: 'Message content'
  }}
  onEdit={() => {
    // Handle edit action
    setShowAlert(false);
  }}
  onSendAnyway={() => {
    // Handle send anyway
    setShowAlert(false);
  }}
/>
```

### BlurredMessage

```tsx
import { BlurredMessage } from '../components/detection/BlurredMessage';

<BlurredMessage
  message={{
    id: '123',
    sender: 'unknown_user',
    preview: 'Tap to reveal',
    fullMessage: 'The actual message content',
    type: 'threat',
    severity: 'high',
    timestamp: new Date(),
    app: 'Instagram'
  }}
  onReveal={() => console.log('Revealed')}
  onReport={() => console.log('Reported')}
  onBlock={() => console.log('Blocked')}
/>
```

---

## Common Patterns

### Creating a New Screen

```tsx
import React from 'react';
import { View, Text, ScrollView, SafeAreaView } from 'react-native';
import { useAppTheme } from '../contexts/ThemeContext';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';

export default function MyScreen() {
  const { theme } = useAppTheme();
  
  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: theme.colors.background }}>
      <ScrollView contentContainerStyle={{ padding: theme.spacing.lg }}>
        <Text style={{ 
          fontSize: theme.typography.heading,
          color: theme.colors.text,
          fontWeight: '700',
          marginBottom: theme.spacing.md
        }}>
          My Screen Title
        </Text>
        
        <Card>
          <Text style={{ color: theme.colors.text }}>
            Screen content here
          </Text>
        </Card>
      </ScrollView>
    </SafeAreaView>
  );
}
```

### Building a Settings Row

```tsx
import { View, Text, Switch } from 'react-native';
import { useAppTheme } from '../contexts/ThemeContext';

function SettingRow({ label, description, value, onValueChange }) {
  const { theme } = useAppTheme();
  
  return (
    <View style={{
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      paddingVertical: theme.spacing.md,
      borderBottomWidth: 1,
      borderBottomColor: theme.colors.border,
    }}>
      <View style={{ flex: 1, marginRight: theme.spacing.md }}>
        <Text style={{
          fontSize: theme.typography.body,
          fontWeight: '600',
          color: theme.colors.text,
          marginBottom: 2,
        }}>
          {label}
        </Text>
        <Text style={{
          fontSize: theme.typography.caption,
          color: theme.colors.subText,
        }}>
          {description}
        </Text>
      </View>
      <Switch
        value={value}
        onValueChange={onValueChange}
        trackColor={{ 
          false: theme.colors.border, 
          true: theme.colors.primary700 
        }}
        thumbColor={value ? theme.colors.primary : '#f4f3f4'}
      />
    </View>
  );
}
```

### Creating an Incident List Item

```tsx
import { TouchableOpacity, View, Text } from 'react-native';
import { useRouter } from 'expo-router';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { useAppTheme } from '../contexts/ThemeContext';

function IncidentItem({ incident }) {
  const { theme } = useAppTheme();
  const router = useRouter();
  
  return (
    <TouchableOpacity 
      onPress={() => router.push(`/incident/${incident.id}`)}
    >
      <Card style={{ marginBottom: theme.spacing.md }}>
        <View style={{
          flexDirection: 'row',
          justifyContent: 'space-between',
          marginBottom: theme.spacing.sm
        }}>
          <Text style={{
            fontSize: theme.typography.body,
            fontWeight: '600',
            color: theme.colors.text,
          }}>
            {incident.sender}
          </Text>
          <Badge label={incident.type} variant="danger" size="small" />
        </View>
        <Text style={{
          fontSize: theme.typography.body,
          color: theme.colors.subText,
          fontStyle: 'italic'
        }} numberOfLines={2}>
          {incident.message}
        </Text>
      </Card>
    </TouchableOpacity>
  );
}
```

---

## Mock Data

### Using Mock Incidents

```tsx
import { 
  mockIncidents, 
  mockStats, 
  getSeverityColor, 
  getTypeLabel, 
  getRelativeTime 
} from '../data/mockData';

// Get all incidents
const incidents = mockIncidents;

// Filter unread
const unreadIncidents = mockIncidents.filter(i => !i.read);

// Get specific incident
const incident = mockIncidents.find(i => i.id === '1');

// Use helper functions
const severityColor = getSeverityColor('high');  // 'danger'
const typeLabel = getTypeLabel('harassment');     // 'Harassment'
const timeAgo = getRelativeTime(new Date());      // '5m ago'
```

### Mock Stats

```tsx
import { mockStats } from '../data/mockData';

const { safetyScore, messagesScanned, threatsBlocked, activeDays } = mockStats;

console.log(safetyScore);      // 87
console.log(messagesScanned);   // 1247
console.log(threatsBlocked);    // 23
console.log(activeDays);        // 14
```

---

## Troubleshooting

### Theme not available

**Error**: "useAppTheme must be used within a ThemeProvider"

**Solution**: Ensure your component is inside the ThemeProvider (check `app/_layout.tsx`)

### Navigation not working

**Error**: Screen doesn't navigate

**Solution**: 
1. Check route is defined in `app/_layout.tsx`
2. Use correct path format: `/settings` not `settings`
3. For dynamic routes: `/incident/${id}` not `/incident/[id]`

### Colors don't meet contrast

**Issue**: Text is hard to read

**Solution**: Always use:
- `theme.colors.text` (#111827) on yellow or light backgrounds
- White text on `theme.colors.danger` or dark backgrounds
- `theme.colors.subText` for secondary text only on white/light surfaces

### Components not importing

**Error**: Cannot find module

**Solution**: Use correct import paths:
```tsx
// From screen files (app/*)
import { Button } from '../components/ui/Button';
import { useAppTheme } from '../contexts/ThemeContext';

// From component files (components/*)
import { useAppTheme } from '../../contexts/ThemeContext';
```

### Shadows not showing on Android

**Issue**: Shadows don't appear

**Solution**: Use `elevation` property from theme:
```tsx
style={{
  ...theme.shadows.md,  // This includes both shadowProps and elevation
}}
```

---

## Best Practices

1. **Always use theme constants** - Never hardcode colors or spacing
2. **Provide accessibility labels** - Every touchable needs `accessibilityLabel`
3. **Test contrast** - Use dark text on yellow backgrounds
4. **Avoid gradients** - Stick to flat fills
5. **Use SafeAreaView** - Wrap top-level screens
6. **Handle loading states** - Show loading indicators for async actions
7. **Provide feedback** - Use haptics, animations, or alerts for user actions

---

## Next Steps

- Explore [demo.tsx](app/demo.tsx) for interactive examples
- Review [README.md](README.md) for full documentation
- Check [mockData.ts](data/mockData.ts) to understand data structures
- Examine existing screens for implementation patterns

---

**Need help?** Check the component source files for inline documentation and TypeScript types.
