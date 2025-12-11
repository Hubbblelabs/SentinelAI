import { Tabs } from 'expo-router';
import { useRouter } from 'expo-router';
import React from 'react';

import { HapticTab } from '@/components/haptic-tab';
import { IconSymbol } from '@/components/ui/icon-symbol';
import { useAppTheme } from '../../contexts/ThemeContext';

export default function TabLayout() {
  const { theme } = useAppTheme();
  const router = useRouter();

  return (
    <Tabs
      screenOptions={{
        tabBarActiveTintColor: theme.colors.primary900,
        tabBarInactiveTintColor: theme.colors.subText,
        tabBarStyle: {
          backgroundColor: theme.colors.surface,
          borderTopColor: theme.colors.border,
        },
        headerShown: false,
        tabBarButton: HapticTab,
      }}>
      <Tabs.Screen
        name="index"
        options={{
          title: 'Dashboard',
          tabBarIcon: ({ color }) => <IconSymbol size={28} name="shield.fill" color={color} />,
        }}
      />
      <Tabs.Screen
        name="explore"
        options={{
          title: 'Settings',
          tabBarIcon: ({ color }) => <IconSymbol size={28} name="gearshape.fill" color={color} />,
          tabBarButton: (props) => (
            <HapticTab
              {...props}
              onPress={() => router.push('/settings')}
            />
          ),
        }}
      />
    </Tabs>
  );
}
