/**
 * Card Component - Surface container with consistent styling
 * Flat design with soft shadows
 */

import React, { ReactNode } from 'react';
import { View, StyleSheet, ViewStyle } from 'react-native';
import { useAppTheme } from '../../contexts/ThemeContext';

interface CardProps {
  children: ReactNode;
  variant?: 'default' | 'muted';
  style?: ViewStyle;
  testID?: string;
}

export const Card: React.FC<CardProps> = ({
  children,
  variant = 'default',
  style,
  testID,
}) => {
  const { theme } = useAppTheme();

  const cardStyle: ViewStyle = {
    backgroundColor: variant === 'muted' ? theme.colors.mutedSurface : theme.colors.surface,
    borderRadius: theme.radii.md,
    padding: theme.spacing.md,
    ...theme.shadows.md,
  };

  return (
    <View style={[cardStyle, style]} testID={testID}>
      {children}
    </View>
  );
};
