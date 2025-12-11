/**
 * Badge Component - Small status indicators and labels
 * Semantic color variants for different states
 */

import React from 'react';
import { View, Text, ViewStyle, TextStyle } from 'react-native';
import { useAppTheme } from '../../contexts/ThemeContext';

interface BadgeProps {
  label: string;
  variant?: 'primary' | 'success' | 'danger' | 'info' | 'neutral';
  size?: 'small' | 'medium';
  style?: ViewStyle;
  testID?: string;
}

export const Badge: React.FC<BadgeProps> = ({
  label,
  variant = 'primary',
  size = 'medium',
  style,
  testID,
}) => {
  const { theme } = useAppTheme();

  const getBackgroundColor = (): string => {
    switch (variant) {
      case 'primary':
        return theme.colors.primary;
      case 'success':
        return theme.colors.success;
      case 'danger':
        return theme.colors.danger;
      case 'info':
        return theme.colors.info;
      case 'neutral':
        return theme.colors.mutedSurface;
      default:
        return theme.colors.primary;
    }
  };

  const getTextColor = (): string => {
    switch (variant) {
      case 'primary':
      case 'neutral':
        return theme.colors.text;
      default:
        return '#FFFFFF';
    }
  };

  const badgeStyle: ViewStyle = {
    backgroundColor: getBackgroundColor(),
    borderRadius: theme.radii.lg,
    paddingVertical: size === 'small' ? theme.spacing.xs : theme.spacing.sm,
    paddingHorizontal: size === 'small' ? theme.spacing.sm : theme.spacing.md,
    alignSelf: 'flex-start',
  };

  const textStyle: TextStyle = {
    color: getTextColor(),
    fontSize: size === 'small' ? theme.typography.caption : theme.typography.body,
    fontWeight: '600',
  };

  return (
    <View style={[badgeStyle, style]} testID={testID}>
      <Text style={textStyle}>{label}</Text>
    </View>
  );
};
