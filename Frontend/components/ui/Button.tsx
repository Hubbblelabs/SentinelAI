/**
 * Button Component - Primary and Secondary variants
 * Follows flat yellow design system with WCAG AA contrast
 */

import React from 'react';
import {
  TouchableOpacity,
  Text,
  ActivityIndicator,
  ViewStyle,
  TextStyle,
} from 'react-native';
import { useAppTheme } from '../../contexts/ThemeContext';

interface ButtonProps {
  title: string;
  onPress: () => void;
  variant?: 'primary' | 'secondary' | 'danger';
  disabled?: boolean;
  loading?: boolean;
  accessibilityLabel?: string;
  style?: ViewStyle;
  testID?: string;
}

export const Button: React.FC<ButtonProps> = ({
  title,
  onPress,
  variant = 'primary',
  disabled = false,
  loading = false,
  accessibilityLabel,
  style,
  testID,
}) => {
  const { theme } = useAppTheme();

  const getButtonStyle = (): ViewStyle => {
    const baseStyle: ViewStyle = {
      paddingVertical: theme.spacing.md,
      paddingHorizontal: theme.spacing.lg,
      borderRadius: theme.radii.lg,
      alignItems: 'center',
      justifyContent: 'center',
      minHeight: 48, // Accessible tap target
      opacity: disabled || loading ? 0.5 : 1,
    };

    switch (variant) {
      case 'primary':
        return {
          ...baseStyle,
          backgroundColor: theme.colors.primary,
        };
      case 'secondary':
        return {
          ...baseStyle,
          backgroundColor: 'transparent',
          borderWidth: 1,
          borderColor: theme.colors.primary700,
        };
      case 'danger':
        return {
          ...baseStyle,
          backgroundColor: theme.colors.danger,
        };
      default:
        return baseStyle;
    }
  };

  const getTextStyle = (): TextStyle => {
    const baseStyle: TextStyle = {
      fontSize: theme.typography.subheading,
      fontWeight: '600',
    };

    switch (variant) {
      case 'primary':
        return {
          ...baseStyle,
          color: theme.colors.text,
        };
      case 'secondary':
        return {
          ...baseStyle,
          color: theme.colors.primary900,
        };
      case 'danger':
        return {
          ...baseStyle,
          color: '#FFFFFF',
        };
      default:
        return baseStyle;
    }
  };

  return (
    <TouchableOpacity
      style={[getButtonStyle(), style]}
      onPress={onPress}
      disabled={disabled || loading}
      accessibilityRole="button"
      accessibilityLabel={accessibilityLabel || title}
      accessibilityState={{ disabled: disabled || loading }}
      testID={testID}
    >
      {loading ? (
        <ActivityIndicator
          color={variant === 'secondary' ? theme.colors.primary900 : theme.colors.text}
        />
      ) : (
        <Text style={getTextStyle()}>{title}</Text>
      )}
    </TouchableOpacity>
  );
};
