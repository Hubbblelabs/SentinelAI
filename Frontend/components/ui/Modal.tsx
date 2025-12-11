/**
 * Modal Component - Accessible dialog overlay
 * Used for detection alerts and confirmations
 */

import React, { ReactNode } from 'react';
import {
  Modal as RNModal,
  View,
  Text,
  TouchableOpacity,
  ViewStyle,
  Dimensions,
  Platform,
  AccessibilityInfo,
} from 'react-native';
import { useAppTheme } from '../../contexts/ThemeContext';

interface ModalProps {
  visible: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
  actions?: ReactNode;
  testID?: string;
}

const { width: SCREEN_WIDTH } = Dimensions.get('window');

export const Modal: React.FC<ModalProps> = ({
  visible,
  onClose,
  title,
  children,
  actions,
  testID,
}) => {
  const { theme } = useAppTheme();

  React.useEffect(() => {
    if (visible && Platform.OS !== 'web') {
      // Announce modal opening for screen readers
      AccessibilityInfo.announceForAccessibility(`Dialog opened: ${title}`);
    }
  }, [visible, title]);

  const overlayStyle: ViewStyle = {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: theme.spacing.lg,
  };

  const modalStyle: ViewStyle = {
    backgroundColor: theme.colors.surface,
    borderRadius: theme.radii.lg,
    padding: theme.spacing.lg,
    width: Math.min(SCREEN_WIDTH - theme.spacing.xl * 2, 400),
    maxWidth: '100%',
    ...theme.shadows.lg,
  };

  const titleStyle = {
    fontSize: theme.typography.heading,
    fontWeight: '700' as const,
    color: theme.colors.text,
    marginBottom: theme.spacing.md,
  };

  return (
    <RNModal
      visible={visible}
      transparent
      animationType="fade"
      onRequestClose={onClose}
      accessibilityViewIsModal
      testID={testID}
    >
      <TouchableOpacity
        style={overlayStyle}
        activeOpacity={1}
        onPress={onClose}
        accessibilityRole="button"
        accessibilityLabel="Close dialog"
      >
        <TouchableOpacity
          activeOpacity={1}
          onPress={(e) => e.stopPropagation()}
          style={modalStyle}
          accessible={false}
        >
          <Text style={titleStyle} accessibilityRole="header">
            {title}
          </Text>
          <View style={{ marginBottom: actions ? theme.spacing.lg : 0 }}>
            {children}
          </View>
          {actions && (
            <View
              style={{
                flexDirection: 'row',
                justifyContent: 'flex-end',
                gap: theme.spacing.sm,
              }}
            >
              {actions}
            </View>
          )}
        </TouchableOpacity>
      </TouchableOpacity>
    </RNModal>
  );
};
