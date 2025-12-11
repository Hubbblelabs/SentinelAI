/**
 * DetectionModal - Alert dialog when harmful content is detected
 * Provides options to edit, send anyway, or block sender
 */

import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useAppTheme } from '../../contexts/ThemeContext';
import { Modal } from '../ui/Modal';
import { Button } from '../ui/Button';
import { Badge } from '../ui/Badge';

export interface DetectionResult {
  type: 'threat' | 'harassment' | 'hate-speech' | 'sexual-content';
  severity: 'low' | 'medium' | 'high';
  confidence: number;
  message: string;
}

interface DetectionModalProps {
  visible: boolean;
  onClose: () => void;
  detection: DetectionResult;
  onEdit: () => void;
  onSendAnyway: () => void;
  onBlock?: () => void;
}

export const DetectionModal: React.FC<DetectionModalProps> = ({
  visible,
  onClose,
  detection,
  onEdit,
  onSendAnyway,
  onBlock,
}) => {
  const { theme } = useAppTheme();

  const getTypeLabel = (): string => {
    switch (detection.type) {
      case 'threat':
        return 'Potential Threat';
      case 'harassment':
        return 'Harassment Detected';
      case 'hate-speech':
        return 'Hate Speech';
      case 'sexual-content':
        return 'Inappropriate Content';
    }
  };

  const getMessage = (): string => {
    switch (detection.type) {
      case 'threat':
        return 'This message contains threatening language that may hurt someone.';
      case 'harassment':
        return 'This message appears to be harassing or bullying in nature.';
      case 'hate-speech':
        return 'This message contains hate speech or discriminatory language.';
      case 'sexual-content':
        return 'This message contains sexually explicit or inappropriate content.';
    }
  };

  const styles = StyleSheet.create({
    content: {
      marginBottom: theme.spacing.md,
    },
    badgeContainer: {
      marginBottom: theme.spacing.md,
    },
    description: {
      fontSize: theme.typography.body,
      color: theme.colors.subText,
      lineHeight: 20,
      marginBottom: theme.spacing.md,
    },
    messagePreview: {
      backgroundColor: theme.colors.mutedSurface,
      padding: theme.spacing.md,
      borderRadius: theme.radii.sm,
      borderLeftWidth: 3,
      borderLeftColor: theme.colors.danger,
      marginBottom: theme.spacing.md,
    },
    messageText: {
      fontSize: theme.typography.body,
      color: theme.colors.text,
      fontStyle: 'italic',
    },
    suggestion: {
      fontSize: theme.typography.caption,
      color: theme.colors.subText,
      fontStyle: 'italic',
    },
    actions: {
      gap: theme.spacing.sm,
    },
  });

  const getSeverityVariant = (): 'success' | 'info' | 'danger' => {
    switch (detection.severity) {
      case 'low':
        return 'success';
      case 'medium':
        return 'info';
      case 'high':
        return 'danger';
    }
  };

  return (
    <Modal
      visible={visible}
      onClose={onClose}
      title={getTypeLabel()}
      testID="detection-modal"
    >
      <View style={styles.content}>
        <View style={styles.badgeContainer}>
          <Badge
            label={`${detection.severity.toUpperCase()} severity - ${Math.round(detection.confidence * 100)}% confident`}
            variant={getSeverityVariant()}
            size="small"
          />
        </View>

        <Text style={styles.description}>{getMessage()}</Text>

        <View style={styles.messagePreview}>
          <Text style={styles.messageText} numberOfLines={3}>
            {detection.message}
          </Text>
        </View>

        <Text style={styles.suggestion}>
          💡 Consider rewording your message to be more respectful and kind.
        </Text>
      </View>

      <View style={styles.actions}>
        <Button
          title="Edit Message"
          onPress={() => {
            onEdit();
            onClose();
          }}
          variant="primary"
          accessibilityLabel="Edit message to make it more appropriate"
        />
        <Button
          title="Send Anyway"
          onPress={() => {
            onSendAnyway();
            onClose();
          }}
          variant="secondary"
          accessibilityLabel="Send message without changes"
        />
        {onBlock && (
          <Button
            title="Cancel"
            onPress={onClose}
            variant="secondary"
            accessibilityLabel="Cancel and don't send"
          />
        )}
      </View>
    </Modal>
  );
};
