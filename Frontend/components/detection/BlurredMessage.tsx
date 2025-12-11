/**
 * BlurredMessage - Incoming message card with blur effect
 * Protects user from viewing harmful content until they choose to reveal
 */

import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Platform } from 'react-native';
import { useAppTheme } from '../../contexts/ThemeContext';
import { Card } from '../ui/Card';
import { Badge } from '../ui/Badge';
import { Button } from '../ui/Button';

export interface MessageData {
  id: string;
  sender: string;
  preview: string;
  fullMessage: string;
  type: 'threat' | 'harassment' | 'hate-speech' | 'sexual-content';
  severity: 'low' | 'medium' | 'high';
  timestamp: Date;
  app: string;
}

interface BlurredMessageProps {
  message: MessageData;
  onReveal?: () => void;
  onReport?: () => void;
  onBlock?: () => void;
}

export const BlurredMessage: React.FC<BlurredMessageProps> = ({
  message,
  onReveal,
  onReport,
  onBlock,
}) => {
  const { theme } = useAppTheme();
  const [isRevealed, setIsRevealed] = useState(false);

  const handleReveal = () => {
    setIsRevealed(true);
    onReveal?.();
  };

  const getTypeLabel = (): string => {
    switch (message.type) {
      case 'threat':
        return 'Threat';
      case 'harassment':
        return 'Harassment';
      case 'hate-speech':
        return 'Hate Speech';
      case 'sexual-content':
        return 'Inappropriate';
    }
  };

  const getSeverityColor = (): 'success' | 'info' | 'danger' => {
    switch (message.severity) {
      case 'low':
        return 'success';
      case 'medium':
        return 'info';
      case 'high':
        return 'danger';
    }
  };

  const styles = StyleSheet.create({
    container: {
      marginBottom: theme.spacing.md,
    },
    header: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: theme.spacing.sm,
    },
    senderInfo: {
      flex: 1,
    },
    sender: {
      fontSize: theme.typography.subheading,
      fontWeight: '600',
      color: theme.colors.text,
    },
    app: {
      fontSize: theme.typography.caption,
      color: theme.colors.subText,
      marginTop: 2,
    },
    messageContainer: {
      position: 'relative',
      minHeight: 80,
      justifyContent: 'center',
      alignItems: 'center',
      backgroundColor: theme.colors.mutedSurface,
      borderRadius: theme.radii.sm,
      padding: theme.spacing.md,
      marginBottom: theme.spacing.md,
    },
    blurredText: {
      fontSize: theme.typography.body,
      color: theme.colors.subText,
      textAlign: 'center',
      ...(Platform.OS === 'web' ? { filter: 'blur(8px)' } : {}),
      opacity: isRevealed ? 1 : 0.3,
    },
    revealOverlay: {
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(255, 255, 255, 0.95)',
      justifyContent: 'center',
      alignItems: 'center',
      borderRadius: theme.radii.sm,
    },
    revealIcon: {
      fontSize: 32,
      marginBottom: theme.spacing.sm,
    },
    revealText: {
      fontSize: theme.typography.body,
      color: theme.colors.text,
      fontWeight: '600',
    },
    revealHint: {
      fontSize: theme.typography.caption,
      color: theme.colors.subText,
      marginTop: theme.spacing.xs,
    },
    revealedMessage: {
      fontSize: theme.typography.body,
      color: theme.colors.text,
      lineHeight: 20,
    },
    actions: {
      flexDirection: 'row',
      gap: theme.spacing.sm,
    },
    actionButton: {
      flex: 1,
    },
  });

  return (
    <Card style={styles.container} testID={`blurred-message-${message.id}`}>
      <View style={styles.header}>
        <View style={styles.senderInfo}>
          <Text style={styles.sender}>{message.sender}</Text>
          <Text style={styles.app}>{message.app}</Text>
        </View>
        <Badge
          label={getTypeLabel()}
          variant={getSeverityColor()}
          size="small"
        />
      </View>

      <View style={styles.messageContainer}>
        <Text style={styles.blurredText}>{message.fullMessage}</Text>
        
        {!isRevealed && (
          <TouchableOpacity
            style={styles.revealOverlay}
            onPress={handleReveal}
            accessibilityRole="button"
            accessibilityLabel="Tap to reveal message content"
          >
            <Text style={styles.revealIcon}>👁️</Text>
            <Text style={styles.revealText}>Tap to Reveal</Text>
            <Text style={styles.revealHint}>This message may be harmful</Text>
          </TouchableOpacity>
        )}
      </View>

      {isRevealed && (
        <View style={styles.actions}>
          {onReport && (
            <Button
              title="Report"
              onPress={onReport}
              variant="danger"
              style={styles.actionButton}
              accessibilityLabel="Report this message"
            />
          )}
          {onBlock && (
            <Button
              title="Block"
              onPress={onBlock}
              variant="secondary"
              style={styles.actionButton}
              accessibilityLabel="Block sender"
            />
          )}
        </View>
      )}
    </Card>
  );
};
