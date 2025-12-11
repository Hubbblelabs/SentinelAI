/**
 * Demo Screen - Showcases detection components
 * For testing and demonstration purposes
 */

import React, { useState } from 'react';
import { View, Text, StyleSheet, ScrollView, SafeAreaView } from 'react-native';
import { useAppTheme } from '../contexts/ThemeContext';
import { Button } from '../components/ui/Button';
import { Card } from '../components/ui/Card';
import { DetectionModal } from '../components/detection/DetectionModal';
import { BlurredMessage, MessageData } from '../components/detection/BlurredMessage';

export default function DemoScreen() {
  const { theme } = useAppTheme();
  const [showDetectionModal, setShowDetectionModal] = useState(false);

  const sampleMessage: MessageData = {
    id: 'demo-1',
    sender: 'unknown_user_123',
    preview: 'This message was flagged',
    fullMessage: 'You\'re worthless and nobody likes you. Just give up already.',
    type: 'harassment',
    severity: 'high',
    timestamp: new Date(),
    app: 'Instagram',
  };

  const styles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: theme.colors.background,
    },
    scrollContent: {
      padding: theme.spacing.lg,
    },
    header: {
      marginBottom: theme.spacing.xl,
    },
    title: {
      fontSize: 28,
      fontWeight: '700',
      color: theme.colors.text,
      marginBottom: theme.spacing.sm,
    },
    subtitle: {
      fontSize: theme.typography.subheading,
      color: theme.colors.subText,
    },
    section: {
      marginBottom: theme.spacing.xl,
    },
    sectionTitle: {
      fontSize: theme.typography.heading,
      fontWeight: '600',
      color: theme.colors.text,
      marginBottom: theme.spacing.md,
    },
    description: {
      fontSize: theme.typography.body,
      color: theme.colors.subText,
      lineHeight: 20,
      marginBottom: theme.spacing.md,
    },
  });

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text style={styles.title}>Detection Demo</Text>
          <Text style={styles.subtitle}>Interactive component showcase</Text>
        </View>

        {/* Detection Modal Demo */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Outgoing Message Detection</Text>
          <Text style={styles.description}>
            When you're about to send a potentially harmful message, this alert appears to give
            you a chance to reconsider.
          </Text>
          <Button
            title="Trigger Detection Alert"
            onPress={() => setShowDetectionModal(true)}
            variant="primary"
          />
        </View>

        {/* Blurred Message Demo */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Incoming Message Protection</Text>
          <Text style={styles.description}>
            Harmful incoming messages are blurred until you choose to reveal them, protecting
            your mental wellbeing.
          </Text>
          <BlurredMessage
            message={sampleMessage}
            onReveal={() => console.log('Message revealed')}
            onReport={() => console.log('Message reported')}
            onBlock={() => console.log('Sender blocked')}
          />
        </View>

        {/* UI Component Showcase */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>UI Components</Text>
          <Text style={styles.description}>Primary design elements:</Text>
          
          <Card style={{ marginBottom: theme.spacing.md }}>
            <Text style={{ fontSize: theme.typography.body, color: theme.colors.text }}>
              This is a default Card component with surface background and subtle shadow.
            </Text>
          </Card>

          <Card variant="muted" style={{ marginBottom: theme.spacing.md }}>
            <Text style={{ fontSize: theme.typography.body, color: theme.colors.text }}>
              This is a muted Card with the yellow-tinted background color.
            </Text>
          </Card>

          <View style={{ gap: theme.spacing.sm }}>
            <Button title="Primary Button" onPress={() => {}} variant="primary" />
            <Button title="Secondary Button" onPress={() => {}} variant="secondary" />
            <Button title="Danger Button" onPress={() => {}} variant="danger" />
            <Button title="Disabled Button" onPress={() => {}} variant="primary" disabled />
          </View>
        </View>

        {/* Info */}
        <Card variant="muted">
          <Text style={styles.description}>
            💡 This demo screen showcases the core detection components and UI primitives. All
            components follow the flat yellow design system with WCAG AA contrast compliance.
          </Text>
        </Card>
      </ScrollView>

      <DetectionModal
        visible={showDetectionModal}
        onClose={() => setShowDetectionModal(false)}
        detection={{
          type: 'harassment',
          severity: 'high',
          confidence: 0.87,
          message: 'You\'re so stupid, nobody wants to be around you.',
        }}
        onEdit={() => {
          console.log('Edit message');
          setShowDetectionModal(false);
        }}
        onSendAnyway={() => {
          console.log('Send anyway');
          setShowDetectionModal(false);
        }}
      />
    </SafeAreaView>
  );
}
