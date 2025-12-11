/**
 * Settings Screen - Configuration and preferences
 * Detection sensitivity, privacy toggles, and app settings
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  SafeAreaView,
  Switch,
  TouchableOpacity,
} from 'react-native';
import { useAppTheme } from '../contexts/ThemeContext';
import { Card } from '../components/ui/Card';

export default function SettingsScreen() {
  const { theme } = useAppTheme();
  
  // Settings state
  const [notifications, setNotifications] = useState(true);
  const [autoBlock, setAutoBlock] = useState(false);
  const [anonymousReporting, setAnonymousReporting] = useState(true);
  const [sensitivity, setSensitivity] = useState<'low' | 'medium' | 'high'>('medium');

  const styles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: theme.colors.background,
    },
    scrollContent: {
      padding: theme.spacing.lg,
    },
    header: {
      marginBottom: theme.spacing.lg,
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
      marginBottom: theme.spacing.lg,
    },
    sectionTitle: {
      fontSize: theme.typography.subheading,
      fontWeight: '600',
      color: theme.colors.text,
      marginBottom: theme.spacing.md,
    },
    settingRow: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      paddingVertical: theme.spacing.md,
      borderBottomWidth: 1,
      borderBottomColor: theme.colors.border,
    },
    settingInfo: {
      flex: 1,
      marginRight: theme.spacing.md,
    },
    settingLabel: {
      fontSize: theme.typography.body,
      fontWeight: '600',
      color: theme.colors.text,
      marginBottom: 2,
    },
    settingDescription: {
      fontSize: theme.typography.caption,
      color: theme.colors.subText,
      lineHeight: 16,
    },
    sensitivityOptions: {
      flexDirection: 'row',
      gap: theme.spacing.sm,
    },
    sensitivityButton: {
      flex: 1,
      paddingVertical: theme.spacing.md,
      paddingHorizontal: theme.spacing.md,
      borderRadius: theme.radii.md,
      borderWidth: 2,
      alignItems: 'center',
    },
    sensitivityButtonActive: {
      backgroundColor: theme.colors.primary,
      borderColor: theme.colors.primary700,
    },
    sensitivityButtonInactive: {
      backgroundColor: 'transparent',
      borderColor: theme.colors.border,
    },
    sensitivityLabel: {
      fontSize: theme.typography.body,
      fontWeight: '600',
    },
    sensitivityLabelActive: {
      color: theme.colors.text,
    },
    sensitivityLabelInactive: {
      color: theme.colors.subText,
    },
    aboutCard: {
      marginTop: theme.spacing.md,
    },
    aboutText: {
      fontSize: theme.typography.body,
      color: theme.colors.subText,
      lineHeight: 20,
      marginBottom: theme.spacing.sm,
    },
    version: {
      fontSize: theme.typography.caption,
      color: theme.colors.subText,
      textAlign: 'center',
      marginTop: theme.spacing.md,
    },
  });

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text style={styles.title}>Settings</Text>
          <Text style={styles.subtitle}>Configure your protection</Text>
        </View>

        {/* Detection Settings */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Detection</Text>
          <Card>
            <View>
              <Text style={styles.settingLabel}>Sensitivity</Text>
              <Text style={styles.settingDescription}>
                Higher sensitivity catches more potential threats but may have more false positives
              </Text>
            </View>
            <View style={[styles.sensitivityOptions, { marginTop: theme.spacing.md }]}>
              {(['low', 'medium', 'high'] as const).map((level) => (
                <TouchableOpacity
                  key={level}
                  style={[
                    styles.sensitivityButton,
                    sensitivity === level
                      ? styles.sensitivityButtonActive
                      : styles.sensitivityButtonInactive,
                  ]}
                  onPress={() => setSensitivity(level)}
                  accessibilityRole="button"
                  accessibilityLabel={`Set sensitivity to ${level}`}
                  accessibilityState={{ selected: sensitivity === level }}
                >
                  <Text
                    style={[
                      styles.sensitivityLabel,
                      sensitivity === level
                        ? styles.sensitivityLabelActive
                        : styles.sensitivityLabelInactive,
                    ]}
                  >
                    {level.charAt(0).toUpperCase() + level.slice(1)}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </Card>
        </View>

        {/* Notification Settings */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Notifications</Text>
          <Card>
            <View style={styles.settingRow}>
              <View style={styles.settingInfo}>
                <Text style={styles.settingLabel}>Enable Notifications</Text>
                <Text style={styles.settingDescription}>
                  Get alerts when harmful content is detected
                </Text>
              </View>
              <Switch
                value={notifications}
                onValueChange={setNotifications}
                trackColor={{ false: theme.colors.border, true: theme.colors.primary700 }}
                thumbColor={notifications ? theme.colors.primary : '#f4f3f4'}
                accessibilityLabel="Toggle notifications"
              />
            </View>
          </Card>
        </View>

        {/* Privacy Settings */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Privacy & Safety</Text>
          <Card>
            <View style={styles.settingRow}>
              <View style={styles.settingInfo}>
                <Text style={styles.settingLabel}>Auto-Block High Threats</Text>
                <Text style={styles.settingDescription}>
                  Automatically block senders of severe threats
                </Text>
              </View>
              <Switch
                value={autoBlock}
                onValueChange={setAutoBlock}
                trackColor={{ false: theme.colors.border, true: theme.colors.primary700 }}
                thumbColor={autoBlock ? theme.colors.primary : '#f4f3f4'}
                accessibilityLabel="Toggle auto-block"
              />
            </View>
            <View style={[styles.settingRow, { borderBottomWidth: 0 }]}>
              <View style={styles.settingInfo}>
                <Text style={styles.settingLabel}>Anonymous Reporting</Text>
                <Text style={styles.settingDescription}>
                  Report incidents without sharing your identity
                </Text>
              </View>
              <Switch
                value={anonymousReporting}
                onValueChange={setAnonymousReporting}
                trackColor={{ false: theme.colors.border, true: theme.colors.primary700 }}
                thumbColor={anonymousReporting ? theme.colors.primary : '#f4f3f4'}
                accessibilityLabel="Toggle anonymous reporting"
              />
            </View>
          </Card>
        </View>

        {/* About Section */}
        <Card style={styles.aboutCard} variant="muted">
          <Text style={styles.aboutText}>
            🛡️ SentinelAI uses advanced AI to detect cyberbullying and harmful content in real-time,
            helping you stay safe online.
          </Text>
          <Text style={styles.aboutText}>
            🔒 All processing happens on your device. Your messages are never sent to external servers.
          </Text>
        </Card>

        <Text style={styles.version}>Version 1.0.0</Text>
      </ScrollView>
    </SafeAreaView>
  );
}
