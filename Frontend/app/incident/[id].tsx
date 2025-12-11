/**
 * Incident Details Screen - Full view of a detected incident
 * Shows complete message, metadata, and action options
 */

import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  SafeAreaView,
  TouchableOpacity,
  Alert,
} from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { useAppTheme } from '../../contexts/ThemeContext';
import { Card } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import { Button } from '../../components/ui/Button';
import { mockIncidents, getSeverityColor, getTypeLabel } from '../../data/mockData';

export default function IncidentDetailsScreen() {
  const { theme } = useAppTheme();
  const router = useRouter();
  const params = useLocalSearchParams();
  const incident = mockIncidents.find(i => i.id === params.id);

  if (!incident) {
    return (
      <SafeAreaView style={[styles(theme).container]}>
        <View style={styles(theme).errorContainer}>
          <Text style={styles(theme).errorText}>Incident not found</Text>
          <Button title="Go Back" onPress={() => router.back()} variant="primary" />
        </View>
      </SafeAreaView>
    );
  }

  const handleBlock = () => {
    Alert.alert(
      'Block Sender',
      `Are you sure you want to block ${incident.sender}?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Block',
          style: 'destructive',
          onPress: () => {
            Alert.alert('Success', 'Sender has been blocked');
            router.back();
          },
        },
      ]
    );
  };

  const handleReport = () => {
    Alert.alert(
      'Report Incident',
      'This incident will be reported to your trusted contacts.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Report',
          onPress: () => {
            Alert.alert('Success', 'Incident has been reported');
          },
        },
      ]
    );
  };

  const handleDelete = () => {
    Alert.alert(
      'Delete Incident',
      'Are you sure you want to remove this incident from your history?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: () => {
            Alert.alert('Success', 'Incident has been deleted');
            router.back();
          },
        },
      ]
    );
  };

  const formatDate = (date: Date) => {
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
    });
  };

  return (
    <SafeAreaView style={styles(theme).container}>
      <ScrollView contentContainerStyle={styles(theme).scrollContent}>
        {/* Header */}
        <View style={styles(theme).header}>
          <TouchableOpacity
            onPress={() => router.back()}
            accessibilityRole="button"
            accessibilityLabel="Go back"
          >
            <Text style={styles(theme).backButton}>← Back</Text>
          </TouchableOpacity>
          <Text style={styles(theme).title}>Incident Details</Text>
        </View>

        {/* Incident Type & Severity */}
        <Card style={styles(theme).section}>
          <View style={styles(theme).badgeRow}>
            <Badge
              label={getTypeLabel(incident.type)}
              variant={getSeverityColor(incident.severity)}
            />
            <Badge label={`${incident.severity} severity`} variant="neutral" />
          </View>
        </Card>

        {/* Message Content */}
        <Card style={styles(theme).section}>
          <Text style={styles(theme).sectionTitle}>Message</Text>
          <View style={styles(theme).messageBox}>
            <Text style={styles(theme).messageText}>{incident.message}</Text>
          </View>
        </Card>

        {/* Metadata */}
        <Card style={styles(theme).section}>
          <Text style={styles(theme).sectionTitle}>Details</Text>
          <View style={styles(theme).metadataRow}>
            <Text style={styles(theme).metadataLabel}>Sender</Text>
            <Text style={styles(theme).metadataValue}>{incident.sender}</Text>
          </View>
          <View style={styles(theme).metadataRow}>
            <Text style={styles(theme).metadataLabel}>App</Text>
            <Text style={styles(theme).metadataValue}>{incident.app}</Text>
          </View>
          <View style={styles(theme).metadataRow}>
            <Text style={styles(theme).metadataLabel}>Timestamp</Text>
            <Text style={styles(theme).metadataValue}>
              {formatDate(incident.timestamp)}
            </Text>
          </View>
        </Card>

        {/* Actions */}
        <View style={styles(theme).actions}>
          <Button
            title="Report to Trusted Contact"
            onPress={handleReport}
            variant="primary"
          />
          <Button
            title="Block Sender"
            onPress={handleBlock}
            variant="danger"
          />
          <Button
            title="Delete Incident"
            onPress={handleDelete}
            variant="secondary"
          />
        </View>

        {/* Info Box */}
        <Card style={styles(theme).infoBox} variant="muted">
          <Text style={styles(theme).infoText}>
            💡 You can configure detection sensitivity and notification preferences in Settings.
          </Text>
        </Card>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = (theme: any) =>
  StyleSheet.create({
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
    backButton: {
      fontSize: theme.typography.subheading,
      color: theme.colors.primary900,
      fontWeight: '600',
      marginBottom: theme.spacing.sm,
    },
    title: {
      fontSize: 28,
      fontWeight: '700',
      color: theme.colors.text,
    },
    section: {
      marginBottom: theme.spacing.md,
    },
    badgeRow: {
      flexDirection: 'row',
      gap: theme.spacing.sm,
      flexWrap: 'wrap',
    },
    sectionTitle: {
      fontSize: theme.typography.subheading,
      fontWeight: '600',
      color: theme.colors.text,
      marginBottom: theme.spacing.sm,
    },
    messageBox: {
      backgroundColor: theme.colors.mutedSurface,
      padding: theme.spacing.md,
      borderRadius: theme.radii.sm,
      borderLeftWidth: 3,
      borderLeftColor: theme.colors.danger,
    },
    messageText: {
      fontSize: theme.typography.body,
      color: theme.colors.text,
      lineHeight: 20,
    },
    metadataRow: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      paddingVertical: theme.spacing.sm,
      borderBottomWidth: 1,
      borderBottomColor: theme.colors.border,
    },
    metadataLabel: {
      fontSize: theme.typography.body,
      color: theme.colors.subText,
    },
    metadataValue: {
      fontSize: theme.typography.body,
      color: theme.colors.text,
      fontWeight: '600',
    },
    actions: {
      gap: theme.spacing.md,
      marginBottom: theme.spacing.lg,
    },
    infoBox: {
      marginTop: theme.spacing.md,
    },
    infoText: {
      fontSize: theme.typography.body,
      color: theme.colors.subText,
      lineHeight: 20,
    },
    errorContainer: {
      flex: 1,
      justifyContent: 'center',
      alignItems: 'center',
      padding: theme.spacing.xl,
    },
    errorText: {
      fontSize: theme.typography.heading,
      color: theme.colors.text,
      marginBottom: theme.spacing.lg,
    },
  });
