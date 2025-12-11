/**
 * Home Dashboard - Main screen showing safety metrics and recent incidents
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  SafeAreaView,
  RefreshControl,
} from 'react-native';
import { useRouter } from 'expo-router';
import { useAppTheme } from '../../contexts/ThemeContext';
import { Card } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import { Button } from '../../components/ui/Button';
import { mockIncidents, mockStats, getSeverityColor, getTypeLabel, getRelativeTime } from '../../data/mockData';

export default function HomeScreen() {
  const { theme } = useAppTheme();
  const router = useRouter();
  const [refreshing, setRefreshing] = useState(false);

  const onRefresh = React.useCallback(() => {
    setRefreshing(true);
    // Simulate refresh
    setTimeout(() => setRefreshing(false), 1000);
  }, []);

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
    safetyScoreCard: {
      alignItems: 'center',
      marginBottom: theme.spacing.lg,
    },
    scoreCircle: {
      width: 120,
      height: 120,
      borderRadius: 60,
      backgroundColor: theme.colors.primary,
      justifyContent: 'center',
      alignItems: 'center',
      marginBottom: theme.spacing.md,
      ...theme.shadows.md,
    },
    scoreText: {
      fontSize: 48,
      fontWeight: '700',
      color: theme.colors.text,
    },
    scoreLabel: {
      fontSize: theme.typography.body,
      color: theme.colors.text,
      fontWeight: '600',
    },
    statsGrid: {
      flexDirection: 'row',
      flexWrap: 'wrap',
      gap: theme.spacing.md,
      marginBottom: theme.spacing.xl,
    },
    statCard: {
      flex: 1,
      minWidth: '47%',
      alignItems: 'center',
      padding: theme.spacing.md,
    },
    statValue: {
      fontSize: 24,
      fontWeight: '700',
      color: theme.colors.text,
      marginBottom: theme.spacing.xs,
    },
    statLabel: {
      fontSize: theme.typography.caption,
      color: theme.colors.subText,
      textAlign: 'center',
    },
    sectionHeader: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: theme.spacing.md,
    },
    sectionTitle: {
      fontSize: theme.typography.heading,
      fontWeight: '700',
      color: theme.colors.text,
    },
    incidentCard: {
      marginBottom: theme.spacing.md,
    },
    incidentHeader: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'flex-start',
      marginBottom: theme.spacing.sm,
    },
    incidentMeta: {
      flex: 1,
    },
    incidentSender: {
      fontSize: theme.typography.body,
      fontWeight: '600',
      color: theme.colors.text,
      marginBottom: 2,
    },
    incidentApp: {
      fontSize: theme.typography.caption,
      color: theme.colors.subText,
    },
    incidentMessage: {
      fontSize: theme.typography.body,
      color: theme.colors.subText,
      marginBottom: theme.spacing.sm,
      fontStyle: 'italic',
    },
    incidentFooter: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
    },
    incidentTime: {
      fontSize: theme.typography.caption,
      color: theme.colors.subText,
    },
    emptyState: {
      alignItems: 'center',
      padding: theme.spacing.xl,
    },
    emptyIcon: {
      fontSize: 64,
      marginBottom: theme.spacing.md,
    },
    emptyText: {
      fontSize: theme.typography.subheading,
      color: theme.colors.subText,
      textAlign: 'center',
    },
  });

  const unreadIncidents = mockIncidents.filter(i => !i.read);

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      >
        <View style={styles.header}>
          <Text style={styles.title}>Dashboard</Text>
          <Text style={styles.subtitle}>Your safety overview</Text>
        </View>

        {/* Safety Score */}
        <Card style={styles.safetyScoreCard} variant="muted">
          <View style={styles.scoreCircle}>
            <Text style={styles.scoreText}>{mockStats.safetyScore}</Text>
          </View>
          <Text style={styles.scoreLabel}>Safety Score</Text>
        </Card>

        {/* Stats Grid */}
        <View style={styles.statsGrid}>
          <Card style={styles.statCard}>
            <Text style={styles.statValue}>{mockStats.messagesScanned.toLocaleString()}</Text>
            <Text style={styles.statLabel}>Messages Scanned</Text>
          </Card>
          <Card style={styles.statCard}>
            <Text style={styles.statValue}>{mockStats.threatsBlocked}</Text>
            <Text style={styles.statLabel}>Threats Blocked</Text>
          </Card>
          <Card style={styles.statCard}>
            <Text style={styles.statValue}>{mockStats.activeDays}</Text>
            <Text style={styles.statLabel}>Active Days</Text>
          </Card>
          <Card style={styles.statCard}>
            <Text style={styles.statValue}>{unreadIncidents.length}</Text>
            <Text style={styles.statLabel}>New Alerts</Text>
          </Card>
        </View>

        {/* Recent Incidents */}
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Recent Incidents</Text>
        </View>

        {mockIncidents.length === 0 ? (
          <Card style={styles.emptyState}>
            <Text style={styles.emptyIcon}>✅</Text>
            <Text style={styles.emptyText}>
              No incidents detected. You're all clear!
            </Text>
          </Card>
        ) : (
          mockIncidents.map((incident) => (
            <TouchableOpacity
              key={incident.id}
              onPress={() => router.push(`/incident/${incident.id}`)}
              accessibilityRole="button"
              accessibilityLabel={`View incident from ${incident.sender}`}
            >
              <Card style={styles.incidentCard}>
                <View style={styles.incidentHeader}>
                  <View style={styles.incidentMeta}>
                    <Text style={styles.incidentSender}>{incident.sender}</Text>
                    <Text style={styles.incidentApp}>{incident.app}</Text>
                  </View>
                  <Badge
                    label={getTypeLabel(incident.type)}
                    variant={getSeverityColor(incident.severity)}
                    size="small"
                  />
                </View>
                <Text style={styles.incidentMessage} numberOfLines={2}>
                  {incident.message}
                </Text>
                <View style={styles.incidentFooter}>
                  <Text style={styles.incidentTime}>
                    {getRelativeTime(incident.timestamp)}
                  </Text>
                  {!incident.read && (
                    <Badge label="New" variant="primary" size="small" />
                  )}
                </View>
              </Card>
            </TouchableOpacity>
          ))
        )}
      </ScrollView>
    </SafeAreaView>
  );
}
