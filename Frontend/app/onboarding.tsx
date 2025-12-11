/**
 * Onboarding Screen - Welcome and permissions setup
 * Introduces SentinelAI and guides permission configuration
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Platform,
  SafeAreaView,
} from 'react-native';
import { useRouter } from 'expo-router';
import { useAppTheme } from '../contexts/ThemeContext';
import { Button } from '../components/ui/Button';

export default function OnboardingScreen() {
  const { theme } = useAppTheme();
  const router = useRouter();
  const [step, setStep] = useState(0);

  const steps = [
    {
      title: 'Welcome to SentinelAI',
      description:
        'Your personal guardian against online harassment. We help you stay safe while communicating.',
      icon: '🛡️',
    },
    {
      title: 'How It Works',
      description:
        'SentinelAI scans incoming messages in real-time and alerts you to potentially harmful content before you see it.',
      icon: '🔍',
    },
    {
      title: 'Your Privacy Matters',
      description:
        'All scanning happens on your device. Your messages are never sent to external servers.',
      icon: '🔒',
    },
    {
      title: Platform.OS === 'android' ? 'Enable Notification Access' : 'Set Up Monitoring',
      description:
        Platform.OS === 'android'
          ? 'To scan messages, we need notification access. This allows us to detect harmful content in real-time.'
          : 'Follow the in-app instructions to enable message monitoring for your messaging apps.',
      icon: '📱',
    },
  ];

  const currentStep = steps[step];
  const isLastStep = step === steps.length - 1;

  const handleNext = () => {
    if (isLastStep) {
      router.replace('/(tabs)');
    } else {
      setStep(step + 1);
    }
  };

  const handleSkip = () => {
    router.replace('/(tabs)');
  };

  const styles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: theme.colors.background,
    },
    content: {
      flex: 1,
      padding: theme.spacing.lg,
      justifyContent: 'center',
    },
    iconContainer: {
      alignItems: 'center',
      marginBottom: theme.spacing.xl,
    },
    icon: {
      fontSize: 80,
    },
    title: {
      fontSize: 28,
      fontWeight: '700',
      color: theme.colors.text,
      textAlign: 'center',
      marginBottom: theme.spacing.md,
    },
    description: {
      fontSize: theme.typography.subheading,
      color: theme.colors.subText,
      textAlign: 'center',
      lineHeight: 24,
      marginBottom: theme.spacing.xl,
    },
    buttonContainer: {
      gap: theme.spacing.md,
    },
    stepIndicator: {
      flexDirection: 'row',
      justifyContent: 'center',
      gap: theme.spacing.sm,
      marginTop: theme.spacing.xl,
    },
    dot: {
      width: 8,
      height: 8,
      borderRadius: 4,
    },
    activeDot: {
      backgroundColor: theme.colors.primary,
    },
    inactiveDot: {
      backgroundColor: theme.colors.border,
    },
  });

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.iconContainer}>
          <Text style={styles.icon}>{currentStep.icon}</Text>
        </View>

        <Text style={styles.title}>{currentStep.title}</Text>
        <Text style={styles.description}>{currentStep.description}</Text>

        <View style={styles.buttonContainer}>
          <Button
            title={isLastStep ? 'Get Started' : 'Next'}
            onPress={handleNext}
            variant="primary"
            accessibilityLabel={isLastStep ? 'Get started with SentinelAI' : 'Continue to next step'}
          />
          
          {!isLastStep && (
            <Button
              title="Skip"
              onPress={handleSkip}
              variant="secondary"
              accessibilityLabel="Skip onboarding"
            />
          )}
        </View>

        <View style={styles.stepIndicator}>
          {steps.map((_, index) => (
            <View
              key={index}
              style={[styles.dot, index === step ? styles.activeDot : styles.inactiveDot]}
            />
          ))}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}
