/**
 * Mock Data - Static data for demonstration
 * Replace with actual API calls in production
 */

export interface Incident {
  id: string;
  type: 'threat' | 'harassment' | 'hate-speech' | 'sexual-content';
  message: string;
  sender: string;
  timestamp: Date;
  severity: 'low' | 'medium' | 'high';
  app: string;
  read: boolean;
}

export const mockIncidents: Incident[] = [
  {
    id: '1',
    type: 'harassment',
    message: 'You\'re so stupid, nobody likes you',
    sender: 'unknown_user_123',
    timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000), // 2 hours ago
    severity: 'high',
    app: 'Instagram',
    read: false,
  },
  {
    id: '2',
    type: 'threat',
    message: 'Watch your back tomorrow',
    sender: 'anonymous_456',
    timestamp: new Date(Date.now() - 5 * 60 * 60 * 1000), // 5 hours ago
    severity: 'high',
    app: 'Messages',
    read: true,
  },
  {
    id: '3',
    type: 'hate-speech',
    message: '[Filtered hate speech content]',
    sender: 'troll_789',
    timestamp: new Date(Date.now() - 24 * 60 * 60 * 1000), // 1 day ago
    severity: 'medium',
    app: 'WhatsApp',
    read: true,
  },
  {
    id: '4',
    type: 'sexual-content',
    message: '[Filtered inappropriate content]',
    sender: 'stranger_101',
    timestamp: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000), // 2 days ago
    severity: 'high',
    app: 'Snapchat',
    read: true,
  },
];

export interface Stats {
  safetyScore: number;
  messagesScanned: number;
  threatsBlocked: number;
  activeDays: number;
}

export const mockStats: Stats = {
  safetyScore: 87,
  messagesScanned: 1247,
  threatsBlocked: 23,
  activeDays: 14,
};

export const getSeverityColor = (severity: 'low' | 'medium' | 'high'): 'success' | 'info' | 'danger' => {
  switch (severity) {
    case 'low':
      return 'success';
    case 'medium':
      return 'info';
    case 'high':
      return 'danger';
  }
};

export const getTypeLabel = (type: Incident['type']): string => {
  switch (type) {
    case 'threat':
      return 'Threat';
    case 'harassment':
      return 'Harassment';
    case 'hate-speech':
      return 'Hate Speech';
    case 'sexual-content':
      return 'Sexual Content';
  }
};

export const getRelativeTime = (date: Date): string => {
  const now = Date.now();
  const diff = now - date.getTime();
  
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);
  const days = Math.floor(diff / 86400000);
  
  if (minutes < 60) {
    return `${minutes}m ago`;
  } else if (hours < 24) {
    return `${hours}h ago`;
  } else {
    return `${days}d ago`;
  }
};
