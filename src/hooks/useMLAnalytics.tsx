import { useState, useEffect, useCallback } from 'react';
import { mlService, MLAnalytics, PredictionResponse } from '@/services/mlService';
import { useToast } from '@/hooks/use-toast';

export interface UseMLAnalyticsReturn {
  analytics: MLAnalytics | null;
  isLoading: boolean;
  isServiceHealthy: boolean;
  refreshAnalytics: () => Promise<void>;
  predictContent: (contentId: string, contentUrl: string, contentType: string) => Promise<PredictionResponse | null>;
  predictFile: (file: File) => Promise<any>;
  sendFeedback: (contentId: string, userId: string, voteType: 'authentic_like' | 'inauthentic_dislike') => Promise<boolean>;
  startTraining: (epochs?: number) => Promise<boolean>;
  analyzeUnanalyzedContent: () => Promise<void>;
}

export const useMLAnalytics = (): UseMLAnalyticsReturn => {
  const [analytics, setAnalytics] = useState<MLAnalytics | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isServiceHealthy, setIsServiceHealthy] = useState(false);
  const { toast } = useToast();

  // Check ML service health
  const checkServiceHealth = useCallback(async () => {
    try {
      const healthy = await mlService.healthCheck();
      setIsServiceHealthy(healthy);
      
      if (!healthy) {
        toast({
          title: "ML Service Unavailable",
          description: "AI analysis features may be limited.",
          variant: "destructive",
        });
      }
    } catch (error) {
      setIsServiceHealthy(false);
    }
  }, [toast]);

  // Refresh analytics data
  const refreshAnalytics = useCallback(async () => {
    if (!isServiceHealthy) return;
    
    setIsLoading(true);
    try {
      const data = await mlService.getAnalyticsSummary();
      setAnalytics(data);
    } catch (error) {
      console.error('Error refreshing analytics:', error);
      toast({
        title: "Error loading analytics",
        description: "Failed to fetch ML analytics data.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  }, [isServiceHealthy, toast]);

  // Predict content authenticity
  const predictContent = useCallback(async (
    contentId: string,
    contentUrl: string,
    contentType: string
  ): Promise<PredictionResponse | null> => {
    if (!isServiceHealthy) {
      toast({
        title: "ML Service Unavailable",
        description: "Cannot analyze content at this time.",
        variant: "destructive",
      });
      return null;
    }

    try {
      const prediction = await mlService.predictAuthenticity({
        content_id: contentId,
        content_url: contentUrl,
        content_type: contentType as 'image' | 'video' | 'text',
      });

      if (prediction) {
        toast({
          title: "AI Analysis Complete",
          description: `Content predicted as ${prediction.prediction} (${Math.round(prediction.confidence * 100)}% confidence)`,
        });
      }

      return prediction;
    } catch (error) {
      console.error('Error predicting content:', error);
      toast({
        title: "Analysis Failed",
        description: "Failed to analyze content. Please try again.",
        variant: "destructive",
      });
      return null;
    }
  }, [isServiceHealthy, toast]);

  // Predict uploaded file
  const predictFile = useCallback(async (file: File) => {
    if (!isServiceHealthy) {
      toast({
        title: "ML Service Unavailable",
        description: "Cannot analyze file at this time.",
        variant: "destructive",
      });
      return null;
    }

    try {
      const prediction = await mlService.predictUploadedFile(file);
      
      if (prediction) {
        toast({
          title: "File Analysis Complete",
          description: `Predicted as ${prediction.prediction} (${Math.round(prediction.confidence * 100)}% confidence)`,
        });
      }

      return prediction;
    } catch (error) {
      console.error('Error predicting file:', error);
      toast({
        title: "Analysis Failed",
        description: "Failed to analyze file. Please try again.",
        variant: "destructive",
      });
      return null;
    }
  }, [isServiceHealthy, toast]);

  // Send feedback to improve model
  const sendFeedback = useCallback(async (
    contentId: string,
    userId: string,
    voteType: 'authentic_like' | 'inauthentic_dislike'
  ): Promise<boolean> => {
    if (!isServiceHealthy) return false;

    try {
      const success = await mlService.sendFeedback(contentId, userId, voteType);
      
      if (success) {
        toast({
          title: "Feedback Sent",
          description: "Thank you for helping improve our AI!",
        });
      }

      return success;
    } catch (error) {
      console.error('Error sending feedback:', error);
      return false;
    }
  }, [isServiceHealthy, toast]);

  // Start model training (admin feature)
  const startTraining = useCallback(async (epochs?: number): Promise<boolean> => {
    if (!isServiceHealthy) return false;

    try {
      const success = await mlService.startTraining(epochs);
      
      if (success) {
        toast({
          title: "Training Started",
          description: "Model training has been initiated in the background.",
        });
      } else {
        toast({
          title: "Training Failed",
          description: "Failed to start model training.",
          variant: "destructive",
        });
      }

      return success;
    } catch (error) {
      console.error('Error starting training:', error);
      toast({
        title: "Training Failed",
        description: "Failed to start model training.",
        variant: "destructive",
      });
      return false;
    }
  }, [isServiceHealthy, toast]);

  // Analyze all unanalyzed content
  const analyzeUnanalyzedContent = useCallback(async (): Promise<void> => {
    if (!isServiceHealthy) return;

    setIsLoading(true);
    try {
      await mlService.analyzeUnanalyzedContent();
      toast({
        title: "Batch Analysis Complete",
        description: "Analyzed all unanalyzed content.",
      });
      
      // Refresh analytics after batch analysis
      await refreshAnalytics();
    } catch (error) {
      console.error('Error in batch analysis:', error);
      toast({
        title: "Batch Analysis Failed",
        description: "Failed to analyze content in batch.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  }, [isServiceHealthy, toast, refreshAnalytics]);

  // Check service health on mount
  useEffect(() => {
    checkServiceHealth();
  }, [checkServiceHealth]);

  // Auto-refresh analytics every 5 minutes
  useEffect(() => {
    const interval = setInterval(() => {
      if (isServiceHealthy) {
        refreshAnalytics();
      }
    }, 5 * 60 * 1000); // 5 minutes

    return () => clearInterval(interval);
  }, [isServiceHealthy, refreshAnalytics]);

  // Initial analytics load
  useEffect(() => {
    if (isServiceHealthy) {
      refreshAnalytics();
    }
  }, [isServiceHealthy, refreshAnalytics]);

  return {
    analytics,
    isLoading,
    isServiceHealthy,
    refreshAnalytics,
    predictContent,
    predictFile,
    sendFeedback,
    startTraining,
    analyzeUnanalyzedContent,
  };
}; 