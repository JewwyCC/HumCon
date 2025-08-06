import { supabase } from '@/integrations/supabase/client';

const ML_SERVICE_URL = process.env.REACT_APP_ML_SERVICE_URL || 'http://localhost:8000';

export interface PredictionResponse {
  content_id: string;
  authentic_probability: number;
  inauthentic_probability: number;
  confidence: number;
  prediction: 'authentic' | 'inauthentic';
  timestamp: string;
}

export interface PredictionRequest {
  content_id: string;
  content_url: string;
  content_type: 'image' | 'video' | 'text';
}

export interface UploadPredictionResponse {
  filename: string;
  content_type: string;
  authentic_probability: number;
  inauthentic_probability: number;
  confidence: number;
  prediction: string;
  timestamp: string;
}

export interface MLAnalytics {
  total_content_analyzed: number;
  authentic_content_ratio: number;
  recent_votes_24h: number;
  model_confidence_avg: number;
  timestamp: string;
}

class MLService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = ML_SERVICE_URL;
  }

  /**
   * Check if ML service is healthy and available
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      const data = await response.json();
      return data.status === 'healthy' && data.model_loaded;
    } catch (error) {
      console.error('ML service health check failed:', error);
      return false;
    }
  }

  /**
   * Predict authenticity for a single content item
   */
  async predictAuthenticity(request: PredictionRequest): Promise<PredictionResponse | null> {
    try {
      const response = await fetch(`${this.baseUrl}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error predicting authenticity:', error);
      return null;
    }
  }

  /**
   * Predict authenticity for multiple content items
   */
  async predictBatch(requests: PredictionRequest[]): Promise<PredictionResponse[]> {
    try {
      const response = await fetch(`${this.baseUrl}/predict/batch`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ content_items: requests }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error predicting batch:', error);
      return [];
    }
  }

  /**
   * Upload and predict authenticity for a file
   */
  async predictUploadedFile(file: File): Promise<UploadPredictionResponse | null> {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${this.baseUrl}/upload/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error predicting uploaded file:', error);
      return null;
    }
  }

  /**
   * Send user feedback to improve the model
   */
  async sendFeedback(
    contentId: string,
    userId: string,
    voteType: 'authentic_like' | 'inauthentic_dislike',
    predictionAccuracy?: number
  ): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content_id: contentId,
          user_id: userId,
          vote_type: voteType,
          prediction_accuracy: predictionAccuracy,
        }),
      });

      return response.ok;
    } catch (error) {
      console.error('Error sending feedback:', error);
      return false;
    }
  }

  /**
   * Get ML analytics summary
   */
  async getAnalyticsSummary(): Promise<MLAnalytics | null> {
    try {
      const response = await fetch(`${this.baseUrl}/analytics/summary`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error getting analytics summary:', error);
      return null;
    }
  }

  /**
   * Get model information
   */
  async getModelInfo(): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/model/info`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error getting model info:', error);
      return null;
    }
  }

  /**
   * Start model training (admin only)
   */
  async startTraining(epochs?: number): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/train`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ epochs }),
      });

      return response.ok;
    } catch (error) {
      console.error('Error starting training:', error);
      return false;
    }
  }

  /**
   * Auto-analyze content when it's uploaded to Supabase
   */
  async autoAnalyzeContent(contentId: string): Promise<void> {
    try {
      // Get content details from Supabase
      const { data: content, error } = await supabase
        .from('content')
        .select('*')
        .eq('id', contentId)
        .single();

      if (error || !content) {
        console.error('Error fetching content for analysis:', error);
        return;
      }

      // Skip if already analyzed
      if (content.ai_confidence_score && content.ai_confidence_score > 0) {
        return;
      }

      // Predict authenticity
      const prediction = await this.predictAuthenticity({
        content_id: contentId,
        content_url: content.file_url || content.thumbnail_url || '',
        content_type: content.content_type,
      });

      if (prediction) {
        console.log(`AI Analysis for ${contentId}:`, prediction);
        
        // The ML service automatically updates the ai_confidence_score in the database
        // But we could also do additional processing here if needed
      }
    } catch (error) {
      console.error('Error in auto-analysis:', error);
    }
  }

  /**
   * Analyze all unanalyzed content in batches
   */
  async analyzeUnanalyzedContent(): Promise<void> {
    try {
      // Get content without AI analysis
      const { data: unanalyzedContent, error } = await supabase
        .from('content')
        .select('*')
        .is('ai_confidence_score', null)
        .eq('is_active', true)
        .limit(50); // Process in batches

      if (error || !unanalyzedContent || unanalyzedContent.length === 0) {
        return;
      }

      // Prepare batch prediction requests
      const requests: PredictionRequest[] = unanalyzedContent
        .filter(content => content.file_url || content.thumbnail_url)
        .map(content => ({
          content_id: content.id,
          content_url: content.file_url || content.thumbnail_url || '',
          content_type: content.content_type,
        }));

      if (requests.length === 0) {
        return;
      }

      // Predict in batch
      const predictions = await this.predictBatch(requests);

      console.log(`Analyzed ${predictions.length} content items`);
    } catch (error) {
      console.error('Error in batch analysis:', error);
    }
  }
}

// Export singleton instance
export const mlService = new MLService(); 