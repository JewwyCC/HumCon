import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  Brain, 
  Activity, 
  TrendingUp, 
  Users, 
  Zap, 
  RefreshCw,
  Play,
  BarChart3
} from 'lucide-react';
import { useMLAnalytics } from '@/hooks/useMLAnalytics';
import { useAuth } from '@/hooks/useAuth';

const MLAnalyticsDashboard: React.FC = () => {
  const { user } = useAuth();
  const {
    analytics,
    isLoading,
    isServiceHealthy,
    refreshAnalytics,
    startTraining,
    analyzeUnanalyzedContent,
  } = useMLAnalytics();

  const handleStartTraining = async () => {
    await startTraining(10); // Train for 10 epochs
  };

  const handleBatchAnalysis = async () => {
    await analyzeUnanalyzedContent();
  };

  if (!isServiceHealthy) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-muted-foreground" />
            AI Analytics Dashboard
          </CardTitle>
          <CardDescription>
            AI analysis service is currently unavailable
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <div className="text-center">
              <Activity className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground">
                The ML service is not running or unreachable.
              </p>
              <p className="text-sm text-muted-foreground mt-2">
                Please check that the ML service is started on port 8000.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">AI Analytics Dashboard</h2>
          <p className="text-muted-foreground">
            Monitor and manage AI-powered content authenticity detection
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant={isServiceHealthy ? "default" : "destructive"}>
            {isServiceHealthy ? "Service Online" : "Service Offline"}
          </Badge>
          <Button
            variant="outline"
            size="sm"
            onClick={refreshAnalytics}
            disabled={isLoading}
          >
            <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
          </Button>
        </div>
      </div>

      {/* Analytics Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Content Analyzed
            </CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {analytics?.total_content_analyzed || 0}
            </div>
            <p className="text-xs text-muted-foreground">
              Total items processed by AI
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Authenticity Rate
            </CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {analytics ? Math.round(analytics.authentic_content_ratio * 100) : 0}%
            </div>
            <Progress 
              value={analytics ? analytics.authentic_content_ratio * 100 : 0} 
              className="mt-2"
            />
            <p className="text-xs text-muted-foreground mt-2">
              Content predicted as authentic
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Recent Votes (24h)
            </CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {analytics?.recent_votes_24h || 0}
            </div>
            <p className="text-xs text-muted-foreground">
              User feedback in last 24 hours
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Model Confidence
            </CardTitle>
            <Brain className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {analytics ? Math.round(analytics.model_confidence_avg * 100) : 0}%
            </div>
            <Progress 
              value={analytics ? analytics.model_confidence_avg * 100 : 0} 
              className="mt-2"
            />
            <p className="text-xs text-muted-foreground mt-2">
              Average prediction confidence
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Action Cards */}
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="h-5 w-5" />
              Batch Analysis
            </CardTitle>
            <CardDescription>
              Analyze all unanalyzed content with AI
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground mb-4">
              Run AI analysis on content that hasn't been processed yet.
              This will help improve the overall accuracy of the system.
            </p>
            <Button 
              onClick={handleBatchAnalysis}
              disabled={isLoading}
              className="w-full"
            >
              <Activity className="h-4 w-4 mr-2" />
              {isLoading ? 'Analyzing...' : 'Start Batch Analysis'}
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5" />
              Model Training
            </CardTitle>
            <CardDescription>
              Retrain the AI model with latest user feedback
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground mb-4">
              Start reinforcement learning training using recent user votes.
              This improves model accuracy based on community feedback.
            </p>
            <Button 
              onClick={handleStartTraining}
              disabled={isLoading}
              variant="secondary"
              className="w-full"
            >
              <Play className="h-4 w-4 mr-2" />
              Start Training (10 epochs)
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Real-time Status */}
      {analytics && (
        <Card>
          <CardHeader>
            <CardTitle>System Status</CardTitle>
            <CardDescription>
              Real-time information about the AI system
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-3">
              <div className="space-y-2">
                <p className="text-sm font-medium">Service Health</p>
                <Badge variant={isServiceHealthy ? "default" : "destructive"}>
                  {isServiceHealthy ? "Healthy" : "Unhealthy"}
                </Badge>
              </div>
              <div className="space-y-2">
                <p className="text-sm font-medium">Last Updated</p>
                <p className="text-sm text-muted-foreground">
                  {analytics.timestamp ? new Date(analytics.timestamp).toLocaleString() : 'Never'}
                </p>
              </div>
              <div className="space-y-2">
                <p className="text-sm font-medium">Data Quality</p>
                <Badge variant="outline">
                  {analytics.recent_votes_24h > 10 ? "Good" : "Limited Data"}
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default MLAnalyticsDashboard; 