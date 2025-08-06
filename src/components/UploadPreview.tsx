import React, { useState, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  Upload, 
  Brain, 
  CheckCircle, 
  AlertCircle, 
  FileImage, 
  FileVideo,
  Loader2
} from 'lucide-react';
import { useMLAnalytics } from '@/hooks/useMLAnalytics';
import { useToast } from '@/hooks/use-toast';

interface UploadPreviewProps {
  file: File | null;
  onFileChange: (file: File | null) => void;
  className?: string;
}

interface PredictionResult {
  filename: string;
  content_type: string;
  authentic_probability: number;
  inauthentic_probability: number;
  confidence: number;
  prediction: string;
  timestamp: string;
}

const UploadPreview: React.FC<UploadPreviewProps> = ({ 
  file, 
  onFileChange, 
  className 
}) => {
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const { predictFile, isServiceHealthy } = useMLAnalytics();
  const { toast } = useToast();

  const handleFileSelect = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0] || null;
    onFileChange(selectedFile);
    setPrediction(null); // Reset prediction when new file is selected
  }, [onFileChange]);

  const handleDrop = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const droppedFile = event.dataTransfer.files[0];
    if (droppedFile && (droppedFile.type.startsWith('image/') || droppedFile.type.startsWith('video/'))) {
      onFileChange(droppedFile);
      setPrediction(null);
    } else {
      toast({
        title: "Invalid file type",
        description: "Please upload an image or video file.",
        variant: "destructive",
      });
    }
  }, [onFileChange, toast]);

  const handleDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  }, []);

  const analyzeFile = async () => {
    if (!file || !isServiceHealthy) return;

    setIsAnalyzing(true);
    try {
      const result = await predictFile(file);
      if (result) {
        setPrediction(result);
      }
    } catch (error) {
      console.error('Error analyzing file:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getFileIcon = () => {
    if (!file) return <Upload className="h-8 w-8 text-muted-foreground" />;
    
    if (file.type.startsWith('image/')) {
      return <FileImage className="h-8 w-8 text-blue-500" />;
    } else if (file.type.startsWith('video/')) {
      return <FileVideo className="h-8 w-8 text-green-500" />;
    }
    
    return <Upload className="h-8 w-8 text-muted-foreground" />;
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getPredictionColor = (prediction: string) => {
    return prediction === 'authentic' ? 'text-green-600' : 'text-red-600';
  };

  const getPredictionIcon = (prediction: string) => {
    return prediction === 'authentic' ? 
      <CheckCircle className="h-4 w-4 text-green-600" /> : 
      <AlertCircle className="h-4 w-4 text-red-600" />;
  };

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Upload className="h-5 w-5" />
          Upload & AI Analysis
        </CardTitle>
        <CardDescription>
          Upload an image or video for instant AI authenticity analysis
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* File Upload Area */}
        <div
          className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-6 text-center hover:border-muted-foreground/50 transition-colors"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
        >
          <input
            type="file"
            accept="image/*,video/*"
            onChange={handleFileSelect}
            className="hidden"
            id="file-upload"
          />
          
          {!file ? (
            <label htmlFor="file-upload" className="cursor-pointer">
              <div className="flex flex-col items-center gap-2">
                {getFileIcon()}
                <p className="text-sm font-medium">Drop files here or click to upload</p>
                <p className="text-xs text-muted-foreground">
                  Supports images and videos up to 10MB
                </p>
              </div>
            </label>
          ) : (
            <div className="flex flex-col items-center gap-2">
              {getFileIcon()}
              <p className="text-sm font-medium">{file.name}</p>
              <p className="text-xs text-muted-foreground">
                {formatFileSize(file.size)} • {file.type}
              </p>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  onFileChange(null);
                  setPrediction(null);
                }}
              >
                Remove
              </Button>
            </div>
          )}
        </div>

        {/* AI Analysis Button */}
        {file && isServiceHealthy && (
          <Button
            onClick={analyzeFile}
            disabled={isAnalyzing}
            className="w-full"
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Brain className="h-4 w-4 mr-2" />
                Analyze with AI
              </>
            )}
          </Button>
        )}

        {/* Service Status */}
        {file && !isServiceHealthy && (
          <div className="bg-muted p-3 rounded-lg">
            <p className="text-sm text-muted-foreground">
              AI analysis is currently unavailable. You can still upload the file.
            </p>
          </div>
        )}

        {/* Prediction Results */}
        {prediction && (
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-lg flex items-center gap-2">
                <Brain className="h-5 w-5" />
                AI Analysis Results
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Main Prediction */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {getPredictionIcon(prediction.prediction)}
                  <span className={`font-medium ${getPredictionColor(prediction.prediction)}`}>
                    {prediction.prediction.charAt(0).toUpperCase() + prediction.prediction.slice(1)}
                  </span>
                </div>
                <Badge variant="outline">
                  {Math.round(prediction.confidence * 100)}% confidence
                </Badge>
              </div>

              {/* Confidence Breakdown */}
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Authentic</span>
                  <span>{Math.round(prediction.authentic_probability * 100)}%</span>
                </div>
                <Progress value={prediction.authentic_probability * 100} className="h-2" />
                
                <div className="flex justify-between text-sm">
                  <span>Inauthentic</span>
                  <span>{Math.round(prediction.inauthentic_probability * 100)}%</span>
                </div>
                <Progress 
                  value={prediction.inauthentic_probability * 100} 
                  className="h-2"
                />
              </div>

              {/* Metadata */}
              <div className="grid grid-cols-2 gap-4 text-xs text-muted-foreground">
                <div>
                  <p className="font-medium">Content Type</p>
                  <p>{prediction.content_type}</p>
                </div>
                <div>
                  <p className="font-medium">Analyzed</p>
                  <p>{new Date(prediction.timestamp).toLocaleString()}</p>
                </div>
              </div>

              {/* Disclaimer */}
              <div className="bg-muted p-3 rounded-lg">
                <p className="text-xs text-muted-foreground">
                  <strong>Disclaimer:</strong> AI predictions are not 100% accurate. 
                  Use this analysis as a guide alongside human judgment and other verification methods.
                </p>
              </div>
            </CardContent>
          </Card>
        )}
      </CardContent>
    </Card>
  );
};

export default UploadPreview; 