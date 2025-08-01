import { Card, CardContent, CardFooter, CardHeader } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Eye, TrendingUp, TrendingDown } from 'lucide-react';
import VoteButton from './VoteButton';
import { cn } from '@/lib/utils';

interface ContentCardProps {
  id: string;
  title: string;
  description?: string;
  contentType: 'video' | 'image' | 'text';
  fileUrl?: string;
  thumbnailUrl?: string;
  aiConfidenceScore?: number;
  authenticVotesCount: number;
  inauthenticVotesCount: number;
  totalEarnings: number;
  creatorName: string;
  creatorAvatar?: string;
  createdAt: string;
  userVote?: 'authentic_like' | 'inauthentic_dislike' | null;
  onVote: (contentId: string, voteType: 'authentic_like' | 'inauthentic_dislike') => void;
  className?: string;
}

const ContentCard = ({
  id,
  title,
  description,
  contentType,
  fileUrl,
  thumbnailUrl,
  aiConfidenceScore,
  authenticVotesCount,
  inauthenticVotesCount,
  totalEarnings,
  creatorName,
  creatorAvatar,
  createdAt,
  userVote,
  onVote,
  className
}: ContentCardProps) => {
  const totalVotes = authenticVotesCount + inauthenticVotesCount;
  const authenticPercentage = totalVotes > 0 ? (authenticVotesCount / totalVotes) * 100 : 0;
  
  const getContentTypeColor = (type: string) => {
    switch (type) {
      case 'video': return 'bg-destructive/10 text-destructive';
      case 'image': return 'bg-success/10 text-success';
      case 'text': return 'bg-warning/10 text-warning';
      default: return 'bg-muted text-muted-foreground';
    }
  };

  const getAIScoreColor = (score?: number) => {
    if (!score) return 'text-muted-foreground';
    if (score >= 0.8) return 'text-success';
    if (score >= 0.6) return 'text-warning';
    return 'text-destructive';
  };

  return (
    <Card className={cn("overflow-hidden hover:shadow-lg transition-all duration-300", className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Avatar className="h-8 w-8">
              <AvatarImage src={creatorAvatar} />
              <AvatarFallback>{creatorName.charAt(0).toUpperCase()}</AvatarFallback>
            </Avatar>
            <div>
              <p className="font-medium text-sm">{creatorName}</p>
              <p className="text-xs text-muted-foreground">
                {new Date(createdAt).toLocaleDateString()}
              </p>
            </div>
          </div>
          <Badge className={getContentTypeColor(contentType)}>
            {contentType.toUpperCase()}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="pb-3">
        {/* Content Preview */}
        {contentType === 'image' && (thumbnailUrl || fileUrl) && (
          <div className="relative mb-4 rounded-lg overflow-hidden">
            <img
              src={thumbnailUrl || fileUrl}
              alt={title}
              className="w-full h-48 object-cover"
            />
            {aiConfidenceScore && (
              <div className="absolute top-2 right-2 bg-black/80 text-white px-2 py-1 rounded text-xs">
                AI: {Math.round(aiConfidenceScore * 100)}%
              </div>
            )}
          </div>
        )}

        {contentType === 'video' && (thumbnailUrl || fileUrl) && (
          <div className="relative mb-4 rounded-lg overflow-hidden">
            <img
              src={thumbnailUrl || fileUrl}
              alt={title}
              className="w-full h-48 object-cover"
            />
            <div className="absolute inset-0 flex items-center justify-center bg-black/30">
              <div className="w-12 h-12 bg-white/90 rounded-full flex items-center justify-center">
                <div className="w-0 h-0 border-l-4 border-l-black border-y-2 border-y-transparent ml-1" />
              </div>
            </div>
            {aiConfidenceScore && (
              <div className="absolute top-2 right-2 bg-black/80 text-white px-2 py-1 rounded text-xs">
                AI: {Math.round(aiConfidenceScore * 100)}%
              </div>
            )}
          </div>
        )}

        {/* Title and Description */}
        <div className="space-y-2">
          <h3 className="font-semibold text-lg line-clamp-2">{title}</h3>
          {description && (
            <p className="text-muted-foreground text-sm line-clamp-3">{description}</p>
          )}
        </div>

        {/* Stats */}
        <div className="flex items-center justify-between mt-4 text-sm">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1 text-success">
              <TrendingUp className="h-4 w-4" />
              <span>{authenticVotesCount}</span>
            </div>
            <div className="flex items-center gap-1 text-destructive">
              <TrendingDown className="h-4 w-4" />
              <span>{inauthenticVotesCount}</span>
            </div>
            <div className="flex items-center gap-1 text-muted-foreground">
              <Eye className="h-4 w-4" />
              <span>{totalVotes}</span>
            </div>
          </div>
          <div className="text-accent font-medium">
            ${totalEarnings.toFixed(2)}
          </div>
        </div>

        {/* Authenticity Bar */}
        {totalVotes > 0 && (
          <div className="mt-3">
            <div className="flex justify-between text-xs text-muted-foreground mb-1">
              <span>Authenticity Score</span>
              <span>{Math.round(authenticPercentage)}%</span>
            </div>
            <div className="h-2 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-success to-primary transition-all duration-500"
                style={{ width: `${authenticPercentage}%` }}
              />
            </div>
          </div>
        )}

        {aiConfidenceScore && (
          <div className="mt-2 text-xs">
            <span className="text-muted-foreground">AI Confidence: </span>
            <span className={getAIScoreColor(aiConfidenceScore)}>
              {Math.round(aiConfidenceScore * 100)}%
            </span>
          </div>
        )}
      </CardContent>

      <CardFooter className="pt-3">
        <VoteButton
          contentId={id}
          initialVote={userVote}
          onVote={onVote}
        />
      </CardFooter>
    </Card>
  );
};

export default ContentCard;