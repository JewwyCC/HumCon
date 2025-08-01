import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { ThumbsUp, ThumbsDown } from 'lucide-react';
import { cn } from '@/lib/utils';

interface VoteButtonProps {
  contentId: string;
  initialVote?: 'authentic_like' | 'inauthentic_dislike' | null;
  onVote: (contentId: string, voteType: 'authentic_like' | 'inauthentic_dislike') => void;
  disabled?: boolean;
}

const VoteButton = ({ contentId, initialVote, onVote, disabled }: VoteButtonProps) => {
  const [currentVote, setCurrentVote] = useState<'authentic_like' | 'inauthentic_dislike' | null>(initialVote || null);
  const [isAnimating, setIsAnimating] = useState(false);

  const handleVote = (voteType: 'authentic_like' | 'inauthentic_dislike') => {
    if (disabled) return;
    
    setIsAnimating(true);
    setCurrentVote(voteType);
    onVote(contentId, voteType);
    
    setTimeout(() => setIsAnimating(false), 300);
  };

  return (
    <div className="flex gap-2">
      <Button
        variant="vote"
        size="sm"
        onClick={() => handleVote('authentic_like')}
        disabled={disabled}
        className={cn(
          "flex items-center gap-2",
          currentVote === 'authentic_like' && "bg-success/20 text-success border-success/50",
          isAnimating && currentVote === 'authentic_like' && "pulse-vote"
        )}
      >
        <ThumbsUp className="h-4 w-4" />
        Authentic & Like
      </Button>
      
      <Button
        variant="vote"
        size="sm"
        onClick={() => handleVote('inauthentic_dislike')}
        disabled={disabled}
        className={cn(
          "flex items-center gap-2",
          currentVote === 'inauthentic_dislike' && "bg-destructive/20 text-destructive border-destructive/50",
          isAnimating && currentVote === 'inauthentic_dislike' && "pulse-vote"
        )}
      >
        <ThumbsDown className="h-4 w-4" />
        Not Authentic
      </Button>
    </div>
  );
};

export default VoteButton;