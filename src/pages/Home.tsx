import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';
import { Search, Plus, Filter, User, LogOut, TrendingUp } from 'lucide-react';
import { useAuth } from '@/hooks/useAuth';
import { useToast } from '@/hooks/use-toast';
import ContentCard from '@/components/ContentCard';
import { supabase } from '@/integrations/supabase/client';

interface Content {
  id: string;
  title: string;
  description?: string;
  content_type: 'video' | 'image' | 'text';
  file_url?: string;
  thumbnail_url?: string;
  ai_confidence_score?: number;
  authentic_votes_count: number;
  inauthentic_votes_count: number;
  total_earnings: number;
  created_at: string;
  profiles: {
    username: string;
    avatar_url?: string;
  };
}

const Home = () => {
  const [content, setContent] = useState<Content[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [filter, setFilter] = useState<'all' | 'trending' | 'recent'>('all');
  const { user, signOut } = useAuth();
  const { toast } = useToast();
  const navigate = useNavigate();

  useEffect(() => {
    if (!user) {
      navigate('/auth');
      return;
    }
    fetchContent();
  }, [user, navigate, filter]);

  const fetchContent = async () => {
    try {
      let query = supabase
        .from('content')
        .select(`
          *,
          profiles (username, avatar_url)
        `)
        .eq('is_active', true);

      if (filter === 'trending') {
        query = query.order('authentic_votes_count', { ascending: false });
      } else if (filter === 'recent') {
        query = query.order('created_at', { ascending: false });
      } else {
        query = query.order('created_at', { ascending: false });
      }

      const { data, error } = await query.limit(20);
      
      if (error) throw error;
      setContent(data || []);
    } catch (error) {
      console.error('Error fetching content:', error);
      toast({
        title: "Error loading content",
        description: "Failed to load content. Please try again.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleVote = async (contentId: string, voteType: 'authentic_like' | 'inauthentic_dislike') => {
    if (!user) return;

    try {
      const { error } = await supabase
        .from('votes')
        .upsert({
          user_id: user.id,
          content_id: contentId,
          vote_type: voteType,
        }, {
          onConflict: 'user_id,content_id'
        });

      if (error) throw error;

      // Send feedback to ML service for reinforcement learning
      try {
        const { mlService } = await import('@/services/mlService');
        await mlService.sendFeedback(contentId, user.id, voteType);
      } catch (mlError) {
        console.log('ML service feedback failed (non-critical):', mlError);
      }

      toast({
        title: "Vote recorded!",
        description: "Thank you for helping build better AI.",
      });

      // Refresh content to show updated vote counts
      fetchContent();
    } catch (error) {
      console.error('Error voting:', error);
      toast({
        title: "Error recording vote",
        description: "Please try again.",
        variant: "destructive",
      });
    }
  };

  const handleSignOut = async () => {
    await signOut();
    navigate('/auth');
  };

  const filteredContent = content.filter(item =>
    item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    (item.description && item.description.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  if (!user) {
    return null;
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-40">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <h1 className="text-2xl font-bold gradient-text">HumCon</h1>
              
              <div className="hidden md:flex items-center gap-4">
                <Button
                  variant={filter === 'all' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setFilter('all')}
                >
                  All
                </Button>
                <Button
                  variant={filter === 'trending' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setFilter('trending')}
                  className="flex items-center gap-2"
                >
                  <TrendingUp className="h-4 w-4" />
                  Trending
                </Button>
                <Button
                  variant={filter === 'recent' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setFilter('recent')}
                >
                  Recent
                </Button>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search content..."
                  className="pl-10 w-64"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>

              <Button
                onClick={() => navigate('/create')}
                className="flex items-center gap-2"
                variant="gradient"
              >
                <Plus className="h-4 w-4" />
                Create
              </Button>

              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="icon">
                    <Avatar className="h-8 w-8">
                      <AvatarFallback>
                        {user?.email?.charAt(0).toUpperCase()}
                      </AvatarFallback>
                    </Avatar>
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem onClick={() => navigate('/profile')}>
                    <User className="h-4 w-4 mr-2" />
                    Profile
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={handleSignOut}>
                    <LogOut className="h-4 w-4 mr-2" />
                    Sign Out
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>

          {/* Mobile filters */}
          <div className="md:hidden flex items-center gap-2 mt-4">
            <Button
              variant={filter === 'all' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setFilter('all')}
            >
              All
            </Button>
            <Button
              variant={filter === 'trending' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setFilter('trending')}
            >
              Trending
            </Button>
            <Button
              variant={filter === 'recent' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setFilter('recent')}
            >
              Recent
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="animate-pulse">
                <div className="bg-muted rounded-lg h-96" />
              </div>
            ))}
          </div>
        ) : filteredContent.length === 0 ? (
          <div className="text-center py-16">
            <h3 className="text-xl font-semibold mb-2">No content found</h3>
            <p className="text-muted-foreground mb-6">
              {searchQuery 
                ? "Try adjusting your search terms" 
                : "Be the first to share authentic human content!"
              }
            </p>
            <Button onClick={() => navigate('/create')} variant="gradient">
              <Plus className="h-4 w-4 mr-2" />
              Create Content
            </Button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredContent.map((item) => (
              <ContentCard
                key={item.id}
                id={item.id}
                title={item.title}
                description={item.description}
                contentType={item.content_type}
                fileUrl={item.file_url}
                thumbnailUrl={item.thumbnail_url}
                aiConfidenceScore={item.ai_confidence_score}
                authenticVotesCount={item.authentic_votes_count}
                inauthenticVotesCount={item.inauthentic_votes_count}
                totalEarnings={item.total_earnings}
                creatorName={item.profiles?.username || 'Anonymous'}
                creatorAvatar={item.profiles?.avatar_url}
                createdAt={item.created_at}
                onVote={handleVote}
              />
            ))}
          </div>
        )}
      </main>
    </div>
  );
};

export default Home;