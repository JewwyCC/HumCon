import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { ArrowLeft, TrendingUp, TrendingDown, DollarSign, Award, Eye, Edit } from 'lucide-react';
import { useAuth } from '@/hooks/useAuth';
import { useToast } from '@/hooks/use-toast';
import { supabase } from '@/integrations/supabase/client';
import ContentCard from '@/components/ContentCard';

interface Profile {
  id: string;
  username: string;
  display_name?: string;
  bio?: string;
  avatar_url?: string;
}

interface UserStats {
  totalContent: number;
  totalVotes: number;
  totalEarnings: number;
  authenticVotes: number;
  inauthenticVotes: number;
}

const Profile = () => {
  const [profile, setProfile] = useState<Profile | null>(null);
  const [userContent, setUserContent] = useState<any[]>([]);
  const [userStats, setUserStats] = useState<UserStats>({
    totalContent: 0,
    totalVotes: 0,
    totalEarnings: 0,
    authenticVotes: 0,
    inauthenticVotes: 0,
  });
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState(false);
  const [editForm, setEditForm] = useState({
    display_name: '',
    bio: '',
  });
  
  const { user } = useAuth();
  const { toast } = useToast();
  const navigate = useNavigate();

  useEffect(() => {
    if (!user) {
      navigate('/auth');
      return;
    }
    fetchProfile();
    fetchUserContent();
    fetchUserStats();
  }, [user, navigate]);

  const fetchProfile = async () => {
    if (!user) return;

    try {
      const { data, error } = await supabase
        .from('profiles')
        .select('*')
        .eq('user_id', user.id)
        .single();

      if (error) throw error;
      
      setProfile(data);
      setEditForm({
        display_name: data.display_name || '',
        bio: data.bio || '',
      });
    } catch (error) {
      console.error('Error fetching profile:', error);
    }
  };

  const fetchUserContent = async () => {
    if (!user) return;

    try {
      const { data, error } = await supabase
        .from('content')
        .select(`
          *,
          profiles (username, avatar_url)
        `)
        .eq('creator_id', user.id)
        .order('created_at', { ascending: false });

      if (error) throw error;
      setUserContent(data || []);
    } catch (error) {
      console.error('Error fetching user content:', error);
    }
  };

  const fetchUserStats = async () => {
    if (!user) return;

    try {
      // Get content stats
      const { data: contentData, error: contentError } = await supabase
        .from('content')
        .select('id, authentic_votes_count, inauthentic_votes_count, total_earnings')
        .eq('creator_id', user.id);

      if (contentError) throw contentError;

      // Get voting stats
      const { data: votesData, error: votesError } = await supabase
        .from('votes')
        .select('vote_type')
        .eq('user_id', user.id);

      if (votesError) throw votesError;

      const totalContent = contentData?.length || 0;
      const totalEarnings = contentData?.reduce((sum, item) => sum + item.total_earnings, 0) || 0;
      const totalVotes = votesData?.length || 0;
      const authenticVotes = votesData?.filter(vote => vote.vote_type === 'authentic_like').length || 0;
      const inauthenticVotes = votesData?.filter(vote => vote.vote_type === 'inauthentic_dislike').length || 0;

      setUserStats({
        totalContent,
        totalVotes,
        totalEarnings,
        authenticVotes,
        inauthenticVotes,
      });
    } catch (error) {
      console.error('Error fetching user stats:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveProfile = async () => {
    if (!user || !profile) return;

    try {
      const { error } = await supabase
        .from('profiles')
        .update({
          display_name: editForm.display_name,
          bio: editForm.bio,
        })
        .eq('user_id', user.id);

      if (error) throw error;

      setProfile({
        ...profile,
        display_name: editForm.display_name,
        bio: editForm.bio,
      });
      setEditing(false);

      toast({
        title: "Profile updated!",
        description: "Your profile has been successfully updated.",
      });
    } catch (error) {
      console.error('Error updating profile:', error);
      toast({
        title: "Error updating profile",
        description: "Failed to update your profile. Please try again.",
        variant: "destructive",
      });
    }
  };

  const handleVote = async (contentId: string, voteType: 'authentic_like' | 'inauthentic_dislike') => {
    // This would typically be handled in the ContentCard component
    // but we can implement it here for the profile page view
    console.log('Vote handled:', contentId, voteType);
  };

  if (!user || loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-40">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => navigate('/')}
            >
              <ArrowLeft className="h-4 w-4" />
            </Button>
            <h1 className="text-2xl font-bold gradient-text">Profile</h1>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8 max-w-4xl">
        <div className="space-y-8">
          {/* Profile Header */}
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-start gap-6">
                <Avatar className="h-24 w-24">
                  <AvatarImage src={profile?.avatar_url} />
                  <AvatarFallback className="text-2xl">
                    {profile?.username?.charAt(0).toUpperCase()}
                  </AvatarFallback>
                </Avatar>
                
                <div className="flex-1">
                  {editing ? (
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <Label htmlFor="display-name">Display Name</Label>
                        <Input
                          id="display-name"
                          placeholder="Your display name"
                          value={editForm.display_name}
                          onChange={(e) => setEditForm({
                            ...editForm,
                            display_name: e.target.value
                          })}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="bio">Bio</Label>
                        <Textarea
                          id="bio"
                          placeholder="Tell us about yourself and your creative work"
                          value={editForm.bio}
                          onChange={(e) => setEditForm({
                            ...editForm,
                            bio: e.target.value
                          })}
                          rows={3}
                        />
                      </div>
                      <div className="flex gap-2">
                        <Button onClick={handleSaveProfile} size="sm">
                          Save Changes
                        </Button>
                        <Button variant="outline" onClick={() => setEditing(false)} size="sm">
                          Cancel
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      <div className="flex items-center gap-3">
                        <h2 className="text-2xl font-bold">
                          {profile?.display_name || profile?.username}
                        </h2>
                        <Button variant="outline" size="sm" onClick={() => setEditing(true)}>
                          <Edit className="h-4 w-4 mr-2" />
                          Edit
                        </Button>
                      </div>
                      <p className="text-muted-foreground">@{profile?.username}</p>
                      {profile?.bio && (
                        <p className="text-foreground">{profile.bio}</p>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Stats Cards */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <Card>
              <CardContent className="pt-6">
                <div className="text-center">
                  <div className="text-2xl font-bold text-primary">{userStats.totalContent}</div>
                  <div className="text-sm text-muted-foreground">Content</div>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-center">
                  <div className="text-2xl font-bold text-success">{userStats.authenticVotes}</div>
                  <div className="text-sm text-muted-foreground">Authentic Votes</div>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-center">
                  <div className="text-2xl font-bold text-destructive">{userStats.inauthenticVotes}</div>
                  <div className="text-sm text-muted-foreground">Not Authentic</div>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-center">
                  <div className="text-2xl font-bold text-accent">${userStats.totalEarnings.toFixed(2)}</div>
                  <div className="text-sm text-muted-foreground">Earnings</div>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-center">
                  <div className="text-2xl font-bold text-secondary">{userStats.totalVotes}</div>
                  <div className="text-sm text-muted-foreground">Votes Cast</div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Content Tabs */}
          <Tabs defaultValue="content" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="content">My Content</TabsTrigger>
              <TabsTrigger value="activity">Voting History</TabsTrigger>
              <TabsTrigger value="impact">AI Impact</TabsTrigger>
            </TabsList>
            
            <TabsContent value="content" className="space-y-6">
              {userContent.length === 0 ? (
                <Card>
                  <CardContent className="pt-6">
                    <div className="text-center py-8">
                      <h3 className="text-lg font-semibold mb-2">No content yet</h3>
                      <p className="text-muted-foreground mb-4">
                        Start sharing your authentic human content!
                      </p>
                      <Button onClick={() => navigate('/create')} variant="gradient">
                        Create Your First Content
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {userContent.map((item) => (
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
                      creatorName={profile?.username || 'You'}
                      creatorAvatar={profile?.avatar_url}
                      createdAt={item.created_at}
                      onVote={handleVote}
                    />
                  ))}
                </div>
              )}
            </TabsContent>
            
            <TabsContent value="activity" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Voting Activity</CardTitle>
                  <CardDescription>
                    Your contribution to building better AI through content evaluation
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-4 border rounded-lg">
                      <div className="flex items-center gap-3">
                        <TrendingUp className="h-5 w-5 text-success" />
                        <div>
                          <p className="font-medium">Authentic Content Votes</p>
                          <p className="text-sm text-muted-foreground">Votes for human-created content</p>
                        </div>
                      </div>
                      <Badge variant="secondary">{userStats.authenticVotes}</Badge>
                    </div>
                    <div className="flex items-center justify-between p-4 border rounded-lg">
                      <div className="flex items-center gap-3">
                        <TrendingDown className="h-5 w-5 text-destructive" />
                        <div>
                          <p className="font-medium">Inauthentic Content Votes</p>
                          <p className="text-sm text-muted-foreground">Votes against AI-generated content</p>
                        </div>
                      </div>
                      <Badge variant="secondary">{userStats.inauthenticVotes}</Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="impact" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Your Impact on AI Development</CardTitle>
                  <CardDescription>
                    How you're helping shape the future of AI content detection
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-6">
                    <div className="text-center p-6 bg-gradient-to-r from-primary/10 to-secondary/10 rounded-lg">
                      <Award className="h-12 w-12 mx-auto text-primary mb-4" />
                      <h3 className="text-lg font-semibold mb-2">AI Training Contributor</h3>
                      <p className="text-muted-foreground">
                        Your {userStats.totalVotes} votes are helping train AI models to better detect authentic human content
                      </p>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="text-center p-4 border rounded-lg">
                        <DollarSign className="h-8 w-8 mx-auto text-accent mb-2" />
                        <div className="text-xl font-bold">${userStats.totalEarnings.toFixed(2)}</div>
                        <div className="text-sm text-muted-foreground">Total Rewards Earned</div>
                      </div>
                      <div className="text-center p-4 border rounded-lg">
                        <Eye className="h-8 w-8 mx-auto text-primary mb-2" />
                        <div className="text-xl font-bold">{userStats.totalContent}</div>
                        <div className="text-sm text-muted-foreground">Content Contributions</div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </main>
    </div>
  );
};

export default Profile;