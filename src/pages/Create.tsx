import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { ArrowLeft, Upload, FileImage, FileVideo, FileText } from 'lucide-react';
import { useAuth } from '@/hooks/useAuth';
import { useToast } from '@/hooks/use-toast';
import { supabase } from '@/integrations/supabase/client';

const Create = () => {
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [contentType, setContentType] = useState<'video' | 'image' | 'text'>('image');
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const { user } = useAuth();
  const { toast } = useToast();
  const navigate = useNavigate();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
    }
  };

  const uploadFile = async (file: File): Promise<string | null> => {
    try {
      const fileExt = file.name.split('.').pop();
      const fileName = `${Math.random()}.${fileExt}`;
      const filePath = `${user?.id}/${fileName}`;

      const { error: uploadError } = await supabase.storage
        .from('content')
        .upload(filePath, file);

      if (uploadError) throw uploadError;

      const { data } = supabase.storage
        .from('content')
        .getPublicUrl(filePath);

      return data.publicUrl;
    } catch (error) {
      console.error('Error uploading file:', error);
      return null;
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!user) {
      navigate('/auth');
      return;
    }

    if (!title.trim()) {
      toast({
        title: "Title required",
        description: "Please provide a title for your content.",
        variant: "destructive",
      });
      return;
    }

    if (contentType !== 'text' && !file) {
      toast({
        title: "File required",
        description: "Please upload a file for your content.",
        variant: "destructive",
      });
      return;
    }

    setUploading(true);

    try {
      let fileUrl = null;
      if (file) {
        fileUrl = await uploadFile(file);
        if (!fileUrl) {
          throw new Error('Failed to upload file');
        }
      }

      const { error } = await (supabase as any)
        .from('content')
        .insert({
          creator_id: user.id,
          title: title.trim(),
          description: description.trim() || null,
          content_type: contentType,
          file_url: fileUrl,
          thumbnail_url: contentType === 'image' ? fileUrl : null,
        });

      if (error) throw error;

      toast({
        title: "Content created!",
        description: "Your content has been uploaded successfully.",
      });

      navigate('/');
    } catch (error) {
      console.error('Error creating content:', error);
      toast({
        title: "Error creating content",
        description: "Failed to upload your content. Please try again.",
        variant: "destructive",
      });
    } finally {
      setUploading(false);
    }
  };

  const getContentIcon = (type: string) => {
    switch (type) {
      case 'video': return <FileVideo className="h-5 w-5" />;
      case 'image': return <FileImage className="h-5 w-5" />;
      case 'text': return <FileText className="h-5 w-5" />;
      default: return <FileImage className="h-5 w-5" />;
    }
  };

  const getAcceptedFileTypes = () => {
    switch (contentType) {
      case 'video': return 'video/*';
      case 'image': return 'image/*';
      case 'text': return '';
      default: return 'image/*,video/*';
    }
  };

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
            <h1 className="text-2xl font-bold gradient-text">Create Content</h1>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8 max-w-2xl">
        <Card>
          <CardHeader>
            <CardTitle>Share Your Authentic Content</CardTitle>
            <CardDescription>
              Upload original human-created content and help train the AI of the future.
              Your content will be evaluated by the community for authenticity.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Title */}
              <div className="space-y-2">
                <Label htmlFor="title">Title *</Label>
                <Input
                  id="title"
                  placeholder="Give your content a compelling title"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  required
                />
              </div>

              {/* Description */}
              <div className="space-y-2">
                <Label htmlFor="description">Description</Label>
                <Textarea
                  id="description"
                  placeholder="Describe your content, your creative process, or inspiration behind it"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  rows={4}
                />
              </div>

              {/* Content Type */}
              <div className="space-y-2">
                <Label htmlFor="contentType">Content Type *</Label>
                <Select value={contentType} onValueChange={(value: 'video' | 'image' | 'text') => setContentType(value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select content type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="image">
                      <div className="flex items-center gap-2">
                        <FileImage className="h-4 w-4" />
                        Image
                      </div>
                    </SelectItem>
                    <SelectItem value="video">
                      <div className="flex items-center gap-2">
                        <FileVideo className="h-4 w-4" />
                        Video
                      </div>
                    </SelectItem>
                    <SelectItem value="text">
                      <div className="flex items-center gap-2">
                        <FileText className="h-4 w-4" />
                        Text
                      </div>
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* File Upload */}
              {contentType !== 'text' && (
                <div className="space-y-2">
                  <Label htmlFor="file">Upload File *</Label>
                  <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-6">
                    <div className="text-center">
                      {file ? (
                        <div className="space-y-2">
                          <div className="flex items-center justify-center gap-2">
                            {getContentIcon(contentType)}
                            <span className="text-sm font-medium">{file.name}</span>
                          </div>
                          <p className="text-xs text-muted-foreground">
                            Size: {(file.size / 1024 / 1024).toFixed(2)} MB
                          </p>
                          <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            onClick={() => setFile(null)}
                          >
                            Remove
                          </Button>
                        </div>
                      ) : (
                        <div className="space-y-4">
                          <Upload className="h-12 w-12 mx-auto text-muted-foreground" />
                          <div>
                            <Label htmlFor="file-upload" className="cursor-pointer">
                              <span className="text-primary hover:text-primary/80">
                                Click to upload
                              </span>
                              <span className="text-muted-foreground"> or drag and drop</span>
                            </Label>
                            <Input
                              id="file-upload"
                              type="file"
                              accept={getAcceptedFileTypes()}
                              onChange={handleFileChange}
                              className="hidden"
                            />
                          </div>
                          <p className="text-xs text-muted-foreground">
                            {contentType === 'video' ? 'MP4, MOV, AVI up to 100MB' : 'PNG, JPG, GIF up to 10MB'}
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Text Content */}
              {contentType === 'text' && (
                <div className="space-y-2">
                  <Label htmlFor="textContent">Your Text Content *</Label>
                  <Textarea
                    id="textContent"
                    placeholder="Share your original writing, poem, story, or thoughts..."
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    rows={8}
                    required
                  />
                </div>
              )}

              {/* Submit Button */}
              <div className="flex gap-4">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => navigate('/')}
                  className="flex-1"
                >
                  Cancel
                </Button>
                <Button
                  type="submit"
                  disabled={uploading}
                  className="flex-1"
                  variant="gradient"
                >
                  {uploading ? "Uploading..." : "Publish Content"}
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>

        {/* Info Card */}
        <Card className="mt-6">
          <CardContent className="pt-6">
            <div className="space-y-4">
              <h3 className="font-semibold">How HumCon Works</h3>
              <div className="space-y-2 text-sm text-muted-foreground">
                <p>• Your content will be reviewed by the community for authenticity</p>
                <p>• Authentic human content earns rewards based on community votes</p>
                <p>• AI-generated or inauthentic content may result in penalties</p>
                <p>• Help train the future of AI content detection</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  );
};

export default Create;