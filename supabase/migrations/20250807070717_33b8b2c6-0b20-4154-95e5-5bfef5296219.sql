-- Add foreign key relationship between content and profiles
ALTER TABLE public.content 
ADD CONSTRAINT content_creator_id_fkey 
FOREIGN KEY (creator_id) REFERENCES public.profiles(user_id);