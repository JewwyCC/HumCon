import asyncio
from typing import List, Dict, Optional, Tuple
from supabase import create_client, Client
from config import settings
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self):
        self.supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    
    async def get_training_data(self, limit: int = 1000) -> List[Dict]:
        """
        Fetch content with votes for training data
        Returns list of content items with aggregated vote information
        """
        try:
            # Get content with vote aggregations
            response = self.supabase.table('content').select(
                '''
                id,
                title,
                description,
                content_type,
                file_url,
                thumbnail_url,
                ai_confidence_score,
                authentic_votes_count,
                inauthentic_votes_count,
                total_earnings,
                created_at,
                creator_id
                '''
            ).eq('is_active', True).limit(limit).execute()
            
            content_data = response.data
            
            # Get detailed votes for each content
            for content_item in content_data:
                votes_response = self.supabase.table('votes').select(
                    'user_id, vote_type, created_at'
                ).eq('content_id', content_item['id']).execute()
                
                content_item['votes'] = votes_response.data
                
            return content_data
            
        except Exception as e:
            logger.error(f"Error fetching training data: {e}")
            return []
    
    async def get_content_by_id(self, content_id: str) -> Optional[Dict]:
        """Get specific content item by ID"""
        try:
            response = self.supabase.table('content').select('*').eq('id', content_id).single().execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching content {content_id}: {e}")
            return None
    
    async def update_ai_confidence(self, content_id: str, confidence_score: float) -> bool:
        """Update AI confidence score for content"""
        try:
            response = self.supabase.table('content').update({
                'ai_confidence_score': confidence_score
            }).eq('id', content_id).execute()
            
            return response.data is not None
        except Exception as e:
            logger.error(f"Error updating AI confidence for {content_id}: {e}")
            return False
    
    async def get_recent_votes(self, hours: int = 24) -> List[Dict]:
        """Get recent votes for online learning"""
        try:
            # Get votes from last N hours
            from datetime import datetime, timedelta
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            response = self.supabase.table('votes').select(
                '''
                id,
                user_id,
                content_id,
                vote_type,
                created_at,
                content (
                    id,
                    file_url,
                    thumbnail_url,
                    content_type,
                    ai_confidence_score
                )
                '''
            ).gte('created_at', cutoff_time.isoformat()).execute()
            
            return response.data
        except Exception as e:
            logger.error(f"Error fetching recent votes: {e}")
            return []
    
    async def get_user_voting_patterns(self, user_id: str) -> Dict:
        """Analyze user voting patterns for personalized learning"""
        try:
            response = self.supabase.table('votes').select(
                'vote_type, created_at'
            ).eq('user_id', user_id).execute()
            
            votes = response.data
            total_votes = len(votes)
            authentic_votes = len([v for v in votes if v['vote_type'] == 'authentic_like'])
            
            return {
                'total_votes': total_votes,
                'authentic_votes': authentic_votes,
                'inauthentic_votes': total_votes - authentic_votes,
                'authentic_ratio': authentic_votes / total_votes if total_votes > 0 else 0.5
            }
        except Exception as e:
            logger.error(f"Error analyzing user {user_id} voting patterns: {e}")
            return {}
    
    async def store_model_prediction(self, content_id: str, prediction: Dict) -> bool:
        """Store model prediction results"""
        try:
            # You might want to create a separate table for model predictions
            # For now, we'll update the content table
            response = self.supabase.table('content').update({
                'ai_confidence_score': prediction.get('confidence', 0.0)
            }).eq('id', content_id).execute()
            
            return response.data is not None
        except Exception as e:
            logger.error(f"Error storing prediction for {content_id}: {e}")
            return False


# Singleton instance
db_manager = DatabaseManager() 