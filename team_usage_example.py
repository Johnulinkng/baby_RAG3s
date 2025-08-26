#!/usr/bin/env python3
"""
Example of how team members would integrate the RAG API into their applications.
This demonstrates different usage patterns and best practices.
"""

import requests
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class RAGClientConfig:
    """Configuration for RAG API client."""
    base_url: str = "http://localhost:8000"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0

class BabyCareRAGClient:
    """
    Team-friendly client for the BabyCare RAG API.
    This is how team members would typically wrap the API.
    """
    
    def __init__(self, config: RAGClientConfig = None):
        self.config = config or RAGClientConfig()
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'BabyCareRAGClient/1.0'
        })
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the RAG service is healthy."""
        try:
            response = self.session.get(
                f"{self.config.base_url}/health",
                timeout=self.config.timeout
            )
            return {
                'healthy': response.status_code == 200,
                'status_code': response.status_code,
                'data': response.json() if response.status_code == 200 else None
            }
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    def ask_question(self, question: str, stream: bool = False) -> Dict[str, Any]:
        """
        Ask a baby care question.
        
        Args:
            question: The question to ask
            stream: Whether to use streaming response
            
        Returns:
            Dictionary with answer, sources, confidence, etc.
        """
        if stream:
            return self._ask_streaming(question)
        else:
            return self._ask_standard(question)
    
    def _ask_standard(self, question: str) -> Dict[str, Any]:
        """Standard non-streaming query."""
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.post(
                    f"{self.config.base_url}/query",
                    json={"question": question},
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        return {
                            'success': True,
                            'answer': data['data']['answer'],
                            'sources': data['data'].get('sources', []),
                            'confidence': data['data'].get('confidence', 0.0),
                            'processing_steps': data['data'].get('processing_steps', [])
                        }
                    else:
                        return {'success': False, 'error': data.get('error', 'Unknown error')}
                else:
                    return {'success': False, 'error': f'HTTP {response.status_code}'}
                    
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                    continue
                return {'success': False, 'error': str(e)}
        
        return {'success': False, 'error': 'Max retries exceeded'}
    
    def _ask_streaming(self, question: str) -> Dict[str, Any]:
        """Streaming query with Server-Sent Events."""
        try:
            response = self.session.post(
                f"{self.config.base_url}/query?stream=true",
                json={"question": question},
                headers={'Accept': 'text/event-stream'},
                stream=True,
                timeout=self.config.timeout
            )
            
            if response.status_code != 200:
                return {'success': False, 'error': f'HTTP {response.status_code}'}
            
            events = []
            final_answer = None
            
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith('event:'):
                    event_type = line.split(':', 1)[1].strip()
                elif line.startswith('data:'):
                    event_data = line.split(':', 1)[1].strip()
                    events.append({'type': event_type, 'data': event_data})
                    
                    # Parse the final message
                    if event_type == 'message':
                        try:
                            final_answer = json.loads(event_data)
                        except json.JSONDecodeError:
                            pass
                    
                    if event_type == 'end':
                        break
            
            if final_answer:
                return {
                    'success': True,
                    'answer': final_answer.get('answer'),
                    'sources': final_answer.get('sources', []),
                    'confidence': final_answer.get('confidence', 0.0),
                    'processing_steps': final_answer.get('processing_steps', []),
                    'streaming_events': len(events)
                }
            else:
                return {'success': False, 'error': 'No final answer received'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}

# Example usage patterns for different team scenarios

def example_web_app_integration():
    """Example: Web application backend integration."""
    print("🌐 Web App Integration Example")
    print("-" * 40)
    
    client = BabyCareRAGClient()
    
    # Health check before serving requests
    health = client.health_check()
    if not health['healthy']:
        print("❌ RAG service is not healthy!")
        return
    
    # Simulate user questions from web app
    user_questions = [
        "What should I do if my baby won't sleep?",
        "How often should I feed my newborn?",
        "When can I start giving my baby solid food?"
    ]
    
    for question in user_questions:
        print(f"\n👤 User asks: {question}")
        result = client.ask_question(question)
        
        if result['success']:
            print(f"🤖 Answer: {result['answer'][:100]}...")
            print(f"📚 Sources: {len(result['sources'])} documents")
            print(f"🎯 Confidence: {result['confidence']:.2f}")
        else:
            print(f"❌ Error: {result['error']}")

def example_mobile_app_integration():
    """Example: Mobile app with streaming for better UX."""
    print("\n📱 Mobile App Integration Example (Streaming)")
    print("-" * 50)
    
    client = BabyCareRAGClient()
    
    question = "What are the signs that my baby is ready for solid foods?"
    print(f"👤 User asks: {question}")
    print("🌊 Streaming response...")
    
    result = client.ask_question(question, stream=True)
    
    if result['success']:
        print(f"✅ Received answer with {result.get('streaming_events', 0)} events")
        print(f"🤖 Answer: {result['answer'][:150]}...")
        print(f"📚 Sources: {', '.join(result['sources'][:2])}")
    else:
        print(f"❌ Streaming failed: {result['error']}")

def example_chatbot_integration():
    """Example: Chatbot with conversation context."""
    print("\n🤖 Chatbot Integration Example")
    print("-" * 40)
    
    client = BabyCareRAGClient()
    
    # Simulate a conversation
    conversation = [
        "My baby is 6 months old. When should I start solid foods?",
        "What are good first foods to try?",
        "How do I know if my baby is allergic to something?"
    ]
    
    for i, question in enumerate(conversation, 1):
        print(f"\n💬 Turn {i}: {question}")
        result = client.ask_question(question)
        
        if result['success']:
            # In a real chatbot, you'd store this context
            answer = result['answer']
            print(f"🤖 Bot: {answer[:120]}...")
            
            # Show processing info for debugging
            steps = result.get('processing_steps', [])
            if steps:
                timing_info = [step for step in steps if 'ms' in step]
                if timing_info:
                    print(f"⏱️  {timing_info[-1]}")  # Show total time
        else:
            print(f"❌ Bot error: {result['error']}")

def example_batch_processing():
    """Example: Batch processing for analytics or content generation."""
    print("\n📊 Batch Processing Example")
    print("-" * 35)
    
    client = BabyCareRAGClient()
    
    # Common questions for FAQ generation
    faq_questions = [
        "What are the ABCs of Safe Sleep?",
        "How often should I burp my baby?",
        "What temperature should the baby's room be?",
        "When do babies start teething?"
    ]
    
    print(f"Processing {len(faq_questions)} FAQ questions...")
    
    faq_results = []
    start_time = time.perf_counter()
    
    for question in faq_questions:
        result = client.ask_question(question)
        if result['success']:
            faq_results.append({
                'question': question,
                'answer': result['answer'],
                'sources': result['sources'],
                'confidence': result['confidence']
            })
        else:
            print(f"❌ Failed: {question}")
    
    end_time = time.perf_counter()
    
    print(f"✅ Generated {len(faq_results)} FAQ entries in {end_time - start_time:.1f}s")
    print(f"📈 Average confidence: {sum(r['confidence'] for r in faq_results) / len(faq_results):.2f}")

if __name__ == "__main__":
    print("🍼 BabyCare RAG Team Integration Examples")
    print("=" * 50)
    
    # Run all examples
    example_web_app_integration()
    example_mobile_app_integration()
    example_chatbot_integration()
    example_batch_processing()
    
    print("\n✅ All integration examples completed!")
    print("\n💡 Key takeaways for team integration:")
    print("   • Always check health before serving requests")
    print("   • Use streaming for better user experience")
    print("   • Implement retry logic for reliability")
    print("   • Monitor response times and confidence scores")
    print("   • Handle errors gracefully in production")
