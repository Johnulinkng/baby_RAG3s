#!/usr/bin/env python3
"""
Test script to simulate team integration scenarios.
This tests the FastAPI server as if it were deployed in a team environment.
"""

import json
import time
import requests
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_QUESTIONS = [
    "What are the ABCs of Safe Sleep?",
    "What is the ideal room temperature for a baby's nursery?",
    "How to soothe a crying baby?",
    "When should I start feeding solid foods to my baby?"
]

class TeamIntegrationTester:
    """Simulates how team members would use the RAG API."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'TeamRAGClient/1.0'
        })
    
    def test_health_check(self) -> Dict[str, Any]:
        """Test health endpoint - critical for monitoring."""
        print("🔍 Testing health check...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            result = {
                'status_code': response.status_code,
                'response_time_ms': round(response.elapsed.total_seconds() * 1000),
                'success': response.status_code == 200,
                'data': response.json() if response.status_code == 200 else None
            }
            print(f"   ✅ Health check: {result['response_time_ms']}ms")
            return result
        except Exception as e:
            print(f"   ❌ Health check failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_non_streaming_query(self, question: str) -> Dict[str, Any]:
        """Test standard JSON API query."""
        print(f"📝 Testing non-streaming: {question[:50]}...")
        try:
            start_time = time.perf_counter()
            response = self.session.post(
                f"{self.base_url}/query",
                json={"question": question}
            )
            end_time = time.perf_counter()
            
            result = {
                'status_code': response.status_code,
                'response_time_ms': round((end_time - start_time) * 1000),
                'success': response.status_code == 200,
                'data': response.json() if response.status_code == 200 else None
            }
            
            if result['success']:
                data = result['data']['data']
                print(f"   ✅ Answer received: {len(data.get('answer', ''))} chars, {len(data.get('sources', []))} sources")
                print(f"   ⏱️  Total time: {result['response_time_ms']}ms")
            else:
                print(f"   ❌ Query failed: {response.status_code}")
            
            return result
        except Exception as e:
            print(f"   ❌ Non-streaming query failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_streaming_query(self, question: str) -> Dict[str, Any]:
        """Test Server-Sent Events streaming."""
        print(f"🌊 Testing streaming: {question[:50]}...")
        try:
            start_time = time.perf_counter()
            events_received = []
            
            response = self.session.post(
                f"{self.base_url}/query?stream=true",
                json={"question": question},
                headers={'Accept': 'text/event-stream'},
                stream=True
            )
            
            if response.status_code != 200:
                return {'success': False, 'status_code': response.status_code}
            
            # Parse SSE events
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith('event:'):
                    event_type = line.split(':', 1)[1].strip()
                elif line.startswith('data:'):
                    event_data = line.split(':', 1)[1].strip()
                    events_received.append({'type': event_type, 'data': event_data})
                    print(f"   📡 Event: {event_type}")
                    
                    if event_type == 'end':
                        break
            
            end_time = time.perf_counter()
            
            result = {
                'success': True,
                'response_time_ms': round((end_time - start_time) * 1000),
                'events_count': len(events_received),
                'events': events_received
            }
            
            print(f"   ✅ Streaming completed: {len(events_received)} events in {result['response_time_ms']}ms")
            return result
            
        except Exception as e:
            print(f"   ❌ Streaming query failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_concurrent_requests(self, question: str, num_requests: int = 3) -> Dict[str, Any]:
        """Test concurrent API calls to simulate team usage."""
        print(f"🚀 Testing {num_requests} concurrent requests...")
        import concurrent.futures
        
        def single_request():
            return self.test_non_streaming_query(question)
        
        start_time = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(single_request) for _ in range(num_requests)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        end_time = time.perf_counter()
        
        successful = sum(1 for r in results if r.get('success'))
        total_time = round((end_time - start_time) * 1000)
        
        print(f"   ✅ Concurrent test: {successful}/{num_requests} successful in {total_time}ms")
        
        return {
            'success': successful == num_requests,
            'total_time_ms': total_time,
            'successful_requests': successful,
            'total_requests': num_requests,
            'results': results
        }

def run_team_integration_tests():
    """Run comprehensive team integration tests."""
    print("🏢 Starting Team Integration Tests")
    print("=" * 50)
    
    tester = TeamIntegrationTester()
    results = {}
    
    # 1. Health check
    results['health'] = tester.test_health_check()
    if not results['health'].get('success'):
        print("❌ Health check failed - server may not be running")
        return results
    
    # 2. Test each question type
    results['non_streaming'] = []
    results['streaming'] = []
    
    for question in TEST_QUESTIONS:
        # Non-streaming test
        non_stream_result = tester.test_non_streaming_query(question)
        results['non_streaming'].append(non_stream_result)
        
        # Streaming test
        stream_result = tester.test_streaming_query(question)
        results['streaming'].append(stream_result)
        
        time.sleep(0.5)  # Brief pause between tests
    
    # 3. Concurrent requests test
    results['concurrent'] = tester.test_concurrent_requests(TEST_QUESTIONS[0], 3)
    
    # 4. Summary
    print("\n📊 Test Summary")
    print("=" * 50)
    
    non_stream_success = sum(1 for r in results['non_streaming'] if r.get('success'))
    stream_success = sum(1 for r in results['streaming'] if r.get('success'))
    
    print(f"Health Check: {'✅' if results['health'].get('success') else '❌'}")
    print(f"Non-streaming: {non_stream_success}/{len(TEST_QUESTIONS)} ✅")
    print(f"Streaming: {stream_success}/{len(TEST_QUESTIONS)} ✅")
    print(f"Concurrent: {'✅' if results['concurrent'].get('success') else '❌'}")
    
    # Average response times
    if results['non_streaming']:
        avg_time = sum(r.get('response_time_ms', 0) for r in results['non_streaming']) / len(results['non_streaming'])
        print(f"Average response time: {round(avg_time)}ms")
    
    return results

if __name__ == "__main__":
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code != 200:
            print("❌ Server is not responding correctly")
            print(f"Status code: {response.status_code}")
            sys.exit(1)
        else:
            print(f"✅ Server is running and healthy")
    except requests.exceptions.RequestException as e:
        print("❌ Server is not running. Please start with:")
        print("   uvicorn server:app --host 0.0.0.0 --port 8000 --workers 2")
        print(f"Error: {e}")
        sys.exit(1)
    
    # Run tests
    test_results = run_team_integration_tests()
    
    # Save results for analysis
    with open('team_integration_results.json', 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"\n📄 Results saved to: team_integration_results.json")
