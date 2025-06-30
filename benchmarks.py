#!/usr/bin/env python3
"""
Hozie AI Benchmarking Suite
Compare Hozie against other AI models using various evaluation methods
"""

import os
import sys
import json
import time
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

# Import Hozie
from LLM import Brain

class HozieBenchmarkSuite:
    def __init__(self):
        self.hozie = Brain(debug=False)
        self.results_dir = "./benchmark_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def benchmark_response_time(self, questions: List[str], iterations: int = 3):
        """Benchmark Hozie's response time"""
        print("⏱️  Benchmarking response times...")
        
        times = []
        for question in questions:
            question_times = []
            for i in range(iterations):
                start_time = time.time()
                try:
                    print(f"  Iteration {i+1}/{iterations} for question: {question[:50]}...")
                    response = self.hozie.answer(question)
                    end_time = time.time()
                    question_times.append(end_time - start_time)
                except Exception as e:
                    print(f"Error with question '{question[:30]}...': {e}")
                    question_times.append(float('inf'))
            
            avg_time = sum(question_times) / len(question_times)
            times.append({
                'question': question[:50] + "..." if len(question) > 50 else question,
                'avg_response_time': avg_time,
                'times': question_times
            })
        
        return times
    
    def benchmark_accuracy(self, qa_pairs: List[Dict[str, str]]):
        """Benchmark Hozie's accuracy on known Q&A pairs"""
        print("🎯 Benchmarking accuracy...")
        
        results = []
        correct = 0
        
        for qa in qa_pairs:
            question = qa['question']
            expected = qa['expected'].lower()
            print(f"  Question: {question[:50]}... (Expected: {expected})")
            try:
                response = self.hozie.answer(question)
                is_correct = self.check_answer_correctness(response.lower(), expected)
                
                if is_correct:
                    correct += 1
                
                results.append({
                    'question': question,
                    'expected': qa['expected'],
                    'got': response,
                    'correct': is_correct
                })
                
            except Exception as e:
                print(f"Error with question '{question[:30]}...': {e}")
                results.append({
                    'question': question,
                    'expected': qa['expected'],
                    'got': f"ERROR: {str(e)}",
                    'correct': False
                })
        
        accuracy = correct / len(qa_pairs) if qa_pairs else 0
        return {
            'accuracy': accuracy,
            'correct_count': correct,
            'total_count': len(qa_pairs),
            'details': results
        }
    
    def check_answer_correctness(self, response: str, expected: str) -> bool:
        """Simple answer correctness checker"""
        # Simple keyword matching - you can make this more sophisticated
        expected_keywords = expected.split()
        response_lower = response.lower()
        
        # Check if most expected keywords are in the response
        matches = sum(1 for keyword in expected_keywords if keyword.lower() in response_lower)
        return matches >= len(expected_keywords) * 0.6  # 60% keyword match threshold
    
    def compare_with_mistral(self, questions: List[str], mistral_api_key: Optional[str] = None):
        """Compare Hozie with Mistral AI"""
        print("🔄 Comparing with Mistral AI...")
        
        results = {
            'hozie_responses': [],
            'mistral_responses': [],
            'comparisons': []
        }
        
        for question in questions:
            print(f"  Comparing question: {question[:50]}...")
            # Get Hozie response
            try:
                hozie_start = time.time()
                hozie_response = self.hozie.answer(question)
                hozie_time = time.time() - hozie_start
            except Exception as e:
                hozie_response = f"ERROR: {str(e)}"
                hozie_time = float('inf')
            
            # Get Mistral response (if API key provided)
            mistral_response = "API key not provided"
            mistral_time = 0
            
            if mistral_api_key:
                try:
                    mistral_start = time.time()
                    mistral_response = self.query_mistral_api(question, mistral_api_key)
                    mistral_time = time.time() - mistral_start
                except Exception as e:
                    mistral_response = f"ERROR: {str(e)}"
                    mistral_time = float('inf')
            
            results['hozie_responses'].append({
                'question': question,
                'response': hozie_response,
                'response_time': hozie_time
            })
            
            results['mistral_responses'].append({
                'question': question,
                'response': mistral_response,
                'response_time': mistral_time
            })
            
            results['comparisons'].append({
                'question': question,
                'hozie_length': len(hozie_response),
                'mistral_length': len(mistral_response),
                'hozie_time': hozie_time,
                'mistral_time': mistral_time
            })
        
        return results
    
    def query_mistral_api(self, question: str, api_key: str) -> str:
        """Query Mistral AI API"""
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': 'mistral-tiny',  # or 'mistral-small', 'mistral-medium'
            'messages': [{'role': 'user', 'content': question}],
            'temperature': 0.7,
            'max_tokens': 500
        }
        
        response = requests.post(
            'https://api.mistral.ai/v1/chat/completions',
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            raise Exception(f"Mistral API error: {response.status_code} - {response.text}")
    
    def run_comprehensive_benchmark(self, mistral_api_key: Optional[str] = None):
        """Run a comprehensive benchmark suite"""
        print("🚀 Starting Comprehensive Hozie Benchmark Suite")
        print("=" * 60)
        
        # Test questions
        test_questions = [
            "What is the capital of France?",
            "Explain machine learning in simple terms.",
            "Name three programming languages.",
            "What causes rain?",
            "How do you make a cup of tea?",
            "What is the largest planet in our solar system?",
            "Explain photosynthesis briefly.",
            "Name two famous painters."
        ]
        
        # Q&A pairs for accuracy testing
        qa_pairs = [
            {"question": "What is the capital of France?", "expected": "paris"},
            {"question": "What color do you get when you mix red and blue?", "expected": "purple violet"},
            {"question": "How many days are in a week?", "expected": "seven 7"},
            {"question": "What is the largest ocean on Earth?", "expected": "pacific"},
        ]
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'model': 'Hozie',
            'benchmarks': {}
        }
        
        # 1. Response Time Benchmark
        print("\n1. Response Time Benchmark")
        response_times = self.benchmark_response_time(test_questions[:5])
        results['benchmarks']['response_times'] = response_times
        
        avg_response_time = sum(r['avg_response_time'] for r in response_times) / len(response_times)
        print(f"   Average response time: {avg_response_time:.2f} seconds")
        
        # 2. Accuracy Benchmark
        print("\n2. Accuracy Benchmark")
        accuracy_results = self.benchmark_accuracy(qa_pairs)
        results['benchmarks']['accuracy'] = accuracy_results
        print(f"   Accuracy: {accuracy_results['accuracy']:.2%}")
        
        # 3. Comparison with Mistral
        print("\n3. Comparison with Mistral AI")
        comparison_results = self.compare_with_mistral(test_questions[:5], mistral_api_key)
        results['benchmarks']['mistral_comparison'] = comparison_results
        
        if mistral_api_key:
            hozie_avg_time = sum(r['response_time'] for r in comparison_results['hozie_responses']) / len(comparison_results['hozie_responses'])
            mistral_avg_time = sum(r['response_time'] for r in comparison_results['mistral_responses']) / len(comparison_results['mistral_responses'])
            print(f"   Hozie avg time: {hozie_avg_time:.2f}s")
            print(f"   Mistral avg time: {mistral_avg_time:.2f}s")
        else:
            print("   Mistral comparison skipped (no API key provided)")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f"hozie_benchmark_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Benchmark completed!")
        print(f"📁 Results saved to: {results_file}")
        
        return results
    
    def create_comparison_report(self, results: Dict[str, Any]):
        """Create a readable comparison report"""
        report_lines = [
            "HOZIE AI BENCHMARK REPORT",
            "=" * 50,
            f"Generated: {results['timestamp']}",
            f"Model: {results['model']}",
            "",
            "PERFORMANCE SUMMARY:",
            "-" * 20,
        ]
        
        # Response times
        if 'response_times' in results['benchmarks']:
            times = results['benchmarks']['response_times']
            avg_time = sum(r['avg_response_time'] for r in times) / len(times)
            report_lines.extend([
                f"Average Response Time: {avg_time:.2f} seconds",
                f"Fastest Response: {min(r['avg_response_time'] for r in times):.2f}s",
                f"Slowest Response: {max(r['avg_response_time'] for r in times):.2f}s",
            ])
        
        # Accuracy
        if 'accuracy' in results['benchmarks']:
            acc = results['benchmarks']['accuracy']
            report_lines.extend([
                f"Accuracy: {acc['accuracy']:.2%} ({acc['correct_count']}/{acc['total_count']})",
            ])
        
        report_content = "\n".join(report_lines)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.results_dir, f"hozie_report_{timestamp}.txt")
        
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        print(f"📊 Report saved to: {report_file}")
        return report_content

def main():
    """Main benchmark runner"""
    print("🤖 Hozie AI Benchmarking Suite")
    print("=" * 40)
    
    # Initialize benchmark suite
    benchmark = HozieBenchmarkSuite()
    
    # Get Mistral API key from environment or user input
    mistral_api_key = os.getenv('MISTRAL_API_KEY')
    if not mistral_api_key:
        mistral_api_key = input("Enter Mistral API key (or press Enter to skip): ").strip()
        if not mistral_api_key:
            mistral_api_key = None
    
    # Run comprehensive benchmark
    try:
        results = benchmark.run_comprehensive_benchmark(mistral_api_key)
        report = benchmark.create_comparison_report(results)
        print("\n" + report)
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
#Average Response Time: 87.71 seconds
#Fastest Response: 3.94s
#Slowest Response: 306.32s

#Average Response Time: 16.39 seconds
#Fastest Response: 8.75s
#Slowest Response: 19.97s