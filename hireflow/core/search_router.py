"""
Intelligent search routing system using LangChain RunnableBranch.
Routes queries to shallow or deep search based on complexity and LLM analysis.
"""

from typing import Dict, Any, List, Optional
import re
from langchain_core.runnables import RunnableBranch
from langchain_google_genai import ChatGoogleGenerativeAI
from core.hybrid_indexer import HybridIndexer
from core.vector_store import VectorStore
from utils.config import GOOGLE_API_KEY, LLM_MODEL
from utils.utils import get_logger, get_rate_limiter

logger = get_logger(__name__)

class SearchRouter:
    """Routes search queries to appropriate search strategy (shallow vs deep)"""
    
    def __init__(self, hybrid_indexer: HybridIndexer = None):
        # Use the indexer that main.py / app.py already filled with resumes.
        # A new HybridIndexer here would be empty, so every search would return nothing.
        self.hybrid_indexer = hybrid_indexer if hybrid_indexer is not None else HybridIndexer()
        # Share its vector store. A fresh VectorStore() is never initialized,
        # so is_ready() would always be False.
        self.vector_store = self.hybrid_indexer.vector_store
        
        self.llm = None
        if GOOGLE_API_KEY:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=LLM_MODEL,
                    google_api_key=GOOGLE_API_KEY,
                    rate_limiter=get_rate_limiter(),  # max 5 requests/minute on the free tier
                )
            except Exception as e:
                pass
        
        self.search_chain = self.build_search_chain()
    
    def build_search_chain(self):
        """Build the RunnableBranch search chain"""
        
        def shallow_search(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Fast vector-only search"""
            query = inputs["query"]
            top_k = inputs.get("top_k", 5)
            filters = inputs.get("filters")
            
            pass
            
            try:
                if self.vector_store.is_ready():
                    results = self.vector_store.search_resumes(query, top_k, filters)
                    
                    # Format results for consistency
                    formatted_results = []
                    for result in results:
                        formatted_results.append({
                            'candidate_id': result.get('metadata', {}).get('candidate_id', 'unknown'),
                            'name': result.get('metadata', {}).get('name', 'Unknown'),
                            'text': result.get('page_content', ''),
                            'score': result.get('score', 0.0),
                            'metadata': result.get('metadata', {}),
                            'search_type': 'shallow_vector',
                            'source_query': query
                        })
                    
                    return {
                        "results": formatted_results,
                        "search_type": "shallow_vector",
                        "query": query,
                        "top_k": top_k
                    }
                elif not inputs.get("_fell_back"):
                    return deep_search(inputs)
                else:
                    return {"results": [], "search_type": "failed", "query": query, "top_k": top_k}

            except Exception as e:
                logger.error(f"Shallow search failed: {e}")
                if inputs.get("_fell_back"):
                    return {"results": [], "search_type": "failed", "query": query, "top_k": top_k}
                return deep_search(inputs)
        
        def deep_search(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Comprehensive hybrid search with multi-query and reranking"""
            query = inputs["query"]
            top_k = inputs.get("top_k", 5)
            filters = inputs.get("filters")
            jd_context = inputs.get("jd_context", "")
            
            pass
            
            try:
                # Perform hybrid search
                # TODO: HW15 [Easy] - Deep search ignores filters
                # deep_search reads 'filters' from the inputs and never uses them.
                # Where should a hard filter (e.g. minimum years) be applied - before
                # ranking or after? What goes wrong if you filter a truncated top-k?
                # Hint: Session 6 - a filter excludes, a score ranks.
                results = self.hybrid_indexer.search_resumes(query, top_k)
                
                if results and self.llm:
                    # LLM reranking
                    reranked_results = self.llm_rerank(results, query, jd_context, top_k)
                    results = reranked_results
                
                # Format results
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        'candidate_id': result.get('candidate_id', 'unknown'),
                        'name': result.get('name', 'Unknown'),
                        'text': result.get('text', ''),
                        'score': result.get('combined_score', 0.0),
                        'metadata': result.get('metadata', {}),
                        'search_type': 'deep_hybrid_reranked',
                        'source_query': query,
                        'query_count': result.get('query_count', 1),
                        'llm_reranked': 'llm_reranked' in result
                    })
                
                return {
                    "results": formatted_results,
                    "search_type": "deep_hybrid_reranked",
                    "query": query,
                    "top_k": top_k
                }
                
            except Exception as e:
                logger.error(f"Deep search failed: {e}")
                # Fall back to shallow search - once. Without the flag, shallow and
                # deep kept falling back to each other forever (RecursionError).
                return shallow_search({**inputs, "_fell_back": True})
        
        def route_search(inputs: Dict[str, Any]) -> str:
            """Route to appropriate search strategy based on query complexity and user preference"""
            query = inputs.get("query", "")
            search_mode = inputs.get("search_mode", "auto")
            
            if search_mode == "shallow":
                return "shallow"
            elif search_mode == "deep":
                return "deep"
            
            if self.llm:
                try:
                    routing_prompt = f"""
                    Analyze this search query and determine if it needs deep search or shallow search.
                    
                    Query: "{query}"
                    
                    Choose:
                    - "shallow" for simple, specific queries (e.g., "accountant", "Python developer")
                    - "deep" for complex, nuanced queries (e.g., "senior accountant with QuickBooks and tax experience")
                    
                    Return only "shallow" or "deep".
                    """
                    
                    response = self.llm.invoke(routing_prompt)
                    decision = response.content.strip().lower()
                    
                    if decision in ["shallow", "deep"]:
                        return decision
                    else:
                        pass
                        
                except Exception as e:
                    pass
            
            query_words = len(query.split())
            if query_words <= 3:
                return "shallow"
            else:
                return "deep"
        
        # TODO: HW13 [Medium] - The route is chosen twice per search
        # RunnableBranch evaluates each condition in turn, and each condition calls
        # route_search() - which calls the LLM. How many LLM calls does one search
        # spend just to choose a route? At 5 requests/minute, that matters.
        # Decide the route once (e.g. a RunnableLambda that adds it to the inputs),
        # then branch on the stored value. Should routing need an LLM at all?
        # Hint: Session 5 - Branched RAG; Session 6 - Runnables.
        chain = RunnableBranch(
            (lambda x: route_search(x) == "shallow", shallow_search),
            (lambda x: route_search(x) == "deep", deep_search),
            deep_search  # Default fallback
        )
        
        return chain
    
    def llm_rerank(self, results: List[Dict], query: str, jd_context: str, top_k: int) -> List[Dict]:
        """Use LLM to intelligently rerank search results"""
        if not results or not self.llm:
            return results
        
        try:
            pass
            
            candidates_info = []
            for i, result in enumerate(results):
                candidate_info = f"{i+1}. {result.get('name', 'Unknown')} - "
                candidate_info += f"Score: {result.get('combined_score', 0):.4f}"
                
                metadata = result.get('metadata', {})
                if 'skills' in metadata:
                    candidate_info += f", Skills: {', '.join(metadata['skills'][:3])}"
                if 'experience' in metadata:
                    candidate_info += f", Experience: {metadata['experience']} years"
                
                candidates_info.append(candidate_info)
            
            # TODO: HW14 [Medium] - The LLM reranks without seeing the resumes
            # The prompt gives the LLM each candidate's name and a retrieval score - and
            # the skills and experience fields, which are always empty (HW1, HW9).
            # How could a recruiter, human or AI, rank candidates from that?
            # Include a short excerpt of each resume.
            rerank_prompt = f"""
            You are an expert recruiter. Given a job search query and candidate results, rerank the candidates by their fit for the role.
            
            Search Query: "{query}"
            Job Context: "{jd_context if jd_context else 'General search'}"
            
            Current Results (ranked by algorithm):
            {chr(10).join(candidates_info)}
            
            Instructions:
            1. Consider the search query and job context
            2. Reorder the candidates by their fit for the role
            3. Return only the numbers in the new order (e.g., "3,1,2,4,5")
            4. Focus on semantic fit, not just scores
            
            Reranked order:
            """
            
            response = self.llm.invoke(rerank_prompt)
            reranked_order = response.content.strip()
            
            try:
                numbers = [int(x) for x in re.findall(r'\d+', reranked_order)]
                
                if len(numbers) == len(results):
                    reranked_results = []
                    for num in numbers:
                        if 1 <= num <= len(results):
                            result = results[num - 1].copy()
                            result['llm_reranked'] = True
                            result['llm_rank'] = len(reranked_results) + 1
                            reranked_results.append(result)
                    
                    return reranked_results
                else:
                    pass
                    
            except Exception as e:
                pass
            
        except Exception as e:
            pass
        
        return results
    
    def search(self, query: str, top_k: int = 5, filters: Optional[Dict] = None, 
               search_mode: str = "auto", jd_context: str = "") -> Dict[str, Any]:
        """Execute search with routing between shallow and deep strategies"""
        
        inputs = {
            "query": query,
            "top_k": top_k,
            "filters": filters,
            "search_mode": search_mode,
            "jd_context": jd_context
        }
        
        try:
            result = self.search_chain.invoke(inputs)
            return result
            
        except Exception as e:
            # Fallback to simple search
            return {
                "results": [],
                "search_type": "fallback",
                "query": query,
                "top_k": top_k,
                "error": str(e)
            }
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search router statistics"""
        return {
            "llm_available": self.llm is not None,
            "vector_store_ready": self.vector_store.is_ready(),
            "hybrid_indexer_ready": self.hybrid_indexer.get_index_stats()['hybrid_ready'],
            "search_modes": ["auto", "shallow", "deep"]
        }

# core/memory_rag.py

"""
Conversation memory system for tracking search history and context.
Uses LangChain's memory to maintain session context across searches.
"""

from langchain_classic.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage
import logging

logger = logging.getLogger(__name__)

class MemoryRAG:
    """Conversation memory system for maintaining search context and history"""
    
    def __init__(self):
        """Initialize LangChain conversation buffer for search tracking"""
        # TODO: HW23 [Easy] - Deprecated memory class
        # Run main.py and read the warning printed here. ConversationBufferMemory is
        # deprecated. This class only stores messages - replace it with a plain list
        # of HumanMessage / AIMessage objects.
        # Hint: Session 6 - Part 11, memory.
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    def record_search(self, query: str, results_count: int):
        """Store search query and result count in conversation memory"""
        self.memory.chat_memory.add_user_message(f"Search: {query}")
        self.memory.chat_memory.add_ai_message(f"Found {results_count} results")
    
    def record_candidate_view(self, candidate_name: str):
        """Record candidate viewing"""
        self.memory.chat_memory.add_user_message(f"Viewed candidate: {candidate_name}")
        self.memory.chat_memory.add_ai_message("Candidate interaction recorded")
    
    def get_search_history(self) -> list:
        """Get recent search queries"""
        queries = []
        for msg in self.memory.chat_memory.messages:
            if isinstance(msg, HumanMessage) and msg.content.startswith("Search:"):
                queries.append(msg.content.replace("Search: ", ""))
        return queries[-5:]
    
    def get_memory_stats(self) -> dict:
        """Get simple memory stats"""
        return {
            'total_messages': len(self.memory.chat_memory.messages),
            'search_count': len([m for m in self.memory.chat_memory.messages 
                               if isinstance(m, HumanMessage) and m.content.startswith("Search:")]),
            'candidate_views': len([m for m in self.memory.chat_memory.messages 
                                  if isinstance(m, HumanMessage) and "candidate" in m.content.lower()])
        }


# core/evaluator.py
"""
RAGAS-based evaluation system for measuring search quality.
Evaluates retrieval and answer generation performance using standard metrics.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from dataclasses import dataclass
from core.search_router import SearchRouter
from utils.utils import get_logger
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy, context_precision, faithfulness, answer_correctness
)

logger = get_logger(__name__)

@dataclass
class RAGEvaluationMetrics:
    """Container for RAGAS evaluation metrics with scoring"""
    answer_relevancy: float     # How relevant the answer is to the question
    context_precision: float    # Precision of retrieved context
    faithfulness: float         # Factual consistency with context
    answer_correctness: float   # Overall answer quality
    overall_score: float        # Weighted average of all metrics
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary format"""
        return {
            'answer_relevancy': self.answer_relevancy,
            'context_precision': self.context_precision,
            'faithfulness': self.faithfulness,
            'answer_correctness': self.answer_correctness,
            'overall_score': self.overall_score
        }

class RAGEvaluator:
    """RAGAS-powered search quality evaluator with history tracking"""
    
    def __init__(self, search_router: SearchRouter = None):
        """Initialize RAGAS metrics and evaluation tracking.

        Pass in the app's SearchRouter. Building a new one here gives an
        empty index, so every search returns nothing and every score is 0.
        """
        self.search_router = search_router
        self.ragas_metrics = [
            answer_relevancy, context_precision, faithfulness, answer_correctness
        ]
        self.evaluation_history = []  # Track all evaluations
    
    # TODO: HW22 [Easy] - Count the cost before you run
    # 4 metrics x top_k samples, and each metric makes 2-3 LLM calls. At 5
    # requests/minute, how long does top_k=5 take? Show the estimate to the user
    # before running, and consider which metrics are worth it.
    def evaluate_search_quality(self, query: str, expected_skills: List[str], 
                              search_mode: str = "deep", top_k: int = 5) -> RAGEvaluationMetrics:
        """Run search and evaluate results using RAGAS quality metrics"""
        
        try:
            search_router = self.search_router or SearchRouter()
            search_result = search_router.search(query, top_k, search_mode=search_mode)
            
            if not search_result.get('results'):
                return self.create_default_metrics()
            
            candidates = search_result['results']
            return self.evaluate_with_ragas(query, candidates, expected_skills)
                
        except Exception as e:
            logger.error(f"Search quality evaluation failed: {e}")
            return self.create_default_metrics()
    
    def create_default_metrics(self) -> RAGEvaluationMetrics:
        """Return zero metrics when search fails or returns no results"""
        return RAGEvaluationMetrics(
            answer_relevancy=0.0,
            context_precision=0.0,
            faithfulness=0.0,
            answer_correctness=0.0,
            overall_score=0.0
        )
    
    def evaluate_with_ragas(self, query: str, candidates: List[Dict], 
                            expected_skills: List[str]) -> RAGEvaluationMetrics:
        """Run RAGAS evaluation on search results and calculate final metrics"""
        
        evaluation_data = self.prepare_ragas_data(query, candidates, expected_skills)
        results = evaluate(
            dataset=self.to_ragas_dataset(evaluation_data),
            metrics=self.ragas_metrics,
            llm=self.judge_llm(),                # without this, RAGAS tries OpenAI
            embeddings=self.judge_embeddings(),
            run_config=self.run_config(),
        )
        table = results.to_pandas()

        metrics = RAGEvaluationMetrics(
            answer_relevancy=self.column_mean(table, 'answer_relevancy'),
            context_precision=self.column_mean(table, 'context_precision'),
            faithfulness=self.column_mean(table, 'faithfulness'),
            answer_correctness=self.column_mean(table, 'answer_correctness'),
            overall_score=0.0
        )
        
        metrics.overall_score = self.calculate_overall_score(metrics)
        self.store_evaluation(query, metrics)
        
        return metrics
    
    # TODO: HW20 [Hard] - What is actually being evaluated?
    # Read this function carefully. The 'context' is built from metadata that is
    # empty (HW1), and the 'answer' is a sentence generated by this code, not by
    # the AI. What does faithfulness mean for a template string checked against
    # 'No metadata'?
    # Redesign it so RAGAS judges something real - for example the re-ranker's
    # assessment (response) against the candidate's resume text (context).
    # Hint: Session 8 - the question / context / answer triple.
    def prepare_ragas_data(self, query: str, candidates: List[Dict], 
                           expected_skills: List[str]) -> pd.DataFrame:
        """Convert search results to RAGAS-compatible DataFrame format"""
        
        data = []
        for candidate in candidates:
            metadata = candidate.get('metadata', {})
            
            # Create context from candidate metadata
            context_parts = []
            if 'name' in metadata:
                context_parts.append(f"Name: {metadata['name']}")
            if 'skills' in metadata:
                context_parts.append(f"Skills: {', '.join(metadata['skills'][:5])}")
            if 'experience' in metadata:
                context_parts.append(f"Experience: {metadata['experience']} years")
            
            context = " | ".join(context_parts) if context_parts else "No metadata"
            
            # Create ground truth and generated answer
            ground_truth = self.create_ground_truth(metadata.get('skills', []), expected_skills)
            generated_answer = f"{metadata.get('name', 'Candidate')} with skills in {', '.join(metadata.get('skills', [])[:3])}"
            
            data.append({
                'question': query,
                'contexts': [context],
                'ground_truth': ground_truth,
                'answer': generated_answer
            })
        
        return pd.DataFrame(data)
    
    # ---- RAGAS 0.4 plumbing ------------------------------------------------

    @staticmethod
    def to_ragas_dataset(frame: pd.DataFrame):
        """RAGAS 0.4 needs an EvaluationDataset with these field names, not a DataFrame."""
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
        samples = []
        for _, row in frame.iterrows():
            samples.append(SingleTurnSample(
                user_input=row['question'],
                retrieved_contexts=list(row['contexts']),
                response=row['answer'],
                reference=row['ground_truth'],
            ))
        return EvaluationDataset(samples=samples)

    @staticmethod
    def judge_llm():
        from ragas.llms import LangchainLLMWrapper
        from langchain_google_genai import ChatGoogleGenerativeAI
        from utils.config import GOOGLE_API_KEY, LLM_MODEL
        from utils.utils import get_rate_limiter
        return LangchainLLMWrapper(ChatGoogleGenerativeAI(
            model=LLM_MODEL, google_api_key=GOOGLE_API_KEY,
            rate_limiter=get_rate_limiter()))   # shares the app's 5/minute budget

    @staticmethod
    def judge_embeddings():
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from utils.utils import get_embeddings
        return LangchainEmbeddingsWrapper(get_embeddings())

    @staticmethod
    def run_config():
        """RAGAS defaults to 16 parallel workers and a 180s timeout. At 5
        requests per minute that means rate-limit errors, then timeouts,
        then every score comes back as NaN."""
        from ragas.run_config import RunConfig
        return RunConfig(max_workers=1, timeout=900, max_retries=10, max_wait=60)

    # TODO: HW21 [Medium] - A score of 0.00 might mean the judge failed
    # RAGAS returns NaN when the judge cannot decide (quota, timeout, bad JSON).
    # This turns NaN - and missing columns - into 0.0, which looks like a terrible
    # score. How would a user tell 'the search is bad' from 'the judge failed'?
    # Hint: Session 8 - the smoke test, and why a number is not proof it worked.
    @staticmethod
    def column_mean(table: pd.DataFrame, column: str) -> float:
        if column not in table.columns:
            return 0.0
        values = table[column].dropna()
        return float(values.mean()) if len(values) else 0.0

    def create_ground_truth(self, candidate_skills: List[str], expected_skills: List[str]) -> str:
        """Generate ground truth labels based on skill matching percentage"""
        if not expected_skills:
            return "No skills specified"
         
        matched_skills = [skill for skill in expected_skills 
                         if skill.lower() in [s.lower() for s in candidate_skills]]
        match_percentage = len(matched_skills) / len(expected_skills)
        
        if match_percentage >= 0.8:
            return f"Excellent match: {len(matched_skills)}/{len(expected_skills)} skills"
        elif match_percentage >= 0.6:
            return f"Good match: {len(matched_skills)}/{len(expected_skills)} skills"
        elif match_percentage >= 0.4:
            return f"Moderate match: {len(matched_skills)}/{len(expected_skills)} skills"
        else:
            return f"Poor match: {len(matched_skills)}/{len(expected_skills)} skills"
    
    def calculate_overall_score(self, metrics: RAGEvaluationMetrics) -> float:
        """Compute weighted average of all RAGAS metrics"""
        weights = {
            'answer_relevancy': 0.30,
            'context_precision': 0.30,
            'faithfulness': 0.20,
            'answer_correctness': 0.20
        }
        
        overall_score = (
            metrics.answer_relevancy * weights['answer_relevancy'] +
            metrics.context_precision * weights['context_precision'] +
            metrics.faithfulness * weights['faithfulness'] +
            metrics.answer_correctness * weights['answer_correctness']
        )
        
        return overall_score
    
    def store_evaluation(self, query: str, metrics: RAGEvaluationMetrics):
        """Save evaluation results to history for later analysis"""
        evaluation_record = {
            'query': query,
            'timestamp': pd.Timestamp.now(),
            'metrics': metrics.to_dict()
        }
        
        self.evaluation_history.append(evaluation_record)
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all evaluations performed"""
        if not self.evaluation_history:
            return {"message": "No evaluations performed yet"}
        
        # Calculate averages
        avg_metrics = {}
        for metric in ['answer_relevancy', 'context_precision', 'faithfulness', 
                      'answer_correctness', 'overall_score']:
            values = [record['metrics'][metric] for record in self.evaluation_history]
            avg_metrics[f'avg_{metric}'] = sum(values) / len(values)
        
        return {
            'total_evaluations': len(self.evaluation_history),
            'average_metrics': avg_metrics,
            'recent_evaluations': self.evaluation_history[-5:]
        }
    
    def export_evaluations(self, filename: str = "rag_evaluations.csv") -> bool:
        """Export all evaluation history to CSV file for analysis"""
        if not self.evaluation_history:
            return False
        
        try:
            export_data = []
            for i, record in enumerate(self.evaluation_history):
                row = {
                    'evaluation_id': i + 1,
                    'query': record['query'],
                    'timestamp': record['timestamp'],
                    **record['metrics']
                }
                export_data.append(row)
            
            df = pd.DataFrame(export_data)
            df.to_csv(filename, index=False)
            return True
            
        except Exception as e:
            logger.error(f"Failed to export evaluations: {e}")
            return False
