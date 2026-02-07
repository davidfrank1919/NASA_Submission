# ragas_eval.py
#Lines 65 - 78 retrieved doing a Google search for "RAGAS evaluation code example" and "RAGAS evaluation with LangChain example". I adapted the code to fit the structure of this project and added error handling for missing dependencies and API keys. The function `evaluate_response_quality` can be called from the chat interface after generating a response to get real-time evaluation metrics.
#found reference for this import via google search
from __future__ import annotations

import os
from typing import Dict, List, Optional

try:
    from ragas import evaluate
    from ragas import SingleTurnSample
    from ragas.metrics import ResponseRelevancy, Faithfulness
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

try:
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False

DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")


def evaluate_response_quality(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None
) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics."""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    if not _HAS_LANGCHAIN:
        return {"error": "LangChain OpenAI wrappers not available. Install langchain-openai."}

    api_key = os.getenv("OPENAI_API_KEY", "voc-2009866425126677464339669333e56cd9164.50366416")
    if not api_key:
        return {"error": "OPENAI_API_KEY missing for RAGAS evaluation"}

    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=os.getenv("OPENAI_EVAL_MODEL", os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5-mini")),
            temperature=0.0,
            api_key=api_key,
            base_url=DEFAULT_BASE_URL
        )
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            api_key=api_key,
            base_url=DEFAULT_BASE_URL
        )
    )

    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    ]

    # Add extra metrics if available in this ragas version
    try:
        from ragas.metrics import ContextPrecision, ContextRecall
        metrics.append(ContextPrecision(llm=evaluator_llm))
        metrics.append(ContextRecall(llm=evaluator_llm))
    except Exception:
        pass

    if ground_truth:
        try:
            from ragas.metrics import AnswerCorrectness
            metrics.append(AnswerCorrectness(llm=evaluator_llm))
        except Exception:
            pass

    # Build sample (try to include reference if supported)
    if ground_truth:
        try:
            sample = SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=ground_truth
            )
        except Exception:
            sample = SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts
            )
    else:
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts
        )

    try:
        # RAGAS evaluate expects a Sample-like object (not necessarily a plain list),
        # pass the sample directly to match the installed ragas API.
        result = evaluate(sample, metrics=metrics)
        try:
            row = result.to_pandas().iloc[0].to_dict()
            return {k: float(v) for k, v in row.items() if isinstance(v, (int, float))}
        except Exception:
            out = {}
            for k, v in dict(result).items():
                if isinstance(v, (int, float)):
                    out[k] = float(v)
            return out if out else {"error": "Could not parse RAGAS output"}
    except Exception as e:
        # If RAGAS/SingleTurnSample compatibility fails (some versions expect
        # different sample shapes), return a helpful error plus sample questions
        # to show to the user so they can try alternate queries.
        samples = []
        try:
            import json, re
            samples_path = os.path.join('.vs', 'test_questions.json')
            if os.path.exists(samples_path):
                raw = open(samples_path, 'r', encoding='utf-8').read()
                raw_clean = re.sub(r'/\*.*?\*/', '', raw, flags=re.S).strip()
                parsed = json.loads(raw_clean)
                samples = [s.get('question') if isinstance(s, dict) else str(s) for s in parsed]
        except Exception:
            samples = []

        return {"error": f"RAGAS evaluation failed: {type(e).__name__}: {e}", "samples": samples}