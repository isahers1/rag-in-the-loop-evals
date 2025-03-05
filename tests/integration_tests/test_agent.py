import pytest
from langsmith import testing as t
from agent import graph
from openevals.llm import create_llm_as_judge
from openevals.prompts import HALLUCINATION_PROMPT, CONCISENESS_PROMPT
from agentevals.langgraph.wrappers.reflection import wrap_graph_with_reflection

hallucination_evaluator = create_llm_as_judge(
    prompt=HALLUCINATION_PROMPT,
    feedback_key="correctness",
    model="openai:o3-mini",
)

conciseness_evaluator = create_llm_as_judge(
    prompt=CONCISENESS_PROMPT,
    feedback_key="conciseness",
    model="openai:o3-mini"
)


@pytest.mark.langsmith(test_suite_name="My-test-dataset")
@pytest.mark.parametrize(
    "question, model",
    [
        ("How many French Indochinese civilians are estimated to have died due to war-related famine and disease?", "mini"),
        ("How many French Indochinese civilians are estimated to have died due to war-related famine and disease?", "4o"),
    ],
)
def test_hallucination(question, model):
    config = {"configurable": {"model": model, "do_eval": False}}
    response = graph.invoke({"question": question}, config=config)
    t.log_inputs({"question": question, "model": model})
    t.log_outputs({"answer": response['messages'][-1].content})

    eval_response = hallucination_evaluator(
        inputs=question,
        outputs=response['messages'][-1].content,
        context=response['context']
    )
    assert eval_response['score'] == 1

@pytest.mark.langsmith(test_suite_name="Test-reflection")
@pytest.mark.parametrize(
    "question",
    [
        ("What was the approximate ratio of Soviet to German military deaths on the Eastern Front?")
    ]
)
def test_reflection(question):
    config = {"configurable": {"model": "mini", "do_eval": False}}
    graph_with_reflection = wrap_graph_with_reflection(graph=graph, evaluator=conciseness_evaluator)
    normal_answer = graph.invoke({"question": question}, config=config)['messages'][-1].content
    answer_with_reflection = graph_with_reflection.invoke({"question": question}, config=config)['messages'][-1].content

    assert len(answer_with_reflection) <= len(normal_answer)
