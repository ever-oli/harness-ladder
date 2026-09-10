from harness_ladder.fewshot import pack_few_shot_messages
from harness_ladder.model import clean_completion

def test_p1_examples_are_explicit_role_turns_and_short_answers():
    messages = pack_few_shot_messages(category="math", max_examples=1)
    assert [(m.role, m.content) for m in messages] == [("user", "Calculate 3 + 4. Give only the final number."), ("assistant", "7")]
    assert all("Reply with exactly" not in m.content for m in messages)
    long_msgs = pack_few_shot_messages(category="long_horizon", max_examples=1)
    assert long_msgs[1].content == "Z"
    assert "DONE" not in long_msgs[0].content

def test_completion_cleanup_removes_thinking_and_special_markup():
    assert clean_completion("<think>work</think>\n\nOK<|endoftext|>") == "OK"
    assert clean_completion("<think>compute 2 + 2\n4") == "4"
