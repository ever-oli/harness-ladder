from harness_ladder.model import TransformersLLM

def test_minicpm_inputs_drop_token_type_ids():
    inputs = {"input_ids": [1], "attention_mask": [1], "token_type_ids": [0]}
    TransformersLLM.clean_generation_inputs(inputs)
    assert "token_type_ids" not in inputs
