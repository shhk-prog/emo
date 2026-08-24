import pytest
from affective_empathy_eval.schemas import parse_affective_state

def test_parse_valid_json():
    text = '{"valence": 0.5, "arousal": -0.5}'
    result = parse_affective_state(text)
    assert result["valence"] == 0.5
    assert result["arousal"] == -0.5

def test_parse_invalid_json():
    text = '{"valence": 0.5, "arousal": '
    with pytest.raises(ValueError):
        parse_affective_state(text)

def test_parse_out_of_bounds():
    text = '{"valence": 1.5, "arousal": 0.0}'
    with pytest.raises(ValueError):
        parse_affective_state(text)
