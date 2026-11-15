import json, pytest
from gemmaloop.model.layer_plan import make_layer_plan, schedule

def cfg(name): return json.load(open(f"tests/fixtures/{name}_text_config.json"))

def test_e2b_plan():
    p = make_layer_plan(cfg("e2b"))
    assert (p.num_layers, p.period, p.first_shared) == (35, 5, 15)
    assert p.full_layers == (4, 9, 14, 19, 24, 29, 34)
    assert (p.prelude, p.loop, p.coda) == (range(0, 15), range(15, 30), range(30, 35))

def test_e4b_plan():
    p = make_layer_plan(cfg("e4b"))
    assert (p.num_layers, p.period, p.first_shared) == (42, 6, 24)
    assert p.full_layers == (5, 11, 17, 23, 29, 35, 41)
    assert (p.prelude, p.loop, p.coda) == (range(0, 24), range(24, 36), range(36, 42))

def test_schedule_lengths():
    assert len(schedule(make_layer_plan(cfg("e2b")), 2)) == 50
    assert len(schedule(make_layer_plan(cfg("e4b")), 2)) == 54
    p = make_layer_plan(cfg("e2b"))
    assert schedule(p, 1) == list(range(35))

def test_loop_must_be_whole_periods():
    c = cfg("e2b"); c["num_kv_shared_layers"] = 18   # first_shared=17, no alineado
    with pytest.raises(ValueError, match="period"):
        make_layer_plan(c)

def test_accepts_transformers_to_dict():
    from transformers import Gemma4TextConfig
    d = Gemma4TextConfig(**cfg("e2b")).to_dict()
    p = make_layer_plan(d)
    assert (p.num_layers, p.period, p.first_shared) == (35, 5, 15)
    assert (p.prelude, p.loop, p.coda) == (range(0, 15), range(15, 30), range(30, 35))
