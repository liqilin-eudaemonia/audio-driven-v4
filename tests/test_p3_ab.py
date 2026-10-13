import copy,sys
from pathlib import Path
import pytest,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pipeline3_ab_policy import comparable,decide,summarize,INDICES,plateau
from omni_av_p3_online.checkpoint import configure_connector
from test_p3_online import make,inputs


def test_fresh_initializer_preserves_planner_and_learns():
    m,prepared=make('C0',opened=False);before={k:v.clone() for k,v in m.state_dict().items()}
    configure_connector({'mode':'fresh'},m)
    for k,v in m.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    layout,target=inputs();opt=torch.optim.AdamW(m.optimizer_groups(.001,.001,.001,.001))
    m(layout,target,prepared,7,1)['loss'].backward();opt.step()
    assert m.connector.video[-1].weight.any() and m.connector.audio[-1].weight.any()
    for k,v in m.state_dict().items():
        if k.startswith('planner_parameters.'):torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    with pytest.raises(ValueError):configure_connector({'mode':'fresh'},m)
    with pytest.raises(ValueError):configure_connector({'mode':'fresh','path':'wrong'},m)


def summary(v=1):return dict(values=dict(video=v,audio=v,cer=v,wer=v),sync={i:dict(distance=v,confidence=1) for i in INDICES},bad_audio=[])

def test_no_early_or_missing_evidence_selection():
    assert decide(100,summary(),summary(.2))['action']=='continue'
    assert decide(300,summary(),summary(.995))['action']=='continue'
    assert decide(500,summary(),summary(.995))['winner']=='warm'
    assert decide(300,summary(),summary(.8))['winner']=='fresh'
    a,b=summary(),summary(.8);a['bad_audio']=[4];b['bad_audio']=[6]
    assert decide(500,a,b)['action']=='continue'
    with pytest.raises(ValueError):summarize({'rows':[]},{})


def test_recipe_only_differs_in_initialization_and_plateau():
    a=dict(seed=4,world_size=12,accumulation=6,connector_initialization={'mode':'warm'})
    b=copy.deepcopy(a);b['connector_initialization']={'mode':'fresh'};assert comparable(a,b)
    b['accumulation']=3;assert not comparable(a,b)
    assert plateau([1,.999,.998]);assert not plateau([1,.99,.98])
