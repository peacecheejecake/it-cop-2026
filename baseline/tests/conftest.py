from pathlib import Path
import pytest
from riskbench.demo import make_demo
from riskbench.data import split_public, internal_partition
from riskbench.views import prepare_view

@pytest.fixture
def data(tmp_path):
    raw=make_demo(str(tmp_path/'raw'))
    split_public(raw['public'],str(tmp_path/'split'),'2024-03-01T00:00:00Z','2024-04-01T00:00:00Z')
    internal_partition(raw['internal'],str(tmp_path/'internal_raw'))
    for role,source in [('train',tmp_path/'split/train'),('valid',tmp_path/'split/valid'),('test',tmp_path/'split/test'),('internal',tmp_path/'internal_raw')]:
        prepare_view(str(source),str(tmp_path/'views'/role),'demo-byte',128,24)
    return {'root':tmp_path,'raw':raw,**{r:str(tmp_path/'views'/r) for r in ('train','valid','test','internal')}}
